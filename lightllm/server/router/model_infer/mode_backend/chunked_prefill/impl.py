import torch
import time
from typing import List, Optional, Callable, Dict, Any
from queue import Queue
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventPack
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend.pre import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import (
    prepare_mtp_prefill_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.common.basemodel.batch_objs import ModelOutput, ModelInput
from lightllm.common.basemodel.triton_kernel.gather_token_id import scatter_token
from lightllm.common.basemodel.triton_kernel.mtp_utils import (
    mtp_scatter_next_token_ids,
)
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.envs_utils import get_env_start_args
from .control_state import ControlState

logger = init_logger(__name__)


def _use_mock_prefill() -> bool:
    """Check if mock prefill mode is enabled via environment variable."""
    import os
    return os.environ.get("MOCK_PREFILL_LOGITS", "").lower() == "true"


class ChunkedPrefillBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

        # 用于控制每一步是执行prefill 和 decode 还是跳过
        self.control_state_machine = ControlState()

        # Mock prefill buffer for performance testing (avoids torch.randn overhead)
        self._mock_logit_buffer = None
        self._mock_kv_buffer = None

        # 在 mtp 模式下切换绑定的prefill 和 decode 函数
        logger.debug(f"MTP mode: {get_env_start_args().mtp_mode}")
        if get_env_start_args().mtp_mode:
            self.prefill = self.prefill_mtp
            self.decode = self.decode_mtp
            self.is_mtp_eagle = get_env_start_args().mtp_mode == "deepseekv3_eagle"
            self.num_mtp_models = 1 if self.is_mtp_eagle else get_env_start_args().mtp_step
            self._draft_decode_func = self._draft_decode_eagle if self.is_mtp_eagle else self._draft_decode_vanilla
        else:
            self.prefill = self.prefill_normal
            self.decode = self.decode_normal

        self.classed_req_strict_prefill = False
        return

    def _sync_batched_lora_state_for_current_reqs(self, current_reqs: List[InferReq]) -> None:
        if not (getattr(self, "lora_support", False) and getattr(self, "use_batched_lora_mode", False)):
            return

        req_bins = None
        enable_detached_lora = False
        if current_reqs:
            from lightllm.server.router.model_infer.infer_batch import Batch

            batch = Batch(current_reqs)
            enable_detached_lora = batch.has_lora_adapters()
            if enable_detached_lora:
                req_bins = self._prepare_batched_lora_for_batch(batch)

        # Disabled: don't switch to single_adapter_mode when there are no LoRA adapters in batch
        # if not enable_detached_lora:
        #     for dispatcher in getattr(self, "lora_dispatchers", []):
        #         switch_mode = getattr(dispatcher, "use_single_adapter_mode", None)
        #         if callable(switch_mode):
        #             switch_mode()

        for layer_infer in self.model.layers_infer:
            layer_infer.use_detached_lora_ = enable_detached_lora
            if hasattr(layer_infer, "set_req_bins"):
                layer_infer.set_req_bins(req_bins)

    def infer_loop(self):
        torch.cuda.set_device(get_current_device_id())
        try:
            while True:
                event_pack = self.overlap_event_manager.get_overlap_event_pack()
                # 关闭overlap 模式
                if not self.support_overlap:
                    event_pack._close_overlap()

                event_pack.wait_to_forward()

                self._try_read_new_reqs()

                # =================================================================
                # S-LoRA Batched LoRA Mode
                # Always use batched mode to support multiple adapters in a single batch
                # =================================================================
                if getattr(self, 'lora_support', False) and getattr(self, 'use_batched_lora_mode', False):
                    current_reqs = list(g_infer_context.requests_mapping.values())
                    self._sync_batched_lora_state_for_current_reqs(current_reqs)
                # =================================================================

                prefill_reqs, decode_reqs = self._get_classed_reqs(
                    no_decode=self.classed_req_no_decode,
                    strict_prefill=self.classed_req_strict_prefill,
                    recover_paused=self.control_state_machine.try_recover_paused_reqs(),
                )

                run_way = self.control_state_machine.select_run_way(prefill_reqs=prefill_reqs, decode_reqs=decode_reqs)

                if run_way.is_prefill():
                    # 进行一次流同步，保证 _try_read_new_reqs 中的一些算子操作，必然已经完成。
                    # 防止后续的推理流程读取到显存中可能存在错误的数据。
                    g_infer_context.get_overlap_stream().wait_stream(torch.cuda.current_stream())
                    self.prefill(
                        event_pack=event_pack,
                        prefill_reqs=prefill_reqs,
                    )
                    continue
                elif run_way.is_decode():
                    # 进行一次流同步，保证 _try_read_new_reqs 中的一些算子操作，必然已经完成。
                    # 防止后续的推理流程读取到显存中可能存在错误的数据。
                    g_infer_context.get_overlap_stream().wait_stream(torch.cuda.current_stream())
                    self.decode(
                        event_pack=event_pack,
                        decode_reqs=decode_reqs,
                    )
                    continue
                elif run_way.is_pass():
                    event_pack.notify_post_handle_and_wait_pre_post_handle()
                    event_pack.notify_forward_and_wait_post_handle()
                    event_pack.notify_pre_post_handle()
                    time.sleep(0.02)
                    continue

        except BaseException as e:
            self.logger.exception(str(e))
            raise e

    def prefill_normal(
        self,
        event_pack: OverlapEventPack,
        prefill_reqs: List[InferReq],
    ):
        # 第一阶段: 模型推理
        model_input, run_reqs = prepare_prefill_inputs(
            prefill_reqs, is_chuncked_mode=not self.disable_chunked_prefill, is_multimodal=self.is_multimodal
        )

        # Mock prefill mode for performance testing (PD disaggregation)
        use_mock = _use_mock_prefill()
        if use_mock:
            logger.debug(f"MOCK MODE ACTIVE: skipping model.forward for {len(prefill_reqs)} requests")
            # Force greedy sampling to avoid _top_p_top_k issues in chunked prefill
            for req in run_reqs:
                req.sampling_param.shm_param.top_k = 1
            mock_logits = self._get_mock_logits(model_input.total_token_num)
            # In chunked prefill mode, logits has shape [total_tokens, vocab_size]
            # where total_tokens is the sum of all sequence lengths.
            # The sample function processes each token's logits individually.
            # We slice to get exactly the needed shape.
            mock_logits = mock_logits[:model_input.total_token_num]
        else:
            mock_logits = None
            logger.debug(f"Don't use mock prefill logits.")

        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            if use_mock:
                # Minimal mock: compute greedy samples and copy to CPU
                model_output = None
                # Greedy sampling: argmax over vocabulary
                next_token_ids = torch.argmax(mock_logits, dim=-1)  # [total_tokens]
                next_token_probs = torch.softmax(mock_logits, dim=-1)
                next_token_probs = torch.gather(next_token_probs, dim=-1, index=next_token_ids.unsqueeze(-1)).squeeze(-1)
                next_token_logprobs = torch.log(next_token_probs)

                # Move to CPU for downstream code
                next_token_ids_cpu = next_token_ids.cpu()
                next_token_logprobs_cpu = next_token_logprobs.cpu()

                # Move b_req_idx for downstream updates
                b_req_idx = model_input.b_req_idx.cuda()
                b_mtp_index = model_input.b_mtp_index.cuda()

                # Manually update req_to_next_token_ids (only for last token of each request in prefill)
                req_to_next = self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids
                for i, req in enumerate(run_reqs):
                    req_to_next[req.req_idx, 0] = next_token_ids_cpu[-1]  # Last token
            else:
                model_output = self.model.forward(model_input)
                b_req_idx = model_input.b_req_idx
                b_mtp_index = model_input.b_mtp_index
                _, next_token_ids_cpu, next_token_logprobs_cpu = self._sample_and_scatter_token(
                    logits=model_output.logits,
                    b_req_idx=b_req_idx,
                    b_mtp_index=b_mtp_index,
                    run_reqs=run_reqs,
                    is_prefill=True,
                    b_prefill_has_output_cpu=model_input.b_prefill_has_output_cpu,
                    mask_func=self.prefill_mask_func,
                )
            sync_event = torch.cuda.Event()
            sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()
        self._post_handle(
            run_reqs=run_reqs,
            next_token_ids=next_token_ids_cpu,
            next_token_logprobs=next_token_logprobs_cpu,
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
            nixl_prefill_chuncked_handle_func=self.nixl_prefill_chuncked_handle_func,
        )
        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def decode_normal(
        self,
        event_pack: OverlapEventPack,
        decode_reqs: List[InferReq],
    ):
        decode_step_id = self._alloc_decode_step_id()

        # Split normal decode requests from COLaRA continuation requests
        normal_decode_reqs = []
        continuation_reqs = []
        for req in decode_reqs:
            if hasattr(req, 'colora_continuation') and req.colora_continuation is not None:
                if req.colora_continuation.completed:
                    continuation_reqs.append(req)
                else:
                    # Still waiting for CPU completion, skip this step
                    pass
            else:
                normal_decode_reqs.append(req)

        run_reqs = []
        next_token_ids_cpu = []
        next_token_logprobs_cpu = []

        # Process normal decode requests
        if normal_decode_reqs:
            model_input, run_reqs_norm = prepare_decode_inputs(normal_decode_reqs, decode_step_id=decode_step_id)
            run_reqs.extend(run_reqs_norm)
            with torch.cuda.stream(g_infer_context.get_overlap_stream()):
                model_output = self.model.forward(model_input)
                _, nti_cpu, ntp_cpu = self._sample_and_scatter_token(
                    logits=model_output.logits,
                    b_req_idx=model_input.b_req_idx,
                    b_mtp_index=model_input.b_mtp_index,
                    run_reqs=run_reqs_norm,
                    is_prefill=False,
                    mask_func=self.decode_mask_func,
                )
                next_token_ids_cpu.extend(nti_cpu)
                next_token_logprobs_cpu.extend(ntp_cpu)
                sync_event = torch.cuda.Event()
                sync_event.record()

        # Process COLaRA continuation requests
        if continuation_reqs:
            from lightllm.common.basemodel.batch_objs import ModelInput

            # Build continuation model input
            batch_size = len(continuation_reqs)
            b_req_idx = []
            b_adapter_bin = []
            b_trace_req_id = []
            b_mtp_index = []
            b_seq_len = []
            resume_from_layer = []
            resumed_hidden = []
            mem_indexes_cpu = []
            multimodal_params = []

            for req in continuation_reqs:
                cont = req.colora_continuation
                b_req_idx.append(req.req_idx)
                from lightllm.server.router.model_infer.infer_batch import get_req_adapter_bin
                adapter_bin = get_req_adapter_bin(req)
                b_adapter_bin.append(adapter_bin)
                b_trace_req_id.append(req.req_id)
                b_mtp_index.append(0)
                seq_len = req.get_cur_total_len()
                b_seq_len.append(seq_len)
                resume_from_layer.append(cont.resume_layer)
                resumed_hidden.append(cont.saved_hidden)
                mem_indexes_cpu.append(cont.mem_index)
                multimodal_params.append(req.multimodal_params)

            assert len(resume_from_layer) == batch_size
            # All resumed hidden should have same resume_from_layer since continuation is after layer L, resume at L+1
            resume_from_layer = resume_from_layer[0]

            # Concatenate all resumed hiddens
            resumed_hidden = torch.cat(resumed_hidden, dim=0)
            b_req_idx = torch.tensor(b_req_idx, dtype=torch.int32, device='cpu')
            b_adapter_bin = torch.tensor(b_adapter_bin, dtype=torch.int32, device='cpu')
            b_trace_req_id = torch.tensor(b_trace_req_id, dtype=torch.int64, device='cpu')
            b_mtp_index = torch.tensor(b_mtp_index, dtype=torch.int32, device='cpu')
            b_seq_len = torch.tensor(b_seq_len, dtype=torch.int32, device='cpu')
            max_len_in_batch = max(b_seq_len)
            max_kv_seq_len = max(b_seq_len)
            max_q_seq_len = 1
            mem_indexes_cpu = torch.cat(mem_indexes_cpu, dim=0)

            # Build continuation model input
            cont_model_input = ModelInput(
                batch_size=batch_size,
                total_token_num=sum(b_seq_len),
                max_len_in_batch=max_len_in_batch,
                max_q_seq_len=max_q_seq_len,
                max_kv_seq_len=max_kv_seq_len,
                max_cache_len=max_len_in_batch,
                input_ids=None,
                mem_indexes_cpu=mem_indexes_cpu,
                b_req_idx=b_req_idx,
                b_adapter_bin=b_adapter_bin,
                b_trace_req_id=b_trace_req_id,
                b_mtp_index=b_mtp_index,
                b_seq_len=b_seq_len,
                is_prefill=False,
                decode_step_id=decode_step_id,
                is_continuation_batch=True,
                resume_from_layer=resume_from_layer,
                resumed_hidden=resumed_hidden,
            )
            cont_model_input.multimodal_params = multimodal_params

            with torch.cuda.stream(g_infer_context.get_overlap_stream()):
                cont_model_output = self.model.forward(cont_model_input)
                _, nti_cpu, ntp_cpu = self._sample_and_scatter_token(
                    logits=cont_model_output.logits,
                    b_req_idx=cont_model_input.b_req_idx,
                    b_mtp_index=cont_model_input.b_mtp_index,
                    run_reqs=continuation_reqs,
                    is_prefill=False,
                    mask_func=self.decode_mask_func,
                )
                # Clear the continuation since we're done with it
                for req in continuation_reqs:
                    req.colora_continuation = None
                next_token_ids_cpu.extend(nti_cpu)
                next_token_logprobs_cpu.extend(ntp_cpu)
                # sync_event is created only if we have normal decoding, else create it
                if not normal_decode_reqs:
                    sync_event = torch.cuda.Event()
                    sync_event.record()

        run_reqs.extend(continuation_reqs)

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=False)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        if normal_decode_reqs or continuation_reqs:
            sync_event.synchronize()
        self._post_handle(
            run_reqs=run_reqs,
            next_token_ids=next_token_ids_cpu,
            next_token_logprobs=next_token_logprobs_cpu,
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )

        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def prefill_mtp(
        self,
        event_pack: OverlapEventPack,
        prefill_reqs: List[InferReq],
    ):
        model_input, run_reqs = prepare_prefill_inputs(
            prefill_reqs, is_chuncked_mode=not self.disable_chunked_prefill, is_multimodal=self.is_multimodal
        )
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output = self.model.forward(model_input)
            next_token_ids, next_token_ids_cpu, next_token_logprobs_cpu = self._sample_and_scatter_token(
                logits=model_output.logits,
                b_req_idx=model_input.b_req_idx,
                b_mtp_index=model_input.b_mtp_index,
                run_reqs=run_reqs,
                is_prefill=True,
                b_prefill_has_output_cpu=model_input.b_prefill_has_output_cpu,
                mask_func=self.prefill_mask_func,
            )
            # mtp kv fill
            self._draft_prefill_forward(
                model_input=model_input, model_output=model_output, next_token_ids=next_token_ids
            )
            sync_event = torch.cuda.Event()
            sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()

        self._post_handle(
            run_reqs=run_reqs,
            next_token_ids=next_token_ids_cpu,
            next_token_logprobs=next_token_logprobs_cpu,
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
            nixl_prefill_chuncked_handle_func=self.nixl_prefill_chuncked_handle_func,
        )

        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def decode_mtp(
        self,
        event_pack: OverlapEventPack,
        decode_reqs: List[InferReq],
    ):
        """
        MTP解码的通用流程，整合eagle和vanilla的共同逻辑
        """
        decode_step_id = self._alloc_decode_step_id()
        model_input, run_reqs = prepare_decode_inputs(decode_reqs, decode_step_id=decode_step_id)

        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            b_mtp_index_cpu = model_input.b_mtp_index
            model_output = self.model.forward(model_input)
            next_token_ids, next_token_logprobs = sample(model_output.logits, run_reqs, self.eos_id)
            # verify the next_token_ids
            b_req_mtp_start_loc = [index for index, mtp_index in enumerate(b_mtp_index_cpu) if mtp_index == 0]
            b_req_mtp_start_loc = g_pin_mem_manager.gen_from_list(
                key="b_req_mtp_start_loc",
                data=b_req_mtp_start_loc,
                dtype=torch.int32,
            ).cuda(non_blocking=True)

            mtp_accept_len, accepted_index = self._verify_mtp_v2(
                new_next_token_ids=next_token_ids,
                b_req_idx=model_input.b_req_idx,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
            )
            accepted_index_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                key="accepted_index",
                gpu_tensor=accepted_index,
            )
            mtp_accept_len_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                key="mtp_accept_len",
                gpu_tensor=mtp_accept_len,
            )
            verify_event = torch.cuda.Event()
            verify_event.record()

            next_token_ids_cpu, next_token_logprobs_cpu = self._async_copy_next_token_infos_to_pin_mem(
                next_token_ids, next_token_logprobs
            )

            # 调用具体的draft decode函数
            additional_mem_indexes_cpu = self._draft_decode_func(
                main_model_input=model_input,
                main_model_output=model_output,
                next_token_ids=next_token_ids,
                mtp_accept_len=mtp_accept_len,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
            )

            g_infer_context.req_sampling_manager.update_reqs_out_token_counter_gpu(
                b_req_idx=model_input.b_req_idx,
                next_token_ids=next_token_ids,
                mask=accepted_index == 1,
            )
            sync_event = torch.cuda.Event()
            sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        verify_event.synchronize()
        verify_ok_reqs = [run_reqs[i] for i in range(len(run_reqs)) if accepted_index_cpu[i] == 1]
        update_packs = self._pre_post_handle(verify_ok_reqs, is_chuncked_mode=False)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()

        # 处理需要释放的内存索引
        need_free_mem_indexes = model_input.mem_indexes_cpu[accepted_index_cpu == 0]
        if additional_mem_indexes_cpu is not None:
            need_free_mem_indexes = torch.cat([need_free_mem_indexes, additional_mem_indexes_cpu], dim=0)

        self._update_mtp_accept_ratio(decode_reqs=decode_reqs, mtp_accept_len_cpu=mtp_accept_len_cpu)
        select_mask = torch.tensor(accepted_index_cpu, dtype=torch.bool, device="cpu")
        self._post_handle(
            run_reqs=verify_ok_reqs,
            next_token_ids=next_token_ids_cpu[select_mask],
            next_token_logprobs=next_token_logprobs_cpu[select_mask],
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )

        if len(need_free_mem_indexes) > 0:
            g_infer_state_lock.acquire()
            g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)
            g_infer_state_lock.release()

        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def _draft_prefill_forward(self, model_input: ModelInput, model_output: ModelOutput, next_token_ids: torch.Tensor):
        # spec prefill: MTP, 这个地方只是为了填充draft model的 kv， 并不会使用生成的token_id。
        draft_model_input = model_input
        draft_model_output = model_output
        draft_next_token_ids_gpu = next_token_ids
        for draft_model_idx in range(self.num_mtp_models):
            draft_model_input = prepare_mtp_prefill_inputs(
                model_input=draft_model_input,
                b_next_token_ids=draft_next_token_ids_gpu,
                deepseekv3_mtp_draft_input_hiddens=draft_model_output.deepseekv3_mtp_main_output_hiddens,
            )
            draft_model_output = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids_gpu = self._gen_argmax_token_ids(draft_model_output)
        return

    def _draft_decode_vanilla(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        mtp_accept_len: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
    ):
        # share some inference info with the main model
        draft_model_input = main_model_input
        draft_model_output = main_model_output
        draft_next_token_ids = next_token_ids
        all_next_token_ids = []
        all_next_token_ids.append(next_token_ids)
        # process the draft model output
        for draft_model_idx in range(self.mtp_step):

            draft_model_input.input_ids = draft_next_token_ids
            draft_model_input.deepseekv3_mtp_draft_input_hiddens = draft_model_output.deepseekv3_mtp_main_output_hiddens
            # spec decode: MTP
            draft_model_output: ModelOutput = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids = self._gen_argmax_token_ids(draft_model_output)
            all_next_token_ids.append(draft_next_token_ids)

        all_next_token_ids = torch.stack(all_next_token_ids, dim=1)  # [batch_size, mtp_step + 1]

        mtp_scatter_next_token_ids(
            req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=main_model_input.b_req_idx,
            mtp_accept_len=mtp_accept_len,
        )
        return None

    def _draft_decode_eagle(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        mtp_accept_len: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
    ):
        batch_size = main_model_input.batch_size
        num_reqs = batch_size // (self.mtp_step + 1)
        g_infer_state_lock.acquire()
        if g_infer_context.radix_cache is not None:
            g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(num_reqs * self.mtp_step)
        eagle_mem_indexes_cpu = g_infer_context.req_manager.mem_manager.alloc(num_reqs * self.mtp_step)
        g_infer_state_lock.release()
        eagle_mem_indexes = eagle_mem_indexes_cpu.cuda(non_blocking=True)

        # share some inference info with the main model
        draft_model_input = main_model_input
        draft_model_output = main_model_output
        draft_next_token_ids = next_token_ids
        all_next_token_ids = []
        all_next_token_ids.append(next_token_ids)
        # process the draft model output
        for _step in range(self.mtp_step):

            draft_model_input.input_ids = draft_next_token_ids
            draft_model_input.deepseekv3_mtp_draft_input_hiddens = draft_model_output.deepseekv3_mtp_main_output_hiddens
            # spec decode: MTP
            draft_model_idx = _step % self.num_mtp_models
            draft_model_output: ModelOutput = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids = self._gen_argmax_token_ids(draft_model_output)
            draft_model_input.b_seq_len += 1
            draft_model_input.max_len_in_batch += 1
            eagle_mem_indexes_i = eagle_mem_indexes[_step * num_reqs : (_step + 1) * num_reqs]
            draft_model_input.mem_indexes = torch.cat(
                [draft_model_input.mem_indexes.view(-1, self.mtp_step + 1)[:, 1:], eagle_mem_indexes_i.view(-1, 1)],
                dim=1,
            ).view(-1)
            all_next_token_ids.append(draft_next_token_ids)

        all_next_token_ids = torch.stack(all_next_token_ids, dim=1)  # [batch_size, mtp_step + 1]

        mtp_scatter_next_token_ids(
            req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=main_model_input.b_req_idx,
            mtp_accept_len=mtp_accept_len,
        )
        return eagle_mem_indexes_cpu

    def _get_mock_logits(self, total_tokens: int) -> torch.Tensor:
        """
        Get a mock logits tensor for prefill performance testing.

        Uses a pre-allocated buffer to avoid torch.randn overhead.
        The buffer is reused across calls to minimize memory allocation.

        Note: This skips actual model forward, so KV cache will be uninitialized.
        For PD mode testing, this is acceptable if you're only testing:
        - Control plane logic
        - Scheduling decisions
        - Network transfer (KV will be garbage but transfer speed is measured)

        For accurate decode performance testing, ensure requests are properly
        initialized with valid KV cache before entering decode stage.

        Returns:
            torch.Tensor: Logits tensor of shape [total_tokens, vocab_size]
        """
        vocab_size = getattr(self.model, "vocab_size", 32000)

        # Initialize attribute if it doesn't exist (safety check)
        if not hasattr(self, "_mock_logit_buffer"):
            self._mock_logit_buffer = None

        # Check if resize is needed
        if self._mock_logit_buffer is None or self._mock_logit_buffer.shape[0] < total_tokens:
            # SAFETY: Lower default to ~2GB (32k tokens) to prevent OOM
            # If total_tokens is huge, allocate exactly what's needed + 10% padding
            safe_default = 32768
            alloc_size = max(total_tokens + 1024, safe_default)

            self._mock_logit_buffer = torch.zeros(
                (alloc_size, vocab_size),
                dtype=torch.float16,
                device="cuda"
            )
            # Deterministic hot-spots to stabilize greedy/top-k sampling
            # Token 100 will be the dominant token (exp(5) >> exp(0))
            self._mock_logit_buffer[:, 100] = 5.0
            self._mock_logit_buffer[:, 1000] = 3.0

        # Return a zero-copy view slice
        return self._mock_logit_buffer[:total_tokens]
