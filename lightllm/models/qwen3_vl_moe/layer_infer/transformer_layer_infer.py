import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
import os
import logging
from functools import partial
from typing import Tuple, Optional, Dict, Any
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.qwen2_vl.triton_kernel.mrope import mrope_triton_fused
from lightllm.models.qwen3_moe.layer_infer.transformer_layer_infer import Qwen3MOETransformerLayerInfer
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo
from lightllm.models.qwen3.triton_kernel.qk_norm import qk_rmsnorm_forward
from lightllm.distributed import all_reduce
from lightllm.utils.dist_utils import get_global_world_size
from lightllm.models.qwen3_vl.triton_kernel.deepstack_multimodal_emb import apply_deepstack_features
from lightllm.models.qwen3_vl_moe.lora_dispatch import SpecJobKey
from lightllm.utils.nvtx_utils import NvtxAnnotate

# Configure logging using global env var
_LOG_LEVEL = os.environ.get("LIGHTLLM_LOGGING", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL, logging.INFO)
logger = logging.getLogger("lightllm.lora.infer")
logger.setLevel(_LOG_LEVEL)


def _parse_spec_submit_layer_whitelist(raw: str):
    whitelist = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            whitelist.add(int(token))
        except ValueError:
            logger.warning("[COLoRA][SpecSubmit] Ignore invalid whitelist token '%s'", token)
    return frozenset(whitelist)


class Qwen3VLMOETransformerLayerInfer(Qwen3MOETransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        self.mrope_section = torch.tensor(
            network_config["rope_scaling"]["mrope_section"], dtype=torch.int32, device="cuda"
        )
        # lora_dispatcher_ and use_detached_lora_ are inherited from Qwen3MOETransformerLayerInfer
        # and set by base_backend.py via set_lora_dispatcher()
        # S-LoRA batched mode support
        self.req_bins_ = None  # Per-request adapter indices for batched LoRA
        self._spec_submit_enabled = os.environ.get("COLORA_SPEC_SUBMIT_ENABLE", "0") == "1"
        self._spec_submit_layer_whitelist = _parse_spec_submit_layer_whitelist(
            os.environ.get("COLORA_SPEC_SUBMIT_LAYER_WHITELIST", "")
        )
        self._spec_submit_layer_enabled = (
            self._spec_submit_enabled and self.layer_num_ in self._spec_submit_layer_whitelist
        )
        self._spec_submit_totals = {"submitted": 0, "skipped": 0, "rejected": 0}
        self._spec_bind_totals = {
            "attempted_bind": 0,
            "successful_bind": 0,
            "stale": 0,
            "not_ready": 0,
            "fallback": 0,
        }
        self._spec_bound_job_keys_current_call = set()
        self._spec_prev_decode_step_id: Optional[int] = None
        self._spec_prev_top1_by_req: Dict[int, Tuple[int, int]] = {}

    def _bind_ffn(self):
        super()._bind_ffn()
        if self.is_moe and os.environ.get("MOE_MODE", "TP") != "EP" and self._spec_submit_layer_enabled:
            self._ffn = partial(Qwen3VLMOETransformerLayerInfer._moe_ffn, self)

    def set_lora_dispatcher(self, dispatcher: Any, use_detached_lora: bool = True):
        super().set_lora_dispatcher(dispatcher, use_detached_lora)
        self._reset_spec_submit_runtime()

    def clear_lora_dispatcher(self):
        self._reset_spec_submit_runtime()
        super().clear_lora_dispatcher()

    def set_req_bins(self, req_bins: torch.Tensor):
        """Set the req_bins tensor for batched mode.

        req_bins[i] contains the adapter index for request i in the batch.

        Debug:
            Logs the req_bins configuration
        """
        self.req_bins_ = req_bins

    def _reset_spec_submit_runtime(self) -> None:
        self._spec_bound_job_keys_current_call = set()
        self._spec_prev_decode_step_id = None
        self._spec_prev_top1_by_req = {}

    def _should_enable_spec_submit(self, infer_state: Optional[Qwen3VLInferStateInfo]) -> bool:
        if not self._spec_submit_layer_enabled:
            return False
        if infer_state is None or getattr(infer_state, "is_prefill", True):
            return False
        if getattr(infer_state, "decode_step_id", None) is None:
            return False
        if not (self.use_detached_lora_ and self.lora_dispatcher_ is not None):
            return False
        if os.environ.get("MOE_MODE", "TP") != "TP":
            return False

        should_hybrid = getattr(self.lora_dispatcher_, "_should_use_hybrid_moe_compute", None)
        if not callable(should_hybrid) or not bool(should_hybrid()):
            return False
        return getattr(self.lora_dispatcher_, "expert_cache_manager", None) is not None

    def _record_spec_submit_outcome(self, status: str) -> None:
        if status in self._spec_submit_totals:
            self._spec_submit_totals[status] += 1

    def _record_spec_bind_counter(self, counter_name: str, colora_stats: Optional[Dict[str, float]] = None) -> None:
        if colora_stats is not None and counter_name in colora_stats:
            colora_stats[counter_name] += 1
        if counter_name in self._spec_bind_totals:
            self._spec_bind_totals[counter_name] += 1

    def _should_enable_spec_bind(self, infer_state: Optional[Qwen3VLInferStateInfo]) -> bool:
        if not self._should_enable_spec_submit(infer_state):
            return False
        return callable(getattr(self.lora_dispatcher_, "try_bind_gate_up_job_with_status", None))

    def _maybe_submit_decode_spec_gate_up(
        self,
        hidden_states: torch.Tensor,
        infer_state: Qwen3VLInferStateInfo,
    ) -> None:
        if not self._should_enable_spec_submit(infer_state):
            return

        step_id = int(infer_state.decode_step_id)
        begin_spec_step = getattr(self.lora_dispatcher_, "begin_spec_step", None)
        if callable(begin_spec_step):
            begin_spec_step(step_id)

        req_idx_tensor = getattr(infer_state, "b_req_idx", None)
        adapter_bin_tensor = getattr(infer_state, "b_adapter_bin", None)
        if req_idx_tensor is None or adapter_bin_tensor is None:
            return

        num_tokens = int(hidden_states.shape[0])
        if req_idx_tensor.shape[0] < num_tokens or adapter_bin_tensor.shape[0] < num_tokens:
            return

        if self._spec_prev_decode_step_id is None or int(self._spec_prev_decode_step_id) + 1 != step_id:
            return

        req_idx_cpu = req_idx_tensor[:num_tokens].detach().to(device="cpu", dtype=torch.int64).view(-1).tolist()
        adapter_bins_cpu = (
            adapter_bin_tensor[:num_tokens].detach().to(device="cpu", dtype=torch.int64).view(-1).tolist()
        )

        predicted_groups: Dict[Tuple[int, int], list[Tuple[int, int]]] = {}
        for row_idx, (req_idx, adapter_bin) in enumerate(zip(req_idx_cpu, adapter_bins_cpu)):
            if int(adapter_bin) < 0:
                continue
            prev_joint = self._spec_prev_top1_by_req.get(int(req_idx))
            if prev_joint is None:
                continue
            prev_adapter_bin, prev_expert_id = prev_joint
            if int(prev_adapter_bin) != int(adapter_bin) or int(prev_expert_id) < 0:
                continue
            predicted_groups.setdefault((int(adapter_bin), int(prev_expert_id)), []).append((int(row_idx), int(req_idx)))

        if not predicted_groups:
            return

        step_counts = {"submitted": 0, "skipped": 0, "rejected": 0}
        reason_counts: Dict[str, int] = {}
        active_adapter_bins = adapter_bin_tensor[:num_tokens]
        submit_fn = getattr(self.lora_dispatcher_, "maybe_submit_fused_gate_up_spec_job", None)
        if not callable(submit_fn):
            return

        for (adapter_bin, expert_id), rows in predicted_groups.items():
            row_indices = torch.tensor([row_idx for row_idx, _ in rows], dtype=torch.long, device=hidden_states.device)
            key = SpecJobKey(
                layer_id=int(self.layer_num_),
                decode_step_id=step_id,
                op_kind="gate_up",
                adapter_bin=int(adapter_bin),
                expert_id=int(expert_id),
                row_group_sig=tuple((int(row_idx), int(req_idx)) for row_idx, req_idx in rows),
            )
            outcome = submit_fn(
                key=key,
                input_tensor=hidden_states,
                req_bins=active_adapter_bins,
                row_indices=row_indices,
            )
            step_counts[outcome.status] += 1
            reason_counts[outcome.reason] = reason_counts.get(outcome.reason, 0) + 1
            self._record_spec_submit_outcome(outcome.status)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[COLoRA][SpecSubmit] layer=%s decode_step_id=%s predicted_jobs=%s submitted=%s skipped=%s rejected=%s reasons=%s totals=%s",
                self.layer_num_,
                step_id,
                len(predicted_groups),
                step_counts["submitted"],
                step_counts["skipped"],
                step_counts["rejected"],
                reason_counts,
                self._spec_submit_totals,
            )

    def _finalize_decode_spec_step_nonblocking(self, infer_state: Optional[Qwen3VLInferStateInfo]) -> None:
        if not self._should_enable_spec_submit(infer_state):
            return
        finalize_fn = getattr(self.lora_dispatcher_, "finalize_spec_step_nonblocking", None)
        if not callable(finalize_fn):
            return
        finalize_fn(int(infer_state.decode_step_id))

    def _eager_retire_remaining_decode_spec_jobs(self, infer_state: Optional[Qwen3VLInferStateInfo]) -> None:
        if not self._should_enable_spec_submit(infer_state):
            return
        retire_fn = getattr(self.lora_dispatcher_, "retire_unbound_gate_up_jobs", None)
        if not callable(retire_fn):
            return
        retire_fn(
            layer_id=int(self.layer_num_),
            decode_step_id=int(infer_state.decode_step_id),
            keep_keys=set(getattr(self, "_spec_bound_job_keys_current_call", set())),
        )

    def _maybe_bind_fused_gate_up_exact(
        self,
        expert_input: torch.Tensor,
        infer_state: Qwen3VLInferStateInfo,
        layer_weight: Qwen3MOETransformerLayerWeight,
        expert_id: int,
        batch_indices: torch.Tensor,
        expert_req_bins: Optional[torch.Tensor],
        pack_meta: Optional[Dict[str, torch.Tensor]],
        colora_stats: Dict[str, float],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Late-bind only exact packed adapter groups for ready fused gate+up speculative jobs."""
        if not self._should_enable_spec_bind(infer_state):
            return None
        if pack_meta is None or expert_req_bins is None:
            return None

        bind_fn = getattr(self.lora_dispatcher_, "try_bind_gate_up_job_with_status", None)
        req_idx_tensor = getattr(infer_state, "b_req_idx", None)
        if not callable(bind_fn) or req_idx_tensor is None:
            return None

        packed_input = pack_meta.get("packed_activations")
        packed_bins = pack_meta.get("packed_bins")
        token_positions = pack_meta.get("token_positions")
        adapter_ids = pack_meta.get("adapter_ids")
        adapter_offsets = pack_meta.get("adapter_offsets")
        if any(item is None for item in (packed_input, packed_bins, token_positions, adapter_ids, adapter_offsets)):
            return None
        if int(packed_bins.numel()) <= 0 or int(adapter_ids.numel()) <= 0:
            return None

        step_counts = {
            "attempted_bind": 0,
            "successful_bind": 0,
            "stale": 0,
            "not_ready": 0,
            "fallback": 0,
        }
        bound_segments = []
        fallback_spans = []

        for group_idx in range(int(adapter_ids.numel())):
            start = int(adapter_offsets[group_idx].item())
            end = int(adapter_offsets[group_idx + 1].item())
            if end <= start:
                continue

            adapter_bin = int(adapter_ids[group_idx].item())
            if adapter_bin < 0:
                fallback_spans.append((start, end))
                continue

            group_token_positions = token_positions[start:end].to(device=batch_indices.device, dtype=torch.long)
            group_batch_indices = batch_indices.index_select(0, group_token_positions)
            if int(group_batch_indices.numel()) <= 0:
                fallback_spans.append((start, end))
                continue

            group_req_ids = req_idx_tensor.index_select(
                0,
                group_batch_indices.to(device=req_idx_tensor.device, dtype=torch.long),
            )
            sort_order_rows = torch.argsort(group_batch_indices)
            sort_order_reqs = sort_order_rows.to(device=group_req_ids.device, dtype=torch.long)
            sorted_rows = group_batch_indices.index_select(0, sort_order_rows)
            sorted_req_ids = group_req_ids.index_select(0, sort_order_reqs)
            row_group_sig = tuple(
                zip(
                    sorted_rows.detach().to(device="cpu", dtype=torch.int64).tolist(),
                    sorted_req_ids.detach().to(device="cpu", dtype=torch.int64).tolist(),
                )
            )
            if not row_group_sig:
                fallback_spans.append((start, end))
                continue

            step_counts["attempted_bind"] += 1
            self._record_spec_bind_counter("attempted_bind", colora_stats)
            outcome = bind_fn(
                SpecJobKey(
                    layer_id=int(self.layer_num_),
                    decode_step_id=int(infer_state.decode_step_id),
                    op_kind="gate_up",
                    adapter_bin=adapter_bin,
                    expert_id=int(expert_id),
                    row_group_sig=tuple((int(row_idx), int(req_idx)) for row_idx, req_idx in row_group_sig),
                )
            )

            if outcome.status != "bound" or outcome.result is None:
                if outcome.status == "stale":
                    step_counts["stale"] += 1
                    self._record_spec_bind_counter("stale", colora_stats)
                elif outcome.status == "not_ready":
                    step_counts["not_ready"] += 1
                    self._record_spec_bind_counter("not_ready", colora_stats)
                step_counts["fallback"] += 1
                self._record_spec_bind_counter("fallback", colora_stats)
                fallback_spans.append((start, end))
                continue

            try:
                gate_group, up_group = outcome.result
                if not torch.is_tensor(gate_group) or not torch.is_tensor(up_group):
                    raise TypeError("speculative gate+up bind result must be a tensor pair")

                gate_group = gate_group.to(device=expert_input.device, dtype=expert_input.dtype, non_blocking=True).contiguous()
                up_group = up_group.to(device=expert_input.device, dtype=expert_input.dtype, non_blocking=True).contiguous()
                expected_rows = end - start
                if int(gate_group.shape[0]) != expected_rows or int(up_group.shape[0]) != expected_rows:
                    raise ValueError(
                        f"speculative gate+up row count mismatch: expected {expected_rows}, "
                        f"got gate={gate_group.shape[0]} up={up_group.shape[0]}"
                    )

                inverse_order = torch.argsort(sort_order_rows.to(device=expert_input.device, dtype=torch.long))
                gate_group = gate_group.index_select(0, inverse_order)
                up_group = up_group.index_select(0, inverse_order)
            except Exception as exc:
                logger.warning(
                    "[COLoRA][SpecBind] layer=%s decode_step_id=%s expert=%s adapter_bin=%s bind result invalid: %s",
                    self.layer_num_,
                    infer_state.decode_step_id,
                    expert_id,
                    adapter_bin,
                    exc,
                )
                step_counts["fallback"] += 1
                self._record_spec_bind_counter("fallback", colora_stats)
                fallback_spans.append((start, end))
                continue

            key = SpecJobKey(
                layer_id=int(self.layer_num_),
                decode_step_id=int(infer_state.decode_step_id),
                op_kind="gate_up",
                adapter_bin=adapter_bin,
                expert_id=int(expert_id),
                row_group_sig=tuple((int(row_idx), int(req_idx)) for row_idx, req_idx in row_group_sig),
            )
            bound_segments.append((start, end, gate_group, up_group))
            bound_keys = getattr(self, "_spec_bound_job_keys_current_call", None)
            if bound_keys is None:
                bound_keys = set()
                self._spec_bound_job_keys_current_call = bound_keys
            bound_keys.add(key)
            step_counts["successful_bind"] += 1
            self._record_spec_bind_counter("successful_bind", colora_stats)

        if logger.isEnabledFor(logging.DEBUG) and step_counts["attempted_bind"] > 0:
            logger.debug(
                "[COLoRA][SpecBind] layer=%s decode_step_id=%s expert=%s attempted=%s success=%s stale=%s not_ready=%s fallback=%s totals=%s",
                self.layer_num_,
                infer_state.decode_step_id,
                expert_id,
                step_counts["attempted_bind"],
                step_counts["successful_bind"],
                step_counts["stale"],
                step_counts["not_ready"],
                step_counts["fallback"],
                self._spec_bind_totals,
            )

        if not bound_segments:
            return None

        packed_gate_out = torch.empty(
            (packed_input.shape[0], bound_segments[0][2].shape[1]),
            dtype=bound_segments[0][2].dtype,
            device=bound_segments[0][2].device,
        )
        packed_up_out = torch.empty(
            (packed_input.shape[0], bound_segments[0][3].shape[1]),
            dtype=bound_segments[0][3].dtype,
            device=bound_segments[0][3].device,
        )
        for start, end, gate_group, up_group in bound_segments:
            packed_gate_out[start:end].copy_(gate_group)
            packed_up_out[start:end].copy_(up_group)

        if fallback_spans:
            fallback_positions = torch.cat(
                [
                    torch.arange(start, end, device=packed_input.device, dtype=torch.long)
                    for start, end in fallback_spans
                ],
                dim=0,
            )
            fallback_input = packed_input.index_select(0, fallback_positions).contiguous()
            fallback_bins = packed_bins.index_select(0, fallback_positions)

            with NvtxAnnotate("MoE_GateLoRA_SpecFallback"):
                fallback_gate = self.lora_dispatcher_.batch_apply_gate_lora(
                    fallback_input,
                    layer_weight.layer_num_,
                    fallback_bins,
                    expert_id=expert_id,
                )
                self._merge_colora_stats(colora_stats)
            with NvtxAnnotate("MoE_UpLoRA_SpecFallback"):
                fallback_up = self.lora_dispatcher_.batch_apply_up_lora(
                    fallback_input,
                    layer_weight.layer_num_,
                    fallback_bins,
                    expert_id=expert_id,
                )
                self._merge_colora_stats(colora_stats)

            packed_gate_out.index_copy_(0, fallback_positions, fallback_gate)
            packed_up_out.index_copy_(0, fallback_positions, fallback_up)

        gate_lora = self._scatter_lora_from_packed(
            packed_output=packed_gate_out,
            token_positions=token_positions,
            total_tokens=expert_input.shape[0],
        )
        up_lora = self._scatter_lora_from_packed(
            packed_output=packed_up_out,
            token_positions=token_positions,
            total_tokens=expert_input.shape[0],
        )
        return gate_lora, up_lora

    def _update_decode_spec_predictor(
        self,
        topk_ids: torch.Tensor,
        infer_state: Qwen3VLInferStateInfo,
        num_tokens: int,
    ) -> None:
        if not self._spec_submit_layer_enabled:
            return
        if infer_state is None or getattr(infer_state, "is_prefill", True):
            self._reset_spec_submit_runtime()
            return
        if os.environ.get("MOE_MODE", "TP") != "TP":
            return

        step_id = getattr(infer_state, "decode_step_id", None)
        req_idx_tensor = getattr(infer_state, "b_req_idx", None)
        adapter_bin_tensor = getattr(infer_state, "b_adapter_bin", None)
        if step_id is None or req_idx_tensor is None or adapter_bin_tensor is None:
            self._reset_spec_submit_runtime()
            return

        num_tokens = min(int(num_tokens), int(req_idx_tensor.shape[0]), int(adapter_bin_tensor.shape[0]))
        if topk_ids is None or topk_ids.numel() == 0 or num_tokens <= 0:
            self._spec_prev_decode_step_id = int(step_id)
            self._spec_prev_top1_by_req = {}
            return

        top1_experts = topk_ids[:num_tokens, 0].detach().to(device="cpu", dtype=torch.int64).view(-1).tolist()
        req_idx_cpu = req_idx_tensor[:num_tokens].detach().to(device="cpu", dtype=torch.int64).view(-1).tolist()
        adapter_bins_cpu = adapter_bin_tensor[:num_tokens].detach().to(device="cpu", dtype=torch.int64).view(-1).tolist()

        next_predictor: Dict[int, Tuple[int, int]] = {}
        for req_idx, adapter_bin, expert_id in zip(req_idx_cpu, adapter_bins_cpu, top1_experts):
            if int(adapter_bin) < 0 or int(expert_id) < 0 or int(expert_id) >= int(self.n_routed_experts):
                continue
            next_predictor[int(req_idx)] = (int(adapter_bin), int(expert_id))

        self._spec_prev_decode_step_id = int(step_id)
        self._spec_prev_top1_by_req = next_predictor

    def _log_router_trace(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_tokens: int,
        infer_state: Qwen3VLInferStateInfo,
    ) -> None:
        super()._log_router_trace(topk_ids, topk_weights, num_tokens, infer_state)
        self._update_decode_spec_predictor(topk_ids, infer_state, num_tokens)

    def _moe_ffn(
        self,
        input: torch.Tensor,
        infer_state: Qwen3VLInferStateInfo,
        layer_weight: Qwen3MOETransformerLayerWeight,
    ) -> torch.Tensor:
        if infer_state is None or getattr(infer_state, "is_prefill", True):
            return super()._moe_ffn(input, infer_state, layer_weight)

        self._spec_bound_job_keys_current_call = set()
        try:
            hidden_states = input.view(-1, self.embed_dim_)
            self._maybe_submit_decode_spec_gate_up(hidden_states, infer_state)
            output = super()._moe_ffn(input, infer_state, layer_weight)
            self._eager_retire_remaining_decode_spec_jobs(infer_state)
            return output
        finally:
            self._spec_bound_job_keys_current_call = set()
            self._finalize_decode_spec_step_nonblocking(infer_state)

    @NvtxAnnotate("Qwen3VL_QKV")
    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: Qwen3VLInferStateInfo,
        layer_weight: Qwen3MOETransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = input.view(-1, self.embed_dim_)
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(input)

        # Apply LoRA using lora_dispatcher_ (set by base_backend.py via init_batched_lora_adapters())
        if self.use_detached_lora_ and self.lora_dispatcher_ is not None:
            # S-LoRA batched mode: use batch_apply_* methods with req_bins
            assert hasattr(self.lora_dispatcher_, 'use_batched_mode'), "lora_dispatcher_ must support batched mode"
            assert self.lora_dispatcher_.use_batched_mode, "lora_dispatcher_ must be in batched mode"
            assert self.req_bins_ is not None, "req_bins_ must be set for batched mode"

            batch_size = input.shape[0]
            logger.info(f"[LoRA Infer] Layer {self.layer_num_}: apply_qkv_lora batch={batch_size}")

            lora_results = self.lora_dispatcher_.get_attn_qkv_lora(
                input, self.layer_num_, self.req_bins_
            )

            q = q + lora_results["q_lora"]
            # cache_kv is [batch, (tp_k + tp_v) * head_dim] with K and V concatenated
            # View to [batch, num_heads, head_dim] to add LoRA to correct heads
            cache_kv = cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
            # Add K-LoRA to first tp_k_head_num heads, V-LoRA to next tp_v_head_num heads
            cache_kv[:, : self.tp_k_head_num_, :] = cache_kv[:, : self.tp_k_head_num_, :] + lora_results["k_lora"].reshape(
                -1, self.tp_k_head_num_, self.head_dim_
            )
            cache_kv[:, self.tp_k_head_num_ :, :] = cache_kv[:, self.tp_k_head_num_ :, :] + lora_results["v_lora"].reshape(
                -1, self.tp_v_head_num_, self.head_dim_
            )
            # View back to 2D for downstream processing
            cache_kv = cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_)

        qk_rmsnorm_forward(
            q,
            weight=layer_weight.q_norm_weight_.weight,
            eps=self.eps_,
        )
        qk_rmsnorm_forward(
            cache_kv[:, : self.tp_k_head_num_ * self.head_dim_],
            weight=layer_weight.k_norm_weight_.weight,
            eps=self.eps_,
        )
        cache_kv = cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        mrope_triton_fused(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
            self.mrope_section,
            is_interleaved=True,
        )
        return q, cache_kv

    @NvtxAnnotate("Qwen3VL_O_Proj")
    def _get_o(
        self, input, infer_state: Qwen3VLInferStateInfo, layer_weight: Qwen3MOETransformerLayerWeight
    ) -> torch.Tensor:
        """Apply o_proj with LoRA support for Qwen3-VL-MoE.

        Computes: o = o_proj(attn_output) + LoRA contribution

        LoRA is applied to: o_proj (optional)
        """
        input = input.view(-1, self.tp_o_head_num_ * self.head_dim_)
        o_tensor = layer_weight.o_proj.mm(input)

        # Apply LoRA using lora_dispatcher_ (set by base_backend.py via init_batched_lora_adapters())
        if self.use_detached_lora_ and self.lora_dispatcher_ is not None:
            # S-LoRA batched mode
            assert hasattr(self.lora_dispatcher_, 'use_batched_mode'), "lora_dispatcher_ must support batched mode"
            assert self.lora_dispatcher_.use_batched_mode, "lora_dispatcher_ must be in batched mode"
            assert self.req_bins_ is not None, "req_bins_ must be set for batched mode"

            batch_size = input.shape[0]
            logger.debug(f"[LoRA Infer] Layer {self.layer_num_}: apply_o_lora batch={batch_size}")

            o_lora = self.lora_dispatcher_.batch_apply_o_lora(input, self.layer_num_, self.req_bins_)
            o_tensor = o_tensor + o_lora

        return o_tensor

    def context_forward(self, input_embdings, infer_state: Qwen3VLInferStateInfo, layer_weight):
            input1 = self._att_norm(input_embdings, infer_state, layer_weight)
            q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
            input1 = None
            self._post_cache_kv(cache_kv, infer_state, layer_weight)
            o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)
            q = None
            o = self._get_o(o, infer_state, layer_weight)
            if self.tp_world_size_ > 1:
                all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
            input_embdings.add_(o.view(-1, self.embed_dim_))
            o = None

            input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
            ffn_out = self._ffn(input1, infer_state, layer_weight)
            input1 = None
            if self.tp_world_size_ > 1:
                all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
            input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
            apply_deepstack_features(
                input_embeddings=input_embdings,
                infer_state=infer_state,
                layer_num=self.layer_num_,
            )
            return input_embdings
