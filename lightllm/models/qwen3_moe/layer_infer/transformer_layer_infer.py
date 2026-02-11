import os
import logging
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
import triton
from typing import Tuple, Optional, Any
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd
from functools import partial
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_global_world_size
from lightllm.distributed.communication_op import all_gather_into_tensor, reduce_scatter_tensor
from lightllm.utils.nvtx_utils import NvtxAnnotate

logger = init_logger(__name__)


class Qwen3MOETransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        self.n_routed_experts = network_config["num_experts"]
        self.is_moe = (
            network_config["num_experts"] > 0
            and layer_num not in network_config["mlp_only_layers"]
            and (layer_num + 1) % network_config["decoder_sparse_step"] == 0
        )
        self.num_experts_per_tok = network_config["num_experts_per_tok"]
        self.norm_topk_prob = network_config["norm_topk_prob"]
        super().__init__(layer_num, network_config, mode)
        self.head_dim_ = network_config["head_dim"]
        self.tp_k_head_num_ = max(self.tp_k_head_num_, 1)
        self.tp_v_head_num_ = max(self.tp_v_head_num_, 1)

        # LoRA dispatcher for MoE (set externally for detached mode)
        self.lora_dispatcher_: Optional[Any] = None
        self.use_detached_lora_: bool = False
        self.req_bins_: Optional[torch.Tensor] = None  # Per-request adapter indices for batched LoRA

        return

    def set_req_bins(self, req_bins: torch.Tensor):
        """Set the req_bins tensor for batched mode.

        Args:
            req_bins: Tensor of shape [batch_size] containing per-request adapter indices
        """
        self.req_bins_ = req_bins

    def _bind_func(self):
        super()._bind_func()
        self._bind_ffn()
        return

    def _bind_ffn(self):
        if self.is_moe:
            moe_mode = os.environ.get("MOE_MODE", "TP")
            if moe_mode == "EP":
                self._ffn = partial(Qwen3MOETransformerLayerInfer._moe_ffn_edp, self)
                self._tpsp_ffn = self._tpsp_ffn_ep
            else:
                self._ffn = partial(Qwen3MOETransformerLayerInfer._moe_ffn, self)
                self._tpsp_ffn = self._tpsp_ffn_tp
        else:
            self._ffn = partial(LlamaTransformerLayerInfer._ffn, self)
            self._tpsp_ffn = self._tpsp_ffn_tp

    def set_lora_dispatcher(self, dispatcher: Any, use_detached_lora: bool = True):
        """Set the LoRA dispatcher for MoE layers.

        Args:
            dispatcher: LoRA dispatcher instance (Qwen3MOELoRADispatcher)
            use_detached_lora: If True, LoRA runs in detached mode (parallel with base)
        """
        self.lora_dispatcher_ = dispatcher
        self.use_detached_lora_ = use_detached_lora

    def clear_lora_dispatcher(self):
        """Clear the LoRA dispatcher."""
        self.lora_dispatcher_ = None
        self.use_detached_lora_ = False

    def _get_local_expert_info(self, layer_weight):
        """Get information about local experts for EP mode.

        Returns:
            is_ep: Whether EP mode is enabled
            local_expert_ids: List of global expert IDs owned by this rank (empty if TP mode)
            local_to_global: Dict mapping local expert ID -> global expert ID
            global_to_local: Dict mapping global expert ID -> local expert ID (-1 if not local)
        """
        moe_mode = os.environ.get("MOE_MODE", "TP")
        if moe_mode != "EP":
            return False, [], {}, {}

        experts = layer_weight.experts
        ep_world_size = get_global_world_size()
        ep_rank = getattr(experts, 'global_rank_', 0)

        n_routed_experts = experts.n_routed_experts
        ep_n_routed_experts = n_routed_experts // ep_world_size

        # Compute global expert IDs owned by this rank
        start_expert = ep_rank * ep_n_routed_experts
        local_expert_ids = list(range(start_expert, start_expert + ep_n_routed_experts))

        # Build mapping dicts
        local_to_global = {i: g for i, g in enumerate(local_expert_ids)}
        global_to_local = {g: i for i, g in enumerate(local_expert_ids)}

        return True, local_expert_ids, local_to_global, global_to_local
        self.lora_dispatcher_ = None
        self.use_detached_lora_ = False

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3MOETransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = input.view(-1, self.embed_dim_)
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(input).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        rmsnorm_forward(
            q.view(-1, self.head_dim_),
            weight=layer_weight.q_norm_weight_.weight,
            eps=self.eps_,
            out=q.view(-1, self.head_dim_),
        )

        cache_kv[:, : self.tp_k_head_num_, :] = rmsnorm_forward(
            cache_kv[:, : self.tp_k_head_num_, :].reshape(-1, cache_kv.shape[-1]),
            weight=layer_weight.k_norm_weight_.weight,
            eps=self.eps_,
        ).view(-1, self.tp_k_head_num_, cache_kv.shape[-1])

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    def _tpsp_get_qkv(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3MOETransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tp_world_size_ > 1:
            sp_token_num, hidden_dim = input.shape
            gather_input = self.alloc_tensor(
                (sp_token_num * self.tp_world_size_, hidden_dim), dtype=input.dtype, device=input.device
            )
            all_gather_into_tensor(gather_input, input, group=infer_state.dist_group, async_op=False)
            input = gather_input[0 : len(infer_state.position_cos), :]

        input = input.view(-1, self.embed_dim_)
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(input).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        rmsnorm_forward(
            q.view(-1, self.head_dim_),
            weight=layer_weight.q_norm_weight_.weight,
            eps=self.eps_,
            out=q.view(-1, self.head_dim_),
        )

        cache_kv[:, : self.tp_k_head_num_, :] = rmsnorm_forward(
            cache_kv[:, : self.tp_k_head_num_, :].reshape(-1, cache_kv.shape[-1]),
            weight=layer_weight.k_norm_weight_.weight,
            eps=self.eps_,
        ).view(-1, self.tp_k_head_num_, cache_kv.shape[-1])

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )

        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)

        return q, cache_kv

    @NvtxAnnotate("MoE_FFN")
    def _moe_ffn(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3MOETransformerLayerWeight
    ) -> torch.Tensor:
        """Per-Expert LoRA Baseline implementation."""
        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape

        # Check if we need per-expert LoRA
        # 确保 dispatcher 存在且开启了 detached lora 模式
        force_slow = getattr(self, 'force_slow_lora_path', False)
        use_per_expert_lora = (self.use_detached_lora_ and self.lora_dispatcher_ is not None) or force_slow

        # ----------------------------------------------------------------
        # Fast Path: 使用 Fused Kernel (无 LoRA)
        # ----------------------------------------------------------------
        if not use_per_expert_lora:
            router_logits = layer_weight.moe_gate.mm(hidden_states)
            layer_weight.experts.experts(
                hidden_states,
                router_logits=router_logits,
                top_k=self.num_experts_per_tok,
                renormalize=self.norm_topk_prob,
                use_grouped_topk=False,
                topk_group=None,
                num_expert_group=None,
            )
            return hidden_states.view(num_tokens, hidden_dim)

        # ----------------------------------------------------------------
        # Slow Path: Per-Expert LoRA Baseline (Explicit Loop, Research purpose)
        # ----------------------------------------------------------------
        from lightllm.common.fused_moe.topk_select import select_experts

        # 1. Router computation
        router_logits = layer_weight.moe_gate.mm(hidden_states)

        # 2. Explicit routing
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            correction_bias=getattr(layer_weight.experts, "e_score_correction_bias", None),
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=False,
            topk_group=None,
            num_expert_group=None,
            scoring_func=getattr(layer_weight.experts, "scoring_func", "softmax"),
        )

        if hasattr(layer_weight.experts, "routed_scaling_factor"):
            topk_weights = topk_weights * layer_weight.experts.routed_scaling_factor

        # 3. Check weights availability
        experts = layer_weight.experts
        # 必须确保使用了 keep_expert_lists=True
        if not hasattr(experts, "experts_gate_projs") or experts.experts_gate_projs[0] is None:
            raise RuntimeError(
                "Per-Expert Baseline requires 'keep_expert_lists=True' in FusedMoeWeightTP."
            )

        final_output = torch.zeros_like(hidden_states)
        total_experts = experts.n_routed_experts

        # EP mode: get local expert info
        is_ep, local_expert_ids, local_to_global, global_to_local = self._get_local_expert_info(layer_weight)

        # 4. Expert Loop
        # In TP mode: iterate over all experts
        # In EP mode: iterate only over local experts
        expert_iter_range = local_expert_ids if is_ep else range(total_experts)

        # Log token distribution per expert before entering the loop
        if logger.isEnabledFor(logging.DEBUG):
            token_counts = {}
            for global_eid in (local_to_global.values() if is_ep else range(total_experts)):
                count = int((topk_ids == global_eid).sum())
                if count > 0:
                    token_counts[global_eid] = count
            logger.debug(f"[MoE] Layer {self.layer_num_}: {num_tokens} tokens -> {token_counts}")

        for local_expert_idx in expert_iter_range:
            with NvtxAnnotate(f"Expert_{local_expert_idx}"):
                # Get global expert ID (used for filtering tokens)
                global_expert_id = local_to_global.get(local_expert_idx, local_expert_idx) if is_ep else local_expert_idx

                # 4.1 Filter tokens assigned to this expert
                mask = topk_ids == global_expert_id
                batch_indices, k_indices = torch.where(mask)

                if batch_indices.shape[0] == 0:
                    continue

                # 4.2 Slice Input
                expert_input = hidden_states[batch_indices]
                expert_req_bins = self.req_bins_[batch_indices]

                # 4.3 Get Weights (use local index for experts list)
                # TODO(FIX): offload here
                w1 = experts.experts_gate_projs[local_expert_idx].cuda()
                w3 = experts.experts_up_projs[local_expert_idx].cuda()
                w2 = experts.w2_list[local_expert_idx].cuda()

                # 4.4 Compute Base
                # input: [N, hidden], w.T: [hidden, inter] -> output: [N, inter]
                gate_out = torch.mm(expert_input, w1.T)
                up_out = torch.mm(expert_input, w3.T)

                # 4.5 Apply Per-Expert LoRA (Gate/Up)
                # Use LOCAL expert index for LoRA buffer (as buffer only stores local experts)
                gate_lora = self.lora_dispatcher_.batch_apply_gate_lora(
                    expert_input, layer_weight.layer_num_, expert_req_bins, expert_id=local_expert_idx
                )
                up_lora = self.lora_dispatcher_.batch_apply_up_lora(
                    expert_input, layer_weight.layer_num_, expert_req_bins, expert_id=local_expert_idx
                )

                gate_out += gate_lora
                up_out += up_lora

                # 4.6 Activation
                current_hidden = torch.nn.functional.silu(gate_out) * up_out

                # 4.7 Down Projection Base
                down_out = torch.mm(current_hidden, w2.T)

                # 4.8 Apply Per-Expert LoRA (Down)
                # Use LOCAL expert index for LoRA buffer
                down_lora = self.lora_dispatcher_.batch_apply_down_lora(
                    current_hidden, layer_weight.layer_num_, expert_req_bins, expert_id=local_expert_idx
                )
                down_out += down_lora

                # 4.9 Weighted Aggregation (Corrected)
                # routing_weights: [num_selected, 1]
                routing_weights = topk_weights[batch_indices, k_indices].view(-1, 1)
                weighted_output = (down_out * routing_weights).to(hidden_states.dtype)

                # 使用 index_add_ 在 final_output 上原地累加
                final_output.index_add_(0, batch_indices, weighted_output)

        return final_output.view(num_tokens, hidden_dim)

    def _moe_ffn_edp(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3MOETransformerLayerWeight
    ) -> torch.Tensor:

        hidden_states = input
        token_num, hidden_dim = hidden_states.shape

        router_logits = layer_weight.moe_gate.mm(hidden_states)

        # Apply moe_gate LoRA using batched S-LoRA API
        if self.use_detached_lora_ and self.lora_dispatcher_ is not None:
            gate_lora = self.lora_dispatcher_.batch_apply_gate_lora(
                hidden_states, layer_weight.layer_num_, self.req_bins_
            )
            router_logits = router_logits + gate_lora

        ep_output = layer_weight.experts.experts(
            hidden_states,
            router_logits=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=False,
            topk_group=None,
            num_expert_group=None,
            is_prefill=infer_state.is_prefill,
        )

        # Apply w2 (down_proj) LoRA using batched S-LoRA API
        if self.use_detached_lora_ and self.lora_dispatcher_ is not None:
            down_lora = self.lora_dispatcher_.batch_apply_down_lora(
                ep_output, layer_weight.layer_num_, self.req_bins_
            )
            ep_output = ep_output + down_lora

        ep_output = ep_output.view(token_num, hidden_dim)
        return ep_output

    def _tpsp_ffn(
        self, input: torch.Tensor, infer_state: LlamaInferStateInfo, layer_weight: Qwen3MOETransformerLayerWeight
    ):
        raise Exception("need bind to real impl")

    def _tpsp_ffn_tp(
        self, input: torch.Tensor, infer_state: LlamaInferStateInfo, layer_weight: Qwen3MOETransformerLayerWeight
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        if self.tp_world_size_ > 1:
            sp_token_num, hidden_dim = input.shape
            gather_input = self.alloc_tensor(
                (sp_token_num * self.tp_world_size_, hidden_dim), dtype=input.dtype, device=input.device
            )
            all_gather_into_tensor(gather_input, input, group=infer_state.dist_group, async_op=False)
            input = gather_input

        ffn2_out = self._ffn(input=input, infer_state=infer_state, layer_weight=layer_weight)

        if self.tp_world_size_ > 1:
            sp_token_num = ffn2_out.shape[0] // self.tp_world_size_
            reduce_o_tensor = self.alloc_tensor(
                (sp_token_num, self.embed_dim_), dtype=ffn2_out.dtype, device=ffn2_out.device
            )
            reduce_scatter_tensor(
                reduce_o_tensor, ffn2_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False
            )
            ffn2_out = reduce_o_tensor
        return ffn2_out

    def _tpsp_ffn_ep(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3MOETransformerLayerWeight
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)

        ffn2_out = self._ffn(input=input, infer_state=infer_state, layer_weight=layer_weight)

        return ffn2_out

    def overlap_tpsp_token_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: Qwen3MOETransformerLayerWeight,
    ):
        if not self.is_moe:
            return super().overlap_tpsp_token_forward(
                input_embdings, input_embdings1, infer_state, infer_state1, layer_weight
            )
        # 0 attention
        _0_input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        _0_q, _0_cache_kv = self._tpsp_get_qkv(_0_input1, infer_state, layer_weight)
        _0_input1 = None
        self._post_cache_kv(_0_cache_kv, infer_state, layer_weight)
        _0_o = self._token_attention_kernel(_0_q, infer_state, layer_weight)
        _0_q = None
        _0_o = self._tpsp_get_o(_0_o, infer_state, layer_weight)
        input_embdings.add_(_0_o.view(-1, self.embed_dim_))
        _0_o = None
        _0_input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        _0_router_logits = layer_weight.moe_gate.mm(_0_input1)
        # 1 hook
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        # 0 dispatch
        (
            _0_recv_x,
            _0_masked_m,
            _0_topk_idx,
            _0_topk_weight,
            _0_handle,
            _0_hook,
        ) = layer_weight.experts.low_latency_dispatch(_0_input1, _0_router_logits)
        infer_state.hook = _0_hook

        # 1 attention
        _1_input1 = self._att_norm(input_embdings1, infer_state1, layer_weight)
        _1_q, _1_cache_kv = self._tpsp_get_qkv(_1_input1, infer_state1, layer_weight)
        _1_input1 = None
        self._post_cache_kv(_1_cache_kv, infer_state1, layer_weight)
        _1_o = self._token_attention_kernel(_1_q, infer_state1, layer_weight)
        _1_q = None
        _1_o = self._tpsp_get_o(_1_o, infer_state1, layer_weight)
        input_embdings1.add_(_1_o.view(-1, self.embed_dim_))
        _1_o = None
        _1_input1 = self._ffn_norm(input_embdings1, infer_state1, layer_weight)
        # to do gate and disptatch

        _1_router_logits = layer_weight.moe_gate.mm(_1_input1)
        # 0 hook
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            infer_state.hook = None

        # 1 dispatch
        (
            _1_recv_x,
            _1_masked_m,
            _1_topk_idx,
            _1_topk_weight,
            _1_handle,
            _1_hook,
        ) = layer_weight.experts.low_latency_dispatch(_1_input1, _1_router_logits)
        infer_state1.hook = _1_hook

        # moe calu
        expected_m = triton.cdiv(
            input_embdings.shape[0] * get_global_world_size() * self.num_experts_per_tok, self.n_routed_experts
        )
        _0_moe_out = layer_weight.experts.masked_group_gemm(_0_recv_x, _0_masked_m, input_embdings.dtype, expected_m)

        # 1 hook
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        # 0 combine
        _0_ffn_out, _0_hook = layer_weight.experts.low_latency_combine(
            _0_moe_out, _0_topk_idx, _0_topk_weight, _0_handle
        )

        infer_state.hook = _0_hook

        # to do moe caclue
        _1_moe_out = layer_weight.experts.masked_group_gemm(_1_recv_x, _1_masked_m, input_embdings1.dtype, expected_m)

        # 0 hook
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            input_embdings.add_(_0_ffn_out.view(-1, self.embed_dim_))
            infer_state.hook = None

        # 1 combine
        _1_ffn_out, _1_hook = layer_weight.experts.low_latency_combine(
            _1_moe_out, _1_topk_idx, _1_topk_weight, _1_handle
        )

        def _1_hook_post():
            _1_hook()
            nonlocal _1_ffn_out
            input_embdings1.add_(_1_ffn_out.view(-1, self.embed_dim_))
            return

        infer_state1.hook = _1_hook_post

        return input_embdings, input_embdings1

    def overlap_tpsp_context_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: Qwen3MOETransformerLayerWeight,
    ):
        if not self.is_moe:
            return super().overlap_tpsp_context_forward(
                input_embdings, input_embdings1, infer_state, infer_state1, layer_weight
            )
        # 0 attention
        _0_input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        _0_q, _0_cache_kv = self._tpsp_get_qkv(_0_input1, infer_state, layer_weight)
        _0_input1 = None
        self._post_cache_kv(_0_cache_kv, infer_state, layer_weight)
        _0_o = self._context_attention_kernel(_0_q, _0_cache_kv, infer_state, layer_weight)
        _0_q = None
        _0_o = self._tpsp_get_o(_0_o, infer_state, layer_weight)
        input_embdings.add_(_0_o.view(-1, self.embed_dim_))
        _0_o = None
        _0_input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        _0_router_logits = layer_weight.moe_gate.mm(_0_input1)

        # wait last 1 combine
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        _0_topk_weight, _0_topk_idx, _0_qinput_tensor = layer_weight.experts.select_experts_and_quant_input(
            _0_input1, _0_router_logits
        )
        from deep_ep import Buffer

        _0_overlap_event = Buffer.capture()

        # 1 attention
        _1_input1 = self._att_norm(input_embdings1, infer_state1, layer_weight)
        _1_q, _1_cache_kv = self._tpsp_get_qkv(_1_input1, infer_state1, layer_weight)
        _1_input1 = None
        self._post_cache_kv(_1_cache_kv, infer_state1, layer_weight)
        _1_o = self._context_attention_kernel(_1_q, _1_cache_kv, infer_state1, layer_weight)
        _1_q = None
        _1_o = self._tpsp_get_o(_1_o, infer_state1, layer_weight)
        input_embdings1.add_(_1_o.view(-1, self.embed_dim_))
        _1_o = None
        _1_input1 = self._ffn_norm(input_embdings1, infer_state1, layer_weight)
        # to do gate and disptatch

        _1_router_logits = layer_weight.moe_gate.mm(_1_input1)

        # 0 dispatch execute
        (
            _0_recv_x,
            _0_recv_topk_idx,
            _0_recv_topk_weight,
            _0_num_recv_tokens_per_expert_list,
            _0_handle,
            _0_hook,
        ) = layer_weight.experts.dispatch(_0_qinput_tensor, _0_topk_idx, _0_topk_weight, overlap_event=_0_overlap_event)
        infer_state.hook = _0_hook

        # wait 0 dispatch
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            infer_state.hook = None

        _1_topk_weight, _1_topk_idx, _1_qinput_tensor = layer_weight.experts.select_experts_and_quant_input(
            _1_input1, _1_router_logits
        )

        _1_overlap_event = Buffer.capture()

        # 0 moe calu
        _0_moe_out = layer_weight.experts.prefilled_group_gemm(
            _0_num_recv_tokens_per_expert_list, _0_recv_x, _0_recv_topk_idx, _0_recv_topk_weight
        )

        # 1 dispatch execute
        (
            _1_recv_x,
            _1_recv_topk_idx,
            _1_recv_topk_weight,
            _1_num_recv_tokens_per_expert_list,
            _1_handle,
            _1_hook,
        ) = layer_weight.experts.dispatch(_1_qinput_tensor, _1_topk_idx, _1_topk_weight, overlap_event=_1_overlap_event)
        infer_state1.hook = _1_hook

        # wait 1 dispatch
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        _0_combine_event = Buffer.capture()
        # 0 combine execute
        _0_ffn_out, _0_hook = layer_weight.experts.combine(_0_moe_out, _0_handle, _0_combine_event)
        infer_state.hook = _0_hook

        # 1 moe calc
        _1_moe_out = layer_weight.experts.prefilled_group_gemm(
            _1_num_recv_tokens_per_expert_list, _1_recv_x, _1_recv_topk_idx, _1_recv_topk_weight
        )

        # wait 0 combine
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            infer_state.hook = None

        _1_combine_event = Buffer.capture()

        input_embdings.add_(_0_ffn_out.view(-1, self.embed_dim_))

        # 1 combine execute
        _1_ffn_out, _1_hook = layer_weight.experts.combine(_1_moe_out, _1_handle, _1_combine_event)

        def _1_hook_post():
            _1_hook()
            nonlocal _1_ffn_out
            input_embdings1.add_(_1_ffn_out.view(-1, self.embed_dim_))
            return

        infer_state1.hook = _1_hook_post

        return input_embdings, input_embdings1
