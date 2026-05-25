import os
import torch
import torch.nn.functional as F
from typing import Optional, Any, Dict, Callable
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.mixtral.layer_infer._custom_ops import fused_topk
from lightllm.models.mixtral.layer_weights.transformer_layer_weight import MixtralTransformerLayerWeight
from lightllm.utils.nvtx_utils import NvtxAnnotate
from lightllm.common.fused_moe.topk_select import select_experts


class MixtralTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        self.num_local_experts = network_config["num_local_experts"]
        self.num_experts_per_tok = network_config["num_experts_per_tok"]
        self.renormalize = True

        self.lora_dispatcher_: Optional[Any] = None
        self.use_detached_lora_: bool = False
        self.req_bins_: Optional[torch.Tensor] = None
        return

    def set_lora_dispatcher(self, dispatcher: Any, use_detached_lora: bool = True):
        self.lora_dispatcher_ = dispatcher
        self.use_detached_lora_ = use_detached_lora

    def clear_lora_dispatcher(self):
        self.lora_dispatcher_ = None
        self.use_detached_lora_ = False

    def set_req_bins(self, req_bins: torch.Tensor):
        self.req_bins_ = req_bins

    def _ffn(
        self, input, infer_state: InferStateInfo, layer_weight: MixtralTransformerLayerWeight
    ) -> torch.Tensor:
        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape

        router_logits = layer_weight.moe_gate.mm(hidden_states)

        use_lora = self.use_detached_lora_ and self.lora_dispatcher_ is not None

        if not use_lora:
            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.num_experts_per_tok,
                renormalize=self.renormalize,
                alloc_tensor_func=self.alloc_tensor,
            )
            from lightllm.common.fused_moe.grouped_fused_moe import fused_experts_impl

            return fused_experts_impl(
                hidden_states=hidden_states,
                w1=layer_weight.experts.w1[0],
                w2=layer_weight.experts.w2[0],
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                use_fp8_w8a8=False,
                w1_scale=None,
                w2_scale=None,
                alloc_tensor_func=self.alloc_tensor,
            )

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            correction_bias=getattr(layer_weight.experts, "e_score_correction_bias", None),
            top_k=self.num_experts_per_tok,
            renormalize=self.renormalize,
            use_grouped_topk=False,
            topk_group=None,
            num_expert_group=None,
            scoring_func=getattr(layer_weight.experts, "scoring_func", "softmax"),
        )

        return self._moe_ffn_slow_path(
            hidden_states, topk_weights, topk_ids, infer_state, layer_weight
        )

    def _moe_ffn_slow_path(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        infer_state: InferStateInfo,
        layer_weight: MixtralTransformerLayerWeight,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        experts = layer_weight.experts
        total_experts = experts.n_routed_experts

        assert hasattr(experts, "experts_gate_projs") and experts.experts_gate_projs[0] is not None, (
            "Per-Expert LoRA requires 'keep_expert_lists=True' in FusedMoeWeightTP."
        )

        final_output = torch.zeros_like(hidden_states)

        flat_topk_ids = topk_ids.flatten()
        expert_counts = torch.bincount(flat_topk_ids, minlength=total_experts)
        sorted_token_indices = torch.argsort(flat_topk_ids)

        global_offsets = [0] * (total_experts + 1)
        for i in range(total_experts):
            global_offsets[i + 1] = global_offsets[i] + int(expert_counts[i].item())

        active_experts_data = []
        for expert_id in range(total_experts):
            count = int(expert_counts[expert_id].item())
            if count == 0:
                continue
            start_idx = global_offsets[expert_id]
            end_idx = start_idx + count
            token_idx = sorted_token_indices[start_idx:end_idx]
            batch_indices = token_idx // self.num_experts_per_tok
            k_indices = token_idx % self.num_experts_per_tok
            active_experts_data.append((expert_id, batch_indices, k_indices))

        dispatcher = self.lora_dispatcher_

        for expert_id, batch_indices, k_indices in active_experts_data:
            with NvtxAnnotate(f"MoE_Expert_{expert_id}"):

                expert_input = hidden_states[batch_indices]
                expert_req_bins = self.req_bins_[batch_indices] if self.req_bins_ is not None else None

                w1 = experts.experts_gate_projs[expert_id].cuda(non_blocking=True)
                w3 = experts.experts_up_projs[expert_id].cuda(non_blocking=True)
                w2 = experts.w2_list[expert_id].cuda(non_blocking=True)

                with NvtxAnnotate("MoE_GateGEMM"):
                    gate_out = torch.mm(expert_input, w1.T)
                with NvtxAnnotate("MoE_GateLoRA"):
                    gate_lora = dispatcher.batch_apply_gate_lora(
                        expert_input, layer_weight.layer_num_, expert_req_bins, expert_id=expert_id
                    )
                    gate_out += gate_lora

                with NvtxAnnotate("MoE_UpGEMM"):
                    up_out = torch.mm(expert_input, w3.T)
                with NvtxAnnotate("MoE_UpLoRA"):
                    up_lora = dispatcher.batch_apply_up_lora(
                        expert_input, layer_weight.layer_num_, expert_req_bins, expert_id=expert_id
                    )
                    up_out += up_lora

                with NvtxAnnotate("MoE_Activation"):
                    current_hidden = torch.nn.functional.silu(gate_out) * up_out

                with NvtxAnnotate("MoE_DownGEMM"):
                    down_out = torch.mm(current_hidden, w2.T)
                with NvtxAnnotate("MoE_DownLoRA"):
                    down_lora = dispatcher.batch_apply_down_lora(
                        current_hidden, layer_weight.layer_num_, expert_req_bins, expert_id=expert_id
                    )
                    down_out += down_lora

                with NvtxAnnotate("MoE_Aggregation"):
                    routing_weights = topk_weights[batch_indices, k_indices].view(-1, 1)
                    weighted_output = (down_out * routing_weights).to(hidden_states.dtype)
                    final_output.index_add_(0, batch_indices, weighted_output)

                del w1, w3, w2

        return final_output.view(num_tokens, hidden_dim)