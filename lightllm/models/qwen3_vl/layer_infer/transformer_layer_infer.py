import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from functools import partial
from typing import Tuple, Optional, Dict, Any
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.qwen2_vl.triton_kernel.mrope import mrope_triton_fused
from lightllm.models.qwen3.layer_infer.transformer_layer_infer import Qwen3TransformerLayerInfer
from lightllm.models.qwen3.layer_weights.transformer_layer_weight import Qwen3TransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo
from lightllm.models.qwen3_vl.layer_weights.transformer_layer_weight import Qwen3VLTransformerLayerWeight
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd
from lightllm.distributed import all_reduce
from lightllm.utils.dist_utils import get_global_world_size
from lightllm.models.qwen3_vl.triton_kernel.deepstack_multimodal_emb import apply_deepstack_features
from lightllm.models.qwen2_vl.layer_infer.transformer_layer_infer import Qwen2VLTransformerLayerInfer
from lightllm.models.qwen3.triton_kernel.qk_norm import qk_rmsnorm_forward
from lightllm.models.qwen3_vl.lora_dispatch import LoRADispatcher, create_lora_dispatcher


class Qwen3VLTransformerLayerInfer(Qwen2VLTransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        self.mrope_section = torch.tensor(
            network_config["rope_scaling"]["mrope_section"], dtype=torch.int32, device="cuda"
        )
        # LoRA configuration
        self.lora_rank = network_config.get("lora_rank", 0)
        self.lora_alpha = network_config.get("lora_alpha", 1.0)
        self.lora_dropout = network_config.get("lora_dropout", 0.0)
        self.has_lora = self.lora_rank > 0
        if self.has_lora:
            self.lora_scaling = self.lora_alpha / self.lora_rank

        # Detached mode: use external dispatcher
        self.lora_dispatcher: Optional[LoRADispatcher] = None
        self.use_detached_lora = False

    def set_lora_dispatcher(self, dispatcher: LoRADispatcher):
        """Set external LoRA dispatcher for detached mode.

        When using detached mode, LoRA weights are managed externally
        and applied through the dispatcher.
        """
        self.lora_dispatcher = dispatcher
        self.use_detached_lora = dispatcher is not None and dispatcher.lora_rank > 0

    def clear_lora_dispatcher(self):
        """Clear the LoRA dispatcher."""
        self.lora_dispatcher = None
        self.use_detached_lora = False

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3TransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = input.view(-1, self.embed_dim_)

        # Base projection
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(input)

        # Apply LoRA to q, k, v projections (detached mode)
        if self.use_detached_lora and self.lora_dispatcher is not None:
            lora_results = self.lora_dispatcher.apply_attention_lora(input, self.layer_num_, infer_state)
            q = q + lora_results["q_lora"]
            cache_kv[:, : self.tp_k_head_num_ * self.head_dim_] = cache_kv[:, : self.tp_k_head_num_ * self.head_dim_] + lora_results["k_lora"]
            cache_kv[:, self.tp_k_head_num_ * self.head_dim_:] = cache_kv[:, self.tp_k_head_num_ * self.head_dim_:] + lora_results["v_lora"]
        elif self.has_lora and layer_weight.has_lora_attn:
            # Merged mode: use weights from layer_weight
            lora_results = self._apply_lora_attn(input, layer_weight)
            q = q + lora_results["q_lora"]
            cache_kv[:, : self.tp_k_head_num_ * self.head_dim_] = cache_kv[:, : self.tp_k_head_num_ * self.head_dim_] + lora_results["k_lora"]
            cache_kv[:, self.tp_k_head_num_ * self.head_dim_:] = cache_kv[:, self.tp_k_head_num_ * self.head_dim_:] + lora_results["v_lora"]

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

    def _ffn(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3VLTransformerLayerWeight
    ) -> torch.Tensor:
        """FFN with LoRA support for Qwen3-VL.

        Computes: ffn_out = down_proj(silu(gate_proj) * up_proj) + LoRA contributions

        LoRA is applied to: gate_proj, up_proj, down_proj

        Supports two modes:
        - Merged mode: LoRA weights stored in layer_weight (applied via mm())
        - Detached mode: LoRA weights managed externally via dispatcher
        """
        input = input.view(-1, self.embed_dim_)

        # Base FFN computation
        up_gate_out = layer_weight.gate_up_proj.mm(input)
        ffn1_out = self.alloc_tensor((input.size(0), up_gate_out.size(1) // 2), input.dtype)
        silu_and_mul_fwd(up_gate_out, ffn1_out)
        up_gate_out = None
        ffn2_out = layer_weight.down_proj.mm(ffn1_out)
        ffn1_out = None

        # Apply LoRA
        if self.use_detached_lora and self.lora_dispatcher is not None:
            # Detached mode: use dispatcher
            lora_out = self.lora_dispatcher.apply_mlp_lora(input, self.layer_num_)
            ffn2_out = ffn2_out + lora_out
        elif self.has_lora and layer_weight.has_lora_mlp:
            # Merged mode: use weights from layer_weight
            lora_out = self._apply_lora_mlp(input, layer_weight)
            ffn2_out = ffn2_out + lora_out

        return ffn2_out

    def _apply_lora_mlp(self, input: torch.Tensor, layer_weight: Qwen3VLTransformerLayerWeight) -> torch.Tensor:
        """Apply LoRA to MLP layers (merged mode).

        LoRA formula: output = lora_B @ (lora_A @ input) * scaling

        For gate_proj, up_proj: compute LoRA contributions and add to gate/up
        For down_proj: compute LoRA on the activated intermediate and add to down
        """
        scaling = self.lora_scaling

        # Compute LoRA for gate_proj: lora_B @ (lora_A @ input)
        gate_lora = layer_weight.gate_proj_b.mm(layer_weight.gate_proj_a.mm(input)) * scaling

        # Compute LoRA for up_proj: lora_B @ (lora_A @ input)
        up_lora = layer_weight.up_proj_b.mm(layer_weight.up_proj_a.mm(input)) * scaling

        # Combine gate and up LoRA contributions (same as base FFN: silu(gate) * up)
        gate_up_lora = torch.cat([gate_lora, up_lora], dim=-1)
        ffn1_lora = self.alloc_tensor((input.size(0), gate_up_lora.size(1) // 2), input.dtype)
        silu_and_mul_fwd(gate_up_lora, ffn1_lora)

        # Compute LoRA for down_proj
        down_lora = layer_weight.down_proj_b.mm(layer_weight.down_proj_a.mm(input)) * scaling

        # Add the LoRA contribution to down_proj output
        ffn1_lora = None  # Free memory
        return down_lora

    def _apply_lora_attn(
        self, input: torch.Tensor, layer_weight: Qwen3VLTransformerLayerWeight
    ) -> Dict[str, torch.Tensor]:
        """Apply LoRA to attention layers (q, k, v projections) in merged mode.

        LoRA formula: output = lora_B @ (lora_A @ input) * scaling

        Args:
            input: Input tensor [batch, hidden_size]
            layer_weight: Layer weights containing LoRA matrices

        Returns:
            Dictionary with q_lora, k_lora, v_lora tensors
        """
        scaling = self.lora_scaling

        # Compute LoRA for q_proj
        q_lora = layer_weight.q_proj_b.mm(layer_weight.q_proj_a.mm(input)) * scaling

        # Compute LoRA for k_proj
        k_lora = layer_weight.k_proj_b.mm(layer_weight.k_proj_a.mm(input)) * scaling

        # Compute LoRA for v_proj
        v_lora = layer_weight.v_proj_b.mm(layer_weight.v_proj_a.mm(input)) * scaling

        return {
            "q_lora": q_lora,
            "k_lora": k_lora,
            "v_lora": v_lora,
        }

    def _get_o(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3VLTransformerLayerWeight
    ) -> torch.Tensor:
        """Apply o_proj with LoRA support for Qwen3-VL.

        Computes: o = o_proj(attn_output) + LoRA contribution

        LoRA is applied to: o_proj

        Supports both merged mode and detached mode.
        """
        input = input.view(-1, self.tp_o_head_num_ * self.head_dim_)
        o_tensor = layer_weight.o_proj.mm(input)

        # Apply LoRA to o_proj
        if self.use_detached_lora and self.lora_dispatcher is not None:
            # Detached mode: use dispatcher
            lora_out = self.lora_dispatcher.apply_o_lora(input, self.layer_num_, infer_state)
            o_tensor = o_tensor + lora_out
        elif self.has_lora and layer_weight.has_lora_attn:
            # Merged mode: use weights from layer_weight
            scaling = self.lora_scaling
            o_lora = layer_weight.o_proj_b.mm(layer_weight.o_proj_a.mm(input)) * scaling
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
