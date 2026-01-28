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

# Configure logging using global env var
_LOG_LEVEL = os.environ.get("LIGHTLLM_LOGGING", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL, logging.INFO)
logger = logging.getLogger("lightllm.lora.infer")
logger.setLevel(_LOG_LEVEL)


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

    def set_req_bins(self, req_bins: torch.Tensor):
        """Set the req_bins tensor for batched mode.

        req_bins[i] contains the adapter index for request i in the batch.

        Debug:
            Logs the req_bins configuration
        """
        self.req_bins_ = req_bins

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
            logger.debug(f"[LoRA Infer] Layer {self.layer_num_}: apply_qkv_lora batch={batch_size}")

            lora_results = self.lora_dispatcher_.get_attn_qkv_lora(
                input, self.layer_num_, self.req_bins_
            )
            logger.debug(f"[LoRA Infer] Layer {self.layer_num_}: q_shape={q.shape}, cache_kv_shape={cache_kv.shape}")
            logger.debug(f"[LoRA Infer]   q_lora_shape={lora_results['q_lora'].shape}")
            logger.debug(f"[LoRA Infer]   k_lora_shape={lora_results['k_lora'].shape}")
            logger.debug(f"[LoRA Infer]   v_lora_shape={lora_results['v_lora'].shape}")
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

            logger.debug(f"[LoRA Infer]   q_lora norm={lora_results['q_lora'].norm().item():.4f}")
            logger.debug(f"[LoRA Infer]   k_lora norm={lora_results['k_lora'].norm().item():.4f}")
            logger.debug(f"[LoRA Infer]   v_lora norm={lora_results['v_lora'].norm().item():.4f}")

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

            logger.debug(f"[LoRA Infer]   o_lora norm={o_lora.norm().item():.4f}")

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
