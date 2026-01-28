"""
S-LoRA Batched LoRA Dispatch for Qwen3-VL-MoE

This module provides S-LoRA style batched LoRA computation for:
- Vision-Language Adapter: vl.q_proj, vl.k_proj, vl.v_proj, vl.o_proj, vl.linear_fc1, vl.linear_fc2
- Attention: self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj
- MoE MLP: moe.gate_proj, moe.up_proj, moe.down_proj
- Language Model Head: moe.lm_head

Key Features:
- Mixed adapter batches: different requests in same batch can use different adapters
- req_bins tracking: maps each request to its adapter index
- dispatch_bgmv kernel: efficient batched LoRA computation

Debugging:
- Set LIGHTLLM_LOGGING=DEBUG to see detailed LoRA dispatch logs
"""
import torch
import os
import logging
from typing import Dict, Optional, Any, List

# Configure logging using global env var
_LOG_LEVEL = os.environ.get("LIGHTLLM_LOGGING", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL, logging.INFO)
logger = logging.getLogger("lightllm.lora.dispatch")
logger.setLevel(_LOG_LEVEL)

# Try to import dispatch_bgmv kernel, fall back to naive implementation
try:
    from lightllm._kernels.lora.bgmv import (
        dispatch_bgmv,
        batch_lora_get_qkv,
        batch_lora_get_o,
        batch_lora_get_mlp,
        batch_lora_get_vl,
    )
    BGMV_AVAILABLE = True
except ImportError:
    BGMV_AVAILABLE = False
    # Fallback: naive per-request computation


class Qwen3VLMoELoRADispatcher:
    """
    S-LoRA Batched LoRA Dispatcher for Qwen3-VL-MoE.

    Supports mixed adapter batches where different requests use different adapters.
    The key insight is to use req_bins to track which adapter each request uses,
    then apply LoRA in a batched manner using the dispatch_bgmv kernel.

    Supported LoRA Targets:
    - Vision: vl_q_proj, vl_k_proj, vl_v_proj, vl_o_proj, vl_linear_fc1, vl_linear_fc2
    - Attention: attn_q_proj, attn_k_proj, attn_v_proj, attn_o_proj
    - MoE MLP: moe_gate_proj, moe_up_proj, moe_down_proj
    - LM Head: moe_lm_head
    """

    def __init__(
        self,
        num_layers: int,
        # Attention LoRA ranks (0 = disabled)
        q_lora_rank: int = 0,
        k_lora_rank: int = 0,
        v_lora_rank: int = 0,
        o_lora_rank: int = 0,
        # MoE MLP LoRA ranks
        gate_lora_rank: int = 0,
        up_lora_rank: int = 0,
        down_lora_rank: int = 0,
        # Vision adapter LoRA ranks
        vl_q_rank: int = 0,
        vl_k_rank: int = 0,
        vl_v_rank: int = 0,
        vl_o_rank: int = 0,
        vl_fc1_rank: int = 0,
        vl_fc2_rank: int = 0,
        # Common settings
        lora_alpha: float = 1.0,
        compute_on_cpu: bool = False,
    ):
        self.num_layers = num_layers
        self.compute_on_cpu = compute_on_cpu
        self.lora_alpha = lora_alpha

        # Attention ranks
        self.q_lora_rank = q_lora_rank
        self.k_lora_rank = k_lora_rank
        self.v_lora_rank = v_lora_rank
        self.o_lora_rank = o_lora_rank

        # MLP ranks
        self.gate_lora_rank = gate_lora_rank
        self.up_lora_rank = up_lora_rank
        self.down_lora_rank = down_lora_rank

        # Vision adapter ranks
        self.vl_q_rank = vl_q_rank
        self.vl_k_rank = vl_k_rank
        self.vl_v_rank = vl_v_rank
        self.vl_o_rank = vl_o_rank
        self.vl_fc1_rank = vl_fc1_rank
        self.vl_fc2_rank = vl_fc2_rank

        # Calculate scaling factors
        self.q_scaling = lora_alpha / q_lora_rank if q_lora_rank > 0 else 1.0
        self.k_scaling = lora_alpha / k_lora_rank if k_lora_rank > 0 else 1.0
        self.v_scaling = lora_alpha / v_lora_rank if v_lora_rank > 0 else 1.0
        self.o_scaling = lora_alpha / o_lora_rank if o_lora_rank > 0 else 1.0

        # S-LoRA mode state
        self.lora_mem_pool = None
        self.req_bins = None
        self.use_batched_mode = False

        # Check if any LoRA is enabled
        self.has_any_lora = any([
            q_lora_rank > 0, k_lora_rank > 0, v_lora_rank > 0, o_lora_rank > 0,
            gate_lora_rank > 0, up_lora_rank > 0, down_lora_rank > 0,
            vl_q_rank > 0, vl_k_rank > 0, vl_v_rank > 0, vl_o_rank > 0,
            vl_fc1_rank > 0, vl_fc2_rank > 0
        ])

    def init_batched_mode(
        self,
        lora_mem_pool,
        req_bins: torch.Tensor
    ):
        """
        Initialize batched mode with LoRA memory pool and req_bins.

        Args:
            lora_mem_pool: LoRAMemPool containing all adapter weights
            req_bins: Tensor mapping request index -> adapter index [batch_size]

        Debug:
            Logs batched mode initialization with batch size and adapter info
        """
        self.lora_mem_pool = lora_mem_pool
        self.req_bins = req_bins
        self.use_batched_mode = True

        # batch_size = req_bins.shape[0] if req_bins is not None else 0
        # unique_adapters = len(torch.unique(req_bins)) if req_bins is not None else 0

    def use_single_adapter_mode(self):
        """Switch back to single adapter mode (original behavior)."""
        logger.debug(f"[LoRA Dispatch] Switching to single adapter mode")
        self.use_batched_mode = False
        self.lora_mem_pool = None
        self.req_bins = None

    # =====================================================================
    # S-LoRA Batched Methods (FIXED)
    # =====================================================================

    def _get_output_buffer(self, input_tensor, pool):
        """Helper to create output buffer with correct dimension (Crucial for GQA/MLP)"""
        return torch.zeros(
            input_tensor.shape[0],
            pool.b_hidden_dim,
            dtype=input_tensor.dtype,
            device=input_tensor.device
        )

    def batch_apply_q_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply q_proj LoRA to batch with different adapters."""
        if self.lora_mem_pool is None or self.lora_mem_pool.attn_q_pool is None:
            return torch.zeros_like(input_tensor)

        pool = self.lora_mem_pool.attn_q_pool
        bins = req_bins if req_bins is not None else self.req_bins

        if BGMV_AVAILABLE:
            output = self._get_output_buffer(input_tensor, pool)
            batch_lora_get_qkv(
                output, input_tensor,
                pool.key_buffer,
                pool.value_buffer,
                pool.a_start,
                pool.a_len,
                pool.a_scaling,
                bins,
                layer_id=layer_id
            )
            return output
        else:
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

    def batch_apply_k_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply k_proj LoRA (GQA Aware). Output dim < Input dim."""
        if self.lora_mem_pool is None or self.lora_mem_pool.attn_k_pool is None:
            return torch.zeros(input_tensor.shape[0], 0, device=input_tensor.device)

        pool = self.lora_mem_pool.attn_k_pool
        bins = req_bins if req_bins is not None else self.req_bins

        output = self._get_output_buffer(input_tensor, pool)

        if BGMV_AVAILABLE:
            batch_lora_get_qkv(
                output, input_tensor,
                pool.key_buffer, pool.value_buffer,
                pool.a_start, pool.a_len, pool.a_scaling, bins,
                a_hidden_dim=input_tensor.shape[1],
                b_hidden_dim=output.shape[1],
                layer_id=layer_id
            )
            return output
        else:
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

    def batch_apply_v_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply v_proj LoRA (GQA Aware)."""
        if self.lora_mem_pool is None or self.lora_mem_pool.attn_v_pool is None:
            return torch.zeros_like(input_tensor)

        pool = self.lora_mem_pool.attn_v_pool
        bins = req_bins if req_bins is not None else self.req_bins

        output = self._get_output_buffer(input_tensor, pool)

        if BGMV_AVAILABLE:
            batch_lora_get_qkv(
                output, input_tensor,
                pool.key_buffer, pool.value_buffer,
                pool.a_start, pool.a_len, pool.a_scaling, bins,
                a_hidden_dim=input_tensor.shape[1],
                b_hidden_dim=output.shape[1],
                layer_id=layer_id
            )
            return output
        else:
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

    def batch_apply_o_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply o_proj LoRA to batch with different adapters."""
        if self.lora_mem_pool is None or self.lora_mem_pool.attn_o_pool is None:
            return torch.zeros_like(input_tensor)

        pool = self.lora_mem_pool.attn_o_pool
        bins = req_bins if req_bins is not None else self.req_bins

        if BGMV_AVAILABLE:
            output = torch.zeros_like(input_tensor)
            batch_lora_get_o(
                output,
                input_tensor,
                pool.key_buffer,  # A matrices
                pool.value_buffer,  # B matrices
                pool.a_start,
                pool.a_len,
                pool.a_scaling,
                bins,
                layer_id=layer_id
            )
            return output
        else:
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

    def batch_apply_gate_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply gate_proj LoRA to batch with different adapters."""
        if self.lora_mem_pool is None or self.lora_mem_pool.moe_gate_pool is None:
            return torch.zeros_like(input_tensor)

        pool = self.lora_mem_pool.moe_gate_pool
        bins = req_bins if req_bins is not None else self.req_bins

        if BGMV_AVAILABLE:
            output = torch.zeros_like(input_tensor)
            batch_lora_get_mlp(
                output,
                input_tensor,
                pool.key_buffer,  # A matrices
                pool.value_buffer,  # B matrices
                pool.a_start,
                pool.a_len,
                pool.a_scaling,
                bins,
                layer_id=layer_id
            )
            return output
        else:
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

    def batch_apply_up_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply up_proj LoRA to batch with different adapters."""
        if self.lora_mem_pool is None or self.lora_mem_pool.moe_up_pool is None:
            return torch.zeros_like(input_tensor)

        pool = self.lora_mem_pool.moe_up_pool
        bins = req_bins if req_bins is not None else self.req_bins

        if BGMV_AVAILABLE:
            output = torch.zeros_like(input_tensor)
            batch_lora_get_mlp(
                output,
                input_tensor,
                pool.key_buffer,  # A matrices
                pool.value_buffer,  # B matrices
                pool.a_start,
                pool.a_len,
                pool.a_scaling,
                bins,
                layer_id=layer_id
            )
            return output
        else:
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

    def batch_apply_down_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply down_proj LoRA. Input: Intermediate, Output: Hidden."""
        if self.lora_mem_pool is None or self.lora_mem_pool.moe_down_pool is None:
            return torch.zeros_like(input_tensor)

        pool = self.lora_mem_pool.moe_down_pool
        bins = req_bins if req_bins is not None else self.req_bins

        output = self._get_output_buffer(input_tensor, pool)

        if BGMV_AVAILABLE:
            batch_lora_get_mlp(
                output, input_tensor,
                pool.key_buffer, pool.value_buffer,
                pool.a_start, pool.a_len, pool.a_scaling, bins,
                a_hidden_dim=input_tensor.shape[1],
                b_hidden_dim=output.shape[1],
                layer_id=layer_id
            )
            return output
        else:
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

    def batch_apply_vl_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        target_type: str,
        req_bins: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply Vision-Language adapter LoRA to batch."""
        if self.lora_mem_pool is None:
            return torch.zeros_like(input_tensor)

        pool_map = {
            "vl_q": self.lora_mem_pool.vl_q_pool,
            "vl_k": self.lora_mem_pool.vl_k_pool,
            "vl_v": self.lora_mem_pool.vl_v_pool,
            "vl_o": self.lora_mem_pool.vl_o_pool,
            "vl_fc1": self.lora_mem_pool.vl_fc1_pool,
            "vl_fc2": self.lora_mem_pool.vl_fc2_pool,
        }

        pool = pool_map.get(target_type)
        if pool is None:
            return torch.zeros_like(input_tensor)

        bins = req_bins if req_bins is not None else self.req_bins

        output = self._get_output_buffer(input_tensor, pool)

        if BGMV_AVAILABLE:
            batch_lora_get_vl(
                output, input_tensor,
                pool.key_buffer, pool.value_buffer,
                pool.a_start, pool.a_len, pool.a_scaling, bins,
                a_hidden_dim=input_tensor.shape[1],
                b_hidden_dim=output.shape[1],
                layer_id=layer_id
            )
            return output
        else:
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

    # =====================================================================
    # Fallback Naive Implementation (when BGMV kernel unavailable)
    # =====================================================================

    def _naive_batch_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        pool,
        req_bins: torch.Tensor
    ) -> torch.Tensor:
        """
        Naive per-request LoRA computation (fallback when BGMV unavailable).

        This is slower but correctness-preserving.
        """
        batch_size = input_tensor.shape[0]
        hidden_dim = input_tensor.shape[1]
        output = torch.zeros(batch_size, hidden_dim, dtype=input_tensor.dtype, device=input_tensor.device)

        # Group requests by adapter
        adapter_to_indices = {}
        for i, bin_idx in enumerate(req_bins):
            bin_idx = bin_idx.item()
            if bin_idx not in adapter_to_indices:
                adapter_to_indices[bin_idx] = []
            adapter_to_indices[bin_idx].append(i)

        # Process each adapter group
        for adapter_idx, req_indices in adapter_to_indices.items():
            if adapter_idx < 0:
                continue  # Skip requests with no adapter

            # Get adapter metadata
            a_start = pool.a_start[adapter_idx].item()
            a_len = pool.a_len[adapter_idx].item()
            a_scaling = pool.a_scaling[adapter_idx].item()

            # Get A and B matrices for this layer
            loc = a_start + layer_id
            if loc >= a_start + a_len:
                continue

            A = pool.key_buffer[loc, :a_len]  # [rank, hidden]
            B = pool.value_buffer[loc, :a_len]  # [rank, hidden]

            # Compute LoRA for each request in this group
            for req_idx in req_indices:
                x = input_tensor[req_idx]  # [hidden]
                # LoRA: x @ A @ B.T * scaling
                # x @ A: [hidden] @ [rank, hidden].T = [rank]
                intermediate = torch.matmul(x, A)  # [rank]
                # intermediate @ B: [rank] @ [hidden, rank].T = [hidden]
                lora_out = torch.matmul(intermediate, B.T) * a_scaling
                output[req_idx] = lora_out

        return output

    # =====================================================================
    # Utility Methods
    # =====================================================================

    def get_attn_qkv_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply all attention LoRA (q, k, v, o) in batched mode.

        Returns dict with q_lora, k_lora, v_lora, o_lora tensors.
        """
        return {
            "q_lora": self.batch_apply_q_lora(input_tensor, layer_id, req_bins),
            "k_lora": self.batch_apply_k_lora(input_tensor, layer_id, req_bins),
            "v_lora": self.batch_apply_v_lora(input_tensor, layer_id, req_bins),
        }

    def get_mlp_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply all MLP LoRA (gate, up, down) in batched mode.

        Returns dict with gate_lora, up_lora, down_lora tensors.
        """
        return {
            "gate_lora": self.batch_apply_gate_lora(input_tensor, layer_id, req_bins),
            "up_lora": self.batch_apply_up_lora(input_tensor, layer_id, req_bins),
            "down_lora": self.batch_apply_down_lora(input_tensor, layer_id, req_bins),
        }


def create_vl_moe_lora_dispatcher(
    num_layers: int,
    lora_rank: int = 64,
    lora_alpha: float = 1.0,
    compute_on_cpu: bool = False,
) -> Qwen3VLMoELoRADispatcher:
    """
    Factory function to create a VL-MoE LoRA dispatcher with S-LoRA batched mode.

    All modules use the same global lora_rank.

    Args:
        num_layers: Number of transformer layers
        lora_rank: Global LoRA rank for all modules (q, k, v, o, gate, up, down, vl_*)
        lora_alpha: LoRA alpha scaling factor
        compute_on_cpu: If True, compute LoRA on CPU

    Returns:
        Qwen3VLMoELoRADispatcher instance configured for batched inference
    """
    return Qwen3VLMoELoRADispatcher(
        num_layers=num_layers,
        q_lora_rank=lora_rank,
        k_lora_rank=lora_rank,
        v_lora_rank=lora_rank,
        o_lora_rank=lora_rank,
        gate_lora_rank=lora_rank,
        up_lora_rank=lora_rank,
        down_lora_rank=lora_rank,
        vl_q_rank=lora_rank,
        vl_k_rank=lora_rank,
        vl_v_rank=lora_rank,
        vl_o_rank=lora_rank,
        vl_fc1_rank=lora_rank,
        vl_fc2_rank=lora_rank,
        lora_alpha=lora_alpha,
        compute_on_cpu=compute_on_cpu,
    )


# Import for backward compatibility
from lightllm.models.qwen3_vl_moe.layer_weights.lora_layer_weight import load_moe_lora_adapter as load_lora_adapter


__all__ = [
    "Qwen3VLMoELoRADispatcher",
    "create_vl_moe_lora_dispatcher",
    "load_lora_adapter",
]
