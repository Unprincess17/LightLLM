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

from lightllm.server.core.objs.lora_compute_config import LoRAComputeConfig

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

# Try to import AVX-512 CPU kernel for CPU offload mode
try:
    from lightllm._kernels.lora.lora_cpu_kernel import (
        batch_lora_avx,
        lora_down_avx,
        lora_up_avx,
        is_available as AVX_AVAILABLE,
    )
    if AVX_AVAILABLE:
        logger.info("AVX-512 BF16 CPU kernel available")
except ImportError:
    AVX_AVAILABLE = False


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
        lora_compute_config: Optional[LoRAComputeConfig] = None,
    ):
        self.num_layers = num_layers
        self.lora_compute_config = lora_compute_config or LoRAComputeConfig()
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

        # CPU Storage + GPU Compute scratchpad buffers (compact, per-batch)
        self.gpu_scratchpad_a = None
        self.gpu_scratchpad_b = None
        self.transfer_stream = None

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
        # Use actual buffer dimension to ensure correct shape for copy operations
        return torch.zeros(
            input_tensor.shape[0],
            pool.value_buffer.shape[2],  # Use actual B buffer dimension
            dtype=input_tensor.dtype,
            device=input_tensor.device
        )

    def _should_use_cpu_compute(self, component: str) -> bool:
        """Check if a component should use CPU computation."""
        if self.lora_compute_config is None:
            return False
        return self.lora_compute_config.should_compute_on_cpu(component)

    def _should_use_cpu_storage(self, component: str) -> bool:
        """Check if component uses CPU storage (requires transfer to GPU for compute)."""
        if self.lora_compute_config is None:
            return False
        return self.lora_compute_config.get_storage_device(component) == "cpu"

    def _ensure_compact_scratchpad(self, pool, device, active_count: int):
        """
        Allocate scratchpad for ONLY active adapters (compact, not full pool).
        Avoids OOM with thousands of adapters.

        NOTE: Always reallocate when pool dimensions change (e.g., switching from Q to K pool).
        """
        max_rank = pool.max_rank
        a_hidden = pool.key_buffer.shape[2]
        b_hidden = pool.value_buffer.shape[2]

        # Check if we need to resize or if dimensions changed
        needs_resize = False
        current_a_hidden = self.gpu_scratchpad_a.shape[2] if self.gpu_scratchpad_a is not None else 0
        current_b_hidden = self.gpu_scratchpad_b.shape[2] if self.gpu_scratchpad_b is not None else 0

        if self.gpu_scratchpad_a is None or self.gpu_scratchpad_a.shape[0] < active_count:
            needs_resize = True
        elif current_a_hidden != a_hidden or current_b_hidden != b_hidden:
            # Pool dimensions changed (e.g., Q pool -> K pool), need to reallocate
            needs_resize = True

        if needs_resize:
            # Allocate for active count only
            self.gpu_scratchpad_a = torch.empty(
                (active_count, max_rank, a_hidden),
                dtype=pool.key_buffer.dtype, device=device
            )
            self.gpu_scratchpad_b = torch.empty(
                (active_count, max_rank, b_hidden),
                dtype=pool.value_buffer.dtype, device=device
            )
            # Create dedicated stream for async transfers
            if device.type == "cuda" and self.transfer_stream is None:
                self.transfer_stream = torch.cuda.Stream(priority=0)

    def _transfer_compact_to_gpu(self, pool, layer_id: int, global_adapter_ids: torch.Tensor,
                                  a_dest: torch.Tensor, b_dest: torch.Tensor):
        """
        Transfer ONLY active adapters from CPU pool to compact GPU scratchpad.
        Maps: Global[5, 999] → Local[0, 1]
        """
        # Get CPU slot indices for each active adapter (indices must be on same device as indexed tensor)
        cpu_slots = pool.a_start[global_adapter_ids.cpu()] + layer_id  # [active_count]

        logger.debug(f"[LoRA Transfer] pool={type(pool).__name__}, layer_id={layer_id}, active_count={len(global_adapter_ids)}")
        logger.debug(f"[LoRA Transfer] a_dest[0].shape={a_dest[0].shape if len(a_dest) > 0 else 'empty'}, b_dest[0].shape={b_dest[0].shape if len(b_dest) > 0 else 'empty'}")
        logger.debug(f"[LoRA Transfer] pool.key_buffer.shape={pool.key_buffer.shape}, pool.value_buffer.shape={pool.value_buffer.shape}")

        # Async transfer using dedicated stream
        if self.transfer_stream is not None:
            with torch.cuda.stream(self.transfer_stream):
                for i in range(len(global_adapter_ids)):
                    adapter_id = global_adapter_ids[i].item()
                    if adapter_id < 0:
                        continue
                    slot = cpu_slots[i].item()
                    logger.debug(f"[LoRA Transfer] i={i}, slot={slot}, key_buffer[{slot}].shape={pool.key_buffer[slot].shape}, value_buffer[{slot}].shape={pool.value_buffer[slot].shape}")
                    a_dest[i].copy_(pool.key_buffer[slot], non_blocking=True)
                    b_dest[i].copy_(pool.value_buffer[slot], non_blocking=True)
            self.transfer_stream.synchronize()  # Sync for baseline correctness
        else:
            # Sync fallback (no CUDA stream available)
            for i in range(len(global_adapter_ids)):
                adapter_id = global_adapter_ids[i].item()
                if adapter_id < 0:
                    continue
                slot = cpu_slots[i].item()
                a_dest[i].copy_(pool.key_buffer[slot])
                b_dest[i].copy_(pool.value_buffer[slot])

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
        if bins is None:
            return torch.zeros_like(input_tensor)

        # Use CPU compute if configured
        if self._should_use_cpu_compute("attn"):
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

        if BGMV_AVAILABLE:
            output = self._get_output_buffer(input_tensor, pool)

            # Compact dispatcher: CPU storage + GPU compute
            if self._should_use_cpu_storage("attn"):
                # 1. Identify ONLY the adapters needed for this batch
                unique_adapters, inverse_indices = torch.unique(bins, return_inverse=True)
                active_count = unique_adapters.size(0)

                # 2. Allocate SMALL scratchpad (only for active adapters)
                self._ensure_compact_scratchpad(pool, input_tensor.device, active_count)

                # 3. Transfer ONLY active adapters to packed scratchpad
                assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None
                self._transfer_compact_to_gpu(
                    pool, layer_id, unique_adapters,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b
                )

                # 4. Create TEMPORARY compact metadata for kernel
                temp_a_start = torch.arange(active_count, device=input_tensor.device, dtype=torch.int32)
                temp_a_len = torch.ones(active_count, device=input_tensor.device, dtype=torch.int32)
                temp_scaling = pool.a_scaling[unique_adapters.long().cpu()].to(input_tensor.device).to(input_tensor.device)

                # 5. Launch kernel with REMAPPED indices
                assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None
                batch_lora_get_qkv(
                    output, input_tensor,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b,
                    temp_a_start, temp_a_len, temp_scaling,
                    inverse_indices,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=0
                )
            else:
                # Standard GPU path
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
        if bins is None:
            return torch.zeros(input_tensor.shape[0], pool.value_buffer.shape[2], dtype=input_tensor.dtype, device=input_tensor.device)

        output = self._get_output_buffer(input_tensor, pool)

        # Use CPU compute if configured
        if self._should_use_cpu_compute("attn"):
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

        if BGMV_AVAILABLE:
            # Compact dispatcher: CPU storage + GPU compute
            if self._should_use_cpu_storage("attn"):
                unique_adapters, inverse_indices = torch.unique(bins, return_inverse=True)
                active_count = unique_adapters.size(0)
                self._ensure_compact_scratchpad(pool, input_tensor.device, active_count)
                assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None
                self._transfer_compact_to_gpu(
                    pool, layer_id, unique_adapters,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b
                )
                temp_a_start = torch.arange(active_count, device=input_tensor.device, dtype=torch.int32)
                temp_a_len = torch.ones(active_count, device=input_tensor.device, dtype=torch.int32)
                temp_scaling = pool.a_scaling[unique_adapters.long().cpu()].to(input_tensor.device)
                batch_lora_get_qkv(
                    output, input_tensor,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b,
                    temp_a_start, temp_a_len, temp_scaling,
                    inverse_indices,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=0
                )
            else:
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
        if bins is None:
            return torch.zeros_like(input_tensor)

        output = self._get_output_buffer(input_tensor, pool)

        # Use CPU compute if configured
        if self._should_use_cpu_compute("attn"):
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

        if BGMV_AVAILABLE:
            # Compact dispatcher: CPU storage + GPU compute
            if self._should_use_cpu_storage("attn"):
                unique_adapters, inverse_indices = torch.unique(bins, return_inverse=True)
                active_count = unique_adapters.size(0)
                self._ensure_compact_scratchpad(pool, input_tensor.device, active_count)
                assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None
                self._transfer_compact_to_gpu(
                    pool, layer_id, unique_adapters,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b
                )
                temp_a_start = torch.arange(active_count, device=input_tensor.device, dtype=torch.int32)
                temp_a_len = torch.ones(active_count, device=input_tensor.device, dtype=torch.int32)
                temp_scaling = pool.a_scaling[unique_adapters.long().cpu()].to(input_tensor.device)
                batch_lora_get_qkv(
                    output, input_tensor,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b,
                    temp_a_start, temp_a_len, temp_scaling,
                    inverse_indices,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=0
                )
            else:
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
        if bins is None:
            return torch.zeros_like(input_tensor)

        # Use CPU compute if configured
        if self._should_use_cpu_compute("attn"):
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

        if BGMV_AVAILABLE:
            output = torch.zeros_like(input_tensor)
            # Compact dispatcher: CPU storage + GPU compute
            if self._should_use_cpu_storage("attn"):
                unique_adapters, inverse_indices = torch.unique(bins, return_inverse=True)
                active_count = unique_adapters.size(0)
                self._ensure_compact_scratchpad(pool, input_tensor.device, active_count)
                assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None
                self._transfer_compact_to_gpu(
                    pool, layer_id, unique_adapters,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b
                )
                temp_a_start = torch.arange(active_count, device=input_tensor.device, dtype=torch.int32)
                temp_a_len = torch.ones(active_count, device=input_tensor.device, dtype=torch.int32)
                temp_scaling = pool.a_scaling[unique_adapters.long().cpu()].to(input_tensor.device)
                batch_lora_get_o(
                    output,
                    input_tensor,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b,
                    temp_a_start, temp_a_len, temp_scaling,
                    inverse_indices,
                    layer_id=0
                )
            else:
                batch_lora_get_o(
                    output,
                    input_tensor,
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

    def batch_apply_gate_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None,
        expert_id: Optional[int] = None
    ) -> torch.Tensor:
        """Apply gate_proj LoRA to batch with different adapters.

        Args:
            input_tensor: Input tensor [batch, hidden]
            layer_id: Layer index
            req_bins: Request to adapter mapping
            expert_id: For MoE, the LOCAL expert index to apply LoRA for.
                       In EP mode, this is the local index (0 to num_local_experts-1).
                       If None, uses base layer_id (for weight loading).
        """
        if self.lora_mem_pool is None or self.lora_mem_pool.moe_gate_pool is None:
            return torch.zeros_like(input_tensor)

        pool = self.lora_mem_pool.moe_gate_pool
        bins = req_bins if req_bins is not None else self.req_bins
        if bins is None:
            return torch.zeros_like(input_tensor)

        # Calculate buffer index with expert dimension
        # expert_id is now always LOCAL (after translation in transformer_layer_infer.py)
        if expert_id is not None:
            # Use pool's actual num_experts which matches local expert count
            buffer_layer_id = layer_id * pool.num_experts + expert_id
        else:
            buffer_layer_id = layer_id

        # Use CPU compute if configured
        if self._should_use_cpu_compute("moe"):
            return self._naive_batch_lora(input_tensor, buffer_layer_id, pool, bins)

        if BGMV_AVAILABLE:
            output = self._get_output_buffer(input_tensor, pool)
            # Compact dispatcher: CPU storage + GPU compute
            if self._should_use_cpu_storage("moe"):
                unique_adapters, inverse_indices = torch.unique(bins, return_inverse=True)
                active_count = unique_adapters.size(0)
                self._ensure_compact_scratchpad(pool, input_tensor.device, active_count)
                assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None
                self._transfer_compact_to_gpu(
                    pool, buffer_layer_id, unique_adapters,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b
                )
                temp_a_start = torch.arange(active_count, device=input_tensor.device, dtype=torch.int32)
                temp_a_len = torch.ones(active_count, device=input_tensor.device, dtype=torch.int32)
                temp_scaling = pool.a_scaling[unique_adapters.long().cpu()].to(input_tensor.device)
                batch_lora_get_mlp(
                    output,
                    input_tensor,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b,
                    temp_a_start, temp_a_len, temp_scaling,
                    inverse_indices,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=0
                )
            else:
                batch_lora_get_mlp(
                    output,
                    input_tensor,
                    pool.key_buffer,
                    pool.value_buffer,
                    pool.a_start,
                    pool.a_len,
                    pool.a_scaling,
                    bins,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=buffer_layer_id
                )
            return output
        else:
            return self._naive_batch_lora(input_tensor, buffer_layer_id, pool, bins)

    def batch_apply_up_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None,
        expert_id: Optional[int] = None
    ) -> torch.Tensor:
        """Apply up_proj LoRA to batch with different adapters.

        Args:
            input_tensor: Input tensor [batch, hidden]
            layer_id: Layer index
            req_bins: Request to adapter mapping
            expert_id: For MoE, the expert index to apply LoRA for.
                       If None, uses base layer_id (for weight loading).
        """
        if self.lora_mem_pool is None or self.lora_mem_pool.moe_up_pool is None:
            return torch.zeros_like(input_tensor)

        pool = self.lora_mem_pool.moe_up_pool
        bins = req_bins if req_bins is not None else self.req_bins
        if bins is None:
            return torch.zeros_like(input_tensor)

        # [MODIFIED] Calculate buffer index with expert dimension
        if expert_id is not None:
            buffer_layer_id = layer_id * pool.num_experts + expert_id
        else:
            buffer_layer_id = layer_id

        # Use CPU compute if configured
        if self._should_use_cpu_compute("moe"):
            return self._naive_batch_lora(input_tensor, buffer_layer_id, pool, bins)

        if BGMV_AVAILABLE:
            output = self._get_output_buffer(input_tensor, pool)
            # Compact dispatcher: CPU storage + GPU compute
            if self._should_use_cpu_storage("moe"):
                unique_adapters, inverse_indices = torch.unique(bins, return_inverse=True)
                active_count = unique_adapters.size(0)
                self._ensure_compact_scratchpad(pool, input_tensor.device, active_count)
                assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None
                self._transfer_compact_to_gpu(
                    pool, buffer_layer_id, unique_adapters,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b
                )
                temp_a_start = torch.arange(active_count, device=input_tensor.device, dtype=torch.int32)
                temp_a_len = torch.ones(active_count, device=input_tensor.device, dtype=torch.int32)
                temp_scaling = pool.a_scaling[unique_adapters.long().cpu()].to(input_tensor.device)
                batch_lora_get_mlp(
                    output,
                    input_tensor,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b,
                    temp_a_start, temp_a_len, temp_scaling,
                    inverse_indices,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=0
                )
            else:
                batch_lora_get_mlp(
                    output,
                    input_tensor,
                    pool.key_buffer,
                    pool.value_buffer,
                    pool.a_start,
                    pool.a_len,
                    pool.a_scaling,
                    bins,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=buffer_layer_id
                )
            return output
        else:
            return self._naive_batch_lora(input_tensor, buffer_layer_id, pool, bins)

    def batch_apply_down_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None,
        expert_id: Optional[int] = None
    ) -> torch.Tensor:
        """Apply down_proj LoRA. Input: Intermediate, Output: Hidden.

        Args:
            input_tensor: Input tensor [batch, hidden]
            layer_id: Layer index
            req_bins: Request to adapter mapping
            expert_id: For MoE, the expert index to apply LoRA for.
                       If None, uses base layer_id (for weight loading).
        """
        if self.lora_mem_pool is None or self.lora_mem_pool.moe_down_pool is None:
            return torch.zeros_like(input_tensor)

        pool = self.lora_mem_pool.moe_down_pool
        bins = req_bins if req_bins is not None else self.req_bins
        if bins is None:
            return torch.zeros_like(input_tensor)

        # [MODIFIED] Calculate buffer index with expert dimension
        if expert_id is not None:
            buffer_layer_id = layer_id * pool.num_experts + expert_id
        else:
            buffer_layer_id = layer_id

        output = self._get_output_buffer(input_tensor, pool)

        # Use CPU compute if configured
        if self._should_use_cpu_compute("moe"):
            return self._naive_batch_lora(input_tensor, buffer_layer_id, pool, bins)

        if BGMV_AVAILABLE:
            # Compact dispatcher: CPU storage + GPU compute
            if self._should_use_cpu_storage("moe"):
                unique_adapters, inverse_indices = torch.unique(bins, return_inverse=True)
                active_count = unique_adapters.size(0)
                self._ensure_compact_scratchpad(pool, input_tensor.device, active_count)
                assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None
                self._transfer_compact_to_gpu(
                    pool, buffer_layer_id, unique_adapters,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b
                )
                temp_a_start = torch.arange(active_count, device=input_tensor.device, dtype=torch.int32)
                temp_a_len = torch.ones(active_count, device=input_tensor.device, dtype=torch.int32)
                temp_scaling = pool.a_scaling[unique_adapters.long().cpu()].to(input_tensor.device)
                batch_lora_get_mlp(
                    output, input_tensor,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b,
                    temp_a_start, temp_a_len, temp_scaling,
                    inverse_indices,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=0
                )
            else:
                batch_lora_get_mlp(
                    output, input_tensor,
                    pool.key_buffer, pool.value_buffer,
                    pool.a_start, pool.a_len, pool.a_scaling, bins,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=buffer_layer_id
                )
            return output
        else:
            return self._naive_batch_lora(input_tensor, buffer_layer_id, pool, bins)

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
        if bins is None:
            return torch.zeros_like(input_tensor)

        output = self._get_output_buffer(input_tensor, pool)

        # Use CPU compute if configured
        if self._should_use_cpu_compute("vl"):
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

        if BGMV_AVAILABLE:
            # Compact dispatcher: CPU storage + GPU compute
            if self._should_use_cpu_storage("vl"):
                unique_adapters, inverse_indices = torch.unique(bins, return_inverse=True)
                active_count = unique_adapters.size(0)
                self._ensure_compact_scratchpad(pool, input_tensor.device, active_count)
                assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None
                self._transfer_compact_to_gpu(
                    pool, layer_id, unique_adapters,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b
                )
                temp_a_start = torch.arange(active_count, device=input_tensor.device, dtype=torch.int32)
                temp_a_len = torch.ones(active_count, device=input_tensor.device, dtype=torch.int32)
                temp_scaling = pool.a_scaling[unique_adapters.long().cpu()].to(input_tensor.device)
                batch_lora_get_vl(
                    output, input_tensor,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b,
                    temp_a_start, temp_a_len, temp_scaling,
                    inverse_indices,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=0
                )
            else:
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
        req_bins: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Naive per-request LoRA computation (fallback when BGMV unavailable).

        Args:
            input_tensor: Input tensor [batch, hidden]
            layer_id: Layer index
            pool: Module pool
            req_bins: Request to adapter mapping (optional, uses self.req_bins if None)

        This is slower but correctness-preserving.
        Uses AVX-512 BF16 kernel when available on CPU.
        """
        # Ensure req_bins is available
        if req_bins is None:
            req_bins = self.req_bins
        if req_bins is None:
            return torch.zeros(input_tensor.shape[0], pool.value_buffer.shape[2], dtype=input_tensor.dtype, device=input_tensor.device)

        batch_size = input_tensor.shape[0]
        # Use pool's B dimension (handles GQA where K/V output != input)
        output_dim = pool.value_buffer.shape[2]
        output = torch.zeros(batch_size, output_dim, dtype=input_tensor.dtype, device=input_tensor.device)

        # Truncate req_bins to match batch_size (handles decode phase with fewer requests)
        if len(req_bins) > batch_size:
            req_bins = req_bins[:batch_size]

        # Group requests by adapter
        adapter_to_indices = {}
        for i, bin_idx in enumerate(req_bins):
            bin_idx = bin_idx.item()
            if bin_idx not in adapter_to_indices:
                adapter_to_indices[bin_idx] = []
            adapter_to_indices[bin_idx].append(i)

        # Determine if we should use AVX kernel
        use_avx = (AVX_AVAILABLE and
                   input_tensor.device.type == 'cpu' and
                   input_tensor.dtype == torch.bfloat16)

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
            if use_avx:
                # Use AVX-512 BF16 kernel for batched computation
                batch_input = input_tensor[req_indices]  # [n, hidden]

                # Convert to bfloat16 if needed
                if A.dtype != torch.bfloat16:
                    A = A.to(dtype=torch.bfloat16)
                    B = B.to(dtype=torch.bfloat16)

                # Ensure contiguous layout
                if not batch_input.is_contiguous():
                    batch_input = batch_input.contiguous()
                if not A.is_contiguous():
                    A = A.contiguous()
                if not B.is_contiguous():
                    B = B.contiguous()

                # Call AVX kernel for batched LoRA
                batch_output = batch_lora_avx(batch_input, A, B, a_scaling)  # [n, hidden]
                output[req_indices] = batch_output
            else:
                # Fallback: PyTorch matmul per request
                # Move to input device and dtype if pool is on different device
                if A.device != input_tensor.device or A.dtype != input_tensor.dtype:
                    A = A.to(dtype=input_tensor.dtype, device=input_tensor.device)
                    B = B.to(dtype=input_tensor.dtype, device=input_tensor.device)

                for req_idx in req_indices:
                    x = input_tensor[req_idx]  # [hidden]
                    # LoRA: x @ A @ B * scaling
                    # A stored as [rank, hidden], need A.T for [hidden, rank]
                    # B stored as [rank, hidden], need B for [hidden, rank]
                    # x @ A.T: [hidden] @ [hidden, rank] = [rank]
                    intermediate = torch.matmul(x, A.T)  # [rank]
                    # intermediate @ B: [rank] @ [hidden, rank] = [hidden]
                    lora_out = torch.matmul(intermediate, B) * a_scaling
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
    lora_compute_config: Optional[LoRAComputeConfig] = None,
) -> Qwen3VLMoELoRADispatcher:
    """
    Factory function to create a VL-MoE LoRA dispatcher with S-LoRA batched mode.

    All modules use the same global lora_rank.

    Args:
        num_layers: Number of transformer layers
        lora_rank: Global LoRA rank for all modules (q, k, v, o, gate, up, down, vl_*)
        lora_alpha: LoRA alpha scaling factor
        lora_compute_config: Configuration for compute location per component

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
        lora_compute_config=lora_compute_config,
    )


# Import for backward compatibility
from lightllm.models.qwen3_vl_moe.layer_weights.lora_layer_weight import load_moe_lora_adapter as load_lora_adapter


__all__ = [
    "Qwen3VLMoELoRADispatcher",
    "create_vl_moe_lora_dispatcher",
    "load_lora_adapter",
]
