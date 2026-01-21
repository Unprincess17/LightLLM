"""
S-LoRA dispatch_bgmv Triton Kernel

This module provides the Batched GPU Memory View (BGMV) kernel for efficient
batched LoRA computation. The key idea is that each request in a batch can use
a different adapter, and req_bins tracks which adapter each request uses.

The kernel computes: y[req] = x[req] @ A[adapter(req)] @ B[adapter(req)] * scaling

Note: A and B matrices are stored in SEPARATE buffers (not interleaved).
- A matrices: stored in a_buffer (key_buffer in the memory pool)
- B matrices: stored in b_buffer (value_buffer in the memory pool)
"""
import torch
import triton
import triton.language as tl
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

@triton.jit
def bgmv_kernel(
    y_ptr, x_ptr, a_buffer_ptr, b_buffer_ptr, a_start_ptr, a_len_ptr, a_scaling_ptr, req_bins_ptr,
    stride_yb, stride_yh,  # y: [batch, h_out]
    stride_xb, stride_xh,  # x: [batch, h_in]
    stride_ab, stride_ar, stride_ah,  # a_buffer: [pool_size, max_rank, h_in]
    stride_bb, stride_br, stride_bh,  # b_buffer: [pool_size, max_rank, h_out]
    stride_ab_meta,  # a_start, a_len, a_scaling: [num_adapters]
    batch_size,
    h_in, h_out,      # CHANGED: Separate input and output dimensions
    max_rank,
    BLOCK_N_IN: tl.constexpr,  # CHANGED: Block size for input dim
    BLOCK_N_OUT: tl.constexpr, # CHANGED: Block size for output dim
    BLOCK_K: tl.constexpr,
):
    """
    Batched GPU Memory View (BGMV) kernel for LoRA with decoupled Input/Output dimensions.

    Supports GQA where Input Dim (Hidden Size) != Output Dim (KV Size).
    """
    req_idx = tl.program_id(0)

    if req_idx >= batch_size:
        return

    # 1. Define separate offsets for Input (A/X) and Output (B/Y)
    offs_n_in = tl.arange(0, BLOCK_N_IN)
    offs_n_out = tl.arange(0, BLOCK_N_OUT)
    offs_k = tl.arange(0, BLOCK_K)

    # -----------------------------------------------------------
    # Phase 1: Input Processing (Reading X and A, Reduction over h_in)
    # -----------------------------------------------------------

    # Load input x[req_idx, :] - [h_in] using BLOCK_N_IN
    x_ptrs = x_ptr + stride_xb * req_idx + stride_xh * offs_n_in
    x_mask = offs_n_in < h_in
    x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

    # Get adapter index and metadata
    adapter_idx = tl.load(req_bins_ptr + req_idx)
    a_start = tl.load(a_start_ptr + adapter_idx)
    a_len = tl.load(a_len_ptr + adapter_idx)
    a_scaling = tl.load(a_scaling_ptr + adapter_idx)

    # Initialize accumulator for output [BLOCK_N_OUT]
    acc = tl.zeros((BLOCK_N_OUT,), dtype=tl.float32)

    # Compute number of complete rank chunks
    num_chunks = (max_rank + BLOCK_K - 1) // BLOCK_K

    for chunk_idx in range(num_chunks):
        k_base = chunk_idx * BLOCK_K

        # Compute actual valid rank offsets for this chunk
        rank_offsets = k_base + offs_k
        rank_valid = rank_offsets < a_len

        # --- Load A (Down Proj) ---
        # A matrix shape: [max_rank, h_in]
        a_loc = a_start
        a_ptr = a_buffer_ptr + stride_ab * a_loc + stride_ar * k_base + stride_ah * 0

        # Load A using offs_n_in
        a_vals = tl.load(
            a_ptr + stride_ar * offs_k[:, None] + stride_ah * offs_n_in[None, :],
            mask=rank_valid[:, None] & (offs_n_in[None, :] < h_in), # Safety mask
            other=0.0
        )

        # Cast to float32
        x_fp32 = tl.cast(x_vals, tl.float32)
        a_fp32 = tl.cast(a_vals, tl.float32)

        # Compute h_chunk = x @ A.T
        # Sum over dimension N_IN: [1, N_IN] * [K, N_IN] -> [K]
        h_chunk = tl.sum(x_fp32[None, :] * a_fp32, axis=1)

        # Zero out invalid ranks
        h_chunk = tl.where(rank_valid, h_chunk, 0.0)

        # -----------------------------------------------------------
        # Phase 2: Output Processing (Reading B, Writing Y)
        # -----------------------------------------------------------

        # --- Load B (Up Proj) ---
        # B matrix shape: [max_rank, h_out] - Uses h_out and BLOCK_N_OUT
        b_loc = a_start
        b_ptr = b_buffer_ptr + stride_bb * b_loc + stride_br * k_base + stride_bh * 0

        # Load B using offs_n_out
        b_vals = tl.load(
            b_ptr + stride_br * offs_k[:, None] + stride_bh * offs_n_out[None, :],
            mask=rank_valid[:, None] & (offs_n_out[None, :] < h_out), # Safety mask
            other=0.0
        )

        b_fp32 = tl.cast(b_vals, tl.float32)

        # Compute acc += h_chunk @ B
        # Broadcast h_chunk [K] against B [K, N_OUT] -> Sum over K -> [N_OUT]
        chunk_contrib = tl.sum(h_chunk[:, None] * b_fp32, axis=0)
        acc = acc + chunk_contrib * a_scaling

    # Store output y[req_idx, :] - [h_out] using BLOCK_N_OUT
    y_ptrs = y_ptr + stride_yb * req_idx + stride_yh * offs_n_out
    y_mask = offs_n_out < h_out
    tl.store(y_ptrs, acc, mask=y_mask)


def dispatch_bgmv(
    y: torch.Tensor,
    x: torch.Tensor,
    a_buffer: torch.Tensor,
    b_buffer: torch.Tensor,
    a_start: torch.Tensor,
    a_len: torch.Tensor,
    a_scaling: torch.Tensor,
    req_bins: torch.Tensor,
    # Added optional dims to support asymmetry
    a_hidden_dim: int | None = None,
    b_hidden_dim: int | None = None,
):
    """
    Batched GPU Memory View (BGMV) dispatch.

    Args:
        a_hidden_dim: The hidden dimension of A matrix (input dim). If None, inferred from x.
        b_hidden_dim: The hidden dimension of B matrix (output dim). If None, inferred from y.
    """
    batch_size = x.shape[0]

    # Infer dimensions if not provided
    h_in = a_hidden_dim if a_hidden_dim is not None else x.shape[1]
    h_out = b_hidden_dim if b_hidden_dim is not None else y.shape[1]
    
    # 检查是否有 NaN/Inf
    assert not torch.isnan(x).any(), "x has NaN"
    assert not torch.isnan(y).any(), "y has NaN"
    
    # 确认 req_bins 有效
    logger.debug(f"req_bins max={req_bins.max()}, a_buffer.shape={a_buffer.shape}")

    # Validation
    pool_size, max_rank, buf_h_in = a_buffer.shape
    _, _, buf_h_out = b_buffer.shape

    assert h_in == buf_h_in, f"Input dim mismatch: x({h_in}) vs a_buffer({buf_h_in})"
    assert h_out == buf_h_out, f"Output dim mismatch: y({h_out}) vs b_buffer({buf_h_out})"

    # Helper to pick block size
    def get_block_n(dim):
        if dim <= 512: return 512
        if dim <= 1024: return 1024
        if dim <= 2048: return 2048
        if dim <= 4096: return 4096
        return 4096 # Cap at 4096 for now, or next_power_of_2(dim)

    BLOCK_N_IN = get_block_n(h_in)
    BLOCK_N_OUT = get_block_n(h_out)
    BLOCK_K = min(64, max_rank)

    grid = (batch_size,)

    bgmv_kernel[grid](
        y, x, a_buffer, b_buffer,
        a_start, a_len, a_scaling, req_bins,
        y.stride(0), y.stride(1),
        x.stride(0), x.stride(1),
        a_buffer.stride(0), a_buffer.stride(1), a_buffer.stride(2),
        b_buffer.stride(0), b_buffer.stride(1), b_buffer.stride(2),
        a_start.stride(0),
        batch_size,
        h_in, h_out,
        max_rank,
        BLOCK_N_IN, BLOCK_N_OUT, BLOCK_K
    )
    return

# --- Updated Wrappers ---

def batch_lora_get_qkv(
    y: torch.Tensor, 
    x: torch.Tensor, 
    a_buffer: torch.Tensor, 
    b_buffer: torch.Tensor,
    a_start: torch.Tensor, a_len: torch.Tensor,
    a_scaling: torch.Tensor, req_bins: torch.Tensor,
    a_hidden_dim: int = None,
    b_hidden_dim: int = None
):
    """
    Supports separate dimensions for GQA K/V projections.
    Pass a_hidden_dim (hidden_size) and b_hidden_dim (kv_dim) for K/V.
    """
    dispatch_bgmv(
        y, x, a_buffer, b_buffer, a_start, a_len, a_scaling, req_bins,
        a_hidden_dim=a_hidden_dim, b_hidden_dim=b_hidden_dim
    )

# Other wrappers remain the same, relying on default None behavior (infer from x/y)
def batch_lora_get_o(
    y: torch.Tensor, x: torch.Tensor, a_buffer: torch.Tensor, b_buffer: torch.Tensor,
    a_start: torch.Tensor, a_len: torch.Tensor,
    a_scaling: torch.Tensor, req_bins: torch.Tensor,
    a_hidden_dim: int = None,
    b_hidden_dim: int = None,
):
    dispatch_bgmv(y, x, a_buffer, b_buffer, a_start, a_len, a_scaling, req_bins,
                  a_hidden_dim=a_hidden_dim, b_hidden_dim=b_hidden_dim)

def batch_lora_get_mlp(
    y: torch.Tensor, x: torch.Tensor, a_buffer: torch.Tensor, b_buffer: torch.Tensor,
    a_start: torch.Tensor, a_len: torch.Tensor,
    a_scaling: torch.Tensor, req_bins: torch.Tensor,
    a_hidden_dim: int = None,
    b_hidden_dim: int = None,
):
    dispatch_bgmv(y, x, a_buffer, b_buffer, a_start, a_len, a_scaling, req_bins,
                  a_hidden_dim=a_hidden_dim, b_hidden_dim=b_hidden_dim)

def batch_lora_get_vl(
    y: torch.Tensor, x: torch.Tensor, a_buffer: torch.Tensor, b_buffer: torch.Tensor,
    a_start: torch.Tensor, a_len: torch.Tensor,
    a_scaling: torch.Tensor, req_bins: torch.Tensor,
    a_hidden_dim: int = None,
    b_hidden_dim: int = None,
):
    dispatch_bgmv(y, x, a_buffer, b_buffer, a_start, a_len, a_scaling, req_bins,
                  a_hidden_dim=a_hidden_dim, b_hidden_dim=b_hidden_dim)
