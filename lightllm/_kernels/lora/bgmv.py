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
    h_in, h_out,      # Separate input and output dimensions
    max_rank,
    layer_id,         # Current layer ID for slot offset
    BLOCK_N_IN: tl.constexpr,  # Block size for input dim
    BLOCK_N_OUT: tl.constexpr, # Block size for output dim
    BLOCK_K: tl.constexpr,
    pool_size = 0, # TODO: for debug only
):
    """
    Batched GPU Memory View (BGMV) kernel for LoRA with decoupled Input/Output dimensions.

    Supports GQA where Input Dim (Hidden Size) != Output Dim (KV Size).
    Each adapter occupies consecutive slots in the buffer; we access slot = a_start + layer_id.

    Handles input/output dimensions larger than BLOCK_N_IN/BLOCK_N_OUT by iterating over blocks.
    """
    req_idx = tl.program_id(0)

    if req_idx >= batch_size:
        return

    # Offsets for blocks
    offs_k = tl.arange(0, BLOCK_K)

    # Get adapter index and metadata
    adapter_idx = tl.load(req_bins_ptr + req_idx)
    a_start = tl.load(a_start_ptr + adapter_idx)
    a_len = tl.load(a_len_ptr + adapter_idx)
    a_scaling = tl.load(a_scaling_ptr + adapter_idx)

    # Compute actual slot location: a_start + layer_id
    slot_loc = a_start + layer_id

    # Compute number of input and output blocks
    num_in_blocks = (h_in + BLOCK_N_IN - 1) // BLOCK_N_IN
    num_out_blocks = (h_out + BLOCK_N_OUT - 1) // BLOCK_N_OUT

    # Iterate over output blocks
    for out_block_idx in range(num_out_blocks):
        out_base = out_block_idx * BLOCK_N_OUT
        offs_n_out = tl.arange(0, BLOCK_N_OUT)

        # Initialize accumulator for this output block
        acc_block = tl.zeros((BLOCK_N_OUT,), dtype=tl.float32)

        # Iterate over input blocks
        for in_block_idx in range(num_in_blocks):
            in_base = in_block_idx * BLOCK_N_IN
            offs_n_in = tl.arange(0, BLOCK_N_IN)

            # Load input x[req_idx, in_block] - [BLOCK_N_IN]
            x_ptrs = x_ptr + stride_xb * req_idx + stride_xh * (offs_n_in + in_base)
            x_mask = (offs_n_in + in_base) < h_in
            x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

            # Compute number of rank chunks
            num_chunks = (max_rank + BLOCK_K - 1) // BLOCK_K

            for chunk_idx in range(num_chunks):
                k_base = chunk_idx * BLOCK_K

                # Valid rank offsets for this chunk
                rank_offsets = k_base + offs_k
                rank_valid = rank_offsets < max_rank

                # --- Load A (Down Proj) ---
                # A matrix: [max_rank, h_in]
                # Load from slot_loc with proper input block offset
                a_ptr = a_buffer_ptr + stride_ab * slot_loc + stride_ar * k_base + stride_ah * in_base

                a_vals = tl.load(
                    a_ptr + stride_ar * offs_k[:, None] + stride_ah * offs_n_in[None, :],
                    mask=rank_valid[:, None] & ((offs_n_in[None, :] + in_base) < h_in),
                    other=0.0
                )

                # Cast to float32
                x_fp32 = tl.cast(x_vals, tl.float32)
                a_fp32 = tl.cast(a_vals, tl.float32)

                # Compute h_chunk = x[in_block] @ A.T for this chunk
                # Sum over dimension N_IN: [N_IN] * [K, N_IN] -> [K]
                h_chunk = tl.sum(x_fp32[None, :] * a_fp32, axis=1)

                # Zero out invalid ranks
                h_chunk = tl.where(rank_valid, h_chunk, 0.0)

                # --- Load B (Up Proj) ---
                # B matrix: [max_rank, h_out]
                # Load from same slot_loc with proper output block offset
                b_ptr = b_buffer_ptr + stride_bb * slot_loc + stride_br * k_base + stride_bh * out_base

                b_vals = tl.load(
                    b_ptr + stride_br * offs_k[:, None] + stride_bh * offs_n_out[None, :],
                    mask=rank_valid[:, None] & ((offs_n_out[None, :] + out_base) < h_out),
                    other=0.0
                )

                b_fp32 = tl.cast(b_vals, tl.float32)

                # Compute acc_block += h_chunk @ B
                # Broadcast h_chunk [K] against B [K, N_OUT] -> Sum over K -> [N_OUT]
                chunk_contrib = tl.sum(h_chunk[:, None] * b_fp32, axis=0)
                acc_block = acc_block + chunk_contrib * a_scaling

        # Store output for this block
        y_ptrs = y_ptr + stride_yb * req_idx + stride_yh * (offs_n_out + out_base)
        y_mask = (offs_n_out + out_base) < h_out
        tl.store(y_ptrs, acc_block, mask=y_mask)


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
    layer_id: int = 0,  # Layer ID for slot offset
):
    """
    Batched GPU Memory View (BGMV) dispatch.

    Args:
        a_hidden_dim: The hidden dimension of A matrix (input dim). If None, inferred from x.
        b_hidden_dim: The hidden dimension of B matrix (output dim). If None, inferred from y.
        layer_id: The current layer ID for computing slot location (slot = a_start + layer_id).
    """
    batch_size = x.shape[0]

    # Infer dimensions if not provided
    h_in = a_hidden_dim if a_hidden_dim is not None else x.shape[1]
    h_out = b_hidden_dim if b_hidden_dim is not None else y.shape[1]

    # Validation
    pool_size, max_rank, buf_h_in = a_buffer.shape
    _, _, buf_h_out = b_buffer.shape

    assert h_in == buf_h_in, f"Input dim mismatch: x({h_in}) vs a_buffer({buf_h_in})"
    assert h_out == buf_h_out, f"Output dim mismatch: y({h_out}) vs b_buffer({buf_h_out})"
    
    assert a_len.shape[0] != 0, f"a_len tensor is empty, cannot proceed with dispatch_bgmv"

    # Helper to pick block size
    def get_block_n(dim):
        if dim <= 512: return 512
        if dim <= 1024: return 1024
        if dim <= 2048: return 2048
        if dim <= 4096: return 4096
        return 4096

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
        layer_id,
        BLOCK_N_IN, BLOCK_N_OUT, BLOCK_K,
        a_buffer.shape[0]
    )
    return

def batch_lora_get_qkv(
    y: torch.Tensor,
    x: torch.Tensor,
    a_buffer: torch.Tensor,
    b_buffer: torch.Tensor,
    a_start: torch.Tensor, a_len: torch.Tensor,
    a_scaling: torch.Tensor, req_bins: torch.Tensor,
    a_hidden_dim: int = None,
    b_hidden_dim: int = None,
    layer_id: int = 0,
):
    """
    Supports separate dimensions for GQA K/V projections.
    Pass a_hidden_dim (hidden_size) and b_hidden_dim (kv_dim) for K/V.
    """
    dispatch_bgmv(
        y, x, a_buffer, b_buffer, a_start, a_len, a_scaling, req_bins,
        a_hidden_dim=a_hidden_dim, b_hidden_dim=b_hidden_dim,
        layer_id=layer_id
    )

def batch_lora_get_o(
    y: torch.Tensor, x: torch.Tensor, a_buffer: torch.Tensor, b_buffer: torch.Tensor,
    a_start: torch.Tensor, a_len: torch.Tensor,
    a_scaling: torch.Tensor, req_bins: torch.Tensor,
    a_hidden_dim: int = None,
    b_hidden_dim: int = None,
    layer_id: int = 0,
):
    dispatch_bgmv(y, x, a_buffer, b_buffer, a_start, a_len, a_scaling, req_bins,
                  a_hidden_dim=a_hidden_dim, b_hidden_dim=b_hidden_dim,
                  layer_id=layer_id)

def batch_lora_get_mlp(
    y: torch.Tensor, x: torch.Tensor, a_buffer: torch.Tensor, b_buffer: torch.Tensor,
    a_start: torch.Tensor, a_len: torch.Tensor,
    a_scaling: torch.Tensor, req_bins: torch.Tensor,
    a_hidden_dim: int = None,
    b_hidden_dim: int = None,
    layer_id: int = 0,
):
    dispatch_bgmv(y, x, a_buffer, b_buffer, a_start, a_len, a_scaling, req_bins,
                  a_hidden_dim=a_hidden_dim, b_hidden_dim=b_hidden_dim,
                  layer_id=layer_id)

def batch_lora_get_vl(
    y: torch.Tensor, x: torch.Tensor, a_buffer: torch.Tensor, b_buffer: torch.Tensor,
    a_start: torch.Tensor, a_len: torch.Tensor,
    a_scaling: torch.Tensor, req_bins: torch.Tensor,
    a_hidden_dim: int = None,
    b_hidden_dim: int = None,
    layer_id: int = 0,
):
    dispatch_bgmv(y, x, a_buffer, b_buffer, a_start, a_len, a_scaling, req_bins,
                  a_hidden_dim=a_hidden_dim, b_hidden_dim=b_hidden_dim,
                  layer_id=layer_id)
