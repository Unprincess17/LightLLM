"""
S-LoRA dispatch_bgmv Triton Kernel

This module provides the Batched GPU Memory View (BGMV) kernel for efficient
batched LoRA computation. The key idea is that each request in a batch can use
a different adapter, and req_bins tracks which adapter each request uses.

The kernel computes: y[req] = x[req] @ A[adapter(req)] @ B[adapter(req)] * scaling

For optimal performance:
- Each thread block processes a subset of the batch
- Within a block, each request loads its adapter's weights dynamically
- The kernel avoids materializing intermediate results for all adapters
"""
import torch
import triton
import triton.language as tl


@triton.jit
def bgmv_kernel(
    y_ptr, x_ptr, w_ptr, a_start_ptr, a_len_ptr, a_scaling_ptr, req_bins_ptr,
    stride_yb, stride_yh,  # y: [batch, hidden]
    stride_xb, stride_xh,  # x: [batch, hidden]
    stride_wp, stride_wr, stride_wh,  # w: [pool_size, rank, hidden]
    stride_ab,  # a_start, a_len, a_scaling: [num_adapters]
    batch_size, hidden_dim, max_rank,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Batched GPU Memory View (BGMV) kernel for LoRA.

    Computes LoRA addition for a batch of requests with different adapters:
        y[req] += x[req] @ A[adapter(req)] @ B[adapter(req)] * scaling

    Args:
        y_ptr: Output tensor [batch, hidden] - will be modified in-place
        x_ptr: Input tensor [batch, hidden]
        w_ptr: LoRA weights [pool_size, max_rank, hidden]
        a_start_ptr: Start index for each adapter [num_adapters]
        a_len_ptr: Length for each adapter [num_adapters]
        a_scaling_ptr: Scaling factor for each adapter [num_adapters]
        req_bins_ptr: Adapter index for each request [batch]
        stride_*: Tensor strides
        batch_size: Number of requests in batch
        hidden_dim: Hidden dimension
        max_rank: Maximum LoRA rank
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(hidden_dim, BLOCK_N)
    pid_n = pid % num_pid_n
    pid_m = pid // num_pid_n

    # Offsets for this program instance
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator for output computation: x @ A @ B.T
    # We compute: result = (x @ A) @ B.T = x @ (A @ B.T)
    # Since A and B are small (rank dimension), we do (x @ A) first, then @ B.T
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, max_rank, BLOCK_K):
        # Load x[req, :] for this block of k
        # x_ptrs: [BLOCK_M, BLOCK_K] for each request
        x_ptrs = x_ptr + stride_xb * offs_m[:, None] + stride_xh * offs_k[None, :]
        x_mask = (offs_m[:, None] < batch_size) & (offs_k[None, :] < max_rank - k)
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # For each request in this block, compute x[req] @ A[adapter(req), k:k+BLOCK_K]
        # and accumulate into a temporary: temp[req, k_offset] = x[req] @ A_adapter
        temp_ptrs = x_ptr + stride_xb * offs_m[:, None] + stride_xh * offs_k[None, :]
        # Actually we need to load A weights for each request's adapter

        # For each request, get its adapter and load A weights
        for i in range(BLOCK_M):
            req_idx = offs_m[i]
            if req_idx >= batch_size:
                continue

            # Get adapter index for this request
            adapter_idx = tl.load(req_bins_ptr + req_idx)

            # Get adapter metadata
            a_start = tl.load(a_start_ptr + adapter_idx)
            a_len = tl.load(a_len_ptr + adapter_idx)
            a_scaling = tl.load(a_scaling_ptr + adapter_idx)

            # Check if this adapter has weights at position k
            if k >= a_len:
                continue

            # Load A weights: w[a_start + k, :rank, :hidden]
            # Layout: w[slot, k_offset, :]
            a_loc = a_start + k
            a_ptrs = w_ptr + stride_wp * a_loc + stride_wr * offs_k + stride_wh * 0  # A matrix
            a_mask = (offs_k[:, None] < a_len - k) & (offs_n[None, :] < hidden_dim)
            a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)

            # Compute: x[req] @ A.T -> [BLOCK_K]
            # x is [1, BLOCK_K], A is [BLOCK_K, hidden], result is [hidden]
            # Actually: x @ A.T where x is [hidden] and A is [rank, hidden]
            # Result: [rank]
            x_req = x_vals[i, :]  # [BLOCK_K]
            a_req = a_vals[:, :]  # [BLOCK_K, BLOCK_N]

            # Compute x @ A.T: result[hidden] = sum_k(x[k] * A[k, hidden])
            # We need x as [1, BLOCK_K] and A as [BLOCK_K, BLOCK_N]
            # Using dot product
            temp_acc = tl.dot(x_req[None, :], a_req)
            acc[i] = acc[i] + temp_acc * a_scaling

    # Store output
    y_ptrs = y_ptr + stride_yb * offs_m[:, None] + stride_yh * offs_n[None, :]
    y_mask = (offs_m[:, None] < batch_size) & (offs_n < hidden_dim)
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def bgmv_expand_kernel(
    y_ptr, x_ptr, w_ptr, a_start_ptr, a_len_ptr, a_scaling_ptr, req_bins_ptr,
    stride_yb, stride_yh,  # y: [batch, hidden]
    stride_xb, stride_xh,  # x: [batch, rank]
    stride_wp, stride_wr, stride_wh,  # w: [pool_size, rank, hidden]
    stride_ab,
    batch_size, rank, hidden_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    BGMV Expand Kernel - for cases where rank > hidden (rare, but for completeness).

    Computes: y[req] = x[req] @ B[adapter(req)].T * scaling
    where x is [batch, rank] and B is [pool_size, rank, hidden]
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(hidden_dim, BLOCK_N)
    pid_n = pid % num_pid_n
    pid_m = pid // num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, rank, BLOCK_K):
        # x_ptrs: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + stride_xb * offs_m[:, None] + stride_xh * (offs_k + k)
        x_mask = (offs_m[:, None] < batch_size) & (offs_k[None, :] < rank - k)
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

        for i in range(BLOCK_M):
            req_idx = offs_m[i]
            if req_idx >= batch_size:
                continue

            adapter_idx = tl.load(req_bins_ptr + req_idx)
            a_start = tl.load(a_start_ptr + adapter_idx)
            a_len = tl.load(a_len_ptr + adapter_idx)
            a_scaling = tl.load(a_scaling_ptr + adapter_idx)

            if k >= a_len:
                continue

            # Load B weights: w[a_start + k, k_offset, :]
            a_loc = a_start + k
            b_ptrs = w_ptr + stride_wp * a_loc + stride_wr * (offs_k + k) + stride_wh * 0
            b_mask = ((offs_k + k)[:, None] < a_len) & (offs_n[None, :] < hidden_dim)
            b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

            # Compute: x[req, k:k+BLOCK_K] @ B[adapter].T
            # x is [BLOCK_K], B is [BLOCK_K, BLOCK_N]
            x_req = x_vals[i, :]
            b_req = b_vals

            temp_acc = tl.dot(x_req[None, :], b_req)
            acc[i] = acc[i] + temp_acc * a_scaling

    y_ptrs = y_ptr + stride_yb * offs_m[:, None] + stride_yh * offs_n[None, :]
    y_mask = (offs_m[:, None] < batch_size) & (offs_n < hidden_dim)
    tl.store(y_ptrs, acc, mask=y_mask)


def dispatch_bgmv(
    y: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    a_start: torch.Tensor,
    a_len: torch.Tensor,
    a_scaling: torch.Tensor,
    req_bins: torch.Tensor,
):
    """
    Batched GPU Memory View (BGMV) for LoRA computation.

    Computes LoRA addition for a batch of requests with different adapters:
        y[req] += x[req] @ A[adapter(req)] @ B[adapter(req)] * scaling

    Args:
        y: Output tensor [batch, hidden] - will be modified in-place
        x: Input tensor [batch, hidden]
        w: LoRA weights from memory pool [pool_size, max_rank, hidden]
        a_start: Start index for each adapter [num_adapters]
        a_len: Length for each adapter [num_adapters]
        a_scaling: Scaling factor for each adapter [num_adapters]
        req_bins: Adapter index for each request [batch]
    """
    batch_size, hidden_dim = x.shape
    pool_size, max_rank, hidden_dim_w = w.shape

    assert hidden_dim == hidden_dim_w, f"hidden_dim mismatch: {hidden_dim} vs {hidden_dim_w}"

    # Choose block sizes based on dimensions
    if hidden_dim <= 512:
        BLOCK_N = 64
    elif hidden_dim <= 1024:
        BLOCK_N = 128
    else:
        BLOCK_N = 256

    BLOCK_M = 32
    BLOCK_K = min(64, max_rank)

    grid = (batch_size * ((hidden_dim + BLOCK_N - 1) // BLOCK_N),)

    bgmv_kernel[grid](
        y, x, w,
        a_start, a_len, a_scaling, req_bins,
        y.stride(0), y.stride(1),
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1), w.stride(2),
        a_start.stride(0),
        batch_size, hidden_dim, max_rank,
        BLOCK_M, BLOCK_N, BLOCK_K
    )


def batch_lora_get_qkv(
    y: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    a_start: torch.Tensor,
    a_len: torch.Tensor,
    a_scaling: torch.Tensor,
    req_bins: torch.Tensor,
):
    """
    Compute LoRA for Q/K/V projections using dispatch_bgmv.

    This is a convenience function that wraps dispatch_bgmv for Q/K/V use cases.
    """
    dispatch_bgmv(y, x, w, a_start, a_len, a_scaling, req_bins)


def batch_lora_get_o(
    y: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    a_start: torch.Tensor,
    a_len: torch.Tensor,
    a_scaling: torch.Tensor,
    req_bins: torch.Tensor,
):
    """
    Compute LoRA for O projection using dispatch_bgmv.
    """
    dispatch_bgmv(y, x, w, a_start, a_len, a_scaling, req_bins)


def batch_lora_get_mlp(
    y: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    a_start: torch.Tensor,
    a_len: torch.Tensor,
    a_scaling: torch.Tensor,
    req_bins: torch.Tensor,
):
    """
    Compute LoRA for MLP projections (gate/up/down) using dispatch_bgmv.
    """
    dispatch_bgmv(y, x, w, a_start, a_len, a_scaling, req_bins)


def batch_lora_get_vl(
    y: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    a_start: torch.Tensor,
    a_len: torch.Tensor,
    a_scaling: torch.Tensor,
    req_bins: torch.Tensor,
):
    """
    Compute LoRA for Vision-Language adapter projections using dispatch_bgmv.
    """
    dispatch_bgmv(y, x, w, a_start, a_len, a_scaling, req_bins)
