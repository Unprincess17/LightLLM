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
import os
import torch
import triton
import triton.language as tl
from lightllm.utils.log_utils import init_logger
from lightllm.utils.nvtx_utils import NvtxAnnotate

logger = init_logger(__name__)


def bgmv_debug_bounds_enabled() -> bool:
    """True when ``LIGHTLLM_BGMV_DEBUG_BOUNDS=1`` (read at call time so tests can toggle)."""
    return os.environ.get("LIGHTLLM_BGMV_DEBUG_BOUNDS", "0") == "1"


def _coerce_req_bins_for_bgmv(req_bins: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return ``req_bins[:batch_size]`` as contiguous ``int32`` on the same device as ``req_bins``."""
    if req_bins.dim() != 1:
        raise ValueError(f"[BGMV] req_bins must be 1-D, got shape {tuple(req_bins.shape)}")
    if req_bins.shape[0] < batch_size:
        raise ValueError(
            f"[BGMV] req_bins length {req_bins.shape[0]} < batch_size {batch_size}"
        )
    head = req_bins[:batch_size]
    if head.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"[BGMV] req_bins must be int32 or int64, got {head.dtype}"
        )
    if head.is_contiguous() and head.dtype == torch.int32:
        return head
    return head.contiguous().to(dtype=torch.int32)


def validate_bgmv_dispatch_inputs(
    *,
    projection: str,
    y: torch.Tensor,
    x: torch.Tensor,
    a_buffer: torch.Tensor,
    b_buffer: torch.Tensor,
    a_start: torch.Tensor,
    a_len: torch.Tensor,
    a_scaling: torch.Tensor,
    req_bins: torch.Tensor,
    h_in: int,
    h_out: int,
    max_rank: int,
    pool_size: int,
    layer_id: int,
    a_rank: torch.Tensor | None = None,
) -> None:
    """Host-side checks before ``bgmv_kernel`` (syncs device tensors).

    Intended for ``LIGHTLLM_BGMV_DEBUG_BOUNDS=1`` and unit tests; keep messages actionable.
    """
    batch_size = int(x.shape[0])
    if req_bins.shape[0] < batch_size:
        raise AssertionError(
            f"[BGMV:{projection}] req_bins len {req_bins.shape[0]} < batch_size {batch_size}"
        )
    rb = req_bins[:batch_size]
    if rb.dim() != 1:
        raise AssertionError(f"[BGMV:{projection}] req_bins must be 1-D, got {tuple(rb.shape)}")
    if not rb.is_contiguous():
        raise AssertionError(
            f"[BGMV:{projection}] req_bins[:batch_size] must be contiguous "
            f"(strides={tuple(rb.stride())})"
        )
    if rb.dtype not in (torch.int32, torch.int64):
        raise AssertionError(
            f"[BGMV:{projection}] req_bins dtype must be int32/int64, got {rb.dtype}"
        )

    for name, t in (
        ("a_start", a_start),
        ("a_len", a_len),
        ("a_scaling", a_scaling),
    ):
        if t.dim() != 1:
            raise AssertionError(f"[BGMV:{projection}] {name} must be 1-D, got {tuple(t.shape)}")
        if not t.is_contiguous():
            raise AssertionError(f"[BGMV:{projection}] {name} must be contiguous")
    if a_start.dtype not in (torch.int32, torch.int64):
        raise AssertionError(f"[BGMV:{projection}] a_start dtype must be integral, got {a_start.dtype}")
    if a_len.dtype not in (torch.int32, torch.int64):
        raise AssertionError(f"[BGMV:{projection}] a_len dtype must be integral, got {a_len.dtype}")
    if not a_scaling.dtype.is_floating_point:
        raise AssertionError(f"[BGMV:{projection}] a_scaling must be floating dtype, got {a_scaling.dtype}")

    n_adapters = int(a_start.shape[0])
    if n_adapters == 0:
        raise AssertionError(f"[BGMV:{projection}] empty metadata (a_start has 0 rows)")
    if a_len.shape[0] != n_adapters or a_scaling.shape[0] != n_adapters:
        raise AssertionError(
            f"[BGMV:{projection}] metadata length mismatch: "
            f"a_start={n_adapters}, a_len={a_len.shape[0]}, a_scaling={a_scaling.shape[0]}"
        )

    if int(y.shape[0]) != batch_size or int(y.shape[0]) != int(x.shape[0]):
        raise AssertionError(
            f"[BGMV:{projection}] y/x batch mismatch: y0={y.shape[0]} x0={x.shape[0]} batch_size={batch_size}"
        )

    rb_min = int(rb.min().item())
    rb_max = int(rb.max().item())
    if rb_min < 0:
        raise AssertionError(
            f"[BGMV:{projection}] req_bins contains negative entries (min={rb_min}); "
            "BGMV does not mask no-adapter rows — fix bins or use CPU/naive LoRA for mixed batches."
        )
    if rb_max >= n_adapters:
        raise AssertionError(
            f"[BGMV:{projection}] req_bins out of range: max={rb_max} >= num_adapters={n_adapters}"
        )

    ua = torch.unique(rb.detach())
    starts = a_start[ua.long()]
    lens = a_len[ua.long()]
    slots = starts.to(dtype=torch.long) + int(layer_id)
    smin = int(slots.min().item())
    smax = int(slots.max().item())
    if smin < 0:
        raise AssertionError(
            f"[BGMV:{projection}] computed slot min {smin} < 0 (layer_id={layer_id})"
        )
    if smax >= int(pool_size):
        raise AssertionError(
            f"[BGMV:{projection}] slot out of pool: max slot {smax} >= pool_size {pool_size} "
            f"(layer_id={layer_id})"
        )
    if bool((lens <= 0).any().item()):
        bad = ua[(lens <= 0)].detach().cpu().tolist()
        raise AssertionError(
            f"[BGMV:{projection}] a_len has non-positive entries for adapter row(s) {bad}"
        )

    # Pool layout: each adapter owns contiguous buffer slots ``[a_start, a_start + a_len)``
    # (``LoRAModulePool``). The kernel loads ``a_start + layer_id``; that index must stay
    # inside the pool, and the declared span must not extend past ``pool_size`` or later
    # layers / other code would read/write OOB relative to ``a_buffer`` / ``b_buffer``.
    span_end_excl = starts.to(dtype=torch.long) + lens.to(dtype=torch.long)
    span_overflow = span_end_excl > int(pool_size)
    if bool(span_overflow.any().item()):
        bad = ua[span_overflow].detach().cpu().tolist()
        worst = int(span_end_excl.max().item())
        raise AssertionError(
            f"[BGMV:{projection}] adapter slot span exceeds pool: max exclusive end "
            f"a_start+a_len={worst} > pool_size={pool_size} (adapter row(s) {bad}, "
            f"layer_id={layer_id})"
        )

    # Optional: full pool metadata should have positive rank for selected adapters.
    if a_rank is not None and int(a_rank.shape[0]) == n_adapters:
        ranks = a_rank[ua.long()]
        if bool((ranks <= 0).any().item()):
            bad = ua[(ranks <= 0)].detach().cpu().tolist()
            raise AssertionError(
                f"[BGMV:{projection}] a_rank non-positive for adapter row(s) {bad} "
                "(uninitialized or zero-rank adapter metadata)"
            )

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


@NvtxAnnotate
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
    *,
    projection: str = "dispatch_bgmv",
    a_rank: torch.Tensor | None = None,
):
    """
    Batched GPU Memory View (BGMV) dispatch.

    Args:
        a_hidden_dim: The hidden dimension of A matrix (input dim). If None, inferred from x.
        b_hidden_dim: The hidden dimension of B matrix (output dim). If None, inferred from y.
        layer_id: The current layer ID for computing slot location (slot = a_start + layer_id).
    """
    batch_size = x.shape[0]
    if batch_size == 0:
        return

    req_bins_launch = _coerce_req_bins_for_bgmv(req_bins, int(batch_size))

    # Infer dimensions if not provided
    h_in = a_hidden_dim if a_hidden_dim is not None else x.shape[1]
    h_out = b_hidden_dim if b_hidden_dim is not None else y.shape[1]

    # Validation
    pool_size, max_rank, buf_h_in = a_buffer.shape
    _, _, buf_h_out = b_buffer.shape

    assert h_in == buf_h_in, f"Input dim mismatch: x({h_in}) vs a_buffer({buf_h_in})"
    assert h_out == buf_h_out, f"Output dim mismatch: y({h_out}) vs b_buffer({buf_h_out})"

    assert a_len.shape[0] != 0, f"a_len tensor is empty, cannot proceed with dispatch_bgmv"

    # Always reject negative bins (kernel has no no-adapter mask).
    if int(req_bins_launch.min().item()) < 0:
        raise ValueError(
            f"[BGMV:{projection}] req_bins contains negative values (min="
            f"{int(req_bins_launch.min().item())}); use naive/CPU LoRA or fix adapter bins."
        )

    if bgmv_debug_bounds_enabled():
        validate_bgmv_dispatch_inputs(
            projection=projection,
            y=y,
            x=x,
            a_buffer=a_buffer,
            b_buffer=b_buffer,
            a_start=a_start,
            a_len=a_len,
            a_scaling=a_scaling,
            req_bins=req_bins_launch,
            h_in=h_in,
            h_out=h_out,
            max_rank=max_rank,
            pool_size=pool_size,
            layer_id=layer_id,
            a_rank=a_rank,
        )

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
        a_start, a_len, a_scaling, req_bins_launch,
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

@NvtxAnnotate
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
    *,
    projection: str = "batch_lora_get_qkv",
    a_rank: torch.Tensor | None = None,
):
    """
    Supports separate dimensions for GQA K/V projections.
    Pass a_hidden_dim (hidden_size) and b_hidden_dim (kv_dim) for K/V.
    """
    dispatch_bgmv(
        y, x, a_buffer, b_buffer, a_start, a_len, a_scaling, req_bins,
        a_hidden_dim=a_hidden_dim, b_hidden_dim=b_hidden_dim,
        layer_id=layer_id,
        projection=projection,
        a_rank=a_rank,
    )
    
@NvtxAnnotate
def batch_lora_get_o(
    y: torch.Tensor, x: torch.Tensor, a_buffer: torch.Tensor, b_buffer: torch.Tensor,
    a_start: torch.Tensor, a_len: torch.Tensor,
    a_scaling: torch.Tensor, req_bins: torch.Tensor,
    a_hidden_dim: int = None,
    b_hidden_dim: int = None,
    layer_id: int = 0,
    *,
    projection: str = "batch_lora_get_o",
    a_rank: torch.Tensor | None = None,
):
    dispatch_bgmv(y, x, a_buffer, b_buffer, a_start, a_len, a_scaling, req_bins,
                  a_hidden_dim=a_hidden_dim, b_hidden_dim=b_hidden_dim,
                  layer_id=layer_id, projection=projection, a_rank=a_rank)

@NvtxAnnotate
def batch_lora_get_mlp(
    y: torch.Tensor, x: torch.Tensor, a_buffer: torch.Tensor, b_buffer: torch.Tensor,
    a_start: torch.Tensor, a_len: torch.Tensor,
    a_scaling: torch.Tensor, req_bins: torch.Tensor,
    a_hidden_dim: int = None,
    b_hidden_dim: int = None,
    layer_id: int = 0,
    *,
    projection: str = "batch_lora_get_mlp",
    a_rank: torch.Tensor | None = None,
):
    dispatch_bgmv(y, x, a_buffer, b_buffer, a_start, a_len, a_scaling, req_bins,
                  a_hidden_dim=a_hidden_dim, b_hidden_dim=b_hidden_dim,
                  layer_id=layer_id, projection=projection, a_rank=a_rank)

def batch_lora_get_vl(
    y: torch.Tensor, x: torch.Tensor, a_buffer: torch.Tensor, b_buffer: torch.Tensor,
    a_start: torch.Tensor, a_len: torch.Tensor,
    a_scaling: torch.Tensor, req_bins: torch.Tensor,
    a_hidden_dim: int = None,
    b_hidden_dim: int = None,
    layer_id: int = 0,
    *,
    projection: str = "batch_lora_get_vl",
    a_rank: torch.Tensor | None = None,
):
    dispatch_bgmv(y, x, a_buffer, b_buffer, a_start, a_len, a_scaling, req_bins,
                  a_hidden_dim=a_hidden_dim, b_hidden_dim=b_hidden_dim,
                  layer_id=layer_id, projection=projection, a_rank=a_rank)
