"""
MoE-Specific AVX-512 BF16 LoRA Kernel for Sapphire Rapids

This kernel is optimized for the irregular and sparse computational
characteristics of Mixture of Experts (MoE) architectures.

Key Features:
1. Dynamic expert kernel selection based on token count
2. Sparse token handling for MoE expert activation
3. Optimized memory access patterns for various expert sizes
4. Efficient scheduling of irregular workloads
5. Specialized kernels for Gate/Up/Down phases of MoE
6. Cache-aware design for small token counts

Created for LightLLM project
"""

import os
import threading
import torch
from torch.utils.cpp_extension import load

# Compile the C++ kernel
_extra_cflags = [
    '-O3',
    '-march=sapphirerapids',  # Required for vdpbf16ps instruction
    '-std=c++17',
    '-fopenmp',  # Enable OpenMP parallelization
    '-ffast-math',  # Enable fast math optimizations
    '-ftree-vectorize',  # Enable tree vectorization
    '-fno-semantic-interposition',  # Optimize for static linking
    '-DNDEBUG',  # Disable assertions for performance
]

_moe_lora_cpu_kernel = None
_load_failed = False
_load_lock = threading.Lock()


def ensure_kernel_loaded() -> None:
    """JIT-compile the extension on first use (import stays non-blocking)."""
    global _moe_lora_cpu_kernel, _load_failed  # noqa: PLW0603

    if _moe_lora_cpu_kernel is not None or _load_failed:
        return
    with _load_lock:
        if _moe_lora_cpu_kernel is not None or _load_failed:
            return
        try:
            _moe_lora_cpu_kernel = load(
                name="moe_lora_cpu_kernel",
                sources=[os.path.join(os.path.dirname(__file__), "moe_lora_cpu_kernel.cpp")],
                extra_cflags=_extra_cflags,
                verbose=False,
            )
        except Exception:
            _moe_lora_cpu_kernel = None
            _load_failed = True


def is_available():
    """Check if the MoE LoRA AVX-512 kernel is available on this system."""
    ensure_kernel_loaded()
    if _moe_lora_cpu_kernel is None:
        return False
    try:
        return bool(_moe_lora_cpu_kernel.is_available())
    except Exception:
        return False


def moe_batch_lora_avx(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scaling: float
) -> torch.Tensor:
    """
    Apply MoE-specific LoRA using AVX-512 BF16 instructions.

    This function dynamically selects the optimal kernel based on the
    number of tokens (x.shape[0]).

    Args:
        x: Input tensor [N, H] where N is number of tokens
        A: LoRA down projection matrix [R, H]
        B: LoRA up projection matrix [R, H]
        scaling: Scaling factor for the LoRA output

    Returns:
        LoRA output [N, H]
    """
    ensure_kernel_loaded()
    if _moe_lora_cpu_kernel is None:
        raise RuntimeError("moe_lora_cpu_kernel extension failed to load")
    output = torch.zeros_like(x)
    _moe_lora_cpu_kernel.moe_batch_lora_avx(
        x, A, B, output,
        x.shape[0], x.shape[1], A.shape[0], scaling
    )
    return output

def moe_batch_lora_gate_avx(
    x: torch.Tensor,
    A: torch.Tensor,
    scaling: float
) -> torch.Tensor:
    """
    Apply LoRA specifically optimized for the Gate phase of MoE.

    Args:
        x: Input tensor [N, H] where N is number of tokens
        A: LoRA gate projection matrix [R, H]
        scaling: Scaling factor for the LoRA output

    Returns:
        LoRA output [N, R]
    """
    ensure_kernel_loaded()
    if _moe_lora_cpu_kernel is None:
        raise RuntimeError("moe_lora_cpu_kernel extension failed to load")
    output = torch.zeros(x.shape[0], A.shape[0], dtype=x.dtype, device=x.device)
    _moe_lora_cpu_kernel.moe_batch_lora_gate_avx(
        x, A, output,
        x.shape[0], x.shape[1], A.shape[0], scaling
    )
    return output

def moe_batch_lora_up_avx(
    x: torch.Tensor,
    B: torch.Tensor,
    scaling: float
) -> torch.Tensor:
    """
    Apply LoRA specifically optimized for the Up phase of MoE.

    Args:
        x: Input tensor [N, R] where N is number of tokens
        B: LoRA up projection matrix [R, H]
        scaling: Scaling factor for the LoRA output

    Returns:
        LoRA output [N, H]
    """
    ensure_kernel_loaded()
    if _moe_lora_cpu_kernel is None:
        raise RuntimeError("moe_lora_cpu_kernel extension failed to load")
    output = torch.zeros(x.shape[0], B.shape[1], dtype=x.dtype, device=x.device)
    _moe_lora_cpu_kernel.moe_batch_lora_up_avx(
        x, B, output,
        x.shape[0], x.shape[1], B.shape[1], scaling
    )
    return output

def moe_batch_lora_down_avx(
    x: torch.Tensor,
    B: torch.Tensor,
    scaling: float
) -> torch.Tensor:
    """
    Apply LoRA specifically optimized for the Down phase of MoE.

    Args:
        x: Input tensor [N, R] where N is number of tokens
        B: LoRA down projection matrix [R, H]
        scaling: Scaling factor for the LoRA output

    Returns:
        LoRA output [N, H]
    """
    ensure_kernel_loaded()
    if _moe_lora_cpu_kernel is None:
        raise RuntimeError("moe_lora_cpu_kernel extension failed to load")
    output = torch.zeros(x.shape[0], B.shape[1], dtype=x.dtype, device=x.device)
    _moe_lora_cpu_kernel.moe_batch_lora_down_avx(
        x, B, output,
        x.shape[0], x.shape[1], B.shape[1], scaling
    )
    return output


def moe_lora_multi_adapter_avx(
    x: torch.Tensor,
    A_all: torch.Tensor,
    B_all: torch.Tensor,
    adapter_ids: torch.Tensor,
    scaling: torch.Tensor,
    uniform_scaling: float = 0.0,
) -> torch.Tensor:
    """Batched multi-adapter LoRA: each token uses its own adapter's weights.

    Eliminates Python per-adapter loop overhead by running the entire
    gate+up computation in a single C++ kernel call.

    Args:
        x: Input tensor [N, H]
        A_all: Stacked A matrices [num_adapters, R, H]
        B_all: Stacked B matrices [num_adapters, R, H]
        adapter_ids: Per-token adapter index [N] (0-based LongTensor, CPU)
        scaling: Per-adapter scaling [num_adapters] (FloatTensor, CPU)
        uniform_scaling: If > 0, use this for all adapters (ignore scaling tensor)

    Returns:
        LoRA output [N, H]
    """
    ensure_kernel_loaded()
    if _moe_lora_cpu_kernel is None:
        raise RuntimeError("moe_lora_cpu_kernel extension failed to load")
    N = x.shape[0]
    H = x.shape[1]
    R = A_all.shape[1]
    num_adapters = A_all.shape[0]
    output = torch.zeros(N, H, dtype=x.dtype, device=x.device)
    # C++ binding expects int32 for adapter_ids
    _adapter_ids_i32 = adapter_ids.to(dtype=torch.int32) if adapter_ids.dtype != torch.int32 else adapter_ids
    _moe_lora_cpu_kernel.moe_lora_multi_adapter_avx(
        x, A_all, B_all, _adapter_ids_i32, scaling, output,
        N, H, R, num_adapters, uniform_scaling,
    )
    return output


def moe_lora_pool_multi_adapter_avx(
    key_buffer: torch.Tensor,
    value_buffer: torch.Tensor,
    x: torch.Tensor,
    adapter_local_ids: torch.Tensor,
    pool_offsets: torch.Tensor,
    pool_ranks: torch.Tensor,
    pool_scaling: torch.Tensor,
    uniform_scaling: float = 0.0,
) -> torch.Tensor:
    """Pool-based multi-adapter LoRA: reads directly from pool buffers.

    Avoids any tensor stacking/copying — reads A/B from the pool's
    key_buffer/value_buffer using per-adapter offsets.

    Args:
        key_buffer: Pool key_buffer [total_slots, max_rank, H] (CPU bf16)
        value_buffer: Pool value_buffer [total_slots, max_rank, H] (CPU bf16)
        x: Input tensor [N, H] (CPU bf16)
        adapter_local_ids: Per-token local adapter ID [N] (int32, CPU)
            Maps each token to its index in pool_offsets/pool_ranks/pool_scaling.
        pool_offsets: Per-adapter slot offset [num_unique] (int32, CPU)
        pool_ranks: Per-adapter rank [num_unique] (int32, CPU)
        pool_scaling: Per-adapter scaling [num_unique] (float32, CPU)
        uniform_scaling: If > 0, use for all adapters

    Returns:
        LoRA output [N, H] (CPU bf16)
    """
    ensure_kernel_loaded()
    if _moe_lora_cpu_kernel is None:
        raise RuntimeError("moe_lora_cpu_kernel extension failed to load")
    N = x.shape[0]
    H = x.shape[1]
    max_rank = key_buffer.shape[1]
    num_unique = pool_offsets.shape[0]
    output = torch.zeros(N, H, dtype=x.dtype, device=x.device)
    _local_ids_i32 = adapter_local_ids.to(dtype=torch.int32) if adapter_local_ids.dtype != torch.int32 else adapter_local_ids
    _offsets_i32 = pool_offsets.to(dtype=torch.int32) if pool_offsets.dtype != torch.int32 else pool_offsets
    _ranks_i32 = pool_ranks.to(dtype=torch.int32) if pool_ranks.dtype != torch.int32 else pool_ranks
    _moe_lora_cpu_kernel.moe_lora_pool_multi_adapter_avx(
        key_buffer, value_buffer, x, output,
        _local_ids_i32, _offsets_i32, _ranks_i32, pool_scaling,
        N, H, max_rank, num_unique, uniform_scaling,
    )
    return output


def moe_lora_pool_fp16_multi_adapter_avx(
    key_buffer: torch.Tensor,
    value_buffer: torch.Tensor,
    x: torch.Tensor,
    adapter_local_ids: torch.Tensor,
    pool_offsets: torch.Tensor,
    pool_ranks: torch.Tensor,
    pool_scaling: torch.Tensor,
    uniform_scaling: float = 0.0,
) -> torch.Tensor:
    """Pool-based multi-adapter LoRA with fp16 weights.

    Reads fp16 weights from pool buffers directly, converts to fp32
    on-the-fly using F16C instructions. Input x is bf16 (activation),
    output is bf16. No tensor copying or stacking in Python.

    Args:
        key_buffer: Pool key_buffer [total_slots, max_rank, H] (CPU fp16)
        value_buffer: Pool value_buffer [total_slots, max_rank, H] (CPU fp16)
        x: Input tensor [N, H] (CPU bf16)
        adapter_local_ids: Per-token local adapter ID [N] (int32, CPU)
        pool_offsets: Per-adapter slot offset [num_unique] (int32, CPU)
        pool_ranks: Per-adapter rank [num_unique] (int32, CPU)
        pool_scaling: Per-adapter scaling [num_unique] (float32, CPU)
        uniform_scaling: If > 0, use for all adapters

    Returns:
        LoRA output [N, H] (CPU bf16)
    """
    ensure_kernel_loaded()
    if _moe_lora_cpu_kernel is None:
        raise RuntimeError("moe_lora_cpu_kernel extension failed to load")
    N = x.shape[0]
    H = x.shape[1]
    max_rank = key_buffer.shape[1]
    num_unique = pool_offsets.shape[0]
    output = torch.zeros(N, H, dtype=x.dtype, device=x.device)
    _local_ids_i32 = adapter_local_ids.to(dtype=torch.int32) if adapter_local_ids.dtype != torch.int32 else adapter_local_ids
    _offsets_i32 = pool_offsets.to(dtype=torch.int32) if pool_offsets.dtype != torch.int32 else pool_offsets
    _ranks_i32 = pool_ranks.to(dtype=torch.int32) if pool_ranks.dtype != torch.int32 else pool_ranks
    _moe_lora_cpu_kernel.moe_lora_pool_fp16_multi_adapter_avx(
        key_buffer, value_buffer, x, output,
        _local_ids_i32, _offsets_i32, _ranks_i32, pool_scaling,
        N, H, max_rank, num_unique, uniform_scaling,
    )
    return output
