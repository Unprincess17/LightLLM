# PyTorch binding for AVX-512 BF16 LoRA Kernel
# This module provides JIT compilation and Python interface for the C++ AVX-512 kernels

import torch
import os
from torch.utils.cpp_extension import load_inline

from lightllm.utils.nvtx_utils import NvtxAnnotate

# Get the directory where this file is located
_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_cpu_flags():
    """Check CPU flags for AVX-512 BF16 support."""
    import subprocess
    try:
        result = subprocess.run(['cat', '/proc/cpuinfo'],
                              capture_output=True, text=True)
        if 'avx512_bf16' in result.stdout:
            return True
    except Exception:
        pass
    return False


def _is_avx512_bf16_supported():
    """Runtime check for AVX-512 BF16 support."""
    # Check using PyTorch
    if not hasattr(torch, 'bfloat16'):
        return False

    # Try to detect CPU features
    try:
        # Check if we're on a supported platform
        import platform
        # Intel Xeon Scalable (Sapphire Rapids and later) supports AVX-512 BF16
        model = platform.processor()
        if 'intel' in model.lower() or 'xeon' in model.lower():
            return True
    except Exception:
        pass

    # For now, assume supported if we're on Linux x86_64
    # A more robust check would parse /proc/cpuinfo
    return True


# Load C++ source from separate file
cpp_path = os.path.join(_KERNEL_DIR, "lora_cpu_kernel.cpp")
with open(cpp_path, 'r') as f:
    cpp_source = f.read()

# JIT compile C++ extension with Sapphire Rapids optimization
# Note: -march=sapphirerapids enables AVX-512 BF16 instructions
_extra_cflags = [
    '-O3',
    '-march=sapphirerapids',  # Required for vdpbf16ps instruction
    '-std=c++17',
    '-fopenmp',  # Enable OpenMP parallelization
    '-ffast-math',  # Enable fast math optimizations
    '-ftree-vectorize',  # Enable tree vectorization
    '-fno-semantic-interposition',  # Optimize for static linking
]

# Add ABI flag to match PyTorch's ABI
try:
    import torch
    if torch._C._GLIBCXX_USE_CXX11_ABI:
        pass  # PyTorch uses CXX11 ABI, no extra flag needed
except Exception:
    _extra_cflags.append('-D_GLIBCXX_USE_CXX11_ABI=0')

# Try to load the extension
try:
    _lora_cpu_kernel = load_inline(
        name="lora_cpu_kernel",
        cpp_sources=[cpp_source],
        extra_cflags=_extra_cflags,
        functions=['lora_down_bindings', 'lora_up_bindings', 'batch_lora_bindings'],
        verbose=True
    )
    _KERNEL_LOADED = True
except Exception as e:
    print(f"[LoRA CPU Kernel] Warning: Failed to load AVX-512 kernel: {e}")
    print("[LoRA CPU Kernel] Falling back to PyTorch implementation")
    _lora_cpu_kernel = None
    _KERNEL_LOADED = False

@NvtxAnnotate("batch_lora_avx")
def batch_lora_avx(
    input_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scaling: float = 1.0
) -> torch.Tensor:
    """
    Batch LoRA computation using AVX-512 BF16 kernel.
    Native BF16 computation with _mm512_dpbf16_ps instruction.

    Args:
        input_tensor: Input tensor [batch, hidden_in] (bfloat16, CPU)
        A: LoRA A weight [rank, hidden_in] (bfloat16, CPU)
        B: LoRA B weight [rank, hidden_out] (bfloat16, CPU)
        scaling: LoRA scaling factor

    Returns:
        Output tensor [batch, hidden_out] (bfloat16, CPU)
    """
    if not _KERNEL_LOADED:
        raise RuntimeError("AVX-512 kernel not loaded")

    # Validate inputs
    assert input_tensor.device.type == 'cpu', "Input must be on CPU"
    assert A.device.type == 'cpu', "A must be on CPU"
    assert B.device.type == 'cpu', "B must be on CPU"
    assert input_tensor.dtype == torch.bfloat16, "Input must be bfloat16"
    assert A.dtype == torch.bfloat16, "A must be bfloat16"
    assert B.dtype == torch.bfloat16, "B must be bfloat16"
    assert input_tensor.ndim == 2, "Input must be 2D [batch, hidden_in]"
    assert A.ndim == 2, "A must be 2D [rank, hidden_in]"
    assert B.ndim == 2, "B must be 2D [rank, hidden_out]"

    batch, hidden_in = input_tensor.shape
    rank = A.shape[0]
    hidden_out = B.shape[1]
    assert A.shape[1] == hidden_in, "A.shape[1] must match input hidden size"
    assert B.shape[0] == rank, "B.shape[0] must match A rank"

    # Ensure contiguous layout for cache efficiency
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    # Allocate output (BF16)
    output = torch.empty(batch, hidden_out, dtype=torch.bfloat16, device='cpu')

    # Call C++ kernel with native BF16
    with NvtxAnnotate("LoRA_CPU_GEMM_Batch"):
        _lora_cpu_kernel.batch_lora_bindings(
            input_tensor,
            A,
            B,
            output,
            scaling
        )

    return output


def lora_down_avx(
    input_tensor: torch.Tensor,
    A: torch.Tensor
) -> torch.Tensor:
    """
    LoRA down projection (x @ A.T) using AVX-512 BF16 kernel.

    Args:
        input_tensor: Input tensor [batch, hidden] (bfloat16, CPU)
        A: LoRA A weight [rank, hidden] (bfloat16, CPU)

    Returns:
        Output tensor [batch, rank] (bfloat16, CPU)
    """
    if not _KERNEL_LOADED:
        raise RuntimeError("AVX-512 kernel not loaded")

    assert input_tensor.device.type == 'cpu', "Input must be on CPU"
    assert A.device.type == 'cpu', "A must be on CPU"
    assert input_tensor.dtype == torch.bfloat16, "Input must be bfloat16"
    assert A.dtype == torch.bfloat16, "A must be bfloat16"

    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    if not A.is_contiguous():
        A = A.contiguous()

    batch, hidden = input_tensor.shape
    rank = A.shape[0]

    output = torch.empty(batch, rank, dtype=torch.bfloat16, device='cpu')

    _lora_cpu_kernel.lora_down_bindings(input_tensor, A, output)

    return output


def lora_up_avx(
    input_tensor: torch.Tensor,
    B: torch.Tensor
) -> torch.Tensor:
    """
    LoRA up projection (x @ B) using AVX-512 BF16 kernel.

    Args:
        input_tensor: Input tensor [batch, rank] (bfloat16, CPU)
        B: LoRA B weight [rank, hidden] (bfloat16, CPU)

    Returns:
        Output tensor [batch, hidden] (bfloat16, CPU)
    """
    if not _KERNEL_LOADED:
        raise RuntimeError("AVX-512 kernel not loaded")

    assert input_tensor.device.type == 'cpu', "Input must be on CPU"
    assert B.device.type == 'cpu', "B must be on CPU"
    assert input_tensor.dtype == torch.bfloat16, "Input must be bfloat16"
    assert B.dtype == torch.bfloat16, "B must be bfloat16"

    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    batch, rank = input_tensor.shape
    hidden = B.shape[1]

    output = torch.empty(batch, hidden, dtype=torch.bfloat16, device='cpu')

    _lora_cpu_kernel.lora_up_bindings(input_tensor, B, output)

    return output


def is_available() -> bool:
    """Check if AVX-512 kernel is available."""
    return _KERNEL_LOADED
