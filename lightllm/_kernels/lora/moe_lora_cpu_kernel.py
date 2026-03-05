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

# Load the compiled kernel
_moe_lora_cpu_kernel = load(
    name='moe_lora_cpu_kernel',
    sources=[
        os.path.join(os.path.dirname(__file__), 'moe_lora_cpu_kernel.cpp')
    ],
    extra_cflags=_extra_cflags,
    verbose=False,
)

def is_available():
    """Check if the MoE LoRA AVX-512 kernel is available on this system."""
    try:
        return bool(_moe_lora_cpu_kernel.is_available())
    except:
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
    output = torch.zeros(x.shape[0], B.shape[1], dtype=x.dtype, device=x.device)
    _moe_lora_cpu_kernel.moe_batch_lora_up_avx(
        x, B, output,
        x.shape[0], x.shape[1], B.shape[0], scaling
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
    output = torch.zeros(x.shape[0], B.shape[1], dtype=x.dtype, device=x.device)
    _moe_lora_cpu_kernel.moe_batch_lora_down_avx(
        x, B, output,
        x.shape[0], x.shape[1], B.shape[0], scaling
    )
    return output
