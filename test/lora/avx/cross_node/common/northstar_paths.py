"""North-star recovery path implementations.

Five paths sharing common boundary:
  T0 = activation ready in inference-GPU input buffer + path selected
  T1 = LoRA residual visible to inference consumer stream on client GPU

All paths use BF16 weights, FP16 activations, FP32 accumulation, FP16 output.
"""
import time
import torch
import torch.cuda
from common.instrumentation import NorthstarTimeline

try:
    from lightllm._kernels.lora.lora_cpu_kernel import batch_lora_avx, ensure_kernel_loaded
    _HAS_AVX_KERNEL = True
except ImportError:
    _HAS_AVX_KERNEL = False


def init_weights(R, H, I, NM, dtype=torch.bfloat16, device="cpu"):
    """Generate NM distinct LoRA weight pairs on the specified device.

    Returns dict with:
      A: [NM, R, H] tensor
      B: [NM, R, I] tensor
    """
    torch.manual_seed(42)
    A = torch.randn(NM, R, H, dtype=dtype, device=device)
    B = torch.randn(NM, R, I, dtype=dtype, device=device)
    return {"A": A, "B": B}


def format_weights_for_path(weights, path, device="cuda"):
    """Prepare weights for a specific path (e.g., copy to GPU for oracle)."""
    if path == "oracle":
        return {
            "A": weights["A"].to(device),
            "B": weights["B"].to(device),
        }
    return weights


def _record_event(stream, timeline, name):
    """Record a host-clock timestamp (microseconds) for a stage boundary."""
    timeline.set(name, time.perf_counter_ns() / 1000.0)


def _sync_record_event(stream, timeline, name):
    """Synchronize stream then record host-clock timestamp."""
    stream.synchronize()
    timeline.set(name, time.perf_counter_ns() / 1000.0)


def cpu_first_recovery(activation_gpu, weights_cpu, R, H, I, NM,
                       num_cores=1, consumer_stream=None):
    """cpu_first path: D2H activation -> AVX compute on CPU -> H2D result.

    Args:
        activation_gpu: [NM, H] FP16 tensor on client GPU
        weights_cpu: dict with A=[NM,R,H] BF16, B=[NM,R,I] BF16 on CPU
        R, H, I, NM: dimensions
        num_cores: number of CPU cores for AVX compute (1 = single-thread)
        consumer_stream: CUDA stream that will consume the result

    Returns: (result_gpu [NM,I] FP16 on client GPU, NorthstarTimeline)
    """
    if consumer_stream is None:
        consumer_stream = torch.cuda.current_stream()

    tl = NorthstarTimeline()
    copy_stream = torch.cuda.Stream()
    copy_stream.wait_stream(consumer_stream)

    # T0: activation ready, path selected
    _record_event(consumer_stream, tl, "T0")

    # cf0: activation pack start
    _record_event(consumer_stream, tl, "cf0")

    # cf1: D2H enqueue on copy stream
    with torch.cuda.stream(copy_stream):
        activation_cpu = activation_gpu.cpu().to(torch.bfloat16)
    _record_event(consumer_stream, tl, "cf1")
    # cf2: D2H observed complete
    _sync_record_event(copy_stream, tl, "cf2")

    # cf3: AVX compute start
    _record_event(consumer_stream, tl, "cf3")
    if _HAS_AVX_KERNEL:
        ensure_kernel_loaded()
        result_cpu = torch.empty(NM, I, dtype=torch.bfloat16, device="cpu")

        # Sequential per-miss loop — each batch_lora_avx call uses OpenMP
        # internally (get_optimal_threads scales with H, R). ThreadPoolExecutor
        # across misses is counterproductive: per-miss GEMMs are too small
        # for thread-spawn overhead. Multi-core benefit comes from N2's
        # concurrent requests (OpenLoopRunner's n_consumers), not per-request.
        for i in range(NM):
            x_i = activation_cpu[i:i+1, :]  # [1, H]
            A_i = weights_cpu["A"][i]        # [R, H]
            B_i = weights_cpu["B"][i]        # [R, I]
            r_i = batch_lora_avx(x_i, A_i, B_i)  # [1, I]
            result_cpu[i:i+1, :] = r_i
    else:
        # Fallback: torch CPU matmul (for testing without AVX kernel)
        x = activation_cpu.to(torch.float32)          # [NM, H]
        A = weights_cpu["A"].to(torch.float32)         # [NM, R, H]
        B = weights_cpu["B"].to(torch.float32)         # [NM, R, I]
        # Per-batch: x_i @ A_i.T @ B_i -> [NM, I]
        z = torch.bmm(x.unsqueeze(1), A.transpose(-1, -2))  # [NM, 1, R]
        y = torch.bmm(z, B)                                   # [NM, 1, I]
        result_cpu = y.squeeze(1).to(torch.bfloat16)          # [NM, I]

    _record_event(consumer_stream, tl, "cf4")  # AVX compute complete

    # cf5: H2D enqueue
    with torch.cuda.stream(copy_stream):
        result_gpu = result_cpu.to(device="cuda", dtype=torch.float16, non_blocking=True)
    _record_event(consumer_stream, tl, "cf5")
    # cf6: H2D observed complete
    _sync_record_event(copy_stream, tl, "cf6")

    # cf7: consumer-stream merge/visibility
    consumer_stream.wait_stream(copy_stream)
    _sync_record_event(consumer_stream, tl, "cf7")

    # T1 = cf7
    tl.set("T1", tl.get("cf7"))

    return result_gpu, tl


def load_then_run_recovery(activation_gpu, weights_cpu, R, H, I, NM,
                           consumer_stream=None):
    """load_then_run path: H2D weights -> GPU compute -> result on GPU.

    Args:
        activation_gpu: [NM, H] FP16 on client GPU
        weights_cpu: dict with A=[NM,R,H] BF16, B=[NM,R,I] BF16 on CPU
        R, H, I, NM: dimensions
        consumer_stream: CUDA stream that will consume the result

    Returns: (result_gpu [NM,I] FP16 on client GPU, NorthstarTimeline)
    """
    if consumer_stream is None:
        consumer_stream = torch.cuda.current_stream()

    tl = NorthstarTimeline()
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    copy_stream.wait_stream(consumer_stream)

    # T0
    _record_event(consumer_stream, tl, "T0")

    # lt0: A/B H2D enqueue
    with torch.cuda.stream(copy_stream):
        A_gpu = weights_cpu["A"].to("cuda", non_blocking=True)  # [NM,R,H] BF16
        B_gpu = weights_cpu["B"].to("cuda", non_blocking=True)  # [NM,R,I] BF16
    _record_event(consumer_stream, tl, "lt0")
    # lt1: H2D complete
    _sync_record_event(copy_stream, tl, "lt1")

    # lt2: GPU compute enqueue
    compute_stream.wait_stream(copy_stream)
    with torch.cuda.stream(compute_stream):
        x = activation_gpu.to(torch.float32)  # [NM, H]
        A_f32 = A_gpu.to(torch.float32)        # [NM, R, H]
        B_f32 = B_gpu.to(torch.float32)        # [NM, R, I]
        # Grouped GEMM: x_i @ A_i.T @ B_i for each i
        # z = torch.einsum("nh,nrh->nr", x, A_f32)  # [NM, R]
        # y = torch.einsum("nr,nri->ni", z, B_f32)  # [NM, I]
        z = torch.bmm(x.unsqueeze(1), A_f32.transpose(-1, -2))  # [NM, 1, R]
        y = torch.bmm(z, B_f32)  # [NM, 1, I]
        result_gpu = y.squeeze(1).to(torch.float16)  # [NM, I]
    _record_event(consumer_stream, tl, "lt2")
    # lt3: GPU compute complete
    _sync_record_event(compute_stream, tl, "lt3")

    # lt4: consumer-stream visibility
    consumer_stream.wait_stream(compute_stream)
    _sync_record_event(consumer_stream, tl, "lt4")

    # T1 = lt4
    tl.set("T1", tl.get("lt4"))

    return result_gpu, tl


def oracle_recovery(activation_gpu, weights_gpu, R, H, I, NM,
                    consumer_stream=None):
    """Oracle path: weights already on GPU, compute only.

    Args:
        activation_gpu: [NM, H] FP16 on client GPU
        weights_gpu: dict with A=[NM,R,H] BF16, B=[NM,R,I] BF16 on GPU
        R, H, I, NM: dimensions
        consumer_stream: CUDA stream

    Returns: (result_gpu [NM,I] FP16 on client GPU, NorthstarTimeline)
    """
    if consumer_stream is None:
        consumer_stream = torch.cuda.current_stream()

    tl = NorthstarTimeline()
    compute_stream = torch.cuda.Stream()
    compute_stream.wait_stream(consumer_stream)

    # T0
    _record_event(consumer_stream, tl, "T0")

    with torch.cuda.stream(compute_stream):
        x = activation_gpu.to(torch.float32)
        A_f32 = weights_gpu["A"].to(torch.float32)
        B_f32 = weights_gpu["B"].to(torch.float32)
        z = torch.bmm(x.unsqueeze(1), A_f32.transpose(-1, -2))
        y = torch.bmm(z, B_f32)
        result_gpu = y.squeeze(1).to(torch.float16)

    # T1: consumer-stream visibility
    consumer_stream.wait_stream(compute_stream)
    _sync_record_event(consumer_stream, tl, "T1")

    return result_gpu, tl
