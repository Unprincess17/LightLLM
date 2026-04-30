import torch
torch.set_num_threads(1)
import time

print('=' * 60)
print('KERNEL-ONLY BENCHMARK (no D2H/H2D overhead)')
print('=' * 60)

H = 4096
R = 8
N = 1
scaling = 1.0

x = torch.randn(N, H, dtype=torch.bfloat16)
A = torch.randn(R, H, dtype=torch.bfloat16)
B = torch.randn(R, H, dtype=torch.bfloat16)
inter = torch.randn(N, R, dtype=torch.bfloat16)

from lightllm._kernels.lora.moe_lora_cpu_kernel import (
    moe_batch_lora_gate_avx, 
    moe_batch_lora_up_avx,
)

iters = 2000

# PyTorch reference
t0 = time.perf_counter()
for _ in range(iters):
    _ = x @ A.T
torch_gate = (time.perf_counter() - t0) / iters * 1e6

t0 = time.perf_counter()
for _ in range(iters):
    _ = inter @ B
torch_up = (time.perf_counter() - t0) / iters * 1e6

# AVX kernels
for _ in range(100): _ = moe_batch_lora_gate_avx(x, A, 1.0)
t0 = time.perf_counter()
for _ in range(iters):
    _ = moe_batch_lora_gate_avx(x, A, 1.0)
avx_gate = (time.perf_counter() - t0) / iters * 1e6

for _ in range(100): _ = moe_batch_lora_up_avx(inter, B, scaling)
t0 = time.perf_counter()
for _ in range(iters):
    _ = moe_batch_lora_up_avx(inter, B, scaling)
avx_up = (time.perf_counter() - t0) / iters * 1e6

print("{:<25} {:>10} {:>10} {:>10}".format("Kernel", "PyTorch", "AVX-512", "Speedup"))
print('-' * 60)
print("{:<25} {:>9.1f}us {:>9.1f}us {:>9.1f}x".format(
    "Gate (x@A^T)", torch_gate, avx_gate, torch_gate/avx_gate))
print("{:<25} {:>9.1f}us {:>9.1f}us {:>9.1f}x".format(
    "Up (inter@B)", torch_up, avx_up, torch_up/avx_up))
print()
print("{:<25} {:>9.1f}us {:>9.1f}us {:>9.1f}x".format(
    "Total Gate + Up", torch_gate + torch_up, avx_gate + avx_up, 
    (torch_gate + torch_up)/(avx_gate + avx_up)))
print()
print('=' * 60)
print('BOTTLENECK ANALYSIS')
print('=' * 60)
print(f'AVX Gate kernel H={H}: {avx_gate:.1f} us / {1/(avx_gate*1e-6)/1e6:.1f} M ops/sec')
print(f'AVX Up kernel H={H}: {avx_up:.1f} us / {1/(avx_up*1e-6)/1e6:.1f} M ops/sec')
print(f'Theoretical peak (3GHz, 32 ops/cycle): 65.5 us for 2*8*4096 ops')
print(f'Gate efficiency: {65.5/(2*avx_gate):.1%} (only half of work)')
print(f'Up efficiency: {65.5/(2*avx_up):.1%}')
