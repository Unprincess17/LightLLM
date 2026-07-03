# test/lora/avx/cross_node/rdma_qp_generator/bench_gdr_latency.py
"""Microbenchmark: GPU-to-GPU RDMA WRITE latency vs message size.

Reports median over 1000 iterations for several message sizes.

Usage:
  Server: python bench_gdr_latency.py server 10.10.1.3
  Client: python bench_gdr_latency.py client 10.10.1.1 10.10.1.3
"""
import statistics
import sys
import time

from gpudirect_transport import GPUDirectTransport

SIZES = [4 * 1024, 16 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024]
ITERS = 1000
WARMUP = 50
BUF_BYTES = 1 << 22  # 4 MB


def main():
    role = sys.argv[1]
    if role == "server":
        bind_ip = sys.argv[2]
        t = GPUDirectTransport(local_ip=bind_ip, remote_ip="",
                                gpu_buffer_bytes=BUF_BYTES)
        t.start_server()
        print("[server] connected, idle 120s", flush=True)
        time.sleep(120)
        t.close()

    elif role == "client":
        local_ip = sys.argv[2]
        remote_ip = sys.argv[3]
        t = GPUDirectTransport(local_ip=local_ip, remote_ip=remote_ip,
                                gpu_buffer_bytes=BUF_BYTES)
        t.start_client()
        print(f"{'size_bytes':>12}  {'median_us':>10}  {'p99_us':>10}",
              flush=True)
        for sz in SIZES:
            # Warmup
            for _ in range(WARMUP):
                t.write_to_remote(0, 0, sz)
            samples = []
            for _ in range(ITERS):
                t0 = time.perf_counter()
                t.write_to_remote(0, 0, sz)
                t1 = time.perf_counter()
                samples.append((t1 - t0) * 1e6)
            med = statistics.median(samples)
            p99 = statistics.quantiles(samples, n=100)[98]
            print(f"{sz:>12}  {med:>10.2f}  {p99:>10.2f}", flush=True)
        t.close()

    else:
        print(f"unknown role: {role}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
