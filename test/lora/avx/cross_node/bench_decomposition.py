"""B0-B5 decomposition harness for the matched-contrast matrix.

B0: local buffers, Python direct, conc=1 (lower bound, external)
B1: persistent TCP, Python direct, conc=1
B2: persistent TCP, Python executor, conc=1
B3: persistent TCP, Python executor, conc=N
B4: per-request TCP, Python executor, conc=1
B5: per-request TCP, Python executor, conc=N (current architecture)

B6-B9 (C++ matched worker) are in a separate plan.
"""
import argparse
import time
from dataclasses import dataclass, asdict
from typing import Optional

CELLS = {
    "B0": {"transport": "local", "runtime": "python_direct", "conc": 1},
    "B1": {"transport": "persistent_tcp", "runtime": "python_direct", "conc": 1},
    "B2": {"transport": "persistent_tcp", "runtime": "python_executor", "conc": 1},
    "B3": {"transport": "persistent_tcp", "runtime": "python_executor", "conc": 8},
    "B4": {"transport": "per_request_tcp", "runtime": "python_executor", "conc": 1},
    "B5": {"transport": "per_request_tcp", "runtime": "python_executor", "conc": 8},
}


@dataclass
class DecompositionConfig:
    cell: str
    nm: int = 8
    rank: int = 64
    n_trials: int = 5
    n_iters: int = 50

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(CELLS[self.cell])
        return d


def run_cell(config: DecompositionConfig, pool=None, act_bf16=None) -> dict:
    """Run one decomposition cell. Returns per-iteration latencies and accounting.

    B0 (local) is fully implemented here. B1-B5 require the live server and
    are wired in the S1-S6 study plans. Calling B1-B5 raises NotImplementedError.
    """
    if config.cell not in CELLS:
        raise ValueError(f"unknown cell {config.cell}; valid: {list(CELLS)}")
    cell_spec = CELLS[config.cell]

    if config.cell == "B0":
        return _run_b0_local(config, cell_spec)
    raise NotImplementedError(
        f"{config.cell} requires live server; see S1-S6 study plans"
    )


def _run_b0_local(config: DecompositionConfig, cell_spec: dict) -> dict:
    """B0: local buffers, Python direct, conc=1. No network, no RDMA.
    Lower bound for the decomposition matrix."""
    import torch
    H, I, R, NM = 2048, 2048, config.rank, config.nm
    device = "cuda"

    x = torch.randn(1, H, dtype=torch.float16, device=device)

    latencies_us = []
    outputs = []
    for trial in range(config.n_trials):
        weights_A = [torch.randn(R, H, dtype=torch.float32, device=device) for _ in range(NM)]
        weights_B = [torch.randn(R, I, dtype=torch.float32, device=device) for _ in range(NM)]

        for _ in range(config.n_iters):
            t0 = time.perf_counter()
            x_f32 = x.to(torch.float32)
            miss_outputs = []
            for i in range(NM):
                inter = x_f32 @ weights_A[i].T
                y = inter @ weights_B[i]
                miss_outputs.append(y)
            torch.cuda.synchronize()
            t18 = time.perf_counter()
            latencies_us.append((t18 - t0) * 1e6)
            if len(outputs) < 1:
                outputs = miss_outputs

    accounting = {
        "cross_domain_residual_us": 0.0,
        "instrumentation_gap_us": 0.0,
        "instrumentation_gap_fraction": 0.0,
    }
    return {
        "config": config.to_dict(),
        "latencies_us": latencies_us,
        "outputs": outputs,
        "accounting": accounting,
        "cell_spec": cell_spec,
    }


def main():
    parser = argparse.ArgumentParser(description="B0-B5 decomposition harness")
    parser.add_argument("--cell", required=True, choices=list(CELLS))
    parser.add_argument("--nm", type=int, default=8)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", default="results/decomposition/decomposition.csv")
    args = parser.parse_args()

    config = DecompositionConfig(
        cell=args.cell, nm=args.nm, rank=args.rank,
        n_trials=args.trials, n_iters=args.iters,
    )
    result = run_cell(config)
    print(f"Cell {args.cell}: {len(result['latencies_us'])} samples")
    print(f"Spec: {result['cell_spec']}")


if __name__ == "__main__":
    main()
