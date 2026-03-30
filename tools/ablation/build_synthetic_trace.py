#!/usr/bin/env python3
"""Build a minimal joined-trace-compatible synthetic workload for ablation debugging."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
CASE_STUDY_DIR = THIS_DIR.parent / "case_study"

import sys

if str(CASE_STUDY_DIR) not in sys.path:
    sys.path.append(str(CASE_STUDY_DIR))

from common import ensure_dir, write_json, write_jsonl  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic joined traces for ablation debugging")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for synthetic joined traces")
    parser.add_argument("--trace_run_id", type=str, default="synthetic_ablation", help="Trace run id")
    parser.add_argument("--request_count", type=int, default=64, help="Number of synthetic requests")
    parser.add_argument("--decode_tokens", type=int, default=16, help="Decode tokens per request")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of synthetic layers")
    parser.add_argument("--experts_per_token", type=int, default=4, help="Experts per token-layer step")
    parser.add_argument("--num_experts", type=int, default=32, help="Total expert ids")
    parser.add_argument("--num_loras", type=int, default=16, help="Total synthetic LoRA ids")
    parser.add_argument("--corr_strength", type=float, default=0.85, help="Adapter repetition probability for the corr trace")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    rng = np.random.default_rng(int(args.seed))

    indep_rows: List[dict] = []
    corr_rows: List[dict] = []
    previous_corr_adapter = None

    for req_idx in range(int(args.request_count)):
        indep_adapter = int(rng.integers(0, int(args.num_loras)))
        if previous_corr_adapter is not None and float(rng.random()) < float(args.corr_strength):
            corr_adapter = int(previous_corr_adapter)
        else:
            corr_adapter = int(rng.integers(0, int(args.num_loras)))
        previous_corr_adapter = corr_adapter

        event_idx = 0
        for phase_name, token_count in (("prefill", 1), ("decode", int(args.decode_tokens))):
            for token_pos in range(token_count):
                for layer_id in range(int(args.num_layers)):
                    experts = rng.choice(int(args.num_experts), size=int(args.experts_per_token), replace=False)
                    for expert_id in experts:
                        base_row = {
                            "arrival_idx": int(req_idx),
                            "req_idx": int(req_idx),
                            "event_idx": int(event_idx),
                            "layer_id": int(layer_id),
                            "token_pos": int(token_pos),
                            "phase": str(phase_name),
                            "expert_id": int(expert_id),
                            "mapping_mode": "synthetic",
                            "duration_ms": int(100 + req_idx),
                            "model_name": "synthetic-moe-lora",
                            "trace_run_id": str(args.trace_run_id),
                            "start_ts": int(req_idx * 10),
                        }
                        indep_rows.append({**base_row, "adapter_id": f"lora_{indep_adapter}"})
                        corr_rows.append({**base_row, "adapter_id": f"lora_{corr_adapter}"})
                    event_idx += 1

    write_jsonl(output_dir / "joined_trace_indep.jsonl", indep_rows)
    write_jsonl(output_dir / "joined_trace_corr.jsonl", corr_rows)
    write_json(
        output_dir / "join_qc_report.json",
        {
            "checks": {
                "per_mode_row_counts": {
                    "indep": len(indep_rows),
                    "corr": len(corr_rows),
                }
            }
        },
    )
    print(f"Wrote synthetic traces to {output_dir}")


if __name__ == "__main__":
    main()
