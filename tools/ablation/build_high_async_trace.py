#!/usr/bin/env python3
"""Build a trace optimized for high async CPU submission ratio.

This creates a synthetic trace where:
1. Adapters are reused frequently (high correlation)
2. Experts are reused within adapters
3. This increases cache hit rate, enabling async CPU overlap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List
import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate high-async-ratio synthetic trace")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--request_count", type=int, default=128, help="Number of requests")
    parser.add_argument("--decode_tokens", type=int, default=64, help="Decode tokens per request")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--experts_per_token", type=int, default=4, help="Experts per token")
    parser.add_argument("--num_experts", type=int, default=32, help="Total expert IDs")
    parser.add_argument("--num_loras", type=int, default=16, help="Total LoRA IDs")
    parser.add_argument("--adapter_reuse_prob", type=float, default=0.95,
                       help="Probability of reusing previous adapter (high = more async)")
    parser.add_argument("--expert_reuse_prob", type=float, default=0.85,
                       help="Probability of reusing previous expert within adapter")
    parser.add_argument("--concurrent_loras", type=int, default=4,
                       help="Number of LoRAs active concurrently (affects overlap)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))

    # Track active adapters to simulate concurrent access patterns
    active_loras = []
    previous_adapter = None

    # Track expert assignment per adapter for reuse
    adapter_experts: dict[int, List[int]] = {}

    output_rows = []
    arrival_idx = 0

    print(f"Generating trace with {args.request_count} requests, {args.decode_tokens} decode tokens each")
    print(f"Adapter reuse prob: {args.adapter_reuse_prob:.2f}, Expert reuse prob: {args.expert_reuse_prob:.2f}")
    print(f"Concurrent LoRAs: {args.concurrent_loras}")

    for req_idx in range(args.request_count):
        # Choose adapter with high reuse probability
        if previous_adapter is not None and float(rng.random()) < float(args.adapter_reuse_prob):
            adapter_id = previous_adapter
        elif active_loras and float(rng.random()) < 0.7:
            # Pick from active set for overlap
            adapter_id = int(rng.choice(active_loras))
        else:
            # New adapter
            adapter_id = int(rng.integers(0, args.num_loras))

            # Manage active LoRA set
            if adapter_id not in active_loras:
                if len(active_loras) >= args.concurrent_loras:
                    active_loras.pop(0)  # Evict oldest
                active_loras.append(adapter_id)

        previous_adapter = adapter_id

        # Ensure this adapter has expert assignments
        if adapter_id not in adapter_experts:
            adapter_experts[adapter_id] = list(range(args.num_experts))
            rng.shuffle(adapter_experts[adapter_id])

        # Generate events for prefill and decode
        for phase_name, token_count in (("prefill", 1), ("decode", args.decode_tokens)):
            for token_pos in range(token_count):
                for layer_id in range(args.num_layers):
                    # Select experts with high reuse probability
                    if float(rng.random()) < float(args.expert_reuse_prob):
                        # Reuse previous experts for this adapter
                        if len(adapter_experts[adapter_id]) >= args.experts_per_token:
                            experts = adapter_experts[adapter_id][:args.experts_per_token]
                        else:
                            experts = adapter_experts[adapter_id]
                    else:
                        # Select new random experts
                        experts = rng.choice(args.num_experts, size=args.experts_per_token,
                                          replace=False).tolist()
                        # Update adapter's expert assignment
                        adapter_experts[adapter_id] = experts

                    # Get weights
                    weights = [float(rng.random()) for _ in experts]
                    total = sum(weights)
                    weights = [w / total for w in weights]

                    # Create event record
                    event = {
                        "event": "router_trace",
                        "arrival_idx": arrival_idx,
                        "req_idx": req_idx,
                        "layer_id": layer_id,
                        "token_pos": token_pos,
                        "phase": phase_name,
                        "topk_experts": [int(e) for e in experts],
                        "topk_weights": weights,
                    }
                    output_rows.append(event)
                    arrival_idx += 1

    # Write output
    output_path = output_dir / "high_async_trace.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Write summary
    summary_path = output_dir / "high_async_trace_summary.json"
    summary = {
        "total_events": len(output_rows),
        "total_requests": args.request_count,
        "decode_tokens_per_request": args.decode_tokens,
        "num_layers": args.num_layers,
        "num_experts": args.num_experts,
        "num_loras": args.num_loras,
        "adapter_reuse_probability": args.adapter_reuse_prob,
        "expert_reuse_probability": args.expert_reuse_prob,
        "concurrent_loras": args.concurrent_loras,
        "expected_async_ratio": "high (target: 70%+)",
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(output_rows)} events to {output_path}")
    print(f"Summary: {summary_path}")
    print(f"\nExpected characteristics:")
    print(f"  - High adapter reuse ({args.adapter_reuse_prob*100:.0f}%) → higher cache hit rate")
    print(f"  - {args.concurrent_loras} concurrent LoRAs → more overlap opportunities")
    print(f"  - Expert reuse ({args.expert_reuse_prob*100:.0f}%) → fewer cache misses")
    print(f"  - Target async CPU ratio: 70%+")

if __name__ == "__main__":
    main()