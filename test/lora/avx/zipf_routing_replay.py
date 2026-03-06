#!/usr/bin/env python3
"""Replay a synthetic Zipfian adapter-expert routing trace for COLoRA cache warmup analysis."""

import argparse
import random

import torch

from lightllm.server.lora.expert_cache import ExpertCacheKey, MoEExpertCacheConfig, MoEExpertCacheManager
from lightllm.server.lora.lora_mem_pool import LoRAModulePool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_adapters", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--zipf_s", type=float, default=1.2)
    parser.add_argument("--cache_budget_mb", type=int, default=256)
    parser.add_argument("--promote_min_hits", type=int, default=2)
    parser.add_argument("--promote_window", type=int, default=128)
    parser.add_argument("--max_promote_per_step", type=int, default=8)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def build_dummy_pool(num_adapters: int) -> LoRAModulePool:
    pool = LoRAModulePool.create(
        pool_size=max(num_adapters + 8, 16),
        max_rank=4,
        input_dim=64,
        output_dim=64,
        dtype=torch.float16,
        device="cpu",
        num_layers=1,
        num_experts=1,
    )

    for adapter_idx in range(num_adapters):
        A = torch.randn(2, 64, dtype=torch.float16)
        B = torch.randn(2, 64, dtype=torch.float16)
        ok = pool.load_adapter(
            adapter_idx=adapter_idx,
            rank=2,
            scaling=1.0,
            layer_weights={0: {"A": A, "B": B}},
        )
        if not ok:
            raise RuntimeError(f"failed to load adapter_idx={adapter_idx}")
    return pool


def sample_zipf_adapter(num_adapters: int, zipf_s: float, rng: random.Random) -> int:
    # Build unnormalized probabilities lazily per draw (small utility script).
    weights = [1.0 / ((i + 1) ** zipf_s) for i in range(num_adapters)]
    total = sum(weights)
    threshold = rng.random() * total
    acc = 0.0
    for idx, w in enumerate(weights):
        acc += w
        if acc >= threshold:
            return idx
    return num_adapters - 1


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    pool = build_dummy_pool(args.num_adapters)
    mgr = MoEExpertCacheManager(
        MoEExpertCacheConfig(
            cache_budget_mb=args.cache_budget_mb,
            promote_min_hits=args.promote_min_hits,
            promote_window=args.promote_window,
            max_promote_per_step=args.max_promote_per_step,
            decay=args.decay,
        )
    )
    mgr.register_projection_pool("gate", pool)

    report_every = max(args.num_steps // 10, 1)
    for step in range(1, args.num_steps + 1):
        adapters = [sample_zipf_adapter(args.num_adapters, args.zipf_s, rng) for _ in range(args.batch_size)]
        unique = sorted(set(adapters))
        keys = [ExpertCacheKey("gate", idx, 0, 0) for idx in unique]

        mgr.apply_completed_promotions()
        mgr.record_access(keys)
        ready = mgr.lookup_many(keys)
        misses = [k for k in keys if k not in ready]
        mgr.schedule_promotion(misses)

        if step % report_every == 0 or step == args.num_steps:
            print(
                f"step={step} "
                f"cache_hit_rate={mgr.get_hit_rate():.4f} "
                f"queue_depth={mgr.get_promotion_queue_depth()} "
                f"dropped_promotions={mgr.get_dropped_promotions()}"
            )


if __name__ == "__main__":
    main()
