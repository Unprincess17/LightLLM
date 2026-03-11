#!/usr/bin/env python3
"""Trace-driven replay utilities for expert-LoRA cache fragmentation studies."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from lightllm.server.lora.expert_cache import ExpertCacheKey, MoEExpertCacheConfig, MoEExpertCacheManager
from lightllm.server.lora.lora_mem_pool import LoRAModulePool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay router and adapter traces for COLoRA case studies")

    # Legacy warmup mode arguments. If --router_trace_path is omitted, the script
    # preserves the original synthetic cache warmup behavior.
    parser.add_argument("--num_adapters", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--zipf_s", type=float, default=1.2)
    parser.add_argument("--cache_budget_mb", type=int, default=256)
    parser.add_argument("--promote_min_hits", type=int, default=2)
    parser.add_argument("--promote_window", type=int, default=128)
    parser.add_argument("--max_promote_per_step", type=int, default=8)
    parser.add_argument("--decay", type=float, default=0.9)

    # Trace-driven replay mode arguments.
    parser.add_argument("--router_trace_path", type=str, default=None, help="Ordered router trace JSONL path")
    parser.add_argument("--adapter_trace_path", type=str, default=None, help="Adapter trace JSONL path")
    parser.add_argument("--output_adapter_trace_path", type=str, default=None, help="Write generated adapter trace JSONL")
    parser.add_argument("--output_joined_trace_path", type=str, default=None, help="Write joined access trace JSONL")
    parser.add_argument("--output_summary_path", type=str, default=None, help="Write replay summary JSON")
    parser.add_argument("--skew", type=str, default="zipf", choices=["uniform", "zipf"])
    parser.add_argument("--correlation", type=str, default="independent", choices=["independent", "class_conditional"])
    parser.add_argument("--session_mean", type=float, default=1.0, help="Mean request count per sticky session")
    parser.add_argument("--burst_probability", type=float, default=0.0, help="Probability of starting a burst")
    parser.add_argument("--burst_factor", type=float, default=4.0, help="Burst length multiplier over session_mean")
    parser.add_argument("--class_locality", type=float, default=0.8, help="Probability of sampling from class-local adapter pool")
    parser.add_argument("--num_classes", type=int, default=8, help="Number of request classes for correlated traces")
    parser.add_argument("--projections", type=str, default="up,down", help="Comma-separated projections for joined trace")
    parser.add_argument("--cache_capacity", type=int, default=128, help="Replay cache capacity in resident objects")
    parser.add_argument(
        "--cache_capacity_fraction",
        type=float,
        default=0.0,
        help="If > 0, derive cache capacity as this fraction of the joint working set",
    )
    parser.add_argument("--hit_cost", type=float, default=1.0)
    parser.add_argument("--miss_cost_prefill", type=float, default=8.0)
    parser.add_argument("--miss_cost_decode", type=float, default=20.0)
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
    weights = [1.0 / ((i + 1) ** zipf_s) for i in range(num_adapters)]
    total = sum(weights)
    threshold = rng.random() * total
    acc = 0.0
    for idx, weight in enumerate(weights):
        acc += weight
        if acc >= threshold:
            return idx
    return num_adapters - 1


def run_legacy_warmup(args: argparse.Namespace) -> None:
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
        misses = [key for key in keys if key not in ready]
        mgr.schedule_promotion(misses)

        if step % report_every == 0 or step == args.num_steps:
            drops = mgr.get_promotion_drop_breakdown()
            print(
                f"step={step} "
                f"cache_hit_rate={mgr.get_hit_rate():.4f} "
                f"queue_depth={mgr.get_promotion_queue_depth()} "
                f"dropped_promotions={mgr.get_dropped_promotions()} "
                f"drop_queue={drops.get('queue_high_watermark', 0)} "
                f"drop_cooldown={drops.get('cooldown', 0)}"
            )


def load_jsonl(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_num} is not a JSON object")
            records.append(payload)
    return records


def load_router_trace(path: str) -> List[dict]:
    records = []
    for record in load_jsonl(path):
        if record.get("event") not in (None, "router_trace"):
            continue
        if "req_idx" not in record or "topk_experts" not in record:
            continue
        records.append(
            {
                "arrival_idx": int(record.get("arrival_idx", len(records))),
                "req_idx": int(record["req_idx"]),
                "phase": str(record.get("phase", "decode")),
                "layer_id": int(record.get("layer_id", 0)),
                "token_pos": int(record.get("token_pos", 0)),
                "topk_experts": [int(expert_id) for expert_id in record.get("topk_experts", [])],
                "topk_weights": [float(weight) for weight in record.get("topk_weights", [])],
            }
        )
    records.sort(key=lambda item: (item["arrival_idx"], item["layer_id"], item["token_pos"]))
    return records


def load_adapter_trace_rows(path: str) -> List[dict]:
    rows = []
    for line_num, record in enumerate(load_jsonl(path), start=1):
        if "adapter_id" not in record:
            raise ValueError(f"{path}:{line_num} missing adapter_id")
        rows.append(
            {
                "arrival_idx": int(record.get("arrival_idx", line_num - 1)),
                "req_idx": int(record.get("req_idx", line_num - 1)),
                "adapter_id": record.get("adapter_id"),
                "session_id": int(record.get("session_id", line_num - 1)),
                "request_class": int(record.get("request_class", -1)),
            }
        )
    rows.sort(key=lambda item: (item["arrival_idx"], item["req_idx"]))
    return rows


def build_request_order(router_events: Sequence[dict]) -> List[int]:
    first_seen: Dict[int, int] = {}
    for record in router_events:
        req_idx = int(record["req_idx"])
        arrival_idx = int(record["arrival_idx"])
        if req_idx not in first_seen or arrival_idx < first_seen[req_idx]:
            first_seen[req_idx] = arrival_idx
    return [req_idx for req_idx, _arrival in sorted(first_seen.items(), key=lambda item: item[1])]


def build_request_classes(router_events: Sequence[dict], num_classes: int) -> Dict[int, int]:
    if num_classes <= 0:
        return {}

    expert_histograms: Dict[int, Counter] = defaultdict(Counter)
    for record in router_events:
        req_idx = int(record["req_idx"])
        for expert_id in record.get("topk_experts", []):
            expert_histograms[req_idx][int(expert_id)] += 1

    request_classes = {}
    for req_idx, histogram in expert_histograms.items():
        dominant_expert = min(histogram.keys()) if not histogram else histogram.most_common(1)[0][0]
        request_classes[req_idx] = int(dominant_expert) % num_classes
    return request_classes


def choose_adapter(
    rng: random.Random,
    adapter_pool: Sequence[int],
    skew: str,
    zipf_s: float,
) -> int:
    if not adapter_pool:
        raise ValueError("adapter pool is empty")
    if skew == "uniform":
        return int(rng.choice(list(adapter_pool)))
    sampled_index = sample_zipf_adapter(len(adapter_pool), zipf_s, rng)
    return int(adapter_pool[sampled_index])


def sample_session_length(session_mean: float, rng: random.Random) -> int:
    if session_mean <= 1.0:
        return 1
    lam = 1.0 / max(session_mean, 1e-6)
    return max(1, int(math.ceil(rng.expovariate(lam))))


def generate_adapter_trace(
    router_events: Sequence[dict],
    num_adapters: int,
    skew: str,
    zipf_s: float,
    correlation: str,
    session_mean: float,
    burst_probability: float,
    burst_factor: float,
    class_locality: float,
    num_classes: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    request_order = build_request_order(router_events)
    request_classes = build_request_classes(router_events, num_classes)
    global_pool = list(range(num_adapters))
    class_pool_size = max(1, math.ceil(num_adapters / max(num_classes, 1)))

    def class_pool_for(req_idx: int) -> List[int]:
        if correlation != "class_conditional":
            return global_pool
        class_id = request_classes.get(req_idx, 0)
        start = (class_id * class_pool_size) % num_adapters
        pool = [(start + offset) % num_adapters for offset in range(min(class_pool_size, num_adapters))]
        return sorted(set(pool))

    def sample_adapter_for_request(req_idx: int) -> int:
        if correlation == "class_conditional" and rng.random() < class_locality:
            return choose_adapter(rng, class_pool_for(req_idx), skew, zipf_s)
        return choose_adapter(rng, global_pool, skew, zipf_s)

    rows: List[dict] = []
    session_id = 0
    session_remaining = 0
    session_adapter = None
    burst_remaining = 0
    burst_adapter = None

    for arrival_idx, req_idx in enumerate(request_order):
        if burst_remaining <= 0 and burst_probability > 0.0 and rng.random() < burst_probability:
            burst_remaining = max(1, int(round(max(session_mean, 1.0) * max(burst_factor, 1.0))))
            burst_adapter = sample_adapter_for_request(req_idx)
            session_id += 1

        if burst_remaining > 0:
            adapter_id = burst_adapter
            burst_remaining -= 1
        else:
            if session_remaining <= 0 or session_adapter is None:
                session_adapter = sample_adapter_for_request(req_idx)
                session_remaining = sample_session_length(session_mean, rng) - 1
                session_id += 1
            else:
                session_remaining -= 1
            adapter_id = session_adapter

        rows.append(
            {
                "arrival_idx": int(arrival_idx),
                "req_idx": int(req_idx),
                "adapter_id": int(adapter_id),
                "session_id": int(session_id),
                "request_class": int(request_classes.get(req_idx, -1)),
            }
        )

    return rows


def align_adapter_trace(request_order: Sequence[int], adapter_rows: Sequence[dict]) -> Dict[int, Optional[int]]:
    direct = {int(row["req_idx"]): row.get("adapter_id") for row in adapter_rows}
    if request_order and all(req_idx in direct for req_idx in request_order):
        return {int(req_idx): direct[int(req_idx)] for req_idx in request_order}

    if len(adapter_rows) < len(request_order):
        raise ValueError(
            f"adapter trace has {len(adapter_rows)} rows but router trace references {len(request_order)} requests"
        )

    assignment = {}
    for req_idx, row in zip(request_order, adapter_rows):
        assignment[int(req_idx)] = row.get("adapter_id")
    return assignment


def parse_projections(raw: str) -> List[str]:
    projections = [token.strip() for token in raw.split(",") if token.strip()]
    if not projections:
        return ["up", "down"]
    return projections


def join_router_and_adapter_traces(
    router_events: Sequence[dict],
    adapter_assignment: Dict[int, object],
    projections: Sequence[str],
) -> List[dict]:
    joined: List[dict] = []
    access_idx = 0
    for record in router_events:
        req_idx = int(record["req_idx"])
        adapter_id = adapter_assignment.get(req_idx)
        experts = [int(expert_id) for expert_id in record.get("topk_experts", []) if int(expert_id) >= 0]
        weights = [float(weight) for weight in record.get("topk_weights", [])]
        for projection in projections:
            for topk_rank, expert_id in enumerate(experts):
                gate_weight = weights[topk_rank] if topk_rank < len(weights) else 0.0
                joined.append(
                    {
                        "access_idx": int(access_idx),
                        "arrival_idx": int(record["arrival_idx"]),
                        "req_idx": req_idx,
                        "phase": str(record.get("phase", "decode")),
                        "layer_id": int(record.get("layer_id", 0)),
                        "token_pos": int(record.get("token_pos", 0)),
                        "projection": projection,
                        "expert_id": expert_id,
                        "adapter_id": adapter_id,
                        "topk_rank": int(topk_rank),
                        "gate_weight": float(gate_weight),
                    }
                )
                access_idx += 1
    return joined


def cache_key(record: dict, collapse_adapter: bool = False) -> Tuple[object, ...]:
    key = (record["projection"], int(record["layer_id"]), int(record["expert_id"]))
    if collapse_adapter:
        return key
    return key + (record.get("adapter_id"),)


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0.0:
        return float(min(values))
    if pct >= 1.0:
        return float(max(values))
    ordered = sorted(float(value) for value in values)
    pos = (len(ordered) - 1) * pct
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def entropy_from_counts(counts: Iterable[int]) -> float:
    values = [int(count) for count in counts if int(count) > 0]
    total = sum(values)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in values:
        prob = count / total
        entropy -= prob * math.log(prob, 2)
    return entropy


def summarize_distribution(records: Sequence[dict], collapse_adapter: bool = False) -> dict:
    counts = Counter(cache_key(record, collapse_adapter=collapse_adapter) for record in records)
    total = sum(counts.values())
    top10 = counts.most_common(10)
    return {
        "total_accesses": int(total),
        "working_set_size": int(len(counts)),
        "top10_coverage": float(sum(count for _key, count in top10) / total) if total else 0.0,
        "entropy_bits": float(entropy_from_counts(counts.values())),
    }


class FenwickTree:
    def __init__(self, size: int):
        self.tree = [0] * (size + 2)

    def add(self, index: int, delta: int) -> None:
        while index < len(self.tree):
            self.tree[index] += delta
            index += index & -index

    def prefix_sum(self, index: int) -> int:
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= index & -index
        return result


def compute_reuse_distance(records: Sequence[dict], collapse_adapter: bool = False) -> dict:
    fenwick = FenwickTree(len(records) + 2)
    last_seen: Dict[Tuple[object, ...], int] = {}
    reuse_distances: List[int] = []
    cold_count = 0

    for position, record in enumerate(records, start=1):
        key = cache_key(record, collapse_adapter=collapse_adapter)
        previous = last_seen.get(key)
        if previous is None:
            cold_count += 1
        else:
            distance = fenwick.prefix_sum(position - 1) - fenwick.prefix_sum(previous)
            reuse_distances.append(int(distance))
            fenwick.add(previous, -1)
        fenwick.add(position, 1)
        last_seen[key] = position

    return {
        "cold_count": int(cold_count),
        "observed_reuses": int(len(reuse_distances)),
        "mean": float(sum(reuse_distances) / len(reuse_distances)) if reuse_distances else 0.0,
        "p50": float(percentile(reuse_distances, 0.50)),
        "p95": float(percentile(reuse_distances, 0.95)),
    }


def summarize_latency(req_stats: Dict[int, dict]) -> dict:
    latencies = [float(stats["latency"]) for stats in req_stats.values()]
    misses = [int(stats["misses"]) for stats in req_stats.values()]
    return {
        "request_count": int(len(req_stats)),
        "mean": float(sum(latencies) / len(latencies)) if latencies else 0.0,
        "p50": float(percentile(latencies, 0.50)),
        "p95": float(percentile(latencies, 0.95)),
        "p99": float(percentile(latencies, 0.99)),
        "mean_misses": float(sum(misses) / len(misses)) if misses else 0.0,
        "p99_misses": float(percentile(misses, 0.99)),
    }


def simulate_lru(
    records: Sequence[dict],
    capacity: int,
    collapse_adapter: bool,
    hit_cost: float,
    miss_cost_prefill: float,
    miss_cost_decode: float,
) -> dict:
    resident: OrderedDict[Tuple[object, ...], None] = OrderedDict()
    req_stats: Dict[int, dict] = defaultdict(lambda: {"hits": 0, "misses": 0, "latency": 0.0})
    hits = 0
    misses = 0

    for record in records:
        req_idx = int(record["req_idx"])
        key = cache_key(record, collapse_adapter=collapse_adapter)
        phase = str(record.get("phase", "decode"))
        miss_cost = miss_cost_prefill if phase == "prefill" else miss_cost_decode

        hit = capacity > 0 and key in resident
        if hit:
            hits += 1
            req_stats[req_idx]["hits"] += 1
            req_stats[req_idx]["latency"] += hit_cost
            resident.move_to_end(key)
        else:
            misses += 1
            req_stats[req_idx]["misses"] += 1
            req_stats[req_idx]["latency"] += miss_cost
            if capacity > 0:
                if key in resident:
                    resident.move_to_end(key)
                else:
                    if len(resident) >= capacity:
                        resident.popitem(last=False)
                    resident[key] = None

    total = hits + misses
    return {
        "cache_capacity": int(capacity),
        "hit_rate": float(hits / total) if total else 0.0,
        "miss_rate": float(misses / total) if total else 0.0,
        "hits": int(hits),
        "misses": int(misses),
        "latency": summarize_latency(req_stats),
    }


def write_jsonl(path: str, rows: Sequence[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def build_trace_driven_summary(args: argparse.Namespace) -> dict:
    router_events = load_router_trace(args.router_trace_path)
    if not router_events:
        raise ValueError(f"router trace is empty: {args.router_trace_path}")

    request_order = build_request_order(router_events)
    if args.adapter_trace_path:
        adapter_rows = load_adapter_trace_rows(args.adapter_trace_path)
    else:
        adapter_rows = generate_adapter_trace(
            router_events=router_events,
            num_adapters=args.num_adapters,
            skew=args.skew,
            zipf_s=args.zipf_s,
            correlation=args.correlation,
            session_mean=args.session_mean,
            burst_probability=args.burst_probability,
            burst_factor=args.burst_factor,
            class_locality=args.class_locality,
            num_classes=args.num_classes,
            seed=args.seed,
        )
        if args.output_adapter_trace_path:
            write_jsonl(args.output_adapter_trace_path, adapter_rows)

    adapter_assignment = align_adapter_trace(request_order, adapter_rows)
    projections = parse_projections(args.projections)
    joined = join_router_and_adapter_traces(router_events, adapter_assignment, projections)
    if args.output_joined_trace_path:
        write_jsonl(args.output_joined_trace_path, joined)

    joint_distribution = summarize_distribution(joined, collapse_adapter=False)
    expert_distribution = summarize_distribution(joined, collapse_adapter=True)
    if args.cache_capacity_fraction > 0.0:
        cache_capacity = max(1, int(math.ceil(joint_distribution["working_set_size"] * args.cache_capacity_fraction)))
    else:
        cache_capacity = max(int(args.cache_capacity), 0)

    summary = {
        "router_events": int(len(router_events)),
        "request_count": int(len(request_order)),
        "joined_events": int(len(joined)),
        "projections": projections,
        "cache_capacity": int(cache_capacity),
        "joint": {
            "distribution": joint_distribution,
            "reuse_distance": compute_reuse_distance(joined, collapse_adapter=False),
            "cache": simulate_lru(
                records=joined,
                capacity=cache_capacity,
                collapse_adapter=False,
                hit_cost=args.hit_cost,
                miss_cost_prefill=args.miss_cost_prefill,
                miss_cost_decode=args.miss_cost_decode,
            ),
        },
        "expert_only": {
            "distribution": expert_distribution,
            "reuse_distance": compute_reuse_distance(joined, collapse_adapter=True),
            "cache": simulate_lru(
                records=joined,
                capacity=cache_capacity,
                collapse_adapter=True,
                hit_cost=args.hit_cost,
                miss_cost_prefill=args.miss_cost_prefill,
                miss_cost_decode=args.miss_cost_decode,
            ),
        },
        "adapter_assignment": {
            "unique_adapters": int(len(set(adapter_assignment.values()))),
            "counts": {
                str(adapter_id): int(count)
                for adapter_id, count in sorted(Counter(adapter_assignment.values()).items(), key=lambda item: str(item[0]))
            },
        },
    }
    return summary


def print_trace_summary(summary: dict) -> None:
    print(
        f"requests={summary['request_count']} "
        f"router_events={summary['router_events']} "
        f"joined_events={summary['joined_events']} "
        f"cache_capacity={summary['cache_capacity']}"
    )
    print(f"adapters={summary['adapter_assignment']['counts']}")
    for label in ("expert_only", "joint"):
        dist = summary[label]["distribution"]
        reuse = summary[label]["reuse_distance"]
        cache = summary[label]["cache"]
        latency = cache["latency"]
        print(
            f"{label}: "
            f"working_set={dist['working_set_size']} "
            f"top10_coverage={dist['top10_coverage']:.4f} "
            f"entropy_bits={dist['entropy_bits']:.4f} "
            f"reuse_p95={reuse['p95']:.2f} "
            f"hit_rate={cache['hit_rate']:.4f} "
            f"miss_rate={cache['miss_rate']:.4f} "
            f"latency_p99={latency['p99']:.2f} "
            f"mean_misses={latency['mean_misses']:.2f}"
        )


def main() -> None:
    args = parse_args()
    if args.router_trace_path is None:
        run_legacy_warmup(args)
        return

    summary = build_trace_driven_summary(args)
    print_trace_summary(summary)

    if args.output_summary_path:
        out_path = Path(args.output_summary_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
