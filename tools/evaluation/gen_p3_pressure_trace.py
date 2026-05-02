#!/usr/bin/env python3
"""Generate controlled pressure traces for P3 cache calibration.

Produces two trace families:

  * ``pressure_disjoint`` – warmup and measurement adapter pools are completely
    disjoint, guaranteeing maximum cold-miss pressure during measurement.
  * ``pressure_low_overlap`` – measurement mostly uses new adapters, but a
    configurable fraction (default 10%) overlaps with warmup adapters,
    creating partial cache residency.

Supports multi-diversity sweeps, LMSYS-backed prompts, and explicit decode
lengths.  Output files are per-diversity, e.g.:

    pressure_disjoint_div16_warmup.jsonl
    pressure_disjoint_div16_measurement.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# Default LMSYS dataset path
DEFAULT_LMSYS_PATH = (
    "/home/shufan/.cache/huggingface/hub/"
    "datasets--lmsys--lmsys-chat-1m/snapshots/"
    "200748d9d3cddcc9d782887541057aca0b18c5da"
)


def _adapter_id(index: int, *, prefix: str = "adapter_") -> str:
    return f"{prefix}{index:03d}"


# ---------------------------------------------------------------------------
# LMSYS prompt loading
# ---------------------------------------------------------------------------

def load_lmsys_prompts(
    lmsys_path: str,
    max_prompts: int = 4096,
    max_input_tokens: int = 1024,
    max_output_tokens: int = 128,
    seed: int = 0,
) -> List[Dict]:
    """Load prompts from LMSYS-Chat-1M parquet shards.

    Returns list of dicts with keys: prompt, input_len, target_output_len.
    Approximates token count as len(text) // 4 (rough word-piece estimate).
    """
    data_dir = os.path.join(lmsys_path, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"LMSYS data directory not found: {data_dir}")

    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required for LMSYS loading: pip install pyarrow")

    rng = random.Random(seed)
    prompts: List[Dict] = []

    parquet_files = sorted(
        f for f in os.listdir(data_dir) if f.endswith(".parquet")
    )

    for pf in parquet_files:
        if len(prompts) >= max_prompts:
            break
        table = pq.read_table(os.path.join(data_dir, pf), columns=["conversation"])
        for row in table.column("conversation"):
            if len(prompts) >= max_prompts:
                break
            try:
                conv = row.as_py()
                if not conv or not isinstance(conv, list):
                    continue
                # Extract first human turn as prompt
                for turn in conv:
                    if isinstance(turn, dict) and turn.get("role") == "human":
                        text = turn.get("content", "").strip()
                        if not text:
                            continue
                        approx_tokens = len(text) // 4
                        if approx_tokens > max_input_tokens:
                            # Truncate roughly
                            text = text[: max_input_tokens * 4]
                            approx_tokens = len(text) // 4
                        # Random target output length up to max
                        target_out = rng.randint(16, max_output_tokens)
                        prompts.append({
                            "prompt": text,
                            "input_len": approx_tokens,
                            "target_output_len": target_out,
                        })
                        break
            except Exception:
                continue

    rng.shuffle(prompts)
    return prompts


def get_fixed_prompts(count: int, decode_target: int, seed: int) -> List[Dict]:
    """Generate fixed-length placeholder prompts for length-mode=fixed."""
    rng = random.Random(seed)
    prompts: List[Dict] = []
    for _ in range(count):
        # Use a short placeholder prompt; input_len is not critical for fixed mode
        prompts.append({
            "prompt": "",
            "input_len": 0,
            "target_output_len": decode_target,
        })
    return prompts


# ---------------------------------------------------------------------------
# Trace generation
# ---------------------------------------------------------------------------

def _build_trace_rows(
    adapter_ids: List[str],
    phase: str,
    prompts: List[Dict],
) -> List[Dict[str, object]]:
    """Build JSONL rows with the P3.1A expanded schema."""
    rows: List[Dict[str, object]] = []
    for req_idx, aid in enumerate(adapter_ids):
        p = prompts[req_idx % len(prompts)]
        rows.append({
            "request_id": f"req_{req_idx:06d}",
            "phase": phase,
            "adapter_id": aid,
            "prompt": p["prompt"],
            "input_len": p["input_len"],
            "target_output_len": p["target_output_len"],
        })
    return rows


def generate_pressure_disjoint(
    *,
    warmup_requests: int,
    measurement_requests: int,
    diversity: int,
    seed: int,
    adapter_prefix: str,
) -> Dict[str, List[Dict[str, object]]]:
    """Warmup and measurement pools are completely disjoint."""
    rng = random.Random(seed)

    warmup_pool = [
        _adapter_id(i, prefix=adapter_prefix) for i in range(diversity)
    ]
    rng.shuffle(warmup_pool)

    measurement_pool = [
        _adapter_id(i, prefix=adapter_prefix) for i in range(diversity, 2 * diversity)
    ]
    rng.shuffle(measurement_pool)

    warmup_ids = rng.choices(warmup_pool, k=warmup_requests)
    measurement_ids = rng.choices(measurement_pool, k=measurement_requests)

    return {
        "warmup_adapter_pool": sorted(warmup_pool),
        "measurement_adapter_pool": sorted(measurement_pool),
        "warmup_ids": warmup_ids,
        "measurement_ids": measurement_ids,
        "meta": {
            "warmup_unique": len(set(warmup_ids)),
            "measurement_unique": len(set(measurement_ids)),
            "pool_size": len(measurement_pool),
            "overlap_count": 0,
            "overlap_fraction": 0.0,
        },
    }


def generate_pressure_low_overlap(
    *,
    warmup_requests: int,
    measurement_requests: int,
    diversity: int,
    overlap_fraction: float,
    seed: int,
    adapter_prefix: str,
) -> Dict[str, List[Dict[str, object]]]:
    """Measurement pool partially overlaps with warmup pool."""
    rng = random.Random(seed)

    warmup_pool = [
        _adapter_id(i, prefix=adapter_prefix) for i in range(diversity)
    ]
    rng.shuffle(warmup_pool)

    overlap_count = max(1, round(diversity * overlap_fraction))
    fresh_count = diversity - overlap_count

    overlap_ids = list(rng.sample(warmup_pool, min(overlap_count, len(warmup_pool))))
    actual_overlap = len(overlap_ids)

    fresh_ids = [
        _adapter_id(i, prefix=adapter_prefix)
        for i in range(diversity, diversity + fresh_count)
    ]

    measurement_pool = overlap_ids + fresh_ids
    rng.shuffle(measurement_pool)

    warmup_ids = rng.choices(warmup_pool, k=warmup_requests)
    measurement_ids = rng.choices(measurement_pool, k=measurement_requests)

    return {
        "warmup_adapter_pool": sorted(warmup_pool),
        "measurement_adapter_pool": sorted(measurement_pool),
        "warmup_ids": warmup_ids,
        "measurement_ids": measurement_ids,
        "meta": {
            "warmup_unique": len(set(warmup_ids)),
            "measurement_unique": len(set(measurement_ids)),
            "pool_size": len(measurement_pool),
            "overlap_count": actual_overlap,
            "overlap_fraction": actual_overlap / max(diversity, 1),
        },
    }


def generate_pressure_mixed_overlap(
    *,
    warmup_requests: int,
    measurement_requests: int,
    diversity: int,
    overlap_fraction: float,
    seed: int,
    adapter_prefix: str,
) -> Dict[str, List[Dict[str, object]]]:
    """Measurement pool overlaps with warmup pool at a higher fraction (default 30%).

    This is a semantic alias for ``generate_pressure_low_overlap`` with a
    higher default overlap fraction, intended for experiments that need
    non-trivial GPU hot-path participation during measurement.
    """
    return generate_pressure_low_overlap(
        warmup_requests=warmup_requests,
        measurement_requests=measurement_requests,
        diversity=diversity,
        overlap_fraction=overlap_fraction,
        seed=seed,
        adapter_prefix=adapter_prefix,
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)
        f.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "artifacts" / "evaluation" / "p3_traces")
    p.add_argument("--trace-types", default="pressure_disjoint,pressure_low_overlap",
                    help="Comma-separated trace types")
    p.add_argument("--diversities", default="16,32,64,128",
                    help="Comma-separated adapter diversity values")
    p.add_argument("--warmup-requests", type=int, default=128)
    p.add_argument("--measurement-requests", type=int, default=512)
    p.add_argument("--overlap-fraction", type=float, default=0.1,
                    help="Overlap fraction for pressure_low_overlap traces")
    p.add_argument("--mixed-overlap-fraction", type=float, default=0.3,
                    help="Overlap fraction for pressure_mixed_overlap traces")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--adapter-prefix", default="adapter_")

    # Prompt / length args
    p.add_argument("--lmsys-path", default=DEFAULT_LMSYS_PATH,
                    help="Path to LMSYS-Chat-1M dataset snapshot")
    p.add_argument("--length-mode", choices=["fixed", "lmsys"], default="fixed",
                    help="Prompt/length source")
    p.add_argument("--decode-target-tokens", type=int, default=64,
                    help="Target output tokens for fixed mode")
    p.add_argument("--max-input-tokens", type=int, default=1024)
    p.add_argument("--max-output-tokens", type=int, default=128)

    return p.parse_args()


TRACE_GENERATORS = {
    "pressure_disjoint": generate_pressure_disjoint,
    "pressure_low_overlap": generate_pressure_low_overlap,
    "pressure_mixed_overlap": generate_pressure_mixed_overlap,
}


def main() -> int:
    args = parse_args()
    out = args.output_dir

    trace_types = [t.strip() for t in args.trace_types.split(",")]
    diversities = [int(d.strip()) for d in args.diversities.split(",")]

    # Load prompts
    total_requests = args.warmup_requests + args.measurement_requests
    if args.length_mode == "lmsys":
        print(f"Loading LMSYS prompts from {args.lmsys_path} ...")
        prompts = load_lmsys_prompts(
            args.lmsys_path,
            max_prompts=max(total_requests, 4096),
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            seed=args.seed,
        )
        print(f"  Loaded {len(prompts)} prompts")
    else:
        prompts = get_fixed_prompts(total_requests, args.decode_target_tokens, args.seed)

    # Split prompts into warmup / measurement pools
    warmup_prompts = prompts[:args.warmup_requests]
    measurement_prompts = prompts[args.warmup_requests:args.warmup_requests + args.measurement_requests]

    all_meta = []

    for tt in trace_types:
        gen_fn = TRACE_GENERATORS.get(tt)
        if gen_fn is None:
            print(f"WARNING: unknown trace type {tt}, skipping", file=__import__("sys").stderr)
            continue

        for div in diversities:
            label = f"{tt}_div{div}"
            gen_kwargs = dict(
                warmup_requests=args.warmup_requests,
                measurement_requests=args.measurement_requests,
                diversity=div,
                seed=args.seed,
                adapter_prefix=args.adapter_prefix,
            )
            if tt == "pressure_low_overlap":
                gen_kwargs["overlap_fraction"] = args.overlap_fraction
            elif tt == "pressure_mixed_overlap":
                gen_kwargs["overlap_fraction"] = args.mixed_overlap_fraction

            result = gen_fn(**gen_kwargs)

            # Build rows with expanded schema
            warmup_rows = _build_trace_rows(
                result["warmup_ids"], "warmup", warmup_prompts,
            )
            measurement_rows = _build_trace_rows(
                result["measurement_ids"], "measurement", measurement_prompts,
            )

            # Write per-diversity files
            write_jsonl(out / f"{tt}_div{div}_warmup.jsonl", warmup_rows)
            write_jsonl(out / f"{tt}_div{div}_measurement.jsonl", measurement_rows)

            m = result["meta"]
            print(f"{label}: {len(warmup_rows)} warmup, {len(measurement_rows)} measurement, "
                  f"unique(warm)={m['warmup_unique']} unique(meas)={m['measurement_unique']} "
                  f"overlap={m['overlap_count']}")

            # Collect metadata
            meta_entry = {
                "trace_name": label,
                "diversity": div,
                "warmup_requests": args.warmup_requests,
                "measurement_requests": args.measurement_requests,
                "warmup_adapter_pool": result["warmup_adapter_pool"],
                "measurement_adapter_pool": result["measurement_adapter_pool"],
                "overlap_fraction": m["overlap_fraction"],
                "seed": args.seed,
                "length_mode": args.length_mode,
                "decode_target_tokens": args.decode_target_tokens,
                "lmsys_path": args.lmsys_path if args.length_mode == "lmsys" else "",
                "max_input_tokens": args.max_input_tokens,
                "max_output_tokens": args.max_output_tokens,
            }
            all_meta.append(meta_entry)

    # Write shared metadata
    meta_path = out / "pressure_trace_meta.json"
    write_json(meta_path, {"traces": all_meta})
    print(f"\nWrote metadata to {meta_path}")

    # Backward compatibility: write legacy filenames if single diversity
    if len(diversities) == 1:
        div = diversities[0]
        for tt in trace_types:
            src_warmup = out / f"{tt}_div{div}_warmup.jsonl"
            src_meas = out / f"{tt}_div{div}_measurement.jsonl"
            dst_warmup = out / f"{tt}_warmup.jsonl"
            dst_meas = out / f"{tt}_measurement.jsonl"
            if src_warmup.exists():
                import shutil
                shutil.copy2(str(src_warmup), str(dst_warmup))
                shutil.copy2(str(src_meas), str(dst_meas))
                print(f"  Legacy: {dst_warmup.name}, {dst_meas.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
