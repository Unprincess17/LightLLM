#!/usr/bin/env python3
"""
Plot adapter-expert routing heatmaps from MoE adapter profiling logs.

Input format: JSON Lines from /tmp/moe_adapter_expert_profile.log with fields:
- layer
- mode
- counts_topk / counts_top1

Example:
    python test/lora/plot_adapter_expert_heatmap.py \
      --log /tmp/moe_adapter_expert_profile.log \
      --output_dir /tmp/moe_heatmaps \
      --metric top1 \
      --mode decode
"""

import argparse
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_int_set(csv_values: Optional[str]) -> Optional[Set[int]]:
    if not csv_values:
        return None
    values: Set[int] = set()
    for token in csv_values.split(","):
        token = token.strip()
        if not token:
            continue
        values.add(int(token))
    return values or None


def parse_label_map(csv_pairs: Optional[str]) -> Dict[int, str]:
    """
    Parse mapping from string like: '0:math,1:coding,2:chat'.
    """
    mapping: Dict[int, str] = {}
    if not csv_pairs:
        return mapping
    for token in csv_pairs.split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        key_str, value = token.split(":", 1)
        try:
            mapping[int(key_str.strip())] = value.strip()
        except ValueError:
            continue
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot adapter/expert routing heatmaps")
    parser.add_argument("--log", type=str, default="/tmp/moe_adapter_expert_profile.log", help="JSONL log path")
    parser.add_argument("--output_dir", type=str, default="/tmp/moe_heatmaps", help="Output directory")
    parser.add_argument("--metric", type=str, default="top1", choices=["top1", "topk"], help="Use counts_top1 or counts_topk")
    parser.add_argument("--mode", type=str, default="decode", choices=["decode", "prefill", "all"], help="Filter mode")
    parser.add_argument("--layers", type=str, default=None, help="Optional comma-separated layer ids to include")
    parser.add_argument("--dedup_consecutive", action="store_true", help="Drop consecutive identical JSON lines")
    parser.add_argument("--log_color", action="store_true", help="Apply log1p to matrix values for plotting")
    parser.add_argument("--max_layers", type=int, default=None, help="Only keep first K layers after filtering")
    parser.add_argument(
        "--layer_adapter_stat",
        type=str,
        default="dominant_expert_id",
        choices=["total_count", "top_expert_share", "active_expert_count", "entropy", "dominant_expert_id"],
        help=(
            "Statistic used for Layer x Adapter heatmap: "
            "total_count (often constant), top_expert_share (skew), "
            "active_expert_count (diversity), entropy (routing spread), "
            "dominant_expert_id (which expert dominates)."
        ),
    )
    parser.add_argument(
        "--adapter_labels",
        type=str,
        default=None,
        help="Optional mapping: '0:math,1:coding,2:chat' for adapter axis labels",
    )
    return parser.parse_args()


def iter_records(log_path: str) -> Iterable[Dict]:
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_filtered_records(
    log_path: str,
    metric: str,
    mode: str,
    layers: Optional[Set[int]],
    dedup_consecutive: bool,
    max_layers: Optional[int],
) -> List[Dict]:
    key = "counts_top1" if metric == "top1" else "counts_topk"
    records: List[Dict] = []
    prev_line_key: Optional[Tuple] = None

    for rec in iter_records(log_path):
        if rec.get("event") != "adapter_expert_routing":
            continue
        if key not in rec:
            continue
        if mode != "all" and rec.get("mode") != mode:
            continue

        layer = rec.get("layer")
        if layer is None:
            continue
        try:
            layer = int(layer)
        except (TypeError, ValueError):
            continue
        if layers is not None and layer not in layers:
            continue

        if dedup_consecutive:
            line_key = (layer, rec.get("mode"), rec.get("num_tokens"), rec.get(key))
            if line_key == prev_line_key:
                continue
            prev_line_key = line_key

        rec["_layer"] = layer
        rec["_counts"] = rec.get(key, {})
        records.append(rec)

    if max_layers is not None and max_layers > 0:
        keep_layers = sorted({r["_layer"] for r in records})[:max_layers]
        keep_layers_set = set(keep_layers)
        records = [r for r in records if r["_layer"] in keep_layers_set]

    return records


def find_matrix_shape(records: List[Dict]) -> Tuple[int, int, int]:
    max_adapter = -1
    max_expert = -1
    max_layer = -1

    for rec in records:
        max_layer = max(max_layer, rec["_layer"])
        counts = rec.get("_counts", {})
        for adapter_str, expert_map in counts.items():
            try:
                adapter_idx = int(adapter_str)
            except (TypeError, ValueError):
                continue
            max_adapter = max(max_adapter, adapter_idx)
            if isinstance(expert_map, dict):
                for expert_str in expert_map.keys():
                    try:
                        expert_idx = int(expert_str)
                    except (TypeError, ValueError):
                        continue
                    max_expert = max(max_expert, expert_idx)

    return max_layer + 1, max_adapter + 1, max_expert + 1


def build_matrices(records: List[Dict], num_layers: int, num_adapters: int, num_experts: int) -> Dict[str, np.ndarray]:
    adapter_expert = np.zeros((num_adapters, num_experts), dtype=np.int64)
    layer_adapter_total = np.zeros((num_layers, num_adapters), dtype=np.int64)
    layer_adapter_seen = np.zeros((num_layers, num_adapters), dtype=np.int64)
    layer_adapter_top_share_sum = np.zeros((num_layers, num_adapters), dtype=np.float64)
    layer_adapter_active_sum = np.zeros((num_layers, num_adapters), dtype=np.float64)
    layer_adapter_entropy_sum = np.zeros((num_layers, num_adapters), dtype=np.float64)
    dominant_votes: Dict[Tuple[int, int], Dict[int, int]] = {}

    for rec in records:
        layer = rec["_layer"]
        counts = rec.get("_counts", {})
        for adapter_str, expert_map in counts.items():
            try:
                adapter_idx = int(adapter_str)
            except (TypeError, ValueError):
                continue
            if not (0 <= adapter_idx < num_adapters):
                continue
            if not isinstance(expert_map, dict):
                continue

            layer_sum = 0
            positive_pairs: List[Tuple[int, int]] = []
            for expert_str, count in expert_map.items():
                try:
                    expert_idx = int(expert_str)
                    count_val = int(count)
                except (TypeError, ValueError):
                    continue
                if count_val <= 0 or not (0 <= expert_idx < num_experts):
                    continue
                adapter_expert[adapter_idx, expert_idx] += count_val
                layer_sum += count_val
                positive_pairs.append((expert_idx, count_val))

            if layer_sum <= 0:
                continue

            layer_adapter_total[layer, adapter_idx] += layer_sum
            layer_adapter_seen[layer, adapter_idx] += 1

            top_expert_id, top_expert_count = max(positive_pairs, key=lambda x: x[1])
            top_share = float(top_expert_count / layer_sum)
            active_expert_count = float(len(positive_pairs))

            probs = np.array([count for _, count in positive_pairs], dtype=np.float64) / float(layer_sum)
            entropy = float(-(probs * np.log(probs)).sum())

            layer_adapter_top_share_sum[layer, adapter_idx] += top_share
            layer_adapter_active_sum[layer, adapter_idx] += active_expert_count
            layer_adapter_entropy_sum[layer, adapter_idx] += entropy
            key = (layer, adapter_idx)
            if key not in dominant_votes:
                dominant_votes[key] = {}
            dominant_votes[key][top_expert_id] = dominant_votes[key].get(top_expert_id, 0) + 1

    with np.errstate(divide="ignore", invalid="ignore"):
        layer_adapter_top_share_avg = np.divide(
            layer_adapter_top_share_sum,
            layer_adapter_seen,
            out=np.zeros_like(layer_adapter_top_share_sum),
            where=layer_adapter_seen > 0,
        )
        layer_adapter_active_avg = np.divide(
            layer_adapter_active_sum,
            layer_adapter_seen,
            out=np.zeros_like(layer_adapter_active_sum),
            where=layer_adapter_seen > 0,
        )
        layer_adapter_entropy_avg = np.divide(
            layer_adapter_entropy_sum,
            layer_adapter_seen,
            out=np.zeros_like(layer_adapter_entropy_sum),
            where=layer_adapter_seen > 0,
        )

    layer_adapter_dominant_expert = np.full((num_layers, num_adapters), -1.0, dtype=np.float64)
    for (layer, adapter_idx), vote_map in dominant_votes.items():
        # Use vote mode; break ties by smaller expert id for determinism.
        dominant_id = sorted(vote_map.items(), key=lambda x: (-x[1], x[0]))[0][0]
        layer_adapter_dominant_expert[layer, adapter_idx] = float(dominant_id)

    return {
        "adapter_expert": adapter_expert,
        "layer_adapter_total": layer_adapter_total,
        "layer_adapter_seen": layer_adapter_seen,
        "layer_adapter_top_share": layer_adapter_top_share_avg,
        "layer_adapter_active": layer_adapter_active_avg,
        "layer_adapter_entropy": layer_adapter_entropy_avg,
        "layer_adapter_dominant_expert": layer_adapter_dominant_expert,
    }


def plot_heatmap(
    matrix: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
    log_color: bool,
    x_values: Optional[np.ndarray] = None,
    y_values: Optional[np.ndarray] = None,
    y_value_label_map: Optional[Dict[int, str]] = None,
    x_tick_step: Optional[int] = None,
    y_tick_step: Optional[int] = None,
) -> None:
    if matrix.size == 0:
        return

    data = np.log1p(matrix.astype(np.float64)) if log_color else matrix.astype(np.float64)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if x_tick_step is None:
        x_tick_step = max(1, math.ceil(matrix.shape[1] / 16))
    if y_tick_step is None:
        y_tick_step = max(1, math.ceil(matrix.shape[0] / 16))

    if x_values is None:
        x_values = np.arange(matrix.shape[1])
    if y_values is None:
        y_values = np.arange(matrix.shape[0])
    if y_value_label_map is None:
        y_value_label_map = {}

    x_ticks = np.arange(0, matrix.shape[1], x_tick_step)
    y_ticks = np.arange(0, matrix.shape[0], y_tick_step)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(x_values[i])) for i in x_ticks])
    ax.set_yticks(y_ticks)
    y_labels = []
    for i in y_ticks:
        raw_idx = int(y_values[i])
        if raw_idx in y_value_label_map and y_value_label_map[raw_idx]:
            y_labels.append(f"{raw_idx}:{y_value_label_map[raw_idx]}")
        else:
            y_labels.append(str(raw_idx))
    ax.set_yticklabels(y_labels)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log1p(count)" if log_color else "count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def compute_skew_summary(layer_adapter: np.ndarray, adapter_ids: Optional[np.ndarray] = None) -> Dict:
    if adapter_ids is None:
        adapter_ids = np.arange(layer_adapter.shape[1], dtype=np.int64)
    if len(adapter_ids) != layer_adapter.shape[1]:
        raise ValueError(
            f"adapter_ids length mismatch: got {len(adapter_ids)}, "
            f"expect {layer_adapter.shape[1]}"
        )

    # Per-layer dominance: max adapter share in each layer
    row_sum = layer_adapter.sum(axis=1)
    max_share = []
    for i in range(layer_adapter.shape[0]):
        if row_sum[i] <= 0:
            max_share.append(0.0)
            continue
        max_share.append(float(layer_adapter[i].max() / row_sum[i]))

    overall = layer_adapter.sum(axis=0)
    total = int(overall.sum())
    overall_share = (overall / total).tolist() if total > 0 else [0.0 for _ in range(layer_adapter.shape[1])]

    top_order = np.argsort(-overall)
    top_adapters = [
        {
            "adapter_idx": int(adapter_ids[idx]),
            "count": int(overall[idx]),
            "share": float(overall_share[idx]),
        }
        for idx in top_order[: min(10, len(top_order))]
        if int(overall[idx]) > 0
    ]

    return {
        "num_layers": int(layer_adapter.shape[0]),
        "num_adapters": int(layer_adapter.shape[1]),
        "max_share_per_layer": max_share,
        "mean_max_share": float(np.mean(max_share)) if max_share else 0.0,
        "top_adapters": top_adapters,
    }


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    adapter_label_map = parse_label_map(args.adapter_labels)

    layer_filter = parse_int_set(args.layers)
    records = load_filtered_records(
        log_path=args.log,
        metric=args.metric,
        mode=args.mode,
        layers=layer_filter,
        dedup_consecutive=args.dedup_consecutive,
        max_layers=args.max_layers,
    )

    if not records:
        print("No matching records found. Check --log / --mode / --layers.")
        return 1

    num_layers, num_adapters, num_experts = find_matrix_shape(records)
    if min(num_layers, num_adapters, num_experts) <= 0:
        print("No valid adapter/expert/layer indices in records.")
        return 1

    matrices = build_matrices(records, num_layers, num_adapters, num_experts)
    adapter_expert = matrices["adapter_expert"]
    layer_adapter_total = matrices["layer_adapter_total"]
    layer_adapter_seen = matrices["layer_adapter_seen"]
    layer_adapter_stat_map = {
        "total_count": layer_adapter_total.astype(np.float64),
        "top_expert_share": matrices["layer_adapter_top_share"],
        "active_expert_count": matrices["layer_adapter_active"],
        "entropy": matrices["layer_adapter_entropy"],
        "dominant_expert_id": matrices["layer_adapter_dominant_expert"],
    }
    selected_layer_adapter = layer_adapter_stat_map[args.layer_adapter_stat]

    # Trim empty rows/cols for cleaner plots
    used_layers = np.where(layer_adapter_seen.sum(axis=1) > 0)[0]
    used_adapters = np.where(adapter_expert.sum(axis=1) > 0)[0]
    used_experts = np.where(adapter_expert.sum(axis=0) > 0)[0]

    if len(used_layers) == 0 or len(used_adapters) == 0 or len(used_experts) == 0:
        print("All counts are zero after filtering.")
        return 1

    layer_adapter_plot = selected_layer_adapter[np.ix_(used_layers, used_adapters)]
    layer_adapter_total_plot = layer_adapter_total[np.ix_(used_layers, used_adapters)]
    adapter_expert_plot = adapter_expert[np.ix_(used_adapters, used_experts)]

    suffix = f"{args.metric}_{args.mode}_{args.layer_adapter_stat}"
    if layer_filter:
        suffix += "_layers" + "-".join(str(x) for x in sorted(layer_filter))

    ae_path = os.path.join(args.output_dir, f"adapter_expert_heatmap_{suffix}.png")
    la_path = os.path.join(args.output_dir, f"layer_adapter_heatmap_{suffix}.png")
    summary_path = os.path.join(args.output_dir, f"routing_skew_summary_{suffix}.json")

    plot_heatmap(
        adapter_expert_plot,
        title=f"Adapter x Expert Routing Heatmap ({args.metric}, mode={args.mode})",
        xlabel="Expert Index",
        ylabel="Adapter Index",
        out_path=ae_path,
        log_color=args.log_color,
        x_values=used_experts,
        y_values=used_adapters,
        y_value_label_map=adapter_label_map,
    )

    plot_heatmap(
        layer_adapter_plot,
        title=f"Layer x Adapter Heatmap ({args.layer_adapter_stat}, {args.metric}, mode={args.mode})",
        xlabel="Adapter Index",
        ylabel="Layer Index",
        out_path=la_path,
        log_color=(args.log_color and args.layer_adapter_stat == "total_count"),
        x_values=used_adapters,
        y_values=used_layers,
    )

    summary = compute_skew_summary(layer_adapter_total_plot, adapter_ids=used_adapters)
    summary["used_layers"] = [int(x) for x in used_layers.tolist()]
    summary["used_adapters"] = [int(x) for x in used_adapters.tolist()]
    summary["used_experts"] = [int(x) for x in used_experts.tolist()]
    summary["num_records"] = len(records)
    summary["layer_adapter_stat"] = args.layer_adapter_stat
    summary["top_adapters_metric"] = "total_count"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Records used: {len(records)}")
    print(f"Heatmap saved: {ae_path}")
    print(f"Heatmap saved: {la_path}")
    print(f"Summary saved: {summary_path}")
    print(f"Mean per-layer max adapter share: {summary['mean_max_share']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
