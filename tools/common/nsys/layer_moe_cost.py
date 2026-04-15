from __future__ import annotations

import argparse
import re
from pathlib import Path
import subprocess

import pandas as pd

# NVTX names like MoE_CoalescedAct_LoRA_GPU/L0/E106/Gate — family is prefix before /Lx/Ey/Role
_MOE_DGU_EVENT_RE = re.compile(r"^(.+)/L(\d+)/E\d+/(Down|Gate|Up)$")

# Defaults when invoked with no CLI args (same workflow as before argv support).
DEFAULT_CSV_FILE = (
    "artifacts/evaluation/live_e2e/nsys_lockdown_colora_conditions_valid_only_01/"
    "paper_runs/colora_load_then_run_sync/nvtx_pushpop_trace.csv"
)
DEFAULT_LAST_DURATION_S = 27.0
DEFAULT_DECODE_NAME = "decode"

# ---------- column names ----------
name_col = "Name"
tid_col = "TID"
start_col = "Start (ms)"
dur_col = "Duration (ms)"
range_id_col = "RangeId"
parent_id_col = "ParentId"
end_col = "End (ms)"


def get_descendants(df, root_range_id, parent_id_col="ParentId", range_id_col="RangeId"):
    descendants = []
    frontier = {root_range_id}
    while frontier:
        children = df[df[parent_id_col].isin(frontier)].copy()
        if children.empty:
            break
        descendants.append(children)
        frontier = set(children[range_id_col].tolist())
    if descendants:
        return pd.concat(descendants, ignore_index=True)
    return df.iloc[0:0].copy()


def parse_moe_dgu_event(event: str) -> tuple[str, str, int] | None:
    """If event matches MoE_* / L<layer>/ E<expert> / {Down,Gate,Up}, return (family, role, layer_from_path)."""
    if not isinstance(event, str):
        return None
    m = _MOE_DGU_EVENT_RE.match(event)
    if not m:
        return None
    return m.group(1), m.group(3), int(m.group(2))


def parse_layer_idx(layer_name):
    if not isinstance(layer_name, str):
        return pd.NA
    parts = layer_name.split(" ")
    if len(parts) != 2:
        return pd.NA
    try:
        return int(parts[1])
    except ValueError:
        return pd.NA


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate MoE_Expert_* and coarse Qwen3VL/MoE NVTX ranges for decodes "
            "whose end time falls in the last --last-duration-s seconds (anchored to last decode)."
        )
    )
    p.add_argument(
        "--csv-file",
        default=DEFAULT_CSV_FILE,
        help="Path to nvtx_pushpop_trace.csv (generated via nsys stats).",
    )
    p.add_argument(
        "--last-duration-s",
        type=float,
        default=DEFAULT_LAST_DURATION_S,
        metavar="SEC",
        help="Include decodes with end time >= (last_decode_end - SEC).",
    )
    p.add_argument(
        "--decode-name",
        default=DEFAULT_DECODE_NAME,
        help='Top-level NVTX range name to treat as a decode step (default: "decode").',
    )
    p.add_argument(
        "--target-tid",
        type=int,
        default=None,
        help="If set, only consider decode ranges on this host thread TID.",
    )
    return p.parse_args(argv)


def run(
    csv_file: str,
    last_duration_s: float,
    target_decode_name: str,
    target_tid: int | None,
) -> None:
    if last_duration_s <= 0:
        raise ValueError("--last-duration-s must be positive")

    csv_path = Path(csv_file)
    if not csv_path.exists():
        print("CSV file does not exist, generating...")
        report_candidates = sorted(csv_path.parent.glob("*.nsys-rep"))
        if not report_candidates:
            raise RuntimeError(f"No .nsys-rep file found in {csv_path.parent}")
        nsys_report = report_candidates[0]
        result = subprocess.run(
            [
                "nsys",
                "stats",
                "--report",
                "nvtx_pushpop_trace",
                "--format",
                "csv",
                "--timeunit",
                "ms",
                str(nsys_report),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = result.stdout.splitlines()
        if len(lines) <= 2:
            raise RuntimeError("Generated CSV output is unexpectedly short")
        csv_path.write_text("\n".join(lines[2:]) + "\n")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df[end_col] = df[start_col] + df[dur_col]

    decode_rows = df[df[name_col] == target_decode_name].copy()
    if target_tid is not None:
        decode_rows = decode_rows[decode_rows[tid_col] == target_tid].copy()

    if decode_rows.empty:
        tid_msg = f" and TID={target_tid}" if target_tid is not None else ""
        raise RuntimeError(f"No decode candidates found for name={target_decode_name}{tid_msg}")

    last_decode_end_ms = decode_rows[end_col].max()
    window_start_ms = last_decode_end_ms - (last_duration_s * 1000.0)
    selected_decodes = decode_rows[decode_rows[end_col] >= window_start_ms].copy()
    selected_decodes = selected_decodes.sort_values(start_col)

    if selected_decodes.empty:
        raise RuntimeError(
            f"No decode events found in the last {last_duration_s} seconds "
            f"(last_decode_end_ms={last_decode_end_ms:.3f}, window_start_ms={window_start_ms:.3f})"
        )

    print(
        f"Selected {len(selected_decodes)} decode events in last {last_duration_s}s "
        f"(window_start_ms={window_start_ms:.3f}, last_decode_end_ms={last_decode_end_ms:.3f})."
    )

    decode_desc_parts = []
    for _, decode_row in selected_decodes.iterrows():
        decode_range_id = decode_row[range_id_col]
        decode_desc = get_descendants(df, decode_range_id, parent_id_col, range_id_col)
        if not decode_desc.empty:
            decode_desc["decode_range_id"] = decode_range_id
            decode_desc["decode_start_ms"] = decode_row[start_col]
            decode_desc_parts.append(decode_desc)

    if not decode_desc_parts:
        raise RuntimeError("Selected decodes contain no descendant events")

    decode_desc_all = pd.concat(decode_desc_parts, ignore_index=True)
    print(f"Total descendant events in selected decodes: {len(decode_desc_all)}")

    layer_rows = decode_desc_all[decode_desc_all[name_col].str.match(r"^Layer \d+$", na=False)].copy()
    layer_map = dict(zip(layer_rows[range_id_col], layer_rows[name_col]))
    parent_lookup = dict(zip(decode_desc_all[range_id_col], decode_desc_all[parent_id_col]))

    def find_parent_layer(range_id):
        cur = range_id
        while cur in parent_lookup:
            parent = parent_lookup[cur]
            if pd.isna(parent):
                return None
            if parent in layer_map:
                return layer_map[parent]
            cur = parent
        return None

    decode_desc_all["layer"] = decode_desc_all[range_id_col].apply(find_parent_layer)
    decode_desc_all["layer_idx"] = decode_desc_all["layer"].apply(parse_layer_idx)

    expert_mask = decode_desc_all[name_col].str.match(r"^MoE_Expert_\d+$", na=False)
    experts = decode_desc_all[expert_mask].copy()

    if experts.empty:
        raise RuntimeError("No expert events found in selected decode window")

    experts["expert_id"] = experts[name_col].str.extract(r"(\d+)$").astype(int)

    overall_avg_ms = experts[dur_col].mean()
    overall_sum_ms = experts[dur_col].sum()
    overall_count = len(experts)

    print("\n=== Overall expert statistics in selected decode window ===")
    print(f"Expert invocation count: {overall_count}")
    print(f"Total expert time (ms): {overall_sum_ms:.6f}")
    print(f"Average expert time per invocation (ms): {overall_avg_ms:.6f}")

    by_expert = (
        experts.groupby("expert_id", as_index=False)
        .agg(
            count=(dur_col, "size"),
            total_ms=(dur_col, "sum"),
            avg_ms=(dur_col, "mean"),
            min_ms=(dur_col, "min"),
            max_ms=(dur_col, "max"),
        )
        .sort_values(["total_ms", "expert_id"], ascending=[False, True])
    )

    by_layer = (
        experts.groupby("layer", dropna=False, as_index=False)
        .agg(
            count=(dur_col, "size"),
            total_ms=(dur_col, "sum"),
            avg_ms=(dur_col, "mean"),
        )
        .sort_values("total_ms", ascending=False)
    )

    detail = experts[
        [
            name_col,
            "expert_id",
            "layer",
            "layer_idx",
            "decode_range_id",
            "decode_start_ms",
            start_col,
            dur_col,
            range_id_col,
            parent_id_col,
        ]
    ].sort_values(start_col)

    coarse_mask = (
        decode_desc_all[name_col].isin(["Qwen3VL_QKV", "Qwen3VL_O_Proj"])
        | decode_desc_all[name_col].str.startswith("MoE_", na=False)
    )
    coarse_events = decode_desc_all[coarse_mask].copy()

    if coarse_events.empty:
        raise RuntimeError("No QKV/O_Proj/MoE events found in selected decode window")

    coarse_by_layer_event = (
        coarse_events.groupby(["layer", "layer_idx", name_col], dropna=False, as_index=False)
        .agg(
            count=(dur_col, "size"),
            total_ms=(dur_col, "sum"),
            avg_ms=(dur_col, "mean"),
            min_ms=(dur_col, "min"),
            max_ms=(dur_col, "max"),
        )
        .rename(columns={name_col: "event"})
        .sort_values(["layer_idx", "layer", "event"], na_position="last")
    )

    coarse_wide_total_ms = (
        coarse_by_layer_event.pivot_table(
            index=["layer", "layer_idx"], columns="event", values="total_ms", fill_value=0.0
        )
        .reset_index()
        .sort_values(["layer_idx", "layer"], na_position="last")
    )

    dgu_parsed = coarse_by_layer_event["event"].apply(parse_moe_dgu_event)
    dgu_mask = dgu_parsed.notna()
    if dgu_mask.any():
        n_mismatch = 0
        for _, r in coarse_by_layer_event.loc[dgu_mask].iterrows():
            p = parse_moe_dgu_event(r["event"])
            if p and pd.notna(r["layer_idx"]) and int(r["layer_idx"]) != p[2]:
                n_mismatch += 1
        if n_mismatch:
            print(
                f"Warning: {n_mismatch} D/G/U event rows have L* in name != layer_idx; "
                "using row layer_idx for grouping."
            )

    def _build_dgu_frame(group_cols: list[str]) -> pd.DataFrame:
        sub = coarse_by_layer_event.loc[dgu_mask].copy()
        sub["family"] = dgu_parsed.loc[dgu_mask].apply(lambda t: t[0])
        sub["role"] = dgu_parsed.loc[dgu_mask].apply(lambda t: t[1])
        out = (
            sub.groupby(group_cols, dropna=False, as_index=False)
            .agg(
                count=("count", "sum"),
                total_ms=("total_ms", "sum"),
                min_ms=("min_ms", "min"),
                max_ms=("max_ms", "max"),
            )
            .assign(avg_ms=lambda df: df["total_ms"] / df["count"])
        )
        value_cols = ["count", "total_ms", "avg_ms", "min_ms", "max_ms"]
        out = out[[*group_cols, *value_cols]]
        sort_cols = ["total_ms"] + [c for c in group_cols if c != "total_ms"]
        ascending = [False] + [True] * (len(sort_cols) - 1)
        return out.sort_values(sort_cols, ascending=ascending)

    _dgu_val_cols = ["count", "total_ms", "avg_ms", "min_ms", "max_ms"]
    dgu_by_layer_family = (
        _build_dgu_frame(["layer", "layer_idx", "family", "role"])
        if dgu_mask.any()
        else pd.DataFrame(columns=["layer", "layer_idx", "family", "role", *_dgu_val_cols])
    )
    dgu_global_by_family = (
        _build_dgu_frame(["family", "role"])
        if dgu_mask.any()
        else pd.DataFrame(columns=["family", "role", *_dgu_val_cols])
    )
    # Sums Down/Gate/Up across NVTX families (e.g. GatherInput + LoRA_GPU) for one layer — combined stage time.
    dgu_by_layer_role_only = (
        _build_dgu_frame(["layer", "layer_idx", "role"])
        if dgu_mask.any()
        else pd.DataFrame(columns=["layer", "layer_idx", "role", *_dgu_val_cols])
    )

    base_dir = csv_path.parent
    by_expert.to_csv(base_dir / "expert_avg_by_id.csv", index=False)
    by_layer.to_csv(base_dir / "expert_avg_by_layer.csv", index=False)
    detail.to_csv(base_dir / "expert_detail_rows.csv", index=False)
    coarse_by_layer_event.to_csv(base_dir / "layer_coarse_summary.csv", index=False)
    coarse_wide_total_ms.to_csv(base_dir / "layer_coarse_summary_wide_total_ms.csv", index=False)
    dgu_by_layer_family.to_csv(base_dir / "layer_moe_dgu_by_layer_family.csv", index=False)
    dgu_global_by_family.to_csv(base_dir / "layer_moe_dgu_global_by_family.csv", index=False)
    dgu_by_layer_role_only.to_csv(base_dir / "layer_moe_dgu_by_layer_role_only.csv", index=False)

    print("\nSaved:")
    print(f"  {base_dir / 'expert_avg_by_id.csv'}")
    print(f"  {base_dir / 'expert_avg_by_layer.csv'}")
    print(f"  {base_dir / 'expert_detail_rows.csv'}")
    print(f"  {base_dir / 'layer_coarse_summary.csv'}")
    print(f"  {base_dir / 'layer_coarse_summary_wide_total_ms.csv'}")
    print(f"  {base_dir / 'layer_moe_dgu_by_layer_family.csv'}")
    print(f"  {base_dir / 'layer_moe_dgu_global_by_family.csv'}")
    print(f"  {base_dir / 'layer_moe_dgu_by_layer_role_only.csv'}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(
        csv_file=args.csv_file,
        last_duration_s=args.last_duration_s,
        target_decode_name=args.decode_name,
        target_tid=args.target_tid,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
