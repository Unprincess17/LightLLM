import argparse
import importlib.util
import json
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]

_API_SPEC = importlib.util.spec_from_file_location(
    "moe_lora_api_test_mod",
    _ROOT / "test/lora/test_moe_lora_api.py",
)
api_mod = importlib.util.module_from_spec(_API_SPEC)
assert _API_SPEC is not None and _API_SPEC.loader is not None
_API_SPEC.loader.exec_module(api_mod)

_REPLAY_SPEC = importlib.util.spec_from_file_location(
    "zipf_replay_mod",
    _ROOT / "test/lora/avx/zipf_routing_replay.py",
)
replay_mod = importlib.util.module_from_spec(_REPLAY_SPEC)
assert _REPLAY_SPEC is not None and _REPLAY_SPEC.loader is not None
_REPLAY_SPEC.loader.exec_module(replay_mod)


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_load_explicit_adapter_trace_orders_by_arrival_idx(tmp_path: Path):
    trace_path = tmp_path / "adapter_trace.jsonl"
    _write_jsonl(
        trace_path,
        [
            {"arrival_idx": 3, "req_idx": 11, "adapter_id": "lora_b"},
            {"arrival_idx": 1, "req_idx": 10, "adapter_id": "base"},
            {"arrival_idx": 2, "req_idx": 12, "adapter_id": "lora_a"},
        ],
    )

    adapter_ids = api_mod.load_explicit_adapter_trace(str(trace_path))
    assert adapter_ids == [None, "lora_a", "lora_b"]


def test_build_request_prompts_uses_namespace_prefix():
    prompts = api_mod.build_request_prompts("Describe the image", 3, "Warmup-Req")
    assert prompts == [
        "[Warmup-Req-0] Describe the image",
        "[Warmup-Req-1] Describe the image",
        "[Warmup-Req-2] Describe the image",
    ]


def test_join_router_and_adapter_traces_expands_projections():
    joined = replay_mod.join_router_and_adapter_traces(
        router_events=[
            {
                "arrival_idx": 0,
                "req_idx": 7,
                "phase": "decode",
                "layer_id": 0,
                "token_pos": 3,
                "topk_experts": [4, 5],
                "topk_weights": [0.8, 0.2],
            }
        ],
        adapter_assignment={7: 9},
        projections=["up", "down"],
    )

    assert len(joined) == 4
    assert [row["projection"] for row in joined] == ["up", "up", "down", "down"]
    assert [row["expert_id"] for row in joined] == [4, 5, 4, 5]
    assert all(row["adapter_id"] == 9 for row in joined)


def test_build_trace_driven_summary_shows_joint_fragmentation(tmp_path: Path):
    router_path = tmp_path / "router_trace.jsonl"
    adapter_path = tmp_path / "adapter_trace.jsonl"

    _write_jsonl(
        router_path,
        [
            {
                "event": "router_trace",
                "arrival_idx": 0,
                "req_idx": 0,
                "phase": "decode",
                "layer_id": 0,
                "token_pos": 0,
                "topk_experts": [1],
                "topk_weights": [1.0],
            },
            {
                "event": "router_trace",
                "arrival_idx": 1,
                "req_idx": 1,
                "phase": "decode",
                "layer_id": 0,
                "token_pos": 0,
                "topk_experts": [1],
                "topk_weights": [1.0],
            },
            {
                "event": "router_trace",
                "arrival_idx": 2,
                "req_idx": 0,
                "phase": "decode",
                "layer_id": 0,
                "token_pos": 1,
                "topk_experts": [1],
                "topk_weights": [1.0],
            },
            {
                "event": "router_trace",
                "arrival_idx": 3,
                "req_idx": 1,
                "phase": "decode",
                "layer_id": 0,
                "token_pos": 1,
                "topk_experts": [1],
                "topk_weights": [1.0],
            },
        ],
    )
    _write_jsonl(
        adapter_path,
        [
            {"arrival_idx": 0, "req_idx": 0, "adapter_id": 0},
            {"arrival_idx": 1, "req_idx": 1, "adapter_id": 1},
        ],
    )

    args = argparse.Namespace(
        num_adapters=2,
        num_steps=0,
        batch_size=0,
        zipf_s=1.2,
        cache_budget_mb=0,
        promote_min_hits=0,
        promote_window=0,
        max_promote_per_step=0,
        decay=0.0,
        router_trace_path=str(router_path),
        adapter_trace_path=str(adapter_path),
        output_adapter_trace_path=None,
        output_joined_trace_path=None,
        output_summary_path=None,
        skew="zipf",
        correlation="independent",
        session_mean=1.0,
        burst_probability=0.0,
        burst_factor=4.0,
        class_locality=0.8,
        num_classes=8,
        projections="up,down",
        cache_capacity=2,
        cache_capacity_fraction=0.0,
        hit_cost=1.0,
        miss_cost_prefill=8.0,
        miss_cost_decode=20.0,
        seed=7,
    )

    summary = replay_mod.build_trace_driven_summary(args)

    assert summary["expert_only"]["distribution"]["working_set_size"] == 2
    assert summary["joint"]["distribution"]["working_set_size"] == 4
    assert summary["joint"]["cache"]["miss_rate"] > summary["expert_only"]["cache"]["miss_rate"]
