import importlib.util
import json
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def _load_module(rel_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


prompt_mod = _load_module("tools/case_study/prepare_prompt_corpus.py", "case_study_prepare_prompt")
azure_mod = _load_module("tools/case_study/preprocess_azure_trace.py", "case_study_preprocess_azure")
mapping_mod = _load_module("tools/case_study/map_adapter_trace.py", "case_study_map_adapter")
collect_mod = _load_module("tools/case_study/collect_router_trace.py", "case_study_collect_router")


def test_extract_sharegpt_example_keeps_context_and_target():
    record = {
        "id": "sharegpt_1",
        "conversations": [
            {"from": "human", "value": "Question 1"},
            {"from": "gpt", "value": "Answer 1"},
            {"from": "human", "value": "Question 2"},
            {"from": "gpt", "value": "Answer 2"},
        ],
    }

    extracted = prompt_mod.extract_sharegpt_example(record, max_history_turns=6)
    assert extracted is not None

    messages, completion_text, source_id, split_tag = extracted
    assert source_id == "sharegpt_1"
    assert split_tag is None
    assert completion_text == "Answer 2"
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]


def test_preprocess_trace_file_derives_sessions_and_outputs_jsonl(tmp_path: Path):
    trace_path = tmp_path / "azure_trace.csv"
    trace_path.write_text(
        "\n".join(
            [
                "app,func,end_timestamp,duration",
                "app_a,func_1,10.000,1.000",
                "app_a,func_1,10.300,0.100",
                "app_a,func_2,30.000,0.050",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    raw_output_path = tmp_path / "adapter_trace_raw.jsonl"

    outputs = azure_mod.preprocess_trace_file(
        trace_path=trace_path,
        raw_output_path=raw_output_path,
        session_gap_ms=5000,
        max_rows=None,
        top_tenants_to_report=10,
    )

    rows = [json.loads(line) for line in raw_output_path.read_text(encoding="utf-8").splitlines()]
    assert outputs["summary"]["row_count"] == 3
    assert [row["session_id"] for row in rows] == ["app_a:1", "app_a:1", "app_a:2"]
    assert rows[0]["duration_ms"] == 1000
    assert rows[1]["start_ts"] == 10200


def test_adapter_mapping_is_deterministic_and_bounded():
    app_counts = mapping_mod.Counter({"app_a": 100, "app_b": 60, "app_c": 30, "app_d": 10})
    mapping = mapping_mod.build_primary_app_mapping(app_counts, cardinality=3, seed=11)

    assert mapping["app_a"] == 0
    assert mapping["app_b"] == 1
    assert mapping["app_c"] == 2
    assert 0 <= mapping["app_d"] < 3

    corr_slot = mapping_mod.compute_correlated_adapter_slot(
        indep_slot=mapping["app_a"],
        cardinality=8,
        func_id="func_x",
        session_id="app_a:1",
        seed=11,
    )
    assert 0 <= corr_slot < 8


def test_canonicalize_router_trace_adds_event_indices(tmp_path: Path):
    raw_trace_path = tmp_path / "router_trace_raw.jsonl"
    raw_trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "router_trace",
                        "arrival_idx": 0,
                        "req_idx": 4,
                        "phase": "prefill",
                        "layer_id": 1,
                        "token_pos": 0,
                        "topk_experts": [1, 2],
                        "topk_weights": [0.8, 0.2],
                    }
                ),
                json.dumps(
                    {
                        "event": "router_trace",
                        "arrival_idx": 1,
                        "req_idx": 4,
                        "phase": "decode",
                        "layer_id": 1,
                        "token_pos": 1,
                        "topk_experts": [2, 3],
                        "topk_weights": [0.7, 0.3],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    canonical_rows = collect_mod.canonicalize_router_trace(
        raw_trace_path=raw_trace_path,
        model_name="test_model",
        trace_run_id="run_x",
    )

    assert [row["event_idx"] for row in canonical_rows] == [0, 1]
    assert canonical_rows[0]["num_selected_experts"] == 2
    assert canonical_rows[0]["trace_run_id"] == "run_x"


def test_canonicalize_router_trace_recovers_concatenated_lines_and_deduplicates_tp_rows(tmp_path: Path):
    raw_trace_path = tmp_path / "router_trace_raw.jsonl"
    first_event = {
        "event": "router_trace",
        "arrival_idx": 0,
        "req_idx": 4,
        "phase": "prefill",
        "layer_id": 1,
        "token_pos": 0,
        "topk_experts": [1, 2],
        "topk_weights": [0.8, 0.2],
    }
    second_event = {
        "event": "router_trace",
        "arrival_idx": 1,
        "req_idx": 4,
        "phase": "decode",
        "layer_id": 1,
        "token_pos": 1,
        "topk_experts": [2, 3],
        "topk_weights": [0.7, 0.3],
    }
    raw_trace_path.write_text(
        json.dumps(first_event) + json.dumps(first_event) + "\n" + json.dumps(second_event) + "\n",
        encoding="utf-8",
    )

    canonical_rows = collect_mod.canonicalize_router_trace(
        raw_trace_path=raw_trace_path,
        model_name="test_model",
        trace_run_id="run_x",
    )

    assert len(canonical_rows) == 2
    assert [row["arrival_idx"] for row in canonical_rows] == [0, 1]
    assert [row["raw_arrival_idx"] for row in canonical_rows] == [0, 1]
    assert [row["event_idx"] for row in canonical_rows] == [0, 1]


def test_canonicalize_router_trace_remaps_server_request_ids(tmp_path: Path):
    raw_trace_path = tmp_path / "router_trace_raw.jsonl"
    raw_trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "router_trace",
                        "arrival_idx": 0,
                        "req_idx": 8,
                        "phase": "prefill",
                        "layer_id": 0,
                        "token_pos": 0,
                        "topk_experts": [1],
                        "topk_weights": [1.0],
                    }
                ),
                json.dumps(
                    {
                        "event": "router_trace",
                        "arrival_idx": 1,
                        "req_idx": 16,
                        "phase": "prefill",
                        "layer_id": 0,
                        "token_pos": 0,
                        "topk_experts": [2],
                        "topk_weights": [1.0],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    request_id_mapping = collect_mod.build_request_id_mapping(
        [
            {"req_idx": 0, "response_id": "8"},
            {"req_idx": 1, "response_id": "16"},
        ]
    )
    canonical_rows = collect_mod.canonicalize_router_trace(
        raw_trace_path=raw_trace_path,
        model_name="test_model",
        trace_run_id="run_x",
        request_id_mapping=request_id_mapping,
    )

    assert [row["req_idx"] for row in canonical_rows] == [0, 1]
    assert [row["server_request_id"] for row in canonical_rows] == ["8", "16"]


def test_build_server_command_uses_bash_and_disables_lora_for_b5():
    command = collect_mod.build_server_command(
        launcher_path=Path("/tmp/start_server.sh"),
        model_dir="/tmp/model",
        port=8040,
        router_trace_phases="prefill,decode",
        raw_router_trace_path=Path("/tmp/router_trace_raw.jsonl"),
        tp=2,
        with_lora=False,
    )

    assert command[:2] == ["bash", "/tmp/start_server.sh"]
    assert "--no_lora" in command


def test_read_log_tail_returns_last_lines(tmp_path: Path):
    log_path = tmp_path / "router_server.log"
    log_path.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")

    assert collect_mod.read_log_tail(log_path, 2) == ["line 2", "line 3"]
