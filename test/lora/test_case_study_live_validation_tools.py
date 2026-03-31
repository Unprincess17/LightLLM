import importlib.util
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def _load_module(rel_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


live_mod = _load_module("tools/case_study/live_validate_small.py", "case_study_live_validate_small")


def test_select_auto_reuse_window_prefers_multi_adapter_reuse():
    request_rows = [
        {"req_idx": idx, "prompt_len_tokens": prompt_len, "target_decode_len": 32}
        for idx, prompt_len in enumerate([20, 20, 30, 30, 40, 40])
    ]
    adapter_rows = [
        {"arrival_idx": idx, "adapter_id": adapter_id}
        for idx, adapter_id in enumerate(["lora_0", "lora_0", "lora_1", "lora_2", "lora_2", "lora_1"])
    ]

    selected = live_mod.select_auto_reuse_window(
        request_rows=request_rows,
        adapter_rows=adapter_rows,
        request_count=4,
        search_limit=6,
        max_decode_tokens=16,
    )

    assert selected["start_index"] == 2
    assert selected["repeated_adapter_count"] == 2
    assert selected["immediate_reuse_pairs"] == 1


def test_build_schedule_rows_maps_offline_adapter_ids_to_live_dummy_names():
    alias_map = live_mod.build_live_adapter_alias_map(
        [
            "/tmp/lora_dummy_0",
            "/tmp/lora_dummy_1",
            "/tmp/lora_dummy_2",
        ]
    )
    schedule_rows = live_mod.build_schedule_rows(
        request_rows=[
            {
                "req_idx": 4,
                "prompt_len_tokens": 128,
                "target_decode_len": 48,
                "messages": [{"role": "user", "content": "hello"}],
                "source_id": "x",
                "split_tag": None,
                "render_mode": "chat_template",
            }
        ],
        adapter_rows=[{"arrival_idx": 4, "adapter_id": "lora_2"}],
        selected_indices=[0],
        alias_map=alias_map,
        requests_path=Path("/tmp/fixed_requests.jsonl"),
        adapter_trace_path=Path("/tmp/adapter_trace.jsonl"),
        adapter_cardinality=8,
        selection_strategy="first_n",
        selection_metadata={"start_index": 0},
        max_decode_tokens=16,
    )

    assert len(schedule_rows) == 1
    assert schedule_rows[0]["adapter_id"] == "lora_2"
    assert schedule_rows[0]["live_adapter_name"] == "lora_dummy_2"
    assert schedule_rows[0]["live_adapter_id"] == "3"
    assert schedule_rows[0]["max_tokens_live"] == 16


def test_parse_and_summarize_colora_log_lines():
    log_text = "\n".join(
        [
            "DEBUG 03-13 12:00:01 [x.py:1] [COLoRA] layer=3 hit_tokens=8 miss_tokens=24 queue_depth=2 hit_rate=0.25 "
            "cache_capacity_slots=96 cache_resident_slots=92 cache_free_slots=4 cache_evictions_total=3 "
            "cpu_compute_time=0.010 gpu_compute_time=0.020 cpu_queue_wait=0.001 d2h_bytes=1024 h2d_bytes=2048 "
            "weight_h2d_bytes=4096 weight_h2d_time=0.003 blocking_promotion_count=1 miss_policy=no_cpu_path overlap_mode=no_overlap "
            "fallback_degrade_count=1 cpu_queue_depth=3 promotion_drop_total=4 promotion_drop_queue=2 "
            "promotion_drop_cooldown=1 promotion_admitted=2 promotion_reject_delta=3 promotion_reject_no_ema=4 "
            "tracker_queue_drop=5 prefetch_submitted=6 prefetch_ready_hits=7 prefetch_not_ready=8 "
            "prefetch_stale=9 prefetch_false_positives=10 prefetch_slot_overwrite=11 "
            "moe_kernel_calls=6 moe_kernel_tokens=24",
            "DEBUG 03-13 12:00:02 [x.py:1] [COLoRA] layer=3 hit_tokens=16 miss_tokens=8 queue_depth=5 hit_rate=0.66 "
            "cache_capacity_slots=96 cache_resident_slots=96 cache_free_slots=0 cache_evictions_total=5 "
            "cpu_compute_time=0.015 gpu_compute_time=0.025 cpu_queue_wait=0.002 d2h_bytes=512 h2d_bytes=1024 "
            "weight_h2d_bytes=2048 weight_h2d_time=0.001 blocking_promotion_count=2 miss_policy=no_deferred_sync overlap_mode=full "
            "fallback_degrade_count=0 cpu_queue_depth=4 promotion_drop_total=5 promotion_drop_queue=2 "
            "promotion_drop_cooldown=1 promotion_admitted=1 promotion_reject_delta=0 promotion_reject_no_ema=1 "
            "tracker_queue_drop=0 prefetch_submitted=2 prefetch_ready_hits=3 prefetch_not_ready=4 "
            "prefetch_stale=5 prefetch_false_positives=6 prefetch_slot_overwrite=12 "
            "moe_kernel_calls=4 moe_kernel_tokens=16",
        ]
    )

    rows = live_mod.parse_colora_log_lines(log_text)
    summary = live_mod.summarize_colora_rows(rows)

    assert len(rows) == 2
    assert summary["colora_hit_tokens"] == 24
    assert summary["colora_miss_tokens"] == 32
    assert summary["observed_hit_rate"] == 24 / 56
    assert summary["cache_capacity_slots_max"] == 96
    assert summary["cache_resident_slots_max"] == 96
    assert summary["cache_free_slots_min"] == 0
    assert summary["cache_evictions_total_end"] == 5
    assert summary["promotion_queue_depth_max"] == 5
    assert summary["cpu_queue_depth_max"] == 4
    assert summary["promotion_drop_total_end"] == 5
    assert summary["promotion_drop_queue_high_watermark_end"] == 2
    assert summary["promotion_drop_cooldown_end"] == 1
    assert summary["promotion_admitted_sum"] == 3
    assert summary["promotion_reject_delta_sum"] == 3
    assert summary["promotion_reject_no_ema_sum"] == 5
    assert summary["tracker_queue_drop_sum"] == 5
    assert summary["prefetch_submitted_sum"] == 8
    assert summary["prefetch_ready_hits_sum"] == 10
    assert summary["prefetch_not_ready_sum"] == 12
    assert summary["prefetch_stale_sum"] == 14
    assert summary["prefetch_false_positives_sum"] == 16
    assert summary["prefetch_slot_overwrite_end"] == 12
    assert summary["fallback_degrade_count_sum"] == 1
    assert summary["weight_h2d_bytes_sum"] == 6144
    assert abs(summary["weight_h2d_time_sum"] - 0.004) < 1e-9
    assert summary["blocking_promotion_count_sum"] == 3
    assert summary["observed_overlap_modes"] == ["full", "no_overlap"]
    assert summary["observed_miss_policies"] == ["no_cpu_path", "no_deferred_sync"]
    assert summary["moe_kernel_calls_sum"] == 10
    assert summary["moe_kernel_tokens_sum"] == 40


def test_summarize_per_window_counters_preserves_extended_colora_fields():
    per_window_rows = [
        {
            "colora_line_count": 1,
            "colora_hit_tokens": 8,
            "colora_miss_tokens": 24,
            "cache_hit_rate_max": 0.25,
            "cache_capacity_slots_max": 96,
            "cache_resident_slots_max": 92,
            "cache_free_slots_min": 4,
            "cache_evictions_total_end": 3,
            "promotion_queue_depth_max": 2,
            "cpu_queue_depth_max": 3,
            "promotion_drop_total_end": 4,
            "promotion_drop_queue_high_watermark_end": 2,
            "promotion_drop_cooldown_end": 1,
            "promotion_admitted_sum": 2,
            "promotion_reject_delta_sum": 3,
            "promotion_reject_no_ema_sum": 4,
            "tracker_queue_drop_sum": 5,
            "prefetch_submitted_sum": 6,
            "prefetch_ready_hits_sum": 7,
            "prefetch_not_ready_sum": 8,
            "prefetch_stale_sum": 9,
            "prefetch_false_positives_sum": 10,
            "prefetch_slot_overwrite_end": 11,
            "fallback_degrade_count_sum": 1,
            "cpu_compute_time_sum": 0.01,
            "gpu_compute_time_sum": 0.02,
            "cpu_queue_wait_time_sum": 0.001,
            "d2h_bytes_sum": 1024.0,
            "h2d_bytes_sum": 2048.0,
            "weight_h2d_bytes_sum": 0.0,
            "weight_h2d_time_sum": 0.0,
            "blocking_promotion_count_sum": 0,
            "observed_overlap_modes": ["none"],
            "observed_miss_policies": ["cpu_first"],
            "moe_kernel_calls_sum": 6,
            "moe_kernel_tokens_sum": 24,
        },
        {
            "colora_line_count": 1,
            "colora_hit_tokens": 16,
            "colora_miss_tokens": 8,
            "cache_hit_rate_max": 0.66,
            "cache_capacity_slots_max": 96,
            "cache_resident_slots_max": 96,
            "cache_free_slots_min": 0,
            "cache_evictions_total_end": 5,
            "promotion_queue_depth_max": 5,
            "cpu_queue_depth_max": 4,
            "promotion_drop_total_end": 5,
            "promotion_drop_queue_high_watermark_end": 2,
            "promotion_drop_cooldown_end": 1,
            "promotion_admitted_sum": 1,
            "promotion_reject_delta_sum": 0,
            "promotion_reject_no_ema_sum": 1,
            "tracker_queue_drop_sum": 0,
            "prefetch_submitted_sum": 2,
            "prefetch_ready_hits_sum": 3,
            "prefetch_not_ready_sum": 4,
            "prefetch_stale_sum": 5,
            "prefetch_false_positives_sum": 6,
            "prefetch_slot_overwrite_end": 12,
            "fallback_degrade_count_sum": 0,
            "cpu_compute_time_sum": 0.015,
            "gpu_compute_time_sum": 0.025,
            "cpu_queue_wait_time_sum": 0.002,
            "d2h_bytes_sum": 512.0,
            "h2d_bytes_sum": 1024.0,
            "weight_h2d_bytes_sum": 0.0,
            "weight_h2d_time_sum": 0.0,
            "blocking_promotion_count_sum": 0,
            "observed_overlap_modes": ["async"],
            "observed_miss_policies": ["cpu_first"],
            "moe_kernel_calls_sum": 4,
            "moe_kernel_tokens_sum": 16,
        },
    ]

    summary = live_mod.summarize_per_window_counters(per_window_rows)

    assert summary["colora_hit_tokens"] == 24
    assert summary["colora_miss_tokens"] == 32
    assert summary["cache_capacity_slots_max"] == 96
    assert summary["cache_resident_slots_max"] == 96
    assert summary["cache_free_slots_min"] == 0
    assert summary["cache_evictions_total_end"] == 5
    assert summary["promotion_admitted_sum"] == 3
    assert summary["promotion_reject_delta_sum"] == 3
    assert summary["promotion_reject_no_ema_sum"] == 5
    assert summary["tracker_queue_drop_sum"] == 5
    assert summary["prefetch_submitted_sum"] == 8
    assert summary["prefetch_ready_hits_sum"] == 10
    assert summary["prefetch_not_ready_sum"] == 12
    assert summary["prefetch_stale_sum"] == 14
    assert summary["prefetch_false_positives_sum"] == 16
    assert summary["prefetch_slot_overwrite_end"] == 12


def test_summarize_reuse_agreement_marks_later_hits_as_agree():
    schedule_rows = [
        {"submit_order": 0, "req_idx": 10, "adapter_id": "lora_6"},
        {"submit_order": 1, "req_idx": 11, "adapter_id": "lora_6"},
        {"submit_order": 2, "req_idx": 12, "adapter_id": "lora_5"},
    ]
    latency_rows = [
        {"submit_order": 0, "latency_ms": 100.0, "status": "ok"},
        {"submit_order": 1, "latency_ms": 90.0, "status": "ok"},
        {"submit_order": 2, "latency_ms": 95.0, "status": "ok"},
    ]
    per_window_counter_rows = [
        {"submit_order": 0, "observed_hit_rate": 0.0, "colora_hit_tokens": 0, "colora_miss_tokens": 32},
        {"submit_order": 1, "observed_hit_rate": 0.5, "colora_hit_tokens": 16, "colora_miss_tokens": 16},
        {"submit_order": 2, "observed_hit_rate": 0.0, "colora_hit_tokens": 0, "colora_miss_tokens": 32},
    ]

    summary = live_mod.summarize_reuse_agreement(
        schedule_rows=schedule_rows,
        latency_rows=latency_rows,
        per_window_counter_rows=per_window_counter_rows,
    )

    assert summary["verdict"] == "agree"
    assert summary["comparable_adapter_count"] == 1
    assert summary["improved_adapter_count"] == 1
    assert summary["comparisons"][0]["adapter_id"] == "lora_6"


def test_parse_args_enables_server_stdout_by_default(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["live_validate_small.py", "--run_id", "router_lora_case_v1"],
    )

    args = live_mod.parse_args()
    assert args.tee_server_output is True

    monkeypatch.setattr(
        "sys.argv",
        ["live_validate_small.py", "--run_id", "router_lora_case_v1", "--no_tee_server_output"],
    )
    args = live_mod.parse_args()
    assert args.tee_server_output is False
