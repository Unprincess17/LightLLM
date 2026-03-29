from lightllm.server.api_cli import make_argument_parser


def test_colora_cli_defaults():
    parser = make_argument_parser()
    args = parser.parse_args([])
    assert args.colora_deferred_promotion_delta_steps == 4
    assert args.colora_promotion_ema_alpha == 0.5
    assert args.colora_miss_policy == "cpu_first"
    assert args.colora_async_fallback == 1
    assert args.colora_cpu_workers == 4
    assert args.colora_cpu_queue_depth == 256
    assert args.colora_cpu_batch_timeout_us == 50
    assert args.colora_temporal_prefetch is False
    assert args.colora_temporal_prefetch_layer_whitelist == ""
    assert args.colora_temporal_hot_cache_slots == 64
    assert args.colora_speculative_dispatch is False
    assert args.colora_spec_layer_whitelist == ""
    # COLoRA request-level skip defaults
    assert args.colora_request_skip == 1
    assert args.colora_max_continuations == 8


def test_colora_cli_overrides():
    parser = make_argument_parser()
    args = parser.parse_args(
        [
            "--colora_miss_policy",
            "load_then_run",
            "--colora_async_fallback",
            "0",
            "--colora_cpu_workers",
            "8",
            "--colora_cpu_queue_depth",
            "64",
            "--colora_cpu_batch_timeout_us",
            "120",
            "--colora_deferred_promotion_delta_steps",
            "7",
            "--colora_promotion_ema_alpha",
            "0.3",
            "--colora_temporal_prefetch",
            "--colora_temporal_prefetch_layer_whitelist",
            "1,5",
            "--colora_temporal_hot_cache_slots",
            "16",
            "--colora_speculative_dispatch",
            "--colora_spec_layer_whitelist",
            "3,7,11",
            "--colora_request_skip",
            "0",
            "--colora_max_continuations",
            "16",
        ]
    )
    assert args.colora_deferred_promotion_delta_steps == 7
    assert args.colora_promotion_ema_alpha == 0.3
    assert args.colora_miss_policy == "load_then_run"
    assert args.colora_async_fallback == 0
    assert args.colora_cpu_workers == 8
    assert args.colora_cpu_queue_depth == 64
    assert args.colora_cpu_batch_timeout_us == 120
    assert args.colora_temporal_prefetch is True
    assert args.colora_temporal_prefetch_layer_whitelist == "1,5"
    assert args.colora_temporal_hot_cache_slots == 16
    assert args.colora_speculative_dispatch is True
    assert args.colora_spec_layer_whitelist == "3,7,11"
    assert args.colora_request_skip == 0
    assert args.colora_max_continuations == 16
