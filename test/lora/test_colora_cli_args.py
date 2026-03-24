from lightllm.server.api_cli import make_argument_parser


def test_colora_cli_defaults():
    parser = make_argument_parser()
    args = parser.parse_args([])
    assert args.colora_miss_policy == "cpu_first"
    assert args.colora_async_fallback == 1
    assert args.colora_cpu_workers == 4
    assert args.colora_cpu_queue_depth == 256
    assert args.colora_cpu_batch_timeout_us == 50
    assert args.colora_speculative_dispatch is False
    assert args.colora_spec_layer_whitelist == ""


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
            "--colora_speculative_dispatch",
            "--colora_spec_layer_whitelist",
            "3,7,11",
        ]
    )
    assert args.colora_miss_policy == "load_then_run"
    assert args.colora_async_fallback == 0
    assert args.colora_cpu_workers == 8
    assert args.colora_cpu_queue_depth == 64
    assert args.colora_cpu_batch_timeout_us == 120
    assert args.colora_speculative_dispatch is True
    assert args.colora_spec_layer_whitelist == "3,7,11"
