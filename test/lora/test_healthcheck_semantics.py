import importlib


health_utils = importlib.import_module("lightllm.server.health_utils")


class _DummySharedInt:
    def __init__(self, value: int):
        self._value = value

    def get_value(self):
        return self._value


class _DummyArgs:
    def __init__(self, run_mode="normal", model_name="dummy-model"):
        self.run_mode = run_mode
        self.model_name = model_name


class _DummyManager:
    def __init__(self):
        self.tokenizer = object()
        self.send_to_router = object()
        self.latest_success_infer_time_mark = _DummySharedInt(123)


def test_lightweight_health_status_fails_before_init():
    status_code, payload = health_utils.build_lightweight_health_status(None, None)
    assert status_code == 503
    assert payload["reason"] == "args_not_initialized"


def test_lightweight_health_status_succeeds_when_manager_ready():
    status_code, payload = health_utils.build_lightweight_health_status(
        _DummyArgs(run_mode="normal", model_name="qwen-test"),
        _DummyManager(),
    )
    assert status_code == 200
    assert payload["check"] == "lightweight"
    assert payload["model_name"] == "qwen-test"
    assert payload["router_socket_ready"] is True
    assert payload["latest_success_infer_time_mark"] == 123
