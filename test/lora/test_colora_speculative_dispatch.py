import importlib.util
import sys
from concurrent.futures import CancelledError, Future
from pathlib import Path


_DISPATCH_PATH = Path(__file__).resolve().parents[2] / "lightllm/models/qwen3_vl_moe/lora_dispatch.py"
_DISPATCH_SPEC = importlib.util.spec_from_file_location("colora_dispatch_mod", _DISPATCH_PATH)
dispatch_mod = importlib.util.module_from_spec(_DISPATCH_SPEC)
assert _DISPATCH_SPEC is not None and _DISPATCH_SPEC.loader is not None
sys.modules[_DISPATCH_SPEC.name] = dispatch_mod
_DISPATCH_SPEC.loader.exec_module(dispatch_mod)


class _ManualFuture:
    def __init__(self, allow_cancel: bool = False):
        self._allow_cancel = bool(allow_cancel)
        self._done = False
        self._cancelled = False
        self._result = None
        self._exception = None
        self._callbacks = []

    def add_done_callback(self, callback):
        if self._done:
            callback(self)
        else:
            self._callbacks.append(callback)

    def done(self):
        return self._done

    def cancel(self):
        if self._done or not self._allow_cancel:
            return False
        self._cancelled = True
        self._done = True
        callbacks = list(self._callbacks)
        self._callbacks.clear()
        for callback in callbacks:
            callback(self)
        return True

    def result(self, timeout=None):
        if not self._done:
            raise TimeoutError("future is not ready")
        if self._cancelled:
            raise CancelledError()
        if self._exception is not None:
            raise self._exception
        return self._result

    def set_result(self, result):
        if self._done:
            raise RuntimeError("future already completed")
        self._result = result
        self._done = True
        callbacks = list(self._callbacks)
        self._callbacks.clear()
        for callback in callbacks:
            callback(self)

    def set_exception(self, exc):
        if self._done:
            raise RuntimeError("future already completed")
        self._exception = exc
        self._done = True
        callbacks = list(self._callbacks)
        self._callbacks.clear()
        for callback in callbacks:
            callback(self)


def _make_dispatcher(queue_depth: int = 2):
    return dispatch_mod.Qwen3VLMoELoRADispatcher(
        num_layers=1,
        gate_lora_rank=1,
        colora_cpu_queue_depth=queue_depth,
        colora_cpu_batch_timeout_us=0,
    )


def _make_key(step_id: int, row_group_sig=((0, 1001), (1, 1002))):
    return dispatch_mod.SpecJobKey(
        layer_id=7,
        decode_step_id=step_id,
        op_kind="gate_up",
        adapter_bin=3,
        expert_id=11,
        row_group_sig=row_group_sig,
    )


def test_try_admit_spec_job_rejects_when_reserve_would_be_consumed():
    dispatcher = _make_dispatcher(queue_depth=1)
    dispatcher.begin_spec_step(1)

    submit_count = 0

    def _submit():
        nonlocal submit_count
        submit_count += 1
        return Future()

    handle = dispatcher.try_admit_spec_job(_make_key(1), _submit)

    assert handle is None
    assert submit_count == 0
    assert dispatcher._get_cpu_queue_depth() == 0
    assert len(dispatcher._spec_active_jobs) == 0


def test_try_admit_spec_job_leaves_capacity_for_real_fallback():
    dispatcher = _make_dispatcher(queue_depth=2)
    dispatcher.begin_spec_step(2)

    future = Future()
    future.set_result(("gate", "up"))
    handle = dispatcher.try_admit_spec_job(_make_key(2), lambda: future)

    assert handle is not None
    assert dispatcher._get_cpu_queue_depth() == 0
    assert dispatcher._reserve_async_queue_slot(reserve_slots=0) is True
    dispatcher._release_async_queue_slot()

    step_stats = dispatcher.end_spec_step(2)
    assert step_stats["retired_active"] == 1
    assert step_stats["reaped_retired"] == 1
    assert dispatcher._get_cpu_queue_depth() == 0


def test_try_bind_gate_up_job_is_opportunistic_and_retires_not_ready_handle():
    dispatcher = _make_dispatcher(queue_depth=2)
    dispatcher.begin_spec_step(3)

    future = _ManualFuture(allow_cancel=False)
    key = _make_key(3)
    handle = dispatcher.try_admit_spec_job(key, lambda: future)

    assert handle is not None
    assert dispatcher._get_cpu_queue_depth() == 1

    result = dispatcher.try_bind_gate_up_job(key)

    assert result is None
    assert key not in dispatcher._spec_active_jobs
    assert len(dispatcher._spec_retired_jobs) == 1
    retired = dispatcher._spec_retired_jobs[0]
    assert retired.retire_reason == "bind_not_ready"
    assert retired.stale is True
    assert dispatcher._get_cpu_queue_depth() == 1

    reap_stats = dispatcher.reap_retired()
    assert reap_stats["reaped_retired"] == 0
    assert reap_stats["pending_retired"] == 1

    future.set_result(("ready_gate", "ready_up"))
    assert dispatcher._get_cpu_queue_depth() == 0

    reap_stats = dispatcher.reap_retired()
    assert reap_stats["reaped_retired"] == 1
    assert reap_stats["pending_retired"] == 0


def test_try_bind_gate_up_job_requires_exact_key_match():
    dispatcher = _make_dispatcher(queue_depth=2)
    dispatcher.begin_spec_step(4)

    future = Future()
    future.set_result(("gate_buf", "up_buf"))
    key = _make_key(4)
    mismatch_key = _make_key(4, row_group_sig=((0, 1001),))
    dispatcher.try_admit_spec_job(key, lambda: future)

    assert dispatcher.try_bind_gate_up_job(mismatch_key) is None
    assert key in dispatcher._spec_active_jobs

    bound = dispatcher.try_bind_gate_up_job(key)
    assert bound == ("gate_buf", "up_buf")
    assert key not in dispatcher._spec_active_jobs
    assert len(dispatcher._spec_retired_jobs) == 0


def test_begin_and_end_spec_step_retire_and_reap_stale_handles():
    dispatcher = _make_dispatcher(queue_depth=2)
    dispatcher.begin_spec_step(5)

    future_step5 = _ManualFuture(allow_cancel=False)
    dispatcher.try_admit_spec_job(_make_key(5), lambda: future_step5)

    begin_stats = dispatcher.begin_spec_step(6)
    assert begin_stats["retired_active"] == 1
    assert begin_stats["pending_retired"] == 1
    assert len(dispatcher._spec_active_jobs) == 0

    future_step5.set_result(("late_gate", "late_up"))
    reap_stats = dispatcher.reap_retired()
    assert reap_stats["reaped_retired"] == 1
    assert reap_stats["pending_retired"] == 0

    future_step6 = Future()
    future_step6.set_result(("unused_gate", "unused_up"))
    dispatcher.try_admit_spec_job(_make_key(6), lambda: future_step6)

    end_stats = dispatcher.end_spec_step(6)
    assert end_stats["retired_active"] == 1
    assert end_stats["reaped_retired"] == 1
    assert end_stats["pending_retired"] == 0
    assert len(dispatcher._spec_active_jobs) == 0
