import importlib.util
import os
import sys

import torch
from concurrent.futures import CancelledError, Future
from pathlib import Path


_DISPATCH_PATH = Path(__file__).resolve().parents[2] / "lightllm/models/qwen3_vl_moe/lora_dispatch.py"
_DISPATCH_SPEC = importlib.util.spec_from_file_location("colora_dispatch_mod", _DISPATCH_PATH)
dispatch_mod = importlib.util.module_from_spec(_DISPATCH_SPEC)
assert _DISPATCH_SPEC is not None and _DISPATCH_SPEC.loader is not None
sys.modules[_DISPATCH_SPEC.name] = dispatch_mod
_DISPATCH_SPEC.loader.exec_module(dispatch_mod)

_LAYER_INFER_PATH = Path(__file__).resolve().parents[2] / "lightllm/models/qwen3_vl_moe/layer_infer/transformer_layer_infer.py"
_LAYER_INFER_SPEC = importlib.util.spec_from_file_location("colora_layer_infer_mod", _LAYER_INFER_PATH)
layer_infer_mod = importlib.util.module_from_spec(_LAYER_INFER_SPEC)
assert _LAYER_INFER_SPEC is not None and _LAYER_INFER_SPEC.loader is not None
sys.modules[_LAYER_INFER_SPEC.name] = layer_infer_mod
_LAYER_INFER_SPEC.loader.exec_module(layer_infer_mod)


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


def test_try_admit_spec_job_rejects_when_no_active_step():
    dispatcher = _make_dispatcher(queue_depth=2)

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
    assert dispatcher.try_bind_gate_up_job(_make_key(1)) is None


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


def test_try_admit_spec_job_rejects_after_end_step_and_bind_rejects():
    dispatcher = _make_dispatcher(queue_depth=2)
    dispatcher.begin_spec_step(3)

    future = Future()
    future.set_result(("gate", "up"))
    key = _make_key(3)
    handle = dispatcher.try_admit_spec_job(key, lambda: future)
    assert handle is not None

    dispatcher.end_spec_step(3)

    submit_count = 0

    def _submit():
        nonlocal submit_count
        submit_count += 1
        return Future()

    assert dispatcher.try_admit_spec_job(key, _submit) is None
    assert submit_count == 0
    assert dispatcher.try_bind_gate_up_job(key) is None
    assert dispatcher._spec_current_step_id is None


def test_try_bind_gate_up_job_is_opportunistic_and_retires_not_ready_handle():
    dispatcher = _make_dispatcher(queue_depth=2)
    dispatcher.begin_spec_step(4)

    future = _ManualFuture(allow_cancel=False)
    key = _make_key(4)
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
    dispatcher.begin_spec_step(5)

    future = Future()
    future.set_result(("gate_buf", "up_buf"))
    key = _make_key(5)
    mismatch_key = _make_key(5, row_group_sig=((0, 1001),))
    dispatcher.try_admit_spec_job(key, lambda: future)

    assert dispatcher.try_bind_gate_up_job(mismatch_key) is None
    assert key in dispatcher._spec_active_jobs

    bound = dispatcher.try_bind_gate_up_job(key)
    assert bound == ("gate_buf", "up_buf")
    assert key not in dispatcher._spec_active_jobs
    assert len(dispatcher._spec_retired_jobs) == 0


def test_cleanup_speculation_state_keeps_pending_retired_handles_tracked_until_done():
    dispatcher = _make_dispatcher(queue_depth=4)
    dispatcher.begin_spec_step(6)

    ready_future = Future()
    ready_future.set_result(("done_gate", "done_up"))
    slow_future = _ManualFuture(allow_cancel=False)

    key_done = _make_key(6, row_group_sig=((0, 1001),))
    key_slow = _make_key(6, row_group_sig=((1, 1002),))

    assert dispatcher.try_admit_spec_job(key_done, lambda: ready_future) is not None
    assert dispatcher.try_admit_spec_job(key_slow, lambda: slow_future) is not None
    assert dispatcher._get_cpu_queue_depth() == 1

    stats = dispatcher.cleanup_speculation_state()

    assert stats["retired_active"] == 2
    assert stats["reaped_retired"] == 1
    assert stats["pending_retired"] == 1
    assert stats["cleared_retired"] == 0
    assert stats["cleared_reserved"] == 0
    assert dispatcher._spec_current_step_id is None
    assert len(dispatcher._spec_active_jobs) == 0
    assert len(dispatcher._spec_active_job_keys_by_step) == 0
    assert len(dispatcher._spec_retired_jobs) == 1
    assert len(dispatcher._spec_reserved_job_keys) == 0
    assert dispatcher._get_cpu_queue_depth() == 1

    slow_future.set_result(("late_gate", "late_up"))
    assert dispatcher._get_cpu_queue_depth() == 0

    reap_stats = dispatcher.reap_retired()
    assert reap_stats["reaped_retired"] == 1
    assert reap_stats["pending_retired"] == 0
    assert len(dispatcher._spec_retired_jobs) == 0


class _ImmediateExecutor:
    def submit(self, fn):
        future = Future()
        try:
            future.set_result(fn())
        except Exception as exc:  # pragma: no cover - test helper
            future.set_exception(exc)
        return future


class _FakePool:
    def __init__(self, num_experts: int = 16):
        self.num_experts = num_experts
        self.max_rank = 1
        self.a_start = torch.tensor([0, 8, 16, 24, 32, 40], dtype=torch.int32)
        self.a_len = torch.tensor([8, 8, 8, 8, 8, 8], dtype=torch.int32)
        self.a_scaling = torch.ones(6, dtype=torch.float32)
        self.key_buffer = torch.zeros((64, 1, 4), dtype=torch.bfloat16)
        self.value_buffer = torch.zeros((64, 1, 4), dtype=torch.bfloat16)


class _FakeMemPool:
    def __init__(self):
        self.moe_gate_pool = _FakePool()
        self.moe_up_pool = _FakePool()


class _FakeCacheManager:
    def __init__(self, ready_keys=()):
        self._ready_keys = set(ready_keys)

    def peek_ready_slots(self, keys):
        return {key: idx for idx, key in enumerate(keys) if key in self._ready_keys}


def _prime_fused_submit_dispatcher(dispatcher, ready_keys=()):
    dispatcher.begin_spec_step(7)
    dispatcher.lora_mem_pool = _FakeMemPool()
    dispatcher.expert_cache_manager = _FakeCacheManager(ready_keys)
    dispatcher._cpu_executor = _ImmediateExecutor()
    dispatcher._should_use_hybrid_moe_compute = lambda: True
    return dispatcher


def test_maybe_submit_fused_gate_up_spec_job_submits_for_cold_joint():
    dispatcher = _prime_fused_submit_dispatcher(_make_dispatcher(queue_depth=4))

    calls = []

    def _fake_strict(input_tensor, layer_id, pool, req_bins, projection, adapter_group_plan=None, return_to_original_device=True):
        calls.append(
            {
                "projection": projection,
                "layer_id": int(layer_id),
                "req_bins": tuple(int(v) for v in req_bins.view(-1).tolist()),
                "return_to_original_device": bool(return_to_original_device),
                "shape": tuple(int(v) for v in input_tensor.shape),
            }
        )
        return torch.zeros((input_tensor.shape[0], 4), dtype=torch.bfloat16), 2, input_tensor.shape[0]

    dispatcher._strict_moe_cpu_batch_lora = _fake_strict

    key = _make_key(7, row_group_sig=((0, 1001), (2, 1002)))
    outcome = dispatcher.maybe_submit_fused_gate_up_spec_job(
        key=key,
        input_tensor=torch.randn(3, 4),
        req_bins=torch.tensor([3, 1, 3], dtype=torch.int32),
        row_indices=torch.tensor([0, 2], dtype=torch.long),
    )

    assert outcome.status == "submitted"
    assert outcome.reason == "admitted"
    assert outcome.handle is not None
    assert dispatcher._get_cpu_queue_depth() == 0
    assert [call["projection"] for call in calls] == ["gate", "up"]
    assert all(call["return_to_original_device"] is False for call in calls)
    assert all(call["req_bins"] == (3, 3) for call in calls)


def test_maybe_submit_fused_gate_up_spec_job_skips_when_joint_ready():
    gate_key = dispatch_mod.ExpertCacheKey(projection="gate", adapter_idx=3, layer_id=7, expert_id=11)
    up_key = dispatch_mod.ExpertCacheKey(projection="up", adapter_idx=3, layer_id=7, expert_id=11)
    dispatcher = _prime_fused_submit_dispatcher(_make_dispatcher(queue_depth=4), ready_keys=(gate_key, up_key))

    outcome = dispatcher.maybe_submit_fused_gate_up_spec_job(
        key=_make_key(7),
        input_tensor=torch.randn(2, 4),
        req_bins=torch.tensor([3, 3], dtype=torch.int32),
        row_indices=torch.tensor([0, 1], dtype=torch.long),
    )

    assert outcome.status == "skipped"
    assert outcome.reason == "joint_gate_up_ready"
    assert outcome.handle is None
    assert dispatcher._get_cpu_queue_depth() == 0
    assert len(dispatcher._spec_active_jobs) == 0


def test_maybe_submit_fused_gate_up_spec_job_rejects_when_queue_reserve_unavailable():
    dispatcher = _prime_fused_submit_dispatcher(_make_dispatcher(queue_depth=1))

    outcome = dispatcher.maybe_submit_fused_gate_up_spec_job(
        key=_make_key(7),
        input_tensor=torch.randn(2, 4),
        req_bins=torch.tensor([3, 3], dtype=torch.int32),
        row_indices=torch.tensor([0, 1], dtype=torch.long),
    )

    assert outcome.status == "rejected"
    assert outcome.reason == "queue_full"
    assert outcome.handle is None
    assert dispatcher._get_cpu_queue_depth() == 0
    assert len(dispatcher._spec_active_jobs) == 0


class _FinalizeSpyDispatcher:
    def __init__(self):
        self.calls = []
        self.expert_cache_manager = object()

    def _should_use_hybrid_moe_compute(self):
        return True

    def finalize_spec_step_nonblocking(self, decode_step_id):
        self.calls.append(int(decode_step_id))
        return {"retired_active": 0, "reaped_retired": 0, "pending_retired": 0}


def _zero_pop_stats():
    return {
        "colora_hit_tokens": 0,
        "colora_miss_tokens": 0,
        "cpu_compute_time": 0.0,
        "gpu_compute_time": 0.0,
        "cpu_queue_wait_time": 0.0,
        "d2h_bytes": 0.0,
        "h2d_bytes": 0.0,
        "fallback_degrade_count": 0,
        "cpu_queue_depth": 0,
        "promotion_drop_total": 0,
        "promotion_drop_queue_high_watermark": 0,
        "promotion_drop_cooldown": 0,
        "moe_kernel_calls": 0,
        "moe_kernel_tokens": 0,
        "overlap_ratio": 0.0,
        "promotion_queue_depth": 0,
        "cache_hit_rate": 0.0,
    }


class _BindOutcomeDispatcher:
    def __init__(self, outcomes=None):
        self.expert_cache_manager = object()
        self._outcomes = dict(outcomes or {})
        self.bind_calls = []
        self.gate_fallback_calls = []
        self.up_fallback_calls = []

    def _should_use_hybrid_moe_compute(self):
        return True

    def try_bind_gate_up_job_with_status(self, key):
        self.bind_calls.append(key)
        return self._outcomes.get(
            key,
            dispatch_mod.SpecBindOutcome(status="missing", reason="bind_missing", result=None),
        )

    def batch_apply_gate_lora(self, input_tensor, layer_id, req_bins=None, expert_id=None):
        self.gate_fallback_calls.append(
            {
                "shape": tuple(int(v) for v in input_tensor.shape),
                "bins": tuple(int(v) for v in req_bins.view(-1).tolist()),
                "layer_id": int(layer_id),
                "expert_id": int(expert_id),
            }
        )
        return torch.full((input_tensor.shape[0], 2), 7.0, dtype=input_tensor.dtype, device=input_tensor.device)

    def batch_apply_up_lora(self, input_tensor, layer_id, req_bins=None, expert_id=None):
        self.up_fallback_calls.append(
            {
                "shape": tuple(int(v) for v in input_tensor.shape),
                "bins": tuple(int(v) for v in req_bins.view(-1).tolist()),
                "layer_id": int(layer_id),
                "expert_id": int(expert_id),
            }
        )
        return torch.full((input_tensor.shape[0], 2), 9.0, dtype=input_tensor.dtype, device=input_tensor.device)

    def pop_colora_stats(self):
        return _zero_pop_stats()


def _make_bind_layer(spec_layer_enabled: bool):
    layer = object.__new__(layer_infer_mod.Qwen3VLMOETransformerLayerInfer)
    layer.is_moe = True
    layer.embed_dim_ = 4
    layer._spec_submit_layer_enabled = bool(spec_layer_enabled)
    layer._temporal_prefetch_layer_enabled = False
    layer.use_detached_lora_ = False
    layer.lora_dispatcher_ = None
    layer._tpsp_ffn_tp = object()
    layer._tpsp_ffn_ep = object()
    return layer


def test_finalize_spec_step_nonblocking_retires_last_decode_step_without_blocking():
    dispatcher = _make_dispatcher(queue_depth=2)
    dispatcher.begin_spec_step(8)

    slow_future = _ManualFuture(allow_cancel=False)
    key = _make_key(8)
    assert dispatcher.try_admit_spec_job(key, lambda: slow_future) is not None
    assert dispatcher._get_cpu_queue_depth() == 1

    stats = dispatcher.finalize_spec_step_nonblocking(8)

    assert stats["retired_active"] == 1
    assert stats["reaped_retired"] == 0
    assert stats["pending_retired"] == 1
    assert dispatcher._spec_current_step_id is None
    assert len(dispatcher._spec_active_jobs) == 0
    assert len(dispatcher._spec_retired_jobs) == 1
    assert dispatcher._get_cpu_queue_depth() == 1

    slow_future.set_result(("late_gate", "late_up"))
    assert dispatcher._get_cpu_queue_depth() == 0
    reap_stats = dispatcher.reap_retired()
    assert reap_stats["reaped_retired"] == 1
    assert reap_stats["pending_retired"] == 0


def test_qwen3_vl_moe_wrapper_finalizes_spec_step_in_production_path():
    prev_mode = os.environ.get("MOE_MODE")
    os.environ["MOE_MODE"] = "TP"
    base_impl = layer_infer_mod.Qwen3MOETransformerLayerInfer._moe_ffn
    try:
        layer_infer_mod.Qwen3MOETransformerLayerInfer._moe_ffn = (
            lambda self, input, infer_state, layer_weight: input + 1
        )
        layer = object.__new__(layer_infer_mod.Qwen3VLMOETransformerLayerInfer)
        layer.embed_dim_ = 4
        layer._spec_submit_layer_enabled = True
        layer.use_detached_lora_ = True
        layer.lora_dispatcher_ = _FinalizeSpyDispatcher()
        layer._maybe_submit_decode_spec_gate_up = lambda hidden_states, infer_state: None

        infer_state = type("InferState", (), {"is_prefill": False, "decode_step_id": 11})()
        output = layer_infer_mod.Qwen3VLMOETransformerLayerInfer._moe_ffn(
            layer,
            torch.zeros(2, 4),
            infer_state,
            None,
        )

        assert torch.equal(output, torch.ones(2, 4))
        assert layer.lora_dispatcher_.calls == [11]
    finally:
        layer_infer_mod.Qwen3MOETransformerLayerInfer._moe_ffn = base_impl
        if prev_mode is None:
            os.environ.pop("MOE_MODE", None)
        else:
            os.environ["MOE_MODE"] = prev_mode


def test_qwen3_vl_moe_wrapper_retires_previous_step_when_next_step_has_missing_adapter_metadata():
    prev_mode = os.environ.get("MOE_MODE")
    os.environ["MOE_MODE"] = "TP"
    base_impl = layer_infer_mod.Qwen3MOETransformerLayerInfer._moe_ffn
    try:
        layer_infer_mod.Qwen3MOETransformerLayerInfer._moe_ffn = (
            lambda self, input, infer_state, layer_weight: input + 1
        )
        dispatcher = _make_dispatcher(queue_depth=2)
        dispatcher._should_use_hybrid_moe_compute = lambda: True
        dispatcher.expert_cache_manager = object()
        dispatcher.begin_spec_step(12)

        slow_future = _ManualFuture(allow_cancel=False)
        key = _make_key(12)
        assert dispatcher.try_admit_spec_job(key, lambda: slow_future) is not None
        assert dispatcher._get_cpu_queue_depth() == 1

        layer = object.__new__(layer_infer_mod.Qwen3VLMOETransformerLayerInfer)
        layer.embed_dim_ = 4
        layer.layer_num_ = 7
        layer._spec_submit_layer_enabled = True
        layer.use_detached_lora_ = True
        layer.lora_dispatcher_ = dispatcher
        layer._spec_bound_job_keys_current_call = set()

        infer_state = type(
            "InferState",
            (),
            {"is_prefill": False, "decode_step_id": 13, "b_req_idx": None, "b_adapter_bin": None},
        )()
        output = layer_infer_mod.Qwen3VLMOETransformerLayerInfer._moe_ffn(
            layer,
            torch.zeros(2, 4),
            infer_state,
            None,
        )

        assert torch.equal(output, torch.ones(2, 4))
        assert dispatcher._spec_current_step_id is None
        assert len(dispatcher._spec_active_jobs) == 0
        assert len(dispatcher._spec_retired_jobs) == 1
        assert dispatcher._spec_retired_jobs[0].key == key
        assert dispatcher._spec_retired_jobs[0].retire_reason == "step_advanced"
        assert dispatcher._get_cpu_queue_depth() == 1

        slow_future.set_result(("late_gate", "late_up"))
        assert dispatcher._get_cpu_queue_depth() == 0
        reap_stats = dispatcher.reap_retired()
        assert reap_stats["reaped_retired"] == 1
        assert reap_stats["pending_retired"] == 0
    finally:
        layer_infer_mod.Qwen3MOETransformerLayerInfer._moe_ffn = base_impl
        if prev_mode is None:
            os.environ.pop("MOE_MODE", None)
        else:
            os.environ["MOE_MODE"] = prev_mode



def test_qwen3_vl_moe_wrapper_finalizes_even_if_spec_submit_setup_raises():
    prev_mode = os.environ.get("MOE_MODE")
    os.environ["MOE_MODE"] = "TP"
    base_impl = layer_infer_mod.Qwen3MOETransformerLayerInfer._moe_ffn
    try:
        layer_infer_mod.Qwen3MOETransformerLayerInfer._moe_ffn = (
            lambda self, input, infer_state, layer_weight: input + 1
        )
        layer = object.__new__(layer_infer_mod.Qwen3VLMOETransformerLayerInfer)
        layer.embed_dim_ = 4
        layer._spec_submit_layer_enabled = True
        layer.use_detached_lora_ = True
        layer.lora_dispatcher_ = _FinalizeSpyDispatcher()

        def _raise_submit(hidden_states, infer_state):
            raise RuntimeError("submit boom")

        layer._maybe_submit_decode_spec_gate_up = _raise_submit

        infer_state = type("InferState", (), {"is_prefill": False, "decode_step_id": 14})()
        try:
            layer_infer_mod.Qwen3VLMOETransformerLayerInfer._moe_ffn(
                layer,
                torch.zeros(2, 4),
                infer_state,
                None,
            )
        except RuntimeError as exc:
            assert str(exc) == "submit boom"
        else:
            raise AssertionError("expected speculative submit setup failure")

        assert layer.lora_dispatcher_.calls == [14]
    finally:
        layer_infer_mod.Qwen3MOETransformerLayerInfer._moe_ffn = base_impl
        if prev_mode is None:
            os.environ.pop("MOE_MODE", None)
        else:
            os.environ["MOE_MODE"] = prev_mode


def test_qwen3_vl_moe_bind_ffn_uses_wrapper_for_moe_layers_even_without_speculation():
    prev_mode = os.environ.get("MOE_MODE")
    os.environ["MOE_MODE"] = "TP"
    try:
        layer = _make_bind_layer(spec_layer_enabled=False)
        layer_infer_mod.Qwen3VLMOETransformerLayerInfer._bind_ffn(layer)
        assert layer._ffn.func is layer_infer_mod.Qwen3VLMOETransformerLayerInfer._moe_ffn
    finally:
        if prev_mode is None:
            os.environ.pop("MOE_MODE", None)
        else:
            os.environ["MOE_MODE"] = prev_mode


def test_qwen3_vl_moe_bind_ffn_uses_wrapper_when_speculation_is_enabled():
    prev_mode = os.environ.get("MOE_MODE")
    os.environ["MOE_MODE"] = "TP"
    try:
        layer = _make_bind_layer(spec_layer_enabled=True)
        layer_infer_mod.Qwen3VLMOETransformerLayerInfer._bind_ffn(layer)
        assert layer._ffn.func is layer_infer_mod.Qwen3VLMOETransformerLayerInfer._moe_ffn
    finally:
        if prev_mode is None:
            os.environ.pop("MOE_MODE", None)
        else:
            os.environ["MOE_MODE"] = prev_mode


def test_qwen3_vl_moe_disabled_speculation_keeps_base_ffn_behavior_unchanged():
    prev_mode = os.environ.get("MOE_MODE")
    os.environ["MOE_MODE"] = "TP"
    base_impl = layer_infer_mod.Qwen3MOETransformerLayerInfer._moe_ffn
    try:
        layer_infer_mod.Qwen3MOETransformerLayerInfer._moe_ffn = (
            lambda self, input, infer_state, layer_weight: input + 5
        )
        layer = _make_bind_layer(spec_layer_enabled=False)
        layer_infer_mod.Qwen3VLMOETransformerLayerInfer._bind_ffn(layer)
        assert layer._ffn.func is layer_infer_mod.Qwen3VLMOETransformerLayerInfer._moe_ffn
        output = layer._ffn(torch.zeros(2, 4), type("InferState", (), {"is_prefill": False, "decode_step_id": 31})(), None)
        assert torch.equal(output, torch.full((2, 4), 5.0))
    finally:
        layer_infer_mod.Qwen3MOETransformerLayerInfer._moe_ffn = base_impl
        if prev_mode is None:
            os.environ.pop("MOE_MODE", None)
        else:
            os.environ["MOE_MODE"] = prev_mode


def test_qwen3_vl_moe_wrapper_bypasses_speculation_completely_on_prefill():
    prev_mode = os.environ.get("MOE_MODE")
    os.environ["MOE_MODE"] = "TP"
    base_impl = layer_infer_mod.Qwen3MOETransformerLayerInfer._moe_ffn
    try:
        layer_infer_mod.Qwen3MOETransformerLayerInfer._moe_ffn = (
            lambda self, input, infer_state, layer_weight: input + 2
        )
        layer = object.__new__(layer_infer_mod.Qwen3VLMOETransformerLayerInfer)
        layer._spec_submit_layer_enabled = True
        layer.use_detached_lora_ = True
        layer.lora_dispatcher_ = _FinalizeSpyDispatcher()
        layer._maybe_submit_decode_spec_gate_up = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("submit should be bypassed on prefill"))
        layer._eager_retire_remaining_decode_spec_jobs = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("eager retire should be bypassed on prefill"))
        layer._finalize_decode_spec_step_nonblocking = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("finalize should be bypassed on prefill"))

        infer_state = type("InferState", (), {"is_prefill": True, "decode_step_id": 32})()
        output = layer_infer_mod.Qwen3VLMOETransformerLayerInfer._moe_ffn(
            layer,
            torch.zeros(2, 4),
            infer_state,
            None,
        )

        assert torch.equal(output, torch.full((2, 4), 2.0))
    finally:
        layer_infer_mod.Qwen3MOETransformerLayerInfer._moe_ffn = base_impl
        if prev_mode is None:
            os.environ.pop("MOE_MODE", None)
        else:
            os.environ["MOE_MODE"] = prev_mode


def test_qwen3_vl_moe_wrapper_eagerly_retires_unbound_jobs_after_actual_routing():
    prev_mode = os.environ.get("MOE_MODE")
    os.environ["MOE_MODE"] = "TP"
    base_impl = layer_infer_mod.Qwen3MOETransformerLayerInfer._moe_ffn
    try:
        layer_infer_mod.Qwen3MOETransformerLayerInfer._moe_ffn = (
            lambda self, input, infer_state, layer_weight: input + 1
        )
        dispatcher = _make_dispatcher(queue_depth=2)
        dispatcher._should_use_hybrid_moe_compute = lambda: True
        dispatcher.expert_cache_manager = object()
        dispatcher.begin_spec_step(33)

        future = _ManualFuture(allow_cancel=False)
        key = dispatch_mod.SpecJobKey(
            layer_id=7,
            decode_step_id=33,
            op_kind="gate_up",
            adapter_bin=3,
            expert_id=11,
            row_group_sig=((0, 1000),),
        )
        assert dispatcher.try_admit_spec_job(key, lambda: future) is not None

        layer = object.__new__(layer_infer_mod.Qwen3VLMOETransformerLayerInfer)
        layer.embed_dim_ = 4
        layer.layer_num_ = 7
        layer._spec_submit_layer_enabled = True
        layer.use_detached_lora_ = True
        layer.lora_dispatcher_ = dispatcher
        layer._spec_bound_job_keys_current_call = set()
        layer._maybe_submit_decode_spec_gate_up = lambda hidden_states, infer_state: None

        infer_state = type("InferState", (), {"is_prefill": False, "decode_step_id": 33})()
        output = layer_infer_mod.Qwen3VLMOETransformerLayerInfer._moe_ffn(
            layer,
            torch.zeros(2, 4),
            infer_state,
            None,
        )

        assert torch.equal(output, torch.ones(2, 4))
        assert key not in dispatcher._spec_active_jobs
        assert len(dispatcher._spec_retired_jobs) == 1
        assert dispatcher._spec_retired_jobs[0].key == key
        assert dispatcher._spec_retired_jobs[0].retire_reason == "bind_unmatched_after_actual"
        assert dispatcher._spec_current_step_id is None

        future.set_result(("late_gate", "late_up"))
        reap_stats = dispatcher.reap_retired()
        assert reap_stats["reaped_retired"] == 1
        assert reap_stats["pending_retired"] == 0
    finally:
        layer_infer_mod.Qwen3MOETransformerLayerInfer._moe_ffn = base_impl
        if prev_mode is None:
            os.environ.pop("MOE_MODE", None)
        else:
            os.environ["MOE_MODE"] = prev_mode


def _make_spec_bind_layer(dispatcher):
    layer = object.__new__(layer_infer_mod.Qwen3VLMOETransformerLayerInfer)
    layer.layer_num_ = 7
    layer._spec_submit_layer_enabled = True
    layer.use_detached_lora_ = True
    layer.lora_dispatcher_ = dispatcher
    layer._spec_bound_job_keys_current_call = set()
    layer._spec_bind_totals = {
        "attempted_bind": 0,
        "successful_bind": 0,
        "stale": 0,
        "not_ready": 0,
        "fallback": 0,
    }
    return layer


def test_try_bind_gate_up_job_with_status_reports_failed_future_and_retires_handle():
    dispatcher = _make_dispatcher(queue_depth=2)
    dispatcher.begin_spec_step(9)

    future = _ManualFuture(allow_cancel=False)
    key = _make_key(9)
    assert dispatcher.try_admit_spec_job(key, lambda: future) is not None

    future.set_exception(RuntimeError("boom"))
    outcome = dispatcher.try_bind_gate_up_job_with_status(key)

    assert outcome.status == "failed"
    assert outcome.reason == "bind_failed"
    assert outcome.result is None
    assert key not in dispatcher._spec_active_jobs


def test_qwen3_vl_moe_exact_bind_consumes_ready_group_and_falls_back_remaining_group():
    prev_mode = os.environ.get("MOE_MODE")
    os.environ["MOE_MODE"] = "TP"
    try:
        expert_input = torch.zeros(3, 2, dtype=torch.float32)
        expert_req_bins = torch.tensor([3, 3, 4], dtype=torch.int32)
        batch_indices = torch.tensor([2, 0, 1], dtype=torch.long)
        infer_state = type(
            "InferState",
            (),
            {"is_prefill": False, "decode_step_id": 21, "b_req_idx": torch.tensor([1000, 1001, 1002], dtype=torch.long)},
        )()
        dispatcher = _BindOutcomeDispatcher(
            outcomes={
                dispatch_mod.SpecJobKey(
                    layer_id=7,
                    decode_step_id=21,
                    op_kind="gate_up",
                    adapter_bin=3,
                    expert_id=11,
                    row_group_sig=((0, 1000), (2, 1002)),
                ): dispatch_mod.SpecBindOutcome(
                    status="bound",
                    reason="bind_success",
                    result=(
                        torch.tensor([[10.0, 11.0], [20.0, 21.0]], dtype=torch.float32),
                        torch.tensor([[30.0, 31.0], [40.0, 41.0]], dtype=torch.float32),
                    ),
                )
            }
        )
        layer = _make_spec_bind_layer(dispatcher)
        pack_meta = layer_infer_mod.Qwen3VLMOETransformerLayerInfer._coalesce_lora_activations(
            layer,
            expert_input,
            expert_req_bins,
        )
        colora_stats = layer_infer_mod.Qwen3MOETransformerLayerInfer._new_colora_stats(layer)

        gate_lora, up_lora = layer_infer_mod.Qwen3VLMOETransformerLayerInfer._maybe_bind_fused_gate_up_exact(
            layer,
            expert_input,
            infer_state,
            type("LayerWeight", (), {"layer_num_": 7})(),
            11,
            batch_indices,
            expert_req_bins,
            pack_meta,
            colora_stats,
        )

        assert gate_lora is not None
        assert up_lora is not None
        assert torch.equal(gate_lora, torch.tensor([[20.0, 21.0], [10.0, 11.0], [7.0, 7.0]]))
        assert torch.equal(up_lora, torch.tensor([[40.0, 41.0], [30.0, 31.0], [9.0, 9.0]]))
        assert len(dispatcher.bind_calls) == 2
        assert dispatcher.gate_fallback_calls == [{"shape": (1, 2), "bins": (4,), "layer_id": 7, "expert_id": 11}]
        assert dispatcher.up_fallback_calls == [{"shape": (1, 2), "bins": (4,), "layer_id": 7, "expert_id": 11}]
        assert colora_stats["attempted_bind"] == 2
        assert colora_stats["successful_bind"] == 1
        assert colora_stats["stale"] == 0
        assert colora_stats["not_ready"] == 0
        assert colora_stats["fallback"] == 1
        assert layer._spec_bind_totals == {
            "attempted_bind": 2,
            "successful_bind": 1,
            "stale": 0,
            "not_ready": 0,
            "fallback": 1,
        }
    finally:
        if prev_mode is None:
            os.environ.pop("MOE_MODE", None)
        else:
            os.environ["MOE_MODE"] = prev_mode


def test_qwen3_vl_moe_exact_bind_returns_none_on_not_ready_group():
    prev_mode = os.environ.get("MOE_MODE")
    os.environ["MOE_MODE"] = "TP"
    try:
        expert_input = torch.zeros(2, 2, dtype=torch.float32)
        expert_req_bins = torch.tensor([3, 3], dtype=torch.int32)
        batch_indices = torch.tensor([1, 0], dtype=torch.long)
        infer_state = type(
            "InferState",
            (),
            {"is_prefill": False, "decode_step_id": 22, "b_req_idx": torch.tensor([1000, 1001], dtype=torch.long)},
        )()
        dispatcher = _BindOutcomeDispatcher(
            outcomes={
                dispatch_mod.SpecJobKey(
                    layer_id=7,
                    decode_step_id=22,
                    op_kind="gate_up",
                    adapter_bin=3,
                    expert_id=11,
                    row_group_sig=((0, 1000), (1, 1001)),
                ): dispatch_mod.SpecBindOutcome(status="not_ready", reason="bind_not_ready", result=None)
            }
        )
        layer = _make_spec_bind_layer(dispatcher)
        pack_meta = layer_infer_mod.Qwen3VLMOETransformerLayerInfer._coalesce_lora_activations(
            layer,
            expert_input,
            expert_req_bins,
        )
        colora_stats = layer_infer_mod.Qwen3MOETransformerLayerInfer._new_colora_stats(layer)

        result = layer_infer_mod.Qwen3VLMOETransformerLayerInfer._maybe_bind_fused_gate_up_exact(
            layer,
            expert_input,
            infer_state,
            type("LayerWeight", (), {"layer_num_": 7})(),
            11,
            batch_indices,
            expert_req_bins,
            pack_meta,
            colora_stats,
        )

        assert result is None
        assert dispatcher.gate_fallback_calls == []
        assert dispatcher.up_fallback_calls == []
        assert colora_stats["attempted_bind"] == 1
        assert colora_stats["successful_bind"] == 0
        assert colora_stats["stale"] == 0
        assert colora_stats["not_ready"] == 1
        assert colora_stats["fallback"] == 1
    finally:
        if prev_mode is None:
            os.environ.pop("MOE_MODE", None)
        else:
            os.environ["MOE_MODE"] = prev_mode


def test_qwen3_vl_moe_exact_bind_counts_stale_group_as_fallback():
    prev_mode = os.environ.get("MOE_MODE")
    os.environ["MOE_MODE"] = "TP"
    try:
        expert_input = torch.zeros(1, 2, dtype=torch.float32)
        expert_req_bins = torch.tensor([3], dtype=torch.int32)
        batch_indices = torch.tensor([0], dtype=torch.long)
        infer_state = type(
            "InferState",
            (),
            {"is_prefill": False, "decode_step_id": 23, "b_req_idx": torch.tensor([1000], dtype=torch.long)},
        )()
        dispatcher = _BindOutcomeDispatcher(
            outcomes={
                dispatch_mod.SpecJobKey(
                    layer_id=7,
                    decode_step_id=23,
                    op_kind="gate_up",
                    adapter_bin=3,
                    expert_id=11,
                    row_group_sig=((0, 1000),),
                ): dispatch_mod.SpecBindOutcome(status="stale", reason="bind_step_mismatch", result=None)
            }
        )
        layer = _make_spec_bind_layer(dispatcher)
        pack_meta = layer_infer_mod.Qwen3VLMOETransformerLayerInfer._coalesce_lora_activations(
            layer,
            expert_input,
            expert_req_bins,
        )
        colora_stats = layer_infer_mod.Qwen3MOETransformerLayerInfer._new_colora_stats(layer)

        result = layer_infer_mod.Qwen3VLMOETransformerLayerInfer._maybe_bind_fused_gate_up_exact(
            layer,
            expert_input,
            infer_state,
            type("LayerWeight", (), {"layer_num_": 7})(),
            11,
            batch_indices,
            expert_req_bins,
            pack_meta,
            colora_stats,
        )

        assert result is None
        assert colora_stats["attempted_bind"] == 1
        assert colora_stats["successful_bind"] == 0
        assert colora_stats["stale"] == 1
        assert colora_stats["not_ready"] == 0
        assert colora_stats["fallback"] == 1
    finally:
        if prev_mode is None:
            os.environ.pop("MOE_MODE", None)
        else:
            os.environ["MOE_MODE"] = prev_mode


class _SpecSubmitSpyDispatcher:
    def __init__(self, hybrid: bool = True, with_cache: bool = True):
        self._hybrid = bool(hybrid)
        self.expert_cache_manager = object() if with_cache else None
        self.begin_calls = []
        self.submit_calls = []

    def _should_use_hybrid_moe_compute(self):
        return self._hybrid

    def begin_spec_step(self, decode_step_id):
        self.begin_calls.append(int(decode_step_id))
        return {"retired_active": 0, "reaped_retired": 0, "pending_retired": 0}

    def maybe_submit_fused_gate_up_spec_job(self, **kwargs):
        self.submit_calls.append(kwargs)
        return dispatch_mod.SpecSubmitOutcome(status="submitted", reason="admitted", handle=object())


def _make_spec_submit_guard_layer(dispatcher, spec_layer_enabled: bool = True):
    layer = object.__new__(layer_infer_mod.Qwen3VLMOETransformerLayerInfer)
    layer.layer_num_ = 7
    layer.n_routed_experts = 16
    layer._spec_submit_layer_enabled = bool(spec_layer_enabled)
    layer._spec_submit_totals = {"submitted": 0, "skipped": 0, "rejected": 0}
    layer._spec_bind_totals = {
        "attempted_bind": 0,
        "successful_bind": 0,
        "stale": 0,
        "not_ready": 0,
        "fallback": 0,
    }
    layer._spec_bound_job_keys_current_call = set()
    layer._spec_prev_decode_step_id = None
    layer._spec_prev_top1_by_req = {}
    layer.use_detached_lora_ = True
    layer.lora_dispatcher_ = dispatcher
    return layer


def test_qwen3_vl_moe_should_enable_spec_submit_returns_false_when_feature_disabled():
    prev_mode = os.environ.get("MOE_MODE")
    os.environ["MOE_MODE"] = "TP"
    try:
        dispatcher = _SpecSubmitSpyDispatcher(hybrid=True, with_cache=True)
        layer = _make_spec_submit_guard_layer(dispatcher, spec_layer_enabled=False)
        infer_state = type("InferState", (), {"is_prefill": False, "decode_step_id": 41})()

        enabled = layer_infer_mod.Qwen3VLMOETransformerLayerInfer._should_enable_spec_submit(layer, infer_state)

        assert enabled is False
    finally:
        if prev_mode is None:
            os.environ.pop("MOE_MODE", None)
        else:
            os.environ["MOE_MODE"] = prev_mode


def test_qwen3_vl_moe_should_enable_spec_submit_is_decode_only():
    prev_mode = os.environ.get("MOE_MODE")
    os.environ["MOE_MODE"] = "TP"
    try:
        dispatcher = _SpecSubmitSpyDispatcher(hybrid=True, with_cache=True)
        layer = _make_spec_submit_guard_layer(dispatcher, spec_layer_enabled=True)

        decode_state = type("InferState", (), {"is_prefill": False, "decode_step_id": 42})()
        prefill_state = type("InferState", (), {"is_prefill": True, "decode_step_id": 42})()
        missing_step_state = type("InferState", (), {"is_prefill": False, "decode_step_id": None})()

        assert layer_infer_mod.Qwen3VLMOETransformerLayerInfer._should_enable_spec_submit(layer, decode_state) is True
        assert layer_infer_mod.Qwen3VLMOETransformerLayerInfer._should_enable_spec_submit(layer, prefill_state) is False
        assert layer_infer_mod.Qwen3VLMOETransformerLayerInfer._should_enable_spec_submit(layer, missing_step_state) is False
    finally:
        if prev_mode is None:
            os.environ.pop("MOE_MODE", None)
        else:
            os.environ["MOE_MODE"] = prev_mode


def test_qwen3_vl_moe_whitelist_miss_skips_submit_path_entirely():
    prev_mode = os.environ.get("MOE_MODE")
    os.environ["MOE_MODE"] = "TP"
    try:
        dispatcher = _SpecSubmitSpyDispatcher(hybrid=True, with_cache=True)
        layer = _make_spec_submit_guard_layer(dispatcher, spec_layer_enabled=False)
        layer._spec_prev_decode_step_id = 42
        layer._spec_prev_top1_by_req = {1000: (3, 11)}

        infer_state = type(
            "InferState",
            (),
            {
                "is_prefill": False,
                "decode_step_id": 43,
                "b_req_idx": torch.tensor([1000], dtype=torch.long),
                "b_adapter_bin": torch.tensor([3], dtype=torch.int32),
            },
        )()

        layer_infer_mod.Qwen3VLMOETransformerLayerInfer._maybe_submit_decode_spec_gate_up(
            layer,
            torch.zeros(1, 4),
            infer_state,
        )

        assert dispatcher.begin_calls == []
        assert dispatcher.submit_calls == []
        assert layer._spec_submit_totals == {"submitted": 0, "skipped": 0, "rejected": 0}
    finally:
        if prev_mode is None:
            os.environ.pop("MOE_MODE", None)
        else:
            os.environ["MOE_MODE"] = prev_mode


def test_qwen3_vl_moe_failed_future_retires_and_returns_none_for_caller_fallback():
    prev_mode = os.environ.get("MOE_MODE")
    os.environ["MOE_MODE"] = "TP"
    try:
        dispatcher = _make_dispatcher(queue_depth=2)
        dispatcher.begin_spec_step(44)
        dispatcher._should_use_hybrid_moe_compute = lambda: True
        dispatcher.expert_cache_manager = object()

        future = _ManualFuture(allow_cancel=False)
        key = _make_key(44, row_group_sig=((0, 1000), (1, 1001)))
        assert dispatcher.try_admit_spec_job(key, lambda: future) is not None

        future.set_exception(RuntimeError("boom"))

        expert_input = torch.zeros(2, 2, dtype=torch.float32)
        expert_req_bins = torch.tensor([3, 3], dtype=torch.int32)
        batch_indices = torch.tensor([0, 1], dtype=torch.long)
        infer_state = type(
            "InferState",
            (),
            {"is_prefill": False, "decode_step_id": 44, "b_req_idx": torch.tensor([1000, 1001], dtype=torch.long)},
        )()
        layer = _make_spec_bind_layer(dispatcher)
        pack_meta = layer_infer_mod.Qwen3VLMOETransformerLayerInfer._coalesce_lora_activations(
            layer,
            expert_input,
            expert_req_bins,
        )
        colora_stats = layer_infer_mod.Qwen3MOETransformerLayerInfer._new_colora_stats(layer)

        result = layer_infer_mod.Qwen3VLMOETransformerLayerInfer._maybe_bind_fused_gate_up_exact(
            layer,
            expert_input,
            infer_state,
            type("LayerWeight", (), {"layer_num_": 7})(),
            11,
            batch_indices,
            expert_req_bins,
            pack_meta,
            colora_stats,
        )

        assert result is None
        assert key not in dispatcher._spec_active_jobs
        assert len(dispatcher._spec_retired_jobs) == 1
        assert dispatcher._spec_retired_jobs[0].retire_reason == "bind_failed"
        assert dispatcher._get_cpu_queue_depth() == 0
        assert colora_stats["attempted_bind"] == 1
        assert colora_stats["successful_bind"] == 0
        assert colora_stats["stale"] == 0
        assert colora_stats["not_ready"] == 0
        assert colora_stats["fallback"] == 1
    finally:
        if prev_mode is None:
            os.environ.pop("MOE_MODE", None)
        else:
            os.environ["MOE_MODE"] = prev_mode


def test_begin_spec_step_retires_previous_step_jobs_and_reaps_late_completion():
    dispatcher = _make_dispatcher(queue_depth=2)
    dispatcher.begin_spec_step(45)

    slow_future = _ManualFuture(allow_cancel=False)
    key = _make_key(45)
    assert dispatcher.try_admit_spec_job(key, lambda: slow_future) is not None
    assert dispatcher._get_cpu_queue_depth() == 1

    stats = dispatcher.begin_spec_step(46)

    assert stats["retired_active"] == 1
    assert stats["reaped_retired"] == 0
    assert stats["pending_retired"] == 1
    assert dispatcher._spec_current_step_id == 46
    assert len(dispatcher._spec_active_jobs) == 0
    assert len(dispatcher._spec_retired_jobs) == 1
    assert dispatcher._spec_retired_jobs[0].retire_reason == "step_advanced"

    slow_future.set_result(("late_gate", "late_up"))
    assert dispatcher._get_cpu_queue_depth() == 0

    reap_stats = dispatcher.reap_retired()
    assert reap_stats["reaped_retired"] == 1
    assert reap_stats["pending_retired"] == 0
