import torch
from contextlib import contextmanager
from types import SimpleNamespace

from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.qwen3_vl_moe.layer_infer.transformer_layer_infer import Qwen3VLMOETransformerLayerInfer
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.mode_backend.dp_backend.impl import DPChunkedPrefillBackend
from lightllm.server.core.objs.lora_compute_config import LoRAComputeConfig


def test_select_active_decode_outputs_filters_paused_rows_and_padding():
    model_output = ModelOutput(
        logits=torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.float32),
        active_request_positions=torch.tensor([1, 3, 4], dtype=torch.long),
    )
    b_req_idx = torch.tensor([100, 101, 102, 103, 104], dtype=torch.int32)
    b_mtp_index = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32)
    run_reqs = ["req0", "req1", "req2"]

    logits, selected_b_req_idx, selected_b_mtp_index, selected_run_reqs = ModeBackend._select_active_decode_outputs(
        SimpleNamespace(),
        model_output,
        b_req_idx,
        b_mtp_index,
        run_reqs,
    )

    assert logits.shape[0] == 1
    assert torch.allclose(logits, torch.tensor([[10.0]], dtype=torch.float32))
    assert selected_b_req_idx.tolist() == [101]
    assert selected_b_mtp_index.tolist() == [0]
    assert selected_run_reqs == ["req1"]


def test_prune_decode_batch_updates_active_positions_and_lengths():
    state = InferStateInfo()
    state.is_prefill = False
    state.b_req_idx = torch.tensor([10, 11, 12], dtype=torch.int32)
    state.b_adapter_bin = torch.tensor([0, 1, 2], dtype=torch.int32)
    state.b_trace_req_id = torch.tensor([110, 111, 112], dtype=torch.int64)
    state.b_mtp_index = torch.tensor([0, 0, 0], dtype=torch.int32)
    state.b_seq_len = torch.tensor([5, 6, 7], dtype=torch.int32)
    state.mem_index = torch.tensor([20, 21, 22], dtype=torch.int32)
    state.active_request_positions = torch.tensor([0, 1, 2], dtype=torch.long)
    state.multimodal_params = [{"idx": 0}, {"idx": 1}, {"idx": 2}]
    state.position_cos = torch.tensor([[1.0], [2.0], [3.0]])
    state.position_sin = torch.tensor([[4.0], [5.0], [6.0]])

    state.prune_decode_batch(torch.tensor([0, 2], dtype=torch.long))

    assert state.b_req_idx.tolist() == [10, 12]
    assert state.b_adapter_bin.tolist() == [0, 2]
    assert state.b_trace_req_id.tolist() == [110, 112]
    assert state.mem_index.tolist() == [20, 22]
    assert state.active_request_positions.tolist() == [0, 2]
    assert state.multimodal_params == [{"idx": 0}, {"idx": 2}]
    assert state.position_cos.tolist() == [[1.0], [3.0]]
    assert state.position_sin.tolist() == [[4.0], [6.0]]
    assert state.batch_size == 2
    assert state.total_token_num == 12
    assert state.max_len_in_batch == 7


def test_complete_paused_layer_on_cpu_merges_hot_and_cold_outputs():
    layer = object.__new__(Qwen3VLMOETransformerLayerInfer)
    layer._ffn_norm_cpu = lambda hidden, _layer_weight: hidden
    layer.use_detached_lora_ = False
    layer.lora_dispatcher_ = None

    experts = SimpleNamespace(
        experts_gate_projs=[torch.eye(2, dtype=torch.float32)],
        experts_up_projs=[torch.eye(2, dtype=torch.float32)],
        w2_list=[torch.eye(2, dtype=torch.float32)],
    )
    layer_weight = SimpleNamespace(experts=experts)

    continuation = SimpleNamespace(saved_hidden=None, completed=False)
    req_obj = SimpleNamespace(colora_continuation=continuation, colora_paused=True)
    hidden = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    partial_ffn = torch.tensor([[0.5, 0.25]], dtype=torch.float32)
    task = SimpleNamespace(
        req_obj=req_obj,
        layer_id=0,
        hidden_after_attention=hidden,
        partial_ffn_output=partial_ffn,
        cold_expert_ids=[0],
        cold_routing_weights=[0.5],
        adapter_bin=0,
        layer_weight=layer_weight,
    )

    Qwen3VLMOETransformerLayerInfer._complete_paused_layer_on_cpu(layer, task)

    cold_ffn = torch.nn.functional.silu(hidden) * hidden * 0.5
    expected_saved_hidden = hidden + partial_ffn + cold_ffn
    assert torch.allclose(req_obj.colora_continuation.saved_hidden, expected_saved_hidden)
    assert req_obj.colora_continuation.completed is True
    assert req_obj.colora_paused is False


def test_init_batched_lora_adapters_disables_request_skip_and_async_fallback_only_in_no_overlap(monkeypatch):
    def _run_init(overlap_mode):
        captured_kwargs = []

        class _DummyPool:
            def __init__(self):
                self.adapter_dirs = []
                self.tp_rank_ = 0
                self.moe_gate_pool = object()
                self.moe_up_pool = object()
                self.moe_down_pool = object()

            def load_adapter(self, adapter_dir, rank, scaling, layer_weights):
                self.adapter_dirs.append(adapter_dir)

        class _DummyAdapter:
            max_rank = 8
            lora_alpha = 16

            @staticmethod
            def get_all_weights():
                return {}

        backend = object.__new__(ModeBackend)
        backend.args = SimpleNamespace(
            colora_async_fallback=1,
            colora_request_skip=1,
            colora_overlap_mode=overlap_mode,
            colora_cpu_workers=2,
            colora_cpu_queue_depth=16,
            colora_cpu_batch_timeout_us=10,
            colora_deferred_promotion_delta_steps=4,
            colora_promotion_ema_alpha=0.5,
            colora_temporal_prefetch=False,
            colora_temporal_hot_cache_slots=8,
            colora_max_continuations=8,
            metric_port=None,
        )
        backend.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)
        backend.model = SimpleNamespace(
            config={
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "intermediate_size": 16,
                "hidden_size": 32,
                "vocab_size": 128,
            },
            data_type=torch.float16,
            layers_num=1,
            layers_infer=[],
        )
        backend.rank_in_node = 0
        backend._lora_compute_config = LoRAComputeConfig(moe_storage="cpu", moe_compute="hybrid")
        backend.colora_metric_client = None
        backend._load_lora_adapter_fn = lambda **kwargs: _DummyAdapter()
        backend._create_lora_dispatcher_fn = lambda **kwargs: captured_kwargs.append(kwargs) or SimpleNamespace()

        import lightllm.server.lora as lora_pkg

        monkeypatch.setattr(lora_pkg, "create_lora_mem_pool", lambda **kwargs: _DummyPool())

        class _DummyCacheManager:
            def register_projection_pool(self, projection, pool):
                return None

            def get_cache_observability_stats(self):
                return {
                    "capacity_slots": 0,
                    "resident_slots": 0,
                    "free_slots": 0,
                    "evictions_total": 0,
                    "gate_capacity_slots": 0,
                    "gate_resident_slots": 0,
                    "gate_free_slots": 0,
                    "gate_evictions_total": 0,
                }

        monkeypatch.setattr(lora_pkg, "MoEExpertCacheManager", lambda cfg: _DummyCacheManager())

        import lightllm.server.router.model_infer.mode_backend.base_backend as base_backend_mod

        monkeypatch.setattr(base_backend_mod, "get_global_world_size", lambda: 1)

        backend.init_batched_lora_adapters({"1": "/tmp/adapter"})

        assert len(captured_kwargs) == 1
        return captured_kwargs[0]

    overlap_kwargs = _run_init("overlap")
    no_overlap_kwargs = _run_init("no_overlap")

    assert overlap_kwargs["colora_request_skip"] is True
    assert overlap_kwargs["colora_async_fallback"] is True
    assert no_overlap_kwargs["colora_request_skip"] is False
    assert no_overlap_kwargs["colora_async_fallback"] is False



def test_decode_normal_groups_completed_continuations_by_resume_layer(monkeypatch):
    forward_inputs = []
    sampled_run_reqs = []
    pre_post_inputs = []

    @contextmanager
    def _noop_stream(_stream):
        yield

    class _DummyEvent:
        def record(self):
            return None

        def synchronize(self):
            return None

    import lightllm.server.router.model_infer.mode_backend.dp_backend.impl as dp_impl

    monkeypatch.setattr(dp_impl.torch.cuda, "stream", _noop_stream)
    monkeypatch.setattr(dp_impl.torch.cuda, "Event", _DummyEvent)
    monkeypatch.setattr(dp_impl.g_infer_context, "get_overlap_stream", lambda: None)

    def _fake_select_active(_model_output, b_req_idx, b_mtp_index, run_reqs):
        logits = torch.zeros((len(run_reqs), 1), dtype=torch.float32)
        return logits, b_req_idx, b_mtp_index, run_reqs

    def _fake_sample(**kwargs):
        sampled_run_reqs.append(kwargs["run_reqs"])
        n = len(kwargs["run_reqs"])
        return None, [11] * n, [0.1] * n

    backend = object.__new__(DPChunkedPrefillBackend)
    backend.model = SimpleNamespace(
        forward=lambda model_input: forward_inputs.append(model_input) or ModelOutput(
            logits=torch.zeros((model_input.batch_size, 4), dtype=torch.float32)
        )
    )
    backend._alloc_decode_step_id = lambda: 7
    backend._select_active_decode_outputs = _fake_select_active
    backend._sample_and_scatter_token = _fake_sample
    backend._pre_post_handle = lambda run_reqs, is_chuncked_mode: pre_post_inputs.append((run_reqs, is_chuncked_mode)) or {}
    backend._post_handle = lambda **kwargs: None
    backend.extra_post_req_handle_func = None

    req_a = SimpleNamespace(
        req_idx=1,
        req_id=101,
        multimodal_params={"id": "a"},
        colora_continuation=SimpleNamespace(
            completed=True,
            resume_layer=3,
            saved_hidden=torch.ones((1, 2), dtype=torch.float32),
            mem_index=torch.tensor([11], dtype=torch.int32),
        ),
        get_cur_total_len=lambda: 5,
    )
    req_b = SimpleNamespace(
        req_idx=2,
        req_id=102,
        multimodal_params={"id": "b"},
        colora_continuation=SimpleNamespace(
            completed=True,
            resume_layer=3,
            saved_hidden=torch.ones((1, 2), dtype=torch.float32) * 2,
            mem_index=torch.tensor([12], dtype=torch.int32),
        ),
        get_cur_total_len=lambda: 6,
    )
    req_c = SimpleNamespace(
        req_idx=3,
        req_id=103,
        multimodal_params={"id": "c"},
        colora_continuation=SimpleNamespace(
            completed=True,
            resume_layer=5,
            saved_hidden=torch.ones((1, 2), dtype=torch.float32) * 3,
            mem_index=torch.tensor([13], dtype=torch.int32),
        ),
        get_cur_total_len=lambda: 7,
    )

    event_pack = SimpleNamespace(
        notify_post_handle_and_wait_pre_post_handle=lambda: None,
        notify_forward_and_wait_post_handle=lambda: None,
        notify_pre_post_handle=lambda: None,
    )

    DPChunkedPrefillBackend.decode_normal(backend, event_pack, [req_a, req_b, req_c])

    assert len(forward_inputs) == 2
    assert {forward_inputs[0].resume_from_layer, forward_inputs[1].resume_from_layer} == {3, 5}
    assert all(inp.is_continuation_batch for inp in forward_inputs)
    assert sorted(inp.batch_size for inp in forward_inputs) == [1, 2]
    assert any(batch == [req_a, req_b] for batch in sampled_run_reqs)
    assert any(batch == [req_c] for batch in sampled_run_reqs)
    assert len(pre_post_inputs) == 1
    assert len(pre_post_inputs[0][0]) == 3
    assert req_a.colora_continuation is None
    assert req_b.colora_continuation is None
    assert req_c.colora_continuation is None
