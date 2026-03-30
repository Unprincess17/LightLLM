import torch
from types import SimpleNamespace

from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.qwen3_vl_moe.layer_infer.transformer_layer_infer import Qwen3VLMOETransformerLayerInfer
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend


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
