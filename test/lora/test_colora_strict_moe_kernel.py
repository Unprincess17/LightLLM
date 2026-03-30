import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lightllm.server.core.objs.lora_compute_config import LoRAComputeConfig
from lightllm.server.lora.lora_mem_pool import LoRAModulePool


_DISPATCH_PATH = Path(__file__).resolve().parents[2] / "lightllm/models/qwen3_vl_moe/lora_dispatch.py"
_DISPATCH_SPEC = importlib.util.spec_from_file_location("colora_dispatch_mod", _DISPATCH_PATH)
dispatch_mod = importlib.util.module_from_spec(_DISPATCH_SPEC)
assert _DISPATCH_SPEC is not None and _DISPATCH_SPEC.loader is not None
sys.modules[_DISPATCH_SPEC.name] = dispatch_mod
_DISPATCH_SPEC.loader.exec_module(dispatch_mod)


def test_strict_moe_kernel_required_for_moe_cpu(monkeypatch):
    monkeypatch.setattr(dispatch_mod, "MOE_AVX_AVAILABLE", False, raising=False)
    with pytest.raises(RuntimeError, match="MoE-specific CPU kernel is required"):
        dispatch_mod.Qwen3VLMoELoRADispatcher(
            num_layers=1,
            gate_lora_rank=2,
            lora_compute_config=LoRAComputeConfig(moe_storage="cpu", moe_compute="cpu"),
        )


def test_strict_moe_kernel_required_for_moe_hybrid(monkeypatch):
    monkeypatch.setattr(dispatch_mod, "MOE_AVX_AVAILABLE", False, raising=False)
    with pytest.raises(RuntimeError, match="MoE-specific CPU kernel is required"):
        dispatch_mod.Qwen3VLMoELoRADispatcher(
            num_layers=1,
            gate_lora_rank=2,
            lora_compute_config=LoRAComputeConfig(moe_storage="cpu", moe_compute="hybrid"),
        )


def test_gpu_moe_mode_does_not_require_moe_cpu_kernel(monkeypatch):
    monkeypatch.setattr(dispatch_mod, "MOE_AVX_AVAILABLE", False, raising=False)
    dispatcher = dispatch_mod.Qwen3VLMoELoRADispatcher(
        num_layers=1,
        gate_lora_rank=2,
        lora_compute_config=LoRAComputeConfig(moe_storage="gpu", moe_compute="gpu"),
    )
    assert dispatcher is not None


def _build_pool():
    pool = LoRAModulePool.create(
        pool_size=8,
        max_rank=4,
        input_dim=4,
        output_dim=4,
        dtype=torch.float16,
        device="cpu",
        num_layers=1,
        num_experts=1,
    )
    ok = pool.load_adapter(
        adapter_idx=0,
        rank=2,
        scaling=1.0,
        layer_weights={0: {"A": torch.randn(2, 4, dtype=torch.float16), "B": torch.randn(2, 4, dtype=torch.float16)}},
    )
    assert ok
    return pool


@pytest.mark.parametrize(
    "method_name,projection",
    [
        ("batch_apply_gate_lora", "gate"),
        ("batch_apply_up_lora", "up"),
        ("batch_apply_down_lora", "down"),
    ],
)
def test_moe_cpu_branches_route_to_strict_helper(monkeypatch, method_name, projection):
    monkeypatch.setattr(dispatch_mod, "MOE_AVX_AVAILABLE", True, raising=False)
    dispatcher = dispatch_mod.Qwen3VLMoELoRADispatcher(
        num_layers=1,
        gate_lora_rank=2,
        up_lora_rank=2,
        down_lora_rank=2,
        lora_compute_config=LoRAComputeConfig(moe_storage="cpu", moe_compute="cpu"),
    )
    pool = _build_pool()
    dispatcher.lora_mem_pool = SimpleNamespace(
        moe_gate_pool=pool,
        moe_up_pool=pool,
        moe_down_pool=pool,
    )
    bins = torch.tensor([0, 0], dtype=torch.long)
    x = torch.randn(2, 4, dtype=torch.float16)

    called = {"projection": None}

    def _fake_strict(input_tensor, layer_id, pool, req_bins, projection, adapter_group_plan=None):
        called["projection"] = projection
        out = torch.zeros(
            input_tensor.shape[0],
            pool.value_buffer.shape[2],
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        return out, 1, int(input_tensor.shape[0])

    monkeypatch.setattr(dispatcher, "_strict_moe_cpu_batch_lora", _fake_strict, raising=True)
    out = getattr(dispatcher, method_name)(x, layer_id=0, req_bins=bins)

    assert out.shape[0] == x.shape[0]
    assert called["projection"] == projection
