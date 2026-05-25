import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from lightllm.server.core.objs.lora_compute_config import LoRAComputeConfig
from lightllm.server.lora import LoRAAdapterLoader, LoRATargetType, create_lora_mem_pool
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend


_CPU_LORA_CONFIG = LoRAComputeConfig(
    vl_storage="cpu",
    vl_compute="gpu",
    attn_storage="cpu",
    attn_compute="gpu",
    moe_storage="cpu",
    moe_compute="gpu",
)


def _write_test_adapter(adapter_dir: Path, expert_delta: float = 0.0) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)

    tensors = {
        "model.language_model.layers.0.self_attn.q_proj.lora_A.weight": (
            torch.arange(16, dtype=torch.float32).reshape(8, 2) + expert_delta
        ),
        "model.language_model.layers.0.self_attn.q_proj.lora_B.weight": (
            torch.arange(16, dtype=torch.float32).reshape(2, 8) + expert_delta
        ),
        "model.language_model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": (
            torch.arange(16, dtype=torch.float32).reshape(8, 2) + 10 + expert_delta
        ),
        "model.language_model.layers.0.mlp.experts.0.gate_proj.lora_B.weight": (
            torch.arange(12, dtype=torch.float32).reshape(2, 6) + 20 + expert_delta
        ),
        "model.language_model.layers.0.mlp.experts.1.gate_proj.lora_A.weight": (
            torch.arange(16, dtype=torch.float32).reshape(8, 2) + 30 + expert_delta
        ),
        "model.language_model.layers.0.mlp.experts.1.gate_proj.lora_B.weight": (
            torch.arange(12, dtype=torch.float32).reshape(2, 6) + 40 + expert_delta
        ),
    }
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4.0}))


def test_moe_adapter_loader_preserves_experts_and_pool_slots(tmp_path):
    adapter_dir = tmp_path / "adapter_a"
    _write_test_adapter(adapter_dir)

    layer_weights = LoRAAdapterLoader.load_from_dir(
        adapter_dir=str(adapter_dir),
        network_config={"num_hidden_layers": 1},
        dtype=torch.float16,
        device="cpu",
    )

    gate_weights = layer_weights[0][LoRATargetType.MOE_EXPERT_GATE]
    assert set(gate_weights.keys()) == {0, 1}
    assert torch.equal(
        gate_weights[0]["A"],
        (torch.arange(16, dtype=torch.float32).reshape(8, 2) + 10).transpose(0, 1).to(torch.float16),
    )
    assert torch.equal(
        layer_weights[0][LoRATargetType.ATTN_Q_PROJ]["A"],
        torch.arange(16, dtype=torch.float32).reshape(8, 2).transpose(0, 1).to(torch.float16),
    )

    pool = create_lora_mem_pool(
        num_layers=1,
        pool_size=4,
        max_rank=2,
        num_heads=2,
        head_dim=4,
        intermediate_dim=6,
        hidden_size=8,
        vocab_size=32,
        dtype=torch.float16,
        lora_compute_config=_CPU_LORA_CONFIG,
        moe_intermediate_dim=6,
        num_experts=2,
    )

    assert pool.load_adapter(
        adapter_dir=str(adapter_dir),
        rank=2,
        scaling=2.0,
        layer_weights=layer_weights,
    )
    assert pool.load_adapter(
        adapter_dir=str(tmp_path / "adapter_b"),
        rank=2,
        scaling=2.0,
        layer_weights=layer_weights,
    )

    assert pool.moe_gate_pool.a_start.tolist() == [0, 2]
    assert pool.moe_gate_pool.a_len.tolist() == [2, 2]
    assert torch.equal(pool.moe_gate_pool.a_buffer[0, :2], gate_weights[0]["A"])
    assert torch.equal(pool.moe_gate_pool.a_buffer[1, :2], gate_weights[1]["A"])
    assert torch.equal(pool.moe_gate_pool.a_buffer[2, :2], gate_weights[0]["A"])
    assert torch.equal(pool.moe_gate_pool.a_buffer[3, :2], gate_weights[1]["A"])


def test_init_batched_lora_adapters_uses_direct_loader_and_sizes_pool(monkeypatch, tmp_path):
    adapter_a = tmp_path / "adapter_a"
    adapter_b = tmp_path / "adapter_b"
    _write_test_adapter(adapter_a)
    _write_test_adapter(adapter_b, expert_delta=100.0)

    captured_pool_kwargs = {}
    captured_loads = []

    class _DummyPool:
        def __init__(self):
            self.adapter_dirs = []
            self.tp_rank_ = 0
            self.moe_gate_pool = object()
            self.moe_up_pool = object()
            self.moe_down_pool = object()

        def load_adapter(self, adapter_dir, rank, scaling, layer_weights):
            captured_loads.append((adapter_dir, rank, scaling, layer_weights))
            self.adapter_dirs.append(adapter_dir)
            return True

    backend = object.__new__(ModeBackend)
    backend.args = SimpleNamespace(
        colora_async_fallback=1,
        colora_request_skip=1,
        colora_overlap_mode="overlap",
        colora_cpu_workers=2,
        colora_cpu_queue_depth=16,
        colora_cpu_batch_timeout_us=10,
        colora_deferred_promotion_delta_steps=4,
        colora_promotion_ema_alpha=0.5,
        colora_temporal_prefetch=False,
        colora_temporal_hot_cache_slots=8,
        colora_max_continuations=8,
        metric_port=None,
        node_rank=0,
    )
    backend.logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )

    DummyModel = type("DummyModel", (), {})
    DummyModel.__module__ = "lightllm.models.qwen3_vl_moe.model"
    backend.model = DummyModel()
    class _HybridConfig:
        def __init__(self, d):
            self.__dict__.update(d)
        def __getitem__(self, key):
            return getattr(self, key)
        def get(self, key, default=None):
            return getattr(self, key, default)
        def __contains__(self, key):
            return hasattr(self, key)

    backend.model.config = _HybridConfig({
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "intermediate_size": 6,
        "moe_intermediate_size": 6,
        "hidden_size": 8,
        "vocab_size": 32,
        "num_local_experts": 2,
    })
    backend.model.data_type = torch.float16
    backend.model.layers_num = 1
    backend.model.layers_infer = []
    backend.rank_in_node = 0
    backend.node_world_size = 1
    backend.global_rank = 0
    backend._lora_compute_config = _CPU_LORA_CONFIG
    backend.colora_metric_client = None
    backend._load_lora_adapter_fn = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("batched loading should bypass the per-adapter object loader")
    )
    backend._create_lora_dispatcher_fn = lambda **kwargs: SimpleNamespace()

    import lightllm.server.lora as lora_pkg
    import lightllm.server.router.model_infer.mode_backend.base_backend as base_backend_mod

    def _fake_create_pool(**kwargs):
        captured_pool_kwargs.update(kwargs)
        return _DummyPool()

    monkeypatch.setattr(lora_pkg, "create_lora_mem_pool", _fake_create_pool)
    monkeypatch.setattr(base_backend_mod, "get_global_world_size", lambda: 1)
    monkeypatch.setattr(base_backend_mod, "get_global_rank", lambda: 0)

    backend.init_batched_lora_adapters(
        {
            "a": str(adapter_a),
            "b": str(adapter_b),
        }
    )

    assert captured_pool_kwargs["num_experts"] == 2
    assert len(captured_loads) == 2
    assert captured_loads[0][1:3] == (2, 2.0)
    assert set(captured_loads[0][3][0][LoRATargetType.MOE_EXPERT_GATE].keys()) == {0, 1}
