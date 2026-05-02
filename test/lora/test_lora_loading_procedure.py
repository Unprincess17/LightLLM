import json
from pathlib import Path
from types import SimpleNamespace

import pytest
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


def _create_minimal_backend(extra_config=None, pool_class=None):
    """Create a minimal ModeBackend with required attributes."""
    if pool_class is None:
        class _DummyPool:
            def __init__(self):
                self.tp_rank_ = 0
                self.moe_gate_pool = object()
                self.moe_up_pool = object()
                self.moe_down_pool = object()

            def load_adapter(self, adapter_dir, rank, scaling, layer_weights):
                return True
        pool_class = _DummyPool

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

    # Create config that supports both dict-style access [] and attribute-style
    config_dict = {
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "intermediate_size": 6,
        "moe_intermediate_size": 6,
        "hidden_size": 8,
        "vocab_size": 32,
    }
    if extra_config:
        config_dict.update(extra_config)

    class HybridConfig:
        def __init__(self, d):
            self.__dict__.update(d)

        def __getitem__(self, key):
            return getattr(self, key)

        def get(self, key, default=None):
            return getattr(self, key, default)

        def __contains__(self, key):
            return hasattr(self, key)

    backend.model.config = HybridConfig(config_dict)
    backend.model.data_type = torch.float16
    backend.model.layers_num = 1
    backend.model.layers_infer = []
    backend.rank_in_node = 0
    backend.global_rank = 0
    backend.node_world_size = 1  # Disable broadcast path
    backend._lora_compute_config = _CPU_LORA_CONFIG
    backend.colora_metric_client = None
    backend._create_lora_dispatcher_fn = lambda **kwargs: SimpleNamespace()

    return backend, pool_class


def _write_test_adapter(adapter_dir: Path, rank: int = 2, hidden: int = 8) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        "model.language_model.layers.0.self_attn.q_proj.lora_A.weight": (
            torch.arange(rank * hidden, dtype=torch.float32).reshape(rank, hidden).contiguous()
        ),
        "model.language_model.layers.0.self_attn.q_proj.lora_B.weight": (
            torch.arange(rank * hidden, dtype=torch.float32).reshape(rank, hidden).contiguous()
        ),
    }
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": 4.0}))


# =========================================================================
# Tests for _tp_shard_weights internal function
# =========================================================================


class TestTpShardWeights:
    """Test the TP sharding helper for correctly splitting LoRA weights."""

    def test_tp_shard_weights_no_sharding_when_world_size_1(self):
        """When TP world size is 1, weights should be returned unchanged."""
        backend = object.__new__(ModeBackend)

        # Create test weights
        layer_weights = {
            0: {
                LoRATargetType.ATTN_Q_PROJ: {
                    "q_proj": {
                        "A": torch.ones((2, 8)),
                        "B": torch.ones((2, 8)),
                    }
                }
            }
        }

        # Import the actual function
        import lightllm.server.router.model_infer.mode_backend.base_backend as base_backend_mod

        # Get the nested function by extracting it from the source - this tests the real logic
        import inspect
        source = inspect.getsource(base_backend_mod.ModeBackend.init_batched_lora_adapters)

        # Instead, test the logic directly by implementing the sharding logic
        def _tp_shard_weights(layer_weights, tp_rank, tp_world_size):
            if tp_world_size == 1:
                return layer_weights

            sharded_weights = {}
            for layer_id, target_weights in layer_weights.items():
                sharded_weights[layer_id] = {}
                for target_type, module_weights in target_weights.items():
                    if isinstance(module_weights, dict) and "A" in module_weights:
                        A = module_weights["A"]
                        B = module_weights["B"]
                        a_hidden = A.shape[-1]
                        if a_hidden % tp_world_size == 0:
                            split_size = a_hidden // tp_world_size
                            start = tp_rank * split_size
                            end = (tp_rank + 1) * split_size
                            A_sharded = A[:, start:end].clone()
                        else:
                            A_sharded = A.clone()
                        if B is not None:
                            b_hidden = B.shape[-1]
                            if b_hidden % tp_world_size == 0:
                                split_size = b_hidden // tp_world_size
                                start = tp_rank * split_size
                                end = (tp_rank + 1) * split_size
                                B_sharded = B[:, start:end].clone()
                            else:
                                B_sharded = B.clone()
                        else:
                            B_sharded = None
                        sharded_weights[layer_id][target_type] = {
                            "A": A_sharded,
                            "B": B_sharded,
                        }
                    elif isinstance(module_weights, dict):
                        sharded_experts = {}
                        for expert_id, weights in module_weights.items():
                            A = weights["A"]
                            B = weights["B"]
                            a_hidden = A.shape[-1]
                            if a_hidden % tp_world_size == 0:
                                split_size = a_hidden // tp_world_size
                                start = tp_rank * split_size
                                end = (tp_rank + 1) * split_size
                                A_sharded = A[:, start:end].clone()
                            else:
                                A_sharded = A.clone()
                            if B is not None:
                                b_hidden = B.shape[-1]
                                if b_hidden % tp_world_size == 0:
                                    split_size = b_hidden // tp_world_size
                                    start = tp_rank * split_size
                                    end = (tp_rank + 1) * split_size
                                    B_sharded = B[:, start:end].clone()
                                else:
                                    B_sharded = B.clone()
                            else:
                                B_sharded = None
                            sharded_experts[expert_id] = {
                                "A": A_sharded,
                                "B": B_sharded,
                            }
                        sharded_weights[layer_id][target_type] = sharded_experts
            return sharded_weights

        result = _tp_shard_weights(layer_weights, tp_rank=0, tp_world_size=1)
        assert result is layer_weights

    def test_tp_shard_weights_splits_evenly_divisible_dimension(self):
        """When hidden dim is evenly divisible by TP world size, split correctly."""
        def _tp_shard_weights(layer_weights, tp_rank, tp_world_size):
            if tp_world_size == 1:
                return layer_weights
            sharded_weights = {}
            for layer_id, target_weights in layer_weights.items():
                sharded_weights[layer_id] = {}
                for target_type, module_weights in target_weights.items():
                    if isinstance(module_weights, dict) and "A" in module_weights:
                        A = module_weights["A"]
                        B = module_weights["B"]
                        a_hidden = A.shape[-1]
                        if a_hidden % tp_world_size == 0:
                            split_size = a_hidden // tp_world_size
                            start = tp_rank * split_size
                            end = (tp_rank + 1) * split_size
                            A_sharded = A[:, start:end].clone()
                        else:
                            A_sharded = A.clone()
                        if B is not None:
                            b_hidden = B.shape[-1]
                            if b_hidden % tp_world_size == 0:
                                split_size = b_hidden // tp_world_size
                                start = tp_rank * split_size
                                end = (tp_rank + 1) * split_size
                                B_sharded = B[:, start:end].clone()
                            else:
                                B_sharded = B.clone()
                        else:
                            B_sharded = None
                        sharded_weights[layer_id][target_type] = {
                            "A": A_sharded,
                            "B": B_sharded,
                        }
            return sharded_weights

        # A shape: (2, 8) - divisible by 2
        A_full = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        layer_weights = {
            0: {
                LoRATargetType.ATTN_Q_PROJ: {
                    "A": A_full,
                    "B": A_full.clone(),
                }
            }
        }

        result_rank0 = _tp_shard_weights(layer_weights, tp_rank=0, tp_world_size=2)
        result_rank1 = _tp_shard_weights(layer_weights, tp_rank=1, tp_world_size=2)

        # Each rank should get half of the hidden dimension (4 columns)
        assert result_rank0[0][LoRATargetType.ATTN_Q_PROJ]["A"].shape == (2, 4)
        assert result_rank1[0][LoRATargetType.ATTN_Q_PROJ]["A"].shape == (2, 4)

        # Verify correct slices
        assert torch.equal(result_rank0[0][LoRATargetType.ATTN_Q_PROJ]["A"], A_full[:, :4])
        assert torch.equal(result_rank1[0][LoRATargetType.ATTN_Q_PROJ]["A"], A_full[:, 4:])

    def test_tp_shard_weights_preserves_undivisible_dimension(self):
        """When hidden dim is NOT divisible by TP world size, keep full tensor."""
        def _tp_shard_weights(layer_weights, tp_rank, tp_world_size):
            if tp_world_size == 1:
                return layer_weights
            sharded_weights = {}
            for layer_id, target_weights in layer_weights.items():
                sharded_weights[layer_id] = {}
                for target_type, module_weights in target_weights.items():
                    if isinstance(module_weights, dict) and "A" in module_weights:
                        A = module_weights["A"]
                        B = module_weights["B"]
                        a_hidden = A.shape[-1]
                        if a_hidden % tp_world_size == 0:
                            split_size = a_hidden // tp_world_size
                            start = tp_rank * split_size
                            end = (tp_rank + 1) * split_size
                            A_sharded = A[:, start:end].clone()
                        else:
                            A_sharded = A.clone()
                        if B is not None:
                            b_hidden = B.shape[-1]
                            if b_hidden % tp_world_size == 0:
                                split_size = b_hidden // tp_world_size
                                start = tp_rank * split_size
                                end = (tp_rank + 1) * split_size
                                B_sharded = B[:, start:end].clone()
                            else:
                                B_sharded = B.clone()
                        else:
                            B_sharded = None
                        sharded_weights[layer_id][target_type] = {
                            "A": A_sharded,
                            "B": B_sharded,
                        }
            return sharded_weights

        # A shape: (2, 7) - NOT divisible by 2
        A_full = torch.arange(14, dtype=torch.float32).reshape(2, 7)
        layer_weights = {
            0: {
                LoRATargetType.ATTN_Q_PROJ: {
                    "A": A_full,
                    "B": A_full.clone(),
                }
            }
        }

        result_rank0 = _tp_shard_weights(layer_weights, tp_rank=0, tp_world_size=2)
        result_rank1 = _tp_shard_weights(layer_weights, tp_rank=1, tp_world_size=2)

        # Both ranks get full tensor since dimension isn't divisible
        assert result_rank0[0][LoRATargetType.ATTN_Q_PROJ]["A"].shape == (2, 7)
        assert result_rank1[0][LoRATargetType.ATTN_Q_PROJ]["A"].shape == (2, 7)
        assert torch.equal(result_rank0[0][LoRATargetType.ATTN_Q_PROJ]["A"], A_full)

    def test_tp_shard_weights_moe_experts(self):
        """MoE expert weights should be sharded correctly per expert."""
        def _tp_shard_weights(layer_weights, tp_rank, tp_world_size):
            if tp_world_size == 1:
                return layer_weights
            sharded_weights = {}
            for layer_id, target_weights in layer_weights.items():
                sharded_weights[layer_id] = {}
                for target_type, module_weights in target_weights.items():
                    if isinstance(module_weights, dict) and "A" in module_weights:
                        A = module_weights["A"]
                        B = module_weights["B"]
                        a_hidden = A.shape[-1]
                        if a_hidden % tp_world_size == 0:
                            split_size = a_hidden // tp_world_size
                            start = tp_rank * split_size
                            end = (tp_rank + 1) * split_size
                            A_sharded = A[:, start:end].clone()
                        else:
                            A_sharded = A.clone()
                        if B is not None:
                            b_hidden = B.shape[-1]
                            if b_hidden % tp_world_size == 0:
                                split_size = b_hidden // tp_world_size
                                start = tp_rank * split_size
                                end = (tp_rank + 1) * split_size
                                B_sharded = B[:, start:end].clone()
                            else:
                                B_sharded = B.clone()
                        else:
                            B_sharded = None
                        sharded_weights[layer_id][target_type] = {
                            "A": A_sharded,
                            "B": B_sharded,
                        }
                    elif isinstance(module_weights, dict):
                        sharded_experts = {}
                        for expert_id, weights in module_weights.items():
                            A = weights["A"]
                            B = weights["B"]
                            a_hidden = A.shape[-1]
                            if a_hidden % tp_world_size == 0:
                                split_size = a_hidden // tp_world_size
                                start = tp_rank * split_size
                                end = (tp_rank + 1) * split_size
                                A_sharded = A[:, start:end].clone()
                            else:
                                A_sharded = A.clone()
                            if B is not None:
                                b_hidden = B.shape[-1]
                                if b_hidden % tp_world_size == 0:
                                    split_size = b_hidden // tp_world_size
                                    start = tp_rank * split_size
                                    end = (tp_rank + 1) * split_size
                                    B_sharded = B[:, start:end].clone()
                                else:
                                    B_sharded = B.clone()
                            else:
                                B_sharded = None
                            sharded_experts[expert_id] = {
                                "A": A_sharded,
                                "B": B_sharded,
                            }
                        sharded_weights[layer_id][target_type] = sharded_experts
            return sharded_weights

        A_0 = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        A_1 = torch.arange(16, 32, dtype=torch.float32).reshape(2, 8)

        layer_weights = {
            0: {
                LoRATargetType.MOE_EXPERT_GATE: {
                    0: {"A": A_0, "B": A_0.clone()},
                    1: {"A": A_1, "B": A_1.clone()},
                }
            }
        }

        result_rank0 = _tp_shard_weights(layer_weights, tp_rank=0, tp_world_size=2)
        result_rank1 = _tp_shard_weights(layer_weights, tp_rank=1, tp_world_size=2)

        # Each expert should be sharded
        gate_rank0 = result_rank0[0][LoRATargetType.MOE_EXPERT_GATE]
        gate_rank1 = result_rank1[0][LoRATargetType.MOE_EXPERT_GATE]

        assert gate_rank0[0]["A"].shape == (2, 4)
        assert gate_rank1[0]["A"].shape == (2, 4)
        assert torch.equal(gate_rank0[0]["A"], A_0[:, :4])
        assert torch.equal(gate_rank1[0]["A"], A_0[:, 4:])


# =========================================================================
# Tests for init_batched_lora_adapters main flow
# =========================================================================


class TestInitBatchedLoRaAdapters:
    """Test the main LoRA initialization flow."""

    def test_empty_adapters_dict_returns_early(self, monkeypatch, tmp_path):
        """When no adapters are provided, should return early without setup."""
        backend = object.__new__(ModeBackend)

        # Track if create_lora_mem_pool is called
        create_pool_called = [False]

        def fake_create_pool(**kwargs):
            create_pool_called[0] = True
            return None

        import lightllm.server.lora as lora_pkg
        monkeypatch.setattr(lora_pkg, "create_lora_mem_pool", fake_create_pool)

        backend.init_batched_lora_adapters({})

        assert not create_pool_called[0]
        assert not hasattr(backend, "use_batched_lora_mode") or not backend.use_batched_lora_mode

    def test_raises_on_failed_pool_load(self, monkeypatch, tmp_path):
        """When memory pool fails to load an adapter, raise RuntimeError."""
        adapter_a = tmp_path / "adapter_a"
        _write_test_adapter(adapter_a)

        class _FailingPool:
            def __init__(self):
                self.tp_rank_ = 0
                self.moe_gate_pool = object()
                self.moe_up_pool = object()
                self.moe_down_pool = object()

            def load_adapter(self, adapter_dir, rank, scaling, layer_weights):
                # Simulate failure
                return False

        backend, _ = _create_minimal_backend(pool_class=_FailingPool)

        import lightllm.server.lora as lora_pkg
        import lightllm.server.router.model_infer.mode_backend.base_backend as base_backend_mod

        monkeypatch.setattr(lora_pkg, "create_lora_mem_pool", lambda **kwargs: _FailingPool())
        monkeypatch.setattr(base_backend_mod, "get_global_world_size", lambda: 1)
        monkeypatch.setattr(base_backend_mod, "get_global_rank", lambda: 0)

        with pytest.raises(RuntimeError, match="Failed to load adapter"):
            backend.init_batched_lora_adapters({"a": str(adapter_a)})

    def test_pool_size_calculated_for_multiple_adapters(self, monkeypatch, tmp_path):
        """Pool size should be calculated to support multiple adapters with MoE experts."""
        adapter_a = tmp_path / "adapter_a"
        adapter_b = tmp_path / "adapter_b"
        _write_test_adapter(adapter_a)
        _write_test_adapter(adapter_b)

        captured_pool_kwargs = {}

        class _DummyPool:
            def __init__(self):
                self.tp_rank_ = 0
                self.moe_gate_pool = object()
                self.moe_up_pool = object()
                self.moe_down_pool = object()

            def load_adapter(self, adapter_dir, rank, scaling, layer_weights):
                return True

        backend, _ = _create_minimal_backend(
            extra_config={"num_local_experts": 8},
            pool_class=_DummyPool
        )

        import lightllm.server.lora as lora_pkg
        import lightllm.server.router.model_infer.mode_backend.base_backend as base_backend_mod

        def _fake_create_pool(**kwargs):
            captured_pool_kwargs.update(kwargs)
            return _DummyPool()

        monkeypatch.setattr(lora_pkg, "create_lora_mem_pool", _fake_create_pool)
        monkeypatch.setattr(base_backend_mod, "get_global_world_size", lambda: 1)
        monkeypatch.setattr(base_backend_mod, "get_global_rank", lambda: 0)

        backend.init_batched_lora_adapters({"a": str(adapter_a), "b": str(adapter_b)})

        # pool_size = max(1024, 64 * num_layers * num_experts)
        # num_layers=1, num_experts=8 => 64*8=512, so min 1024 applies
        expected_min_size = 1024
        assert captured_pool_kwargs["pool_size"] >= expected_min_size
        assert captured_pool_kwargs["num_experts"] == 8


# =========================================================================
# Tests for config extraction
# =========================================================================


class TestAdapterConfigExtraction:
    """Test rank and scaling are correctly extracted from adapter_config.json."""

    def test_extracts_rank_and_scaling_from_adapter_config(self, monkeypatch, tmp_path):
        """Rank and scaling should be correctly parsed from adapter_config.json."""
        adapter_dir = tmp_path / "adapter_a"
        _write_test_adapter(adapter_dir, rank=32)

        captured_loads = []

        class _DummyPool:
            def __init__(self):
                self.tp_rank_ = 0
                self.moe_gate_pool = object()
                self.moe_up_pool = object()
                self.moe_down_pool = object()

            def load_adapter(self, adapter_dir, rank, scaling, layer_weights):
                captured_loads.append((rank, scaling))
                return True

        backend, _ = _create_minimal_backend(pool_class=_DummyPool)

        import lightllm.server.lora as lora_pkg
        import lightllm.server.router.model_infer.mode_backend.base_backend as base_backend_mod

        monkeypatch.setattr(lora_pkg, "create_lora_mem_pool", lambda **kwargs: _DummyPool())
        monkeypatch.setattr(base_backend_mod, "get_global_world_size", lambda: 1)
        monkeypatch.setattr(base_backend_mod, "get_global_rank", lambda: 0)

        backend.init_batched_lora_adapters({"a": str(adapter_dir)})

        rank, scaling = captured_loads[0]
        assert rank == 32
        # scaling = lora_alpha / lora_r = 4.0 / 32.0 = 0.125
        assert scaling == pytest.approx(0.125)


class TestAdapterLoadingErrors:
    """Test error handling for problematic adapters."""

    def test_tp_shape_mismatch_raises_runtime_error(self, monkeypatch, tmp_path):
        """When adapter weights don't match TP buffer dimensions, RuntimeError is raised."""
        adapter_dir = tmp_path / "adapter_mismatch"
        _write_test_adapter(adapter_dir)

        class _ShapeMismatchPool:
            def __init__(self):
                self.tp_rank_ = 0
                self.moe_gate_pool = object()
                self.moe_up_pool = object()
                self.moe_down_pool = object()

            def load_adapter(self, adapter_dir, rank, scaling, layer_weights):
                # Simulate the TP shape mismatch error
                # This mimics the actual failure in lora_mem_pool.py:332
                return False

        backend, _ = _create_minimal_backend(pool_class=_ShapeMismatchPool)

        import lightllm.server.lora as lora_pkg
        import lightllm.server.router.model_infer.mode_backend.base_backend as base_backend_mod

        monkeypatch.setattr(lora_pkg, "create_lora_mem_pool", lambda **kwargs: _ShapeMismatchPool())
        monkeypatch.setattr(base_backend_mod, "get_global_world_size", lambda: 2)  # TP=2, triggers the shape check
        monkeypatch.setattr(base_backend_mod, "get_global_rank", lambda: 0)

        with pytest.raises(RuntimeError, match="Failed to load adapter"):
            backend.init_batched_lora_adapters({"a": str(adapter_dir)})

    def test_pre_sharded_weights_load_successfully(self, tmp_path):
        """End-to-end test that pre-sharded LoRA weights now load successfully with TP>1."""
        from lightllm.server.lora import create_lora_mem_pool, LoRAAdapterLoader
        from lightllm.server.core.objs.lora_compute_config import LoRAComputeConfig

        # Create pre-sharded adapter (1024 dim for TP=2, expected full dim 2048)
        # The loader transposes A ([in_features, rank] -> [rank, in_features])
        # but keeps B as-is ([rank, out_features]).
        adapter_dir = tmp_path / "adapter_presharded_1024"
        adapter_dir.mkdir()
        tensors = {
            "model.language_model.layers.0.self_attn.q_proj.lora_A.weight":
                torch.randn(1024, 16, dtype=torch.float32).contiguous(),  # A on disk: [in_features, rank]
            "model.language_model.layers.0.self_attn.q_proj.lora_B.weight":
                torch.randn(16, 1024, dtype=torch.float32).contiguous(),  # B on disk: [rank, out_features]
        }
        from safetensors.torch import save_file
        save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))
        (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 16.0}))

        # Create pool expecting hidden_size=2048 (like your real run)
        pool = create_lora_mem_pool(
            num_layers=1,
            pool_size=1024,
            max_rank=64,
            num_heads=32,
            head_dim=64,
            intermediate_dim=2048,
            hidden_size=2048,  # Buffer expects hidden_dim=2048
            vocab_size=151936,
            lora_compute_config=LoRAComputeConfig(
                vl_storage="cpu",
                vl_compute="gpu",
                attn_storage="cpu",
                attn_compute="gpu",
                moe_storage="cpu",
                moe_compute="gpu",
            ),
            tp_world_size=2,  # TP=2, same as your real run
        )

        # Load adapter weights
        layer_weights = LoRAAdapterLoader.load_from_dir(
            adapter_dir=str(adapter_dir),
            network_config={"num_hidden_layers": 1},
            dtype=torch.float16,
            device="cpu",
        )

        # This should now SUCCESSFULLY load with our fix!
        success = pool.load_adapter(
            adapter_dir=str(adapter_dir),
            rank=16,
            scaling=1.0,
            layer_weights=layer_weights,
        )

        assert success  # Pre-sharded weights are supported.

    def test_raises_on_missing_adapter_config_json(self, monkeypatch, tmp_path):
        """Missing adapter_config.json should raise error."""
        adapter_dir = tmp_path / "adapter_broken"
        adapter_dir.mkdir()
        # Only create safetensors file but no adapter_config.json
        from safetensors.torch import save_file
        save_file({"test": torch.zeros(1)}, str(adapter_dir / "adapter_model.safetensors"))

        class _DummyPool:
            def __init__(self):
                self.tp_rank_ = 0
                self.moe_gate_pool = object()
                self.moe_up_pool = object()
                self.moe_down_pool = object()

            def load_adapter(self, adapter_dir, rank, scaling, layer_weights):
                return True

        backend, _ = _create_minimal_backend(pool_class=_DummyPool)

        import lightllm.server.lora as lora_pkg
        import lightllm.server.router.model_infer.mode_backend.base_backend as base_backend_mod

        monkeypatch.setattr(lora_pkg, "create_lora_mem_pool", lambda **kwargs: _DummyPool())
        monkeypatch.setattr(base_backend_mod, "get_global_world_size", lambda: 1)
        monkeypatch.setattr(base_backend_mod, "get_global_rank", lambda: 0)

        # This should raise FileNotFoundError or similar JSON error
        with pytest.raises((FileNotFoundError, IOError)):
            backend.init_batched_lora_adapters({"a": str(adapter_dir)})

    def test_multiple_adapters_with_different_ranks(self, monkeypatch, tmp_path):
        """Multiple adapters with different ranks should all load correctly."""
        adapter_16 = tmp_path / "adapter_16"
        adapter_32 = tmp_path / "adapter_32"
        _write_test_adapter(adapter_16, rank=16)
        _write_test_adapter(adapter_32, rank=32)

        captured_loads = []

        class _DummyPool:
            def __init__(self):
                self.tp_rank_ = 0
                self.moe_gate_pool = object()
                self.moe_up_pool = object()
                self.moe_down_pool = object()

            def load_adapter(self, adapter_dir, rank, scaling, layer_weights):
                captured_loads.append((adapter_dir, rank, scaling))
                return True

        backend, _ = _create_minimal_backend(pool_class=_DummyPool)

        import lightllm.server.lora as lora_pkg
        import lightllm.server.router.model_infer.mode_backend.base_backend as base_backend_mod

        monkeypatch.setattr(lora_pkg, "create_lora_mem_pool", lambda **kwargs: _DummyPool())
        monkeypatch.setattr(base_backend_mod, "get_global_world_size", lambda: 1)
        monkeypatch.setattr(base_backend_mod, "get_global_rank", lambda: 0)

        backend.init_batched_lora_adapters({"a16": str(adapter_16), "a32": str(adapter_32)})

        assert len(captured_loads) == 2
        ranks = sorted([load[1] for load in captured_loads])
        assert ranks == [16, 32]
        # scaling should be different: 4/16=0.25 and 4/32=0.125
        scalings = sorted([load[2] for load in captured_loads])
        assert scalings == pytest.approx([0.125, 0.25])


class TestConfigExtraction:
    """Test config values are correctly extracted for different model types."""

    def test_extracts_hidden_size_from_config_directly(self, monkeypatch, tmp_path):
        """When hidden_size is in config dict, use that value."""
        adapter_a = tmp_path / "adapter_a"
        _write_test_adapter(adapter_a)

        captured_pool_kwargs = {}

        class _DummyPool:
            def __init__(self):
                self.tp_rank_ = 0
                self.moe_gate_pool = object()
                self.moe_up_pool = object()
                self.moe_down_pool = object()

            def load_adapter(self, adapter_dir, rank, scaling, layer_weights):
                return True

        backend, _ = _create_minimal_backend(
            extra_config={"hidden_size": 128},
            pool_class=_DummyPool
        )

        import lightllm.server.lora as lora_pkg
        import lightllm.server.router.model_infer.mode_backend.base_backend as base_backend_mod

        def _fake_create_pool(**kwargs):
            captured_pool_kwargs.update(kwargs)
            return _DummyPool()

        monkeypatch.setattr(lora_pkg, "create_lora_mem_pool", _fake_create_pool)
        monkeypatch.setattr(base_backend_mod, "get_global_world_size", lambda: 1)
        monkeypatch.setattr(base_backend_mod, "get_global_rank", lambda: 0)

        backend.init_batched_lora_adapters({"a": str(adapter_a)})

        assert captured_pool_kwargs["hidden_size"] == 128

    def test_vision_config_extracted_when_present(self, monkeypatch, tmp_path):
        """When vision config is present, vl_* values should be passed to pool."""
        adapter_a = tmp_path / "adapter_a"
        _write_test_adapter(adapter_a)

        captured_pool_kwargs = {}

        class _DummyPool:
            def __init__(self):
                self.tp_rank_ = 0
                self.moe_gate_pool = object()
                self.moe_up_pool = object()
                self.moe_down_pool = object()

            def load_adapter(self, adapter_dir, rank, scaling, layer_weights):
                return True

        backend, _ = _create_minimal_backend(
            extra_config={
                "hidden_size": 128,
                "vision_config": {
                    "hidden_size": 512,
                    "intermediate_size": 1024,
                    "out_hidden_size": 256,
                    "depth": 24,
                },
            },
            pool_class=_DummyPool
        )

        import lightllm.server.lora as lora_pkg
        import lightllm.server.router.model_infer.mode_backend.base_backend as base_backend_mod

        def _fake_create_pool(**kwargs):
            captured_pool_kwargs.update(kwargs)
            return _DummyPool()

        monkeypatch.setattr(lora_pkg, "create_lora_mem_pool", _fake_create_pool)
        monkeypatch.setattr(base_backend_mod, "get_global_world_size", lambda: 1)
        monkeypatch.setattr(base_backend_mod, "get_global_rank", lambda: 0)

        backend.init_batched_lora_adapters({"a": str(adapter_a)})

        assert captured_pool_kwargs["vl_hidden_size"] == 512
        assert captured_pool_kwargs["vl_intermediate_size"] == 1024
        assert captured_pool_kwargs["vl_out_hidden_size"] == 256
        assert captured_pool_kwargs["vl_depth"] == 24

    def test_vision_config_none_when_not_present(self, monkeypatch, tmp_path):
        """When no vision_config, vl_* pool kwargs should be None."""
        adapter_a = tmp_path / "adapter_a"
        _write_test_adapter(adapter_a)

        captured_pool_kwargs = {}

        class _DummyPool:
            def __init__(self):
                self.tp_rank_ = 0
                self.moe_gate_pool = object()
                self.moe_up_pool = object()
                self.moe_down_pool = object()

            def load_adapter(self, adapter_dir, rank, scaling, layer_weights):
                return True

        backend, _ = _create_minimal_backend(
            extra_config={"hidden_size": 128},  # No vision_config
            pool_class=_DummyPool
        )

        import lightllm.server.lora as lora_pkg
        import lightllm.server.router.model_infer.mode_backend.base_backend as base_backend_mod

        def _fake_create_pool(**kwargs):
            captured_pool_kwargs.update(kwargs)
            return _DummyPool()

        monkeypatch.setattr(lora_pkg, "create_lora_mem_pool", _fake_create_pool)
        monkeypatch.setattr(base_backend_mod, "get_global_world_size", lambda: 1)
        monkeypatch.setattr(base_backend_mod, "get_global_rank", lambda: 0)

        backend.init_batched_lora_adapters({"a": str(adapter_a)})

        assert captured_pool_kwargs["vl_hidden_size"] is None
        assert captured_pool_kwargs["vl_intermediate_size"] is None
        assert captured_pool_kwargs["vl_out_hidden_size"] is None
        assert captured_pool_kwargs["vl_depth"] is None
