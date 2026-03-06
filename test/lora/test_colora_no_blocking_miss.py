import torch
import importlib.util
from pathlib import Path

from lightllm.server.core.objs.lora_compute_config import LoRAComputeConfig
from lightllm.server.lora.expert_cache import MoEExpertCacheConfig, MoEExpertCacheManager
from lightllm.server.lora.lora_mem_pool import LoRAModulePool


_DISPATCH_PATH = Path(__file__).resolve().parents[2] / "lightllm/models/qwen3_vl_moe/lora_dispatch.py"
_DISPATCH_SPEC = importlib.util.spec_from_file_location("colora_dispatch_mod", _DISPATCH_PATH)
dispatch_mod = importlib.util.module_from_spec(_DISPATCH_SPEC)
assert _DISPATCH_SPEC is not None and _DISPATCH_SPEC.loader is not None
_DISPATCH_SPEC.loader.exec_module(dispatch_mod)



def _build_pool_single_adapter(dtype=torch.float32):
    pool = LoRAModulePool.create(
        pool_size=8,
        max_rank=4,
        input_dim=4,
        output_dim=4,
        dtype=dtype,
        device="cpu",
        num_layers=1,
        num_experts=1,
    )
    ok = pool.load_adapter(
        adapter_idx=0,
        rank=2,
        scaling=1.0,
        layer_weights={
            0: {
                "A": torch.randn(2, 4, dtype=dtype),
                "B": torch.randn(2, 4, dtype=dtype),
            }
        },
    )
    assert ok
    return pool


def test_colora_hybrid_miss_path_returns_without_waiting_for_promotion():
    pool = _build_pool_single_adapter()

    cache_mgr = MoEExpertCacheManager(
        MoEExpertCacheConfig(
            cache_budget_mb=16,
            promote_min_hits=100,  # ensure this call does not trigger promotion
            promote_window=8,
            max_promote_per_step=1,
            decay=0.9,
        )
    )
    cache_mgr.register_projection_pool("gate", pool)

    dispatcher = dispatch_mod.Qwen3VLMoELoRADispatcher(
        num_layers=1,
        gate_lora_rank=2,
        lora_compute_config=LoRAComputeConfig(moe_storage="cpu", moe_compute="hybrid"),
    )
    dispatcher.expert_cache_manager = cache_mgr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 4, dtype=torch.float32, device=device)
    bins = torch.tensor([0, 0], dtype=torch.long, device=device)

    out = dispatcher._batch_apply_moe_lora_hybrid(
        input_tensor=x,
        layer_id=0,
        buffer_layer_id=0,
        pool=pool,
        bins=bins,
        projection="gate",
        expert_id=0,
    )

    assert out.shape == x.shape
    stats = dispatcher.pop_colora_stats()
    assert stats["colora_miss_tokens"] == 2
    assert stats["promotion_queue_depth"] == 0
