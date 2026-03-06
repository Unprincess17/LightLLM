import torch
import importlib.util
from pathlib import Path

from lightllm.server.core.objs.lora_compute_config import LoRAComputeConfig
from lightllm.server.lora.lora_mem_pool import LoRAModulePool


_DISPATCH_PATH = Path(__file__).resolve().parents[2] / "lightllm/models/qwen3_vl_moe/lora_dispatch.py"
_DISPATCH_SPEC = importlib.util.spec_from_file_location("colora_dispatch_mod", _DISPATCH_PATH)
dispatch_mod = importlib.util.module_from_spec(_DISPATCH_SPEC)
assert _DISPATCH_SPEC is not None and _DISPATCH_SPEC.loader is not None
_DISPATCH_SPEC.loader.exec_module(dispatch_mod)

Qwen3VLMoELoRADispatcher = dispatch_mod.Qwen3VLMoELoRADispatcher


def _build_pool_with_rank_mismatch_case():
    pool = LoRAModulePool.create(
        pool_size=8,
        max_rank=8,
        input_dim=4,
        output_dim=3,
        dtype=torch.float32,
        device="cpu",
        num_layers=1,
        num_experts=1,
    )

    # rank=2 but a_len for this adapter is slot count (=1), which used to be a bug source.
    A = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.5, -1.0, 0.0, 2.0]],
        dtype=torch.float32,
    )
    B = torch.tensor(
        [[1.0, 2.0, 3.0], [0.1, -0.2, 0.3]],
        dtype=torch.float32,
    )

    ok = pool.load_adapter(
        adapter_idx=0,
        rank=2,
        scaling=1.0,
        layer_weights={0: {"A": A, "B": B}},
    )
    assert ok
    return pool, A, B


def test_colora_pool_records_adapter_rank_metadata():
    pool, _, _ = _build_pool_with_rank_mismatch_case()

    assert hasattr(pool, "a_rank")
    assert int(pool.a_rank.shape[0]) == 1
    assert int(pool.a_rank[0].item()) == 2
    # Slot count is intentionally different from rank in this test case.
    assert int(pool.a_len[0].item()) == 1


def test_colora_naive_cpu_uses_a_rank_instead_of_a_len():
    pool, A, B = _build_pool_with_rank_mismatch_case()
    dispatcher = Qwen3VLMoELoRADispatcher(
        num_layers=1,
        gate_lora_rank=2,
        lora_compute_config=LoRAComputeConfig(moe_storage="cpu", moe_compute="cpu"),
    )

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    bins = torch.tensor([0], dtype=torch.long)

    out = dispatcher._naive_batch_lora(
        input_tensor=x,
        layer_id=0,
        pool=pool,
        req_bins=bins,
        force_cpu=True,
    )

    ref = (x @ A.T) @ B
    assert out.shape == ref.shape
    # CPU fallback may use AVX BF16 kernel; validate within BF16 tolerance.
    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2)
