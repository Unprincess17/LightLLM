"""Regression: ``a_len`` must reserve the full per-adapter span (see LoRAModulePool.load_adapter)."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightllm.server.lora.lora_mem_pool import LoRAModulePool


def test_sparse_layer_keys_reserve_num_layers_slots():
    """Sparse checkpoints must not shrink ``a_len`` below ``num_layers`` (prevents buffer overlap)."""
    num_layers = 8
    pool = LoRAModulePool.create(
        pool_size=256,
        max_rank=4,
        input_dim=16,
        output_dim=16,
        dtype=torch.float32,
        device="cpu",
        num_layers=num_layers,
        num_experts=1,
    )
    rank = 4
    # Only two layer entries, far apart — old bug used ``valid_layers==2`` and packed the next adapter at +2.
    layer_weights = {
        0: {"A": torch.randn(rank, 16), "B": torch.randn(rank, 16)},
        7: {"A": torch.randn(rank, 16), "B": torch.randn(rank, 16)},
    }
    assert pool.load_adapter(0, rank, 1.0, layer_weights)
    assert int(pool.a_len[0].item()) == num_layers
    assert int(pool._compute_location()) == num_layers

    layer_weights_b = {
        i: {"A": torch.randn(rank, 16), "B": torch.randn(rank, 16)} for i in range(num_layers)
    }
    assert pool.load_adapter(1, rank, 1.0, layer_weights_b)
    assert int(pool.a_start[1].item()) == num_layers
