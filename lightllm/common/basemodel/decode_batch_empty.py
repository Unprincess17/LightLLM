"""
Decode-batch emptiness checks for token forward (COLoRA prune, etc.).

CUDA may report ``invalid configuration argument`` on a later innocent op (e.g. ``fill_``)
when an earlier kernel launched with invalid grid/block. Unit tests cannot reliably
reproduce that *error site*, but they *can* lock the contract: never run ``post_infer`` /
LM head when the hidden has 0 rows or ``batch_size`` is 0.
"""

from __future__ import annotations

from typing import Any

import torch

from lightllm.common.basemodel.infer_struct import InferStateInfo


def should_break_layer_loop(infer_state: InferStateInfo, input_embs: torch.Tensor) -> bool:
    """After one transformer layer, stop if there are no rows left to decode."""
    if infer_state.batch_size == 0:
        return True
    if isinstance(input_embs, torch.Tensor) and input_embs.ndim >= 1 and input_embs.shape[0] == 0:
        infer_state.batch_size = 0
        return True
    return False


def should_emit_empty_logits(infer_state: InferStateInfo, input_embs: Any) -> bool:
    """True iff we must skip ``post_infer`` and return logits with shape ``(0, vocab)``."""
    if infer_state.batch_size == 0:
        return True
    if isinstance(input_embs, torch.Tensor) and input_embs.ndim >= 1 and input_embs.shape[0] == 0:
        return True
    return False
