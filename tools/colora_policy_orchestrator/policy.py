"""CoLoRA hybrid recovery path selector.

Iteration 0 baseline: always pick S1 (remote weight transfer). This is the
naive "fetch on miss" policy from Slide 5, used as the reference point.

Each autoresearch iteration evolves choose_path() to reduce mean foreground
recovery latency across the (rank, n_tokens, ep_bw_pct) grid.
"""
from __future__ import annotations

from dataclasses import dataclass

S1_REMOTE_WEIGHT = 1
S2_REMOTE_ACTIVATION = 2
S3_REMOTE_RELAY = 3
LOCAL_CPU = 10
LOCAL_GPU = 11


@dataclass(frozen=True)
class RecoveryContext:
    rank: int
    n_tokens: int
    ep_bw_pct: int


def choose_path(ctx: RecoveryContext) -> int:
    """Return the chosen recovery strategy id for this miss context."""
    return S1_REMOTE_WEIGHT
