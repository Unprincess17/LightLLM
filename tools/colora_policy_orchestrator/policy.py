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
    # S3 (pre-cached relay) is the default winner. S2 (remote activation)
    # carve-outs target high-margin cells from the cross-node sweep.
    if ctx.rank == 128 and ctx.n_tokens == 1 and ctx.ep_bw_pct in (0, 50, 90):
        return S2_REMOTE_ACTIVATION
    if ctx.rank == 16 and ctx.n_tokens == 4 and ctx.ep_bw_pct >= 75:
        return S2_REMOTE_ACTIVATION
    if ctx.rank == 16 and ctx.n_tokens == 4 and ctx.ep_bw_pct == 0:
        return S2_REMOTE_ACTIVATION
    if ctx.rank == 16 and ctx.n_tokens == 1 and 25 <= ctx.ep_bw_pct <= 50:
        return S2_REMOTE_ACTIVATION
    if ctx.rank == 32 and ctx.n_tokens == 1 and ctx.ep_bw_pct == 0:
        return S2_REMOTE_ACTIVATION
    if ctx.rank == 16 and ctx.n_tokens == 2 and ctx.ep_bw_pct == 90:
        return S2_REMOTE_ACTIVATION
    if ctx.rank == 128 and ctx.n_tokens == 2 and ctx.ep_bw_pct == 75:
        return S2_REMOTE_ACTIVATION
    if ctx.rank == 32 and ctx.n_tokens == 1 and ctx.ep_bw_pct == 50:
        return S2_REMOTE_ACTIVATION
    if ctx.rank == 128 and ctx.n_tokens == 2 and ctx.ep_bw_pct in (25, 75):
        return S2_REMOTE_ACTIVATION
    return S3_REMOTE_RELAY
