"""Forced-cold invariant: every required expert-LoRA object is nonresident
on the client GPU before every measured request.

Implementation: no-reuse object-ID stream. Each request gets a distinct
weight pair from a large pool. Pool size must exceed the measurement window
(N_requests per trial). The oracle path is exempt (warm weights by definition).
"""
import torch

class ForcedColdWeightPool:
    """Pre-generated pool of distinct LoRA weight pairs.

    Ensures no weight reuse within the measurement window, so no object
    can become a hit through residual cache state.

    Attributes:
        R, H, I: LoRA dimensions
        pool_size: number of distinct weight pairs
        dtype: weight dtype (BF16 for all paths)
        device: where weights live ("cpu" for cpu_first/load_then_run)
    """

    def __init__(self, R, H, I, pool_size, dtype=torch.bfloat16,
                 device="cpu", seed=42):
        self.R = R
        self.H = H
        self.I = I
        self.pool_size = pool_size
        self.dtype = dtype
        self.device = device

        g = torch.Generator(device=device)
        g.manual_seed(seed)
        # Pre-generate all weight pairs
        self._A = torch.randn(pool_size, R, H, dtype=dtype, device=device, generator=g)
        self._B = torch.randn(pool_size, R, I, dtype=dtype, device=device, generator=g)

    def get(self, index):
        """Get the i-th weight pair (cold on client GPU).

        Returns: dict with A=[R,H], B=[R,I] tensors (single object, not batched).
        """
        if index >= self.pool_size:
            raise IndexError(f"Weight pool exhausted: index {index} >= pool_size {self.pool_size}")
        return {
            "A": self._A[index],
            "B": self._B[index],
        }

    def get_batch(self, start_index, NM):
        """Get NM consecutive weight pairs as a batched tensor.

        Returns: dict with A=[NM,R,H], B=[NM,R,I]
        """
        if start_index + NM > self.pool_size:
            raise IndexError(
                f"Weight pool exhausted: {start_index}+{NM} > pool_size {self.pool_size}"
            )
        return {
            "A": self._A[start_index:start_index + NM],
            "B": self._B[start_index:start_index + NM],
        }

    def get_oracle_weights(self, index, device="cuda"):
        """Get weight pair on GPU (warm, for oracle path only)."""
        w = self.get(index)
        return {"A": w["A"].to(device), "B": w["B"].to(device)}

    def get_oracle_batch(self, start_index, NM, device="cuda"):
        """Get NM weight pairs on GPU (warm, for oracle path only)."""
        w = self.get_batch(start_index, NM)
        return {"A": w["A"].to(device), "B": w["B"].to(device)}
