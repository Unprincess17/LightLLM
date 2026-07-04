"""Scheduling analysis helpers for Client-SJF and Server-SJF evaluation.

Functions to measure whether the intended submission order (Client-SJF)
actually controls GPU execution order on the server.

Key concepts:
  - submission_seq: order in which the client sent the request (req_id order)
  - handler_start_seq: order in which the server handler started processing
  - completion_seq: order in which the server handler finished
  - NM (num_miss): the predicted service time (shorter = higher priority)
"""
from typing import List, Dict, Optional
from itertools import combinations


def compute_priority_fidelity(requests: List[Dict]) -> float:
    """Fraction of decisions where the light request starts first when both
    light and heavy are queued.

    For each pair (light, heavy) where light has lower NM than heavy,
    we check whether light has a lower handler_start_seq.  Only pairs
    where both were submitted before either started handling are counted
    (i.e., both were in the queue simultaneously).

    requests: list of dicts with keys:
        req_id, nm, submission_seq, handler_start_seq, completion_seq

    Returns: float in [0, 1] (0.0 = never respected, 1.0 = always respected).
    Returns 1.0 if there are no comparable pairs.
    """
    sortable = [
        r for r in requests
        if r.get("handler_start_seq") is not None
        and r.get("submission_seq") is not None
        and r.get("nm") is not None
    ]
    if len(sortable) < 2:
        return 1.0

    total = 0
    respected = 0
    for a, b in combinations(sortable, 2):
        # Identify light (lower NM) and heavy (higher NM)
        if a["nm"] < b["nm"]:
            light, heavy = a, b
        elif a["nm"] > b["nm"]:
            light, heavy = b, a
        else:
            continue  # same NM, no priority decision to make

        # Only count if both were submitted before either started handling.
        # "Both queued" means the later submission happened before the
        # earlier handler start.
        later_submit = max(light["submission_seq"], heavy["submission_seq"])
        earlier_start = min(light["handler_start_seq"], heavy["handler_start_seq"])
        if later_submit >= earlier_start:
            continue  # not both in queue at the same time

        total += 1
        if light["handler_start_seq"] < heavy["handler_start_seq"]:
            respected += 1

    if total == 0:
        return 1.0
    return respected / total


def compute_inversion_count(requests: List[Dict]) -> int:
    """Number of times a heavy request starts (handler_start_seq) while an
    earlier-submitted light request remains queued.

    An inversion occurs when:
      - request A (light, lower NM) was submitted before request B (heavy, higher NM)
      - but B's handler_start_seq < A's handler_start_seq
      - and A was still queued when B started (A's handler_start_seq > B's handler_start_seq
        and A's submission_seq < B's handler_start_seq)

    requests: list of dicts with keys:
        req_id, nm, submission_seq, handler_start_seq

    Returns: integer count of inversions.
    """
    sortable = [
        r for r in requests
        if r.get("handler_start_seq") is not None
        and r.get("submission_seq") is not None
        and r.get("nm") is not None
    ]
    if len(sortable) < 2:
        return 0

    inversions = 0
    for a, b in combinations(sortable, 2):
        # a is lighter (lower NM), b is heavier
        if a["nm"] >= b["nm"]:
            if a["nm"] > b["nm"]:
                light, heavy = b, a
            else:
                continue
        else:
            light, heavy = a, b

        # Inversion: heavy started before light, but light was submitted first
        if (heavy["handler_start_seq"] < light["handler_start_seq"]
                and light["submission_seq"] < heavy["submission_seq"]):
            inversions += 1

    return inversions


def compute_order_correlation(intended_order: List[int],
                               actual_order: List[int]) -> float:
    """Kendall's tau between intended (submission) and actual (handler-start) order.

    Both inputs are lists of request IDs in the intended and actual order
    respectively.  For example:
        intended_order = [1, 2, 3, 4]  (submission order)
        actual_order   = [1, 3, 2, 4]  (handler_start order)

    Returns: float in [-1, 1].  1.0 = perfect agreement, -1.0 = perfect reversal,
             0.0 = no correlation.  Returns 0.0 for lists shorter than 2.
    """
    n = len(intended_order)
    if n < 2 or len(actual_order) != n:
        return 0.0

    # Build rank maps: req_id -> position
    intended_rank = {req_id: i for i, req_id in enumerate(intended_order)}
    actual_rank = {req_id: i for i, req_id in enumerate(actual_order)}

    # All req_ids must be present in both lists
    if set(intended_order) != set(actual_order):
        raise ValueError("intended_order and actual_order must contain the same request IDs")

    # Count concordant and discordant pairs
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            req_i = intended_order[i]
            req_j = intended_order[j]
            di = actual_rank[req_i] - actual_rank[req_j]
            if di > 0:
                discordant += 1
            elif di < 0:
                concordant += 1
            # ties (di == 0) are impossible since actual_order is a permutation

    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return 0.0
    return (concordant - discordant) / total_pairs
