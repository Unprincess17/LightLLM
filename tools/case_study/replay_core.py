#!/usr/bin/env python3
"""Shared replay kernels for B8/B9/B10 cache and latency analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numba import njit

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent))

from common import numeric_summary


POLICY_LRU = "lru"

PHASE_PREFILL = 0
PHASE_DECODE = 1
PHASE_UNKNOWN = 2


@dataclass
class ConditionAccessBuffer:
    condition: str
    access_ids: np.ndarray
    total_events: int
    max_object_id: int


@dataclass
class BudgetReplaySummary:
    condition: str
    cache_budget: int
    total_events: int
    hits: int
    misses: int
    hit_rate: float
    miss_rate: float
    cold_misses: int
    cold_miss_rate: float
    capacity_misses: int
    capacity_miss_rate: float
    total_distinct_objects: int
    total_evictions: int
    eviction_frequency: float
    unique_evicted_objects: int
    mean_residency_if_available: Optional[float]
    per_request_hit_count_summary: dict
    per_request_miss_count_summary: dict
    per_request_hits: np.ndarray
    per_request_misses: np.ndarray


@dataclass
class PhaseReplaySummary:
    condition: str
    cache_budget: int
    total_events: int
    hits: int
    misses: int
    cold_misses: int
    capacity_misses: int
    total_evictions: int
    unique_evicted_objects: int
    mean_residency_if_available: Optional[float]
    per_request_hits: np.ndarray
    per_request_misses: np.ndarray
    per_request_cold_misses: np.ndarray
    per_request_prefill_misses: np.ndarray
    per_request_decode_misses: np.ndarray


@dataclass
class PhaseReplayBitmapSummary:
    condition: str
    cache_budget: int
    total_events: int
    hits: int
    misses: int
    cold_misses: int
    capacity_misses: int
    total_evictions: int
    unique_evicted_objects: int
    mean_residency_if_available: Optional[float]
    per_request_hits: np.ndarray
    per_request_misses: np.ndarray
    per_request_cold_misses: np.ndarray
    per_request_prefill_misses: np.ndarray
    per_request_decode_misses: np.ndarray
    miss_flags: np.ndarray
    cold_miss_flags: np.ndarray


@njit(cache=False)
def lru_detach(prev_link: np.ndarray, next_link: np.ndarray, object_id: int, head: int, tail: int) -> Tuple[int, int]:
    previous_id = int(prev_link[object_id])
    next_id = int(next_link[object_id])

    if previous_id != 0:
        next_link[previous_id] = next_id
    else:
        head = next_id

    if next_id != 0:
        prev_link[next_id] = previous_id
    else:
        tail = previous_id

    prev_link[object_id] = 0
    next_link[object_id] = 0
    return head, tail


@njit(cache=False)
def lru_append(prev_link: np.ndarray, next_link: np.ndarray, object_id: int, head: int, tail: int) -> Tuple[int, int]:
    prev_link[object_id] = tail
    next_link[object_id] = 0
    if tail != 0:
        next_link[tail] = object_id
    else:
        head = object_id
    tail = object_id
    return head, tail


@njit(cache=False)
def simulate_lru_accesses(
    access_ids: np.ndarray,
    request_offsets: np.ndarray,
    max_object_id: int,
    capacity: int,
) -> Tuple[int, int, int, int, int, int, float, np.ndarray, np.ndarray]:
    seen_before = np.zeros(max_object_id + 1, dtype=np.uint8)
    request_count = request_offsets.shape[0] - 1
    per_request_hits = np.zeros(request_count, dtype=np.int64)
    per_request_misses = np.zeros(request_count, dtype=np.int64)

    hits = 0
    misses = 0
    cold_misses = 0
    capacity_misses = 0
    total_evictions = 0
    unique_evicted_objects = 0
    residency_sum = 0.0
    residency_count = 0

    if capacity <= 0:
        request_ordinal = 0
        for event_index in range(access_ids.shape[0]):
            while request_ordinal + 1 < request_offsets.shape[0] and event_index >= request_offsets[request_ordinal + 1]:
                request_ordinal += 1

            object_id = int(access_ids[event_index])
            misses += 1
            per_request_misses[request_ordinal] += 1
            if seen_before[object_id] != 0:
                capacity_misses += 1
            else:
                seen_before[object_id] = 1
                cold_misses += 1

        return (
            hits,
            misses,
            cold_misses,
            capacity_misses,
            total_evictions,
            unique_evicted_objects,
            0.0,
            per_request_hits,
            per_request_misses,
        )

    in_cache = np.zeros(max_object_id + 1, dtype=np.uint8)
    prev_link = np.zeros(max_object_id + 1, dtype=np.int32)
    next_link = np.zeros(max_object_id + 1, dtype=np.int32)
    inserted_at = np.zeros(max_object_id + 1, dtype=np.int64)
    ever_evicted = np.zeros(max_object_id + 1, dtype=np.uint8)

    cache_size = 0
    head = 0
    tail = 0
    request_ordinal = 0

    for event_index in range(access_ids.shape[0]):
        while request_ordinal + 1 < request_offsets.shape[0] and event_index >= request_offsets[request_ordinal + 1]:
            request_ordinal += 1

        object_id = int(access_ids[event_index])
        if in_cache[object_id] != 0:
            hits += 1
            per_request_hits[request_ordinal] += 1
            if tail != object_id:
                head, tail = lru_detach(prev_link, next_link, object_id, head, tail)
                head, tail = lru_append(prev_link, next_link, object_id, head, tail)
            continue

        misses += 1
        per_request_misses[request_ordinal] += 1
        if seen_before[object_id] != 0:
            capacity_misses += 1
        else:
            seen_before[object_id] = 1
            cold_misses += 1

        if cache_size >= capacity:
            evicted_object_id = head
            head, tail = lru_detach(prev_link, next_link, evicted_object_id, head, tail)
            in_cache[evicted_object_id] = 0
            total_evictions += 1
            if ever_evicted[evicted_object_id] == 0:
                ever_evicted[evicted_object_id] = 1
                unique_evicted_objects += 1
            residency_sum += float((event_index + 1) - inserted_at[evicted_object_id])
            residency_count += 1
        else:
            cache_size += 1

        head, tail = lru_append(prev_link, next_link, object_id, head, tail)
        in_cache[object_id] = 1
        inserted_at[object_id] = event_index + 1

    mean_residency = residency_sum / residency_count if residency_count > 0 else 0.0
    return (
        hits,
        misses,
        cold_misses,
        capacity_misses,
        total_evictions,
        unique_evicted_objects,
        mean_residency,
        per_request_hits,
        per_request_misses,
    )


@njit(cache=False)
def simulate_lru_accesses_with_phase(
    access_ids: np.ndarray,
    phase_ids: np.ndarray,
    request_offsets: np.ndarray,
    max_object_id: int,
    capacity: int,
) -> Tuple[int, int, int, int, int, int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seen_before = np.zeros(max_object_id + 1, dtype=np.uint8)
    request_count = request_offsets.shape[0] - 1
    per_request_hits = np.zeros(request_count, dtype=np.int64)
    per_request_misses = np.zeros(request_count, dtype=np.int64)
    per_request_cold_misses = np.zeros(request_count, dtype=np.int64)
    per_request_prefill_misses = np.zeros(request_count, dtype=np.int64)
    per_request_decode_misses = np.zeros(request_count, dtype=np.int64)

    hits = 0
    misses = 0
    cold_misses = 0
    capacity_misses = 0
    total_evictions = 0
    unique_evicted_objects = 0
    residency_sum = 0.0
    residency_count = 0

    if capacity <= 0:
        request_ordinal = 0
        for event_index in range(access_ids.shape[0]):
            while request_ordinal + 1 < request_offsets.shape[0] and event_index >= request_offsets[request_ordinal + 1]:
                request_ordinal += 1

            object_id = int(access_ids[event_index])
            phase_id = int(phase_ids[event_index])
            misses += 1
            per_request_misses[request_ordinal] += 1
            if phase_id == PHASE_PREFILL:
                per_request_prefill_misses[request_ordinal] += 1
            elif phase_id == PHASE_DECODE:
                per_request_decode_misses[request_ordinal] += 1
            if seen_before[object_id] != 0:
                capacity_misses += 1
            else:
                seen_before[object_id] = 1
                cold_misses += 1
                per_request_cold_misses[request_ordinal] += 1

        return (
            hits,
            misses,
            cold_misses,
            capacity_misses,
            total_evictions,
            unique_evicted_objects,
            0.0,
            per_request_hits,
            per_request_misses,
            per_request_cold_misses,
            per_request_prefill_misses,
            per_request_decode_misses,
        )

    in_cache = np.zeros(max_object_id + 1, dtype=np.uint8)
    prev_link = np.zeros(max_object_id + 1, dtype=np.int32)
    next_link = np.zeros(max_object_id + 1, dtype=np.int32)
    inserted_at = np.zeros(max_object_id + 1, dtype=np.int64)
    ever_evicted = np.zeros(max_object_id + 1, dtype=np.uint8)

    cache_size = 0
    head = 0
    tail = 0
    request_ordinal = 0

    for event_index in range(access_ids.shape[0]):
        while request_ordinal + 1 < request_offsets.shape[0] and event_index >= request_offsets[request_ordinal + 1]:
            request_ordinal += 1

        object_id = int(access_ids[event_index])
        phase_id = int(phase_ids[event_index])
        if in_cache[object_id] != 0:
            hits += 1
            per_request_hits[request_ordinal] += 1
            if tail != object_id:
                head, tail = lru_detach(prev_link, next_link, object_id, head, tail)
                head, tail = lru_append(prev_link, next_link, object_id, head, tail)
            continue

        misses += 1
        per_request_misses[request_ordinal] += 1
        if phase_id == PHASE_PREFILL:
            per_request_prefill_misses[request_ordinal] += 1
        elif phase_id == PHASE_DECODE:
            per_request_decode_misses[request_ordinal] += 1
        if seen_before[object_id] != 0:
            capacity_misses += 1
        else:
            seen_before[object_id] = 1
            cold_misses += 1
            per_request_cold_misses[request_ordinal] += 1

        if cache_size >= capacity:
            evicted_object_id = head
            head, tail = lru_detach(prev_link, next_link, evicted_object_id, head, tail)
            in_cache[evicted_object_id] = 0
            total_evictions += 1
            if ever_evicted[evicted_object_id] == 0:
                ever_evicted[evicted_object_id] = 1
                unique_evicted_objects += 1
            residency_sum += float((event_index + 1) - inserted_at[evicted_object_id])
            residency_count += 1
        else:
            cache_size += 1

        head, tail = lru_append(prev_link, next_link, object_id, head, tail)
        in_cache[object_id] = 1
        inserted_at[object_id] = event_index + 1

    mean_residency = residency_sum / residency_count if residency_count > 0 else 0.0
    return (
        hits,
        misses,
        cold_misses,
        capacity_misses,
        total_evictions,
        unique_evicted_objects,
        mean_residency,
        per_request_hits,
        per_request_misses,
        per_request_cold_misses,
        per_request_prefill_misses,
        per_request_decode_misses,
    )


@njit(cache=False)
def simulate_lru_accesses_with_phase_and_bitmaps(
    access_ids: np.ndarray,
    phase_ids: np.ndarray,
    request_offsets: np.ndarray,
    max_object_id: int,
    capacity: int,
) -> Tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    seen_before = np.zeros(max_object_id + 1, dtype=np.uint8)
    request_count = request_offsets.shape[0] - 1
    per_request_hits = np.zeros(request_count, dtype=np.int64)
    per_request_misses = np.zeros(request_count, dtype=np.int64)
    per_request_cold_misses = np.zeros(request_count, dtype=np.int64)
    per_request_prefill_misses = np.zeros(request_count, dtype=np.int64)
    per_request_decode_misses = np.zeros(request_count, dtype=np.int64)
    miss_flags = np.zeros(access_ids.shape[0], dtype=np.uint8)
    cold_miss_flags = np.zeros(access_ids.shape[0], dtype=np.uint8)

    hits = 0
    misses = 0
    cold_misses = 0
    capacity_misses = 0
    total_evictions = 0
    unique_evicted_objects = 0
    residency_sum = 0.0
    residency_count = 0

    if capacity <= 0:
        request_ordinal = 0
        for event_index in range(access_ids.shape[0]):
            while request_ordinal + 1 < request_offsets.shape[0] and event_index >= request_offsets[request_ordinal + 1]:
                request_ordinal += 1

            object_id = int(access_ids[event_index])
            phase_id = int(phase_ids[event_index])
            misses += 1
            miss_flags[event_index] = 1
            per_request_misses[request_ordinal] += 1
            if phase_id == PHASE_PREFILL:
                per_request_prefill_misses[request_ordinal] += 1
            elif phase_id == PHASE_DECODE:
                per_request_decode_misses[request_ordinal] += 1
            if seen_before[object_id] != 0:
                capacity_misses += 1
            else:
                seen_before[object_id] = 1
                cold_misses += 1
                cold_miss_flags[event_index] = 1
                per_request_cold_misses[request_ordinal] += 1

        return (
            hits,
            misses,
            cold_misses,
            capacity_misses,
            total_evictions,
            unique_evicted_objects,
            0.0,
            per_request_hits,
            per_request_misses,
            per_request_cold_misses,
            per_request_prefill_misses,
            per_request_decode_misses,
            miss_flags,
            cold_miss_flags,
        )

    in_cache = np.zeros(max_object_id + 1, dtype=np.uint8)
    prev_link = np.zeros(max_object_id + 1, dtype=np.int32)
    next_link = np.zeros(max_object_id + 1, dtype=np.int32)
    inserted_at = np.zeros(max_object_id + 1, dtype=np.int64)
    ever_evicted = np.zeros(max_object_id + 1, dtype=np.uint8)

    cache_size = 0
    head = 0
    tail = 0
    request_ordinal = 0

    for event_index in range(access_ids.shape[0]):
        while request_ordinal + 1 < request_offsets.shape[0] and event_index >= request_offsets[request_ordinal + 1]:
            request_ordinal += 1

        object_id = int(access_ids[event_index])
        phase_id = int(phase_ids[event_index])
        if in_cache[object_id] != 0:
            hits += 1
            per_request_hits[request_ordinal] += 1
            if tail != object_id:
                head, tail = lru_detach(prev_link, next_link, object_id, head, tail)
                head, tail = lru_append(prev_link, next_link, object_id, head, tail)
            continue

        misses += 1
        miss_flags[event_index] = 1
        per_request_misses[request_ordinal] += 1
        if phase_id == PHASE_PREFILL:
            per_request_prefill_misses[request_ordinal] += 1
        elif phase_id == PHASE_DECODE:
            per_request_decode_misses[request_ordinal] += 1
        if seen_before[object_id] != 0:
            capacity_misses += 1
        else:
            seen_before[object_id] = 1
            cold_misses += 1
            cold_miss_flags[event_index] = 1
            per_request_cold_misses[request_ordinal] += 1

        if cache_size >= capacity:
            evicted_object_id = head
            head, tail = lru_detach(prev_link, next_link, evicted_object_id, head, tail)
            in_cache[evicted_object_id] = 0
            total_evictions += 1
            if ever_evicted[evicted_object_id] == 0:
                ever_evicted[evicted_object_id] = 1
                unique_evicted_objects += 1
            residency_sum += float((event_index + 1) - inserted_at[evicted_object_id])
            residency_count += 1
        else:
            cache_size += 1

        head, tail = lru_append(prev_link, next_link, object_id, head, tail)
        in_cache[object_id] = 1
        inserted_at[object_id] = event_index + 1

    mean_residency = residency_sum / residency_count if residency_count > 0 else 0.0
    return (
        hits,
        misses,
        cold_misses,
        capacity_misses,
        total_evictions,
        unique_evicted_objects,
        mean_residency,
        per_request_hits,
        per_request_misses,
        per_request_cold_misses,
        per_request_prefill_misses,
        per_request_decode_misses,
        miss_flags,
        cold_miss_flags,
    )


def simulate_cache_policy(
    policy: str,
    access_buffer: ConditionAccessBuffer,
    request_offsets: np.ndarray,
    cache_budget: int,
) -> BudgetReplaySummary:
    if policy != POLICY_LRU:
        raise ValueError(f"unsupported cache policy: {policy}")

    (
        hits,
        misses,
        cold_misses,
        capacity_misses,
        total_evictions,
        unique_evicted_objects,
        mean_residency,
        per_request_hits,
        per_request_misses,
    ) = simulate_lru_accesses(
        access_ids=access_buffer.access_ids,
        request_offsets=request_offsets,
        max_object_id=access_buffer.max_object_id,
        capacity=cache_budget,
    )

    total_events = int(access_buffer.total_events)
    total_distinct_objects = int(cold_misses)
    if hits + misses != total_events:
        raise ValueError(
            "cache replay accounting mismatch: "
            f"condition={access_buffer.condition}, budget={cache_budget}, "
            f"hits={hits}, misses={misses}, total_events={total_events}"
        )
    if cold_misses + capacity_misses != misses:
        raise ValueError(
            "cache miss accounting mismatch: "
            f"condition={access_buffer.condition}, budget={cache_budget}, "
            f"cold={cold_misses}, capacity={capacity_misses}, misses={misses}"
        )
    if int(np.sum(per_request_hits)) != hits or int(np.sum(per_request_misses)) != misses:
        raise ValueError(
            "per-request cache accounting mismatch: "
            f"condition={access_buffer.condition}, budget={cache_budget}"
        )

    hit_rate = float(hits / total_events) if total_events else 0.0
    miss_rate = float(misses / total_events) if total_events else 0.0
    cold_miss_rate = float(cold_misses / total_events) if total_events else 0.0
    capacity_miss_rate = float(capacity_misses / total_events) if total_events else 0.0
    eviction_frequency = float(total_evictions / total_events) if total_events else 0.0

    return BudgetReplaySummary(
        condition=access_buffer.condition,
        cache_budget=int(cache_budget),
        total_events=total_events,
        hits=int(hits),
        misses=int(misses),
        hit_rate=hit_rate,
        miss_rate=miss_rate,
        cold_misses=int(cold_misses),
        cold_miss_rate=cold_miss_rate,
        capacity_misses=int(capacity_misses),
        capacity_miss_rate=capacity_miss_rate,
        total_distinct_objects=total_distinct_objects,
        total_evictions=int(total_evictions),
        eviction_frequency=eviction_frequency,
        unique_evicted_objects=int(unique_evicted_objects),
        mean_residency_if_available=(float(mean_residency) if total_evictions > 0 else None),
        per_request_hit_count_summary=numeric_summary(per_request_hits.tolist()),
        per_request_miss_count_summary=numeric_summary(per_request_misses.tolist()),
        per_request_hits=per_request_hits,
        per_request_misses=per_request_misses,
    )


def simulate_phase_replay(
    condition_buffer: ConditionAccessBuffer,
    phase_ids: np.ndarray,
    request_offsets: np.ndarray,
    cache_budget: int,
) -> PhaseReplaySummary:
    (
        hits,
        misses,
        cold_misses,
        capacity_misses,
        total_evictions,
        unique_evicted_objects,
        mean_residency,
        per_request_hits,
        per_request_misses,
        per_request_cold_misses,
        per_request_prefill_misses,
        per_request_decode_misses,
    ) = simulate_lru_accesses_with_phase(
        access_ids=condition_buffer.access_ids,
        phase_ids=phase_ids,
        request_offsets=request_offsets,
        max_object_id=condition_buffer.max_object_id,
        capacity=cache_budget,
    )

    total_events = int(condition_buffer.total_events)
    if hits + misses != total_events:
        raise ValueError(
            "cache replay accounting mismatch: "
            f"condition={condition_buffer.condition}, budget={cache_budget}, "
            f"hits={hits}, misses={misses}, total_events={total_events}"
        )
    if cold_misses + capacity_misses != misses:
        raise ValueError(
            "cache miss accounting mismatch: "
            f"condition={condition_buffer.condition}, budget={cache_budget}, "
            f"cold={cold_misses}, capacity={capacity_misses}, misses={misses}"
        )

    return PhaseReplaySummary(
        condition=condition_buffer.condition,
        cache_budget=int(cache_budget),
        total_events=total_events,
        hits=int(hits),
        misses=int(misses),
        cold_misses=int(cold_misses),
        capacity_misses=int(capacity_misses),
        total_evictions=int(total_evictions),
        unique_evicted_objects=int(unique_evicted_objects),
        mean_residency_if_available=(float(mean_residency) if total_evictions > 0 else None),
        per_request_hits=per_request_hits,
        per_request_misses=per_request_misses,
        per_request_cold_misses=per_request_cold_misses,
        per_request_prefill_misses=per_request_prefill_misses,
        per_request_decode_misses=per_request_decode_misses,
    )


def simulate_phase_replay_with_bitmaps(
    condition_buffer: ConditionAccessBuffer,
    phase_ids: np.ndarray,
    request_offsets: np.ndarray,
    cache_budget: int,
) -> PhaseReplayBitmapSummary:
    (
        hits,
        misses,
        cold_misses,
        capacity_misses,
        total_evictions,
        unique_evicted_objects,
        mean_residency,
        per_request_hits,
        per_request_misses,
        per_request_cold_misses,
        per_request_prefill_misses,
        per_request_decode_misses,
        miss_flags,
        cold_miss_flags,
    ) = simulate_lru_accesses_with_phase_and_bitmaps(
        access_ids=condition_buffer.access_ids,
        phase_ids=phase_ids,
        request_offsets=request_offsets,
        max_object_id=condition_buffer.max_object_id,
        capacity=cache_budget,
    )

    total_events = int(condition_buffer.total_events)
    if hits + misses != total_events:
        raise ValueError(
            "cache replay accounting mismatch: "
            f"condition={condition_buffer.condition}, budget={cache_budget}, "
            f"hits={hits}, misses={misses}, total_events={total_events}"
        )
    if cold_misses + capacity_misses != misses:
        raise ValueError(
            "cache miss accounting mismatch: "
            f"condition={condition_buffer.condition}, budget={cache_budget}, "
            f"cold={cold_misses}, capacity={capacity_misses}, misses={misses}"
        )
    if int(np.sum(miss_flags)) != misses:
        raise ValueError(
            "event-level miss bitmap mismatch: "
            f"condition={condition_buffer.condition}, budget={cache_budget}, "
            f"miss_bitmap_sum={int(np.sum(miss_flags))}, misses={misses}"
        )
    if int(np.sum(cold_miss_flags)) != cold_misses:
        raise ValueError(
            "event-level cold-miss bitmap mismatch: "
            f"condition={condition_buffer.condition}, budget={cache_budget}, "
            f"cold_bitmap_sum={int(np.sum(cold_miss_flags))}, cold_misses={cold_misses}"
        )

    return PhaseReplayBitmapSummary(
        condition=condition_buffer.condition,
        cache_budget=int(cache_budget),
        total_events=total_events,
        hits=int(hits),
        misses=int(misses),
        cold_misses=int(cold_misses),
        capacity_misses=int(capacity_misses),
        total_evictions=int(total_evictions),
        unique_evicted_objects=int(unique_evicted_objects),
        mean_residency_if_available=(float(mean_residency) if total_evictions > 0 else None),
        per_request_hits=per_request_hits,
        per_request_misses=per_request_misses,
        per_request_cold_misses=per_request_cold_misses,
        per_request_prefill_misses=per_request_prefill_misses,
        per_request_decode_misses=per_request_decode_misses,
        miss_flags=miss_flags,
        cold_miss_flags=cold_miss_flags,
    )
