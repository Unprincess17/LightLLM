import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import torch


class ExpertCacheSlotState:
    INVALID = "invalid"
    LOADING = "loading"
    READY = "ready"


@dataclass(frozen=True)
class ExpertCacheKey:
    projection: str
    adapter_idx: int
    layer_id: int
    expert_id: int


@dataclass
class MoEExpertCacheConfig:
    cache_budget_mb: int = 2048
    promote_min_hits: int = 2
    promote_window: int = 128
    max_promote_per_step: int = 8
    decay: float = 0.9
    # Default COLoRA miss policy: do not block request on promotion.
    miss_policy: str = "cpu_first"
    # Promotion queue soft cap. If None, derive from promote_window.
    queue_high_watermark: Optional[int] = None
    # Cooldown in schedule steps before same key can be queued again.
    promote_cooldown_steps: int = 4


@dataclass
class _CacheEntry:
    slot_id: int
    state: str
    utility: float
    last_access: float
    access_count: int
    last_queued_step: int = -1


@dataclass
class _ProjectionState:
    projection: str
    a_buffer: Optional[torch.Tensor]
    b_buffer: Optional[torch.Tensor]
    max_slots: int
    max_rank: int
    entries: Dict[ExpertCacheKey, _CacheEntry] = field(default_factory=dict)
    slot_to_key: Dict[int, ExpertCacheKey] = field(default_factory=dict)
    free_slots: List[int] = field(default_factory=list)
    promotion_queue: Deque[ExpertCacheKey] = field(default_factory=deque)
    queued: set = field(default_factory=set)


class MoEExpertCacheManager:
    """Expert-level GPU hot cache for COLoRA hybrid decode path."""

    def __init__(self, config: Optional[MoEExpertCacheConfig] = None):
        self.config = config or MoEExpertCacheConfig()
        self._lock = threading.Lock()
        self._states: Dict[str, _ProjectionState] = {}
        self._source_pools: Dict[str, object] = {}

        self._lookup_total = 0
        self._lookup_hits = 0
        self._dropped_promotions = 0
        self._dropped_promotions_by_queue = 0
        self._dropped_promotions_by_cooldown = 0
        self._dropped_promotions_by_no_slot = 0
        self._dropped_promotions_by_missing_source = 0
        self._schedule_step = 0

    def register_projection_pool(self, projection: str, pool) -> None:
        """Register source CPU pool and allocate GPU cache buffers for one projection."""
        self._source_pools[projection] = pool

        if pool is None:
            return

        budget_bytes_total = max(int(self.config.cache_budget_mb), 1) * 1024 * 1024
        budget_bytes_per_proj = max(budget_bytes_total // 3, 1)

        dtype = pool.key_buffer.dtype
        elem_size = torch.empty((), dtype=dtype).element_size()
        max_rank = int(pool.max_rank)
        slot_bytes = max_rank * (pool.key_buffer.shape[2] + pool.value_buffer.shape[2]) * elem_size
        max_slots = max(int(budget_bytes_per_proj // max(slot_bytes, 1)), 1)

        if not torch.cuda.is_available():
            max_slots = 0

        a_buffer = None
        b_buffer = None
        if max_slots > 0:
            a_buffer = torch.empty(
                (max_slots, max_rank, pool.key_buffer.shape[2]),
                dtype=dtype,
                device="cuda",
            )
            b_buffer = torch.empty(
                (max_slots, max_rank, pool.value_buffer.shape[2]),
                dtype=dtype,
                device="cuda",
            )

        self._states[projection] = _ProjectionState(
            projection=projection,
            a_buffer=a_buffer,
            b_buffer=b_buffer,
            max_slots=max_slots,
            max_rank=max_rank,
            free_slots=list(range(max_slots)),
        )

    def get_projection_buffers(self, projection: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        state = self._states.get(projection)
        if state is None:
            return None, None
        return state.a_buffer, state.b_buffer

    def get_promotion_queue_depth(self) -> int:
        return sum(len(state.promotion_queue) for state in self._states.values())

    def get_hit_rate(self) -> float:
        if self._lookup_total <= 0:
            return 0.0
        return float(self._lookup_hits) / float(self._lookup_total)

    def get_dropped_promotions(self) -> int:
        return self._dropped_promotions

    def get_promotion_drop_breakdown(self) -> Dict[str, int]:
        return {
            "total": int(self._dropped_promotions),
            "queue_high_watermark": int(self._dropped_promotions_by_queue),
            "cooldown": int(self._dropped_promotions_by_cooldown),
            "no_slot": int(self._dropped_promotions_by_no_slot),
            "missing_source": int(self._dropped_promotions_by_missing_source),
        }

    def _get_queue_high_watermark(self) -> int:
        queue_limit = max(int(self.config.promote_window), int(self.config.max_promote_per_step), 1)
        if self.config.queue_high_watermark is None:
            return queue_limit
        return max(int(self.config.queue_high_watermark), 1)

    def _promotion_priority(self, entry: _CacheEntry, now_ts: float) -> float:
        """Higher score means stronger hot-rebound tendency."""
        age = max(now_ts - entry.last_access, 0.0)
        rebound = 1.0 / (1.0 + age)
        return float(entry.utility) + 0.1 * float(entry.access_count) + 2.0 * rebound

    def lookup_many(self, keys: List[ExpertCacheKey]) -> Dict[ExpertCacheKey, int]:
        """Return READY slot IDs for keys currently cached on GPU."""
        ready: Dict[ExpertCacheKey, int] = {}

        with self._lock:
            self._lookup_total += len(keys)
            for key in keys:
                state = self._states.get(key.projection)
                if state is None:
                    continue

                entry = state.entries.get(key)
                if entry is None:
                    continue
                if entry.state != ExpertCacheSlotState.READY:
                    continue
                if entry.slot_id < 0:
                    continue

                now_ts = time.time()
                age = max(now_ts - entry.last_access, 0.0)
                entry.utility = entry.utility * self.config.decay + 1.0 / (1.0 + age)
                entry.last_access = now_ts
                ready[key] = entry.slot_id

            self._lookup_hits += len(ready)

        return ready

    def peek_ready_slots(self, keys: List[ExpertCacheKey]) -> Dict[ExpertCacheKey, int]:
        """Return READY slot IDs without mutating cache accounting or utility."""
        ready: Dict[ExpertCacheKey, int] = {}

        with self._lock:
            for key in keys:
                state = self._states.get(key.projection)
                if state is None:
                    continue

                entry = state.entries.get(key)
                if entry is None:
                    continue
                if entry.state != ExpertCacheSlotState.READY:
                    continue
                if entry.slot_id < 0:
                    continue

                ready[key] = entry.slot_id

        return ready

    def record_access(self, keys: List[ExpertCacheKey]) -> None:
        now_ts = time.time()
        with self._lock:
            for key in keys:
                state = self._states.get(key.projection)
                if state is None:
                    continue

                entry = state.entries.get(key)
                if entry is None:
                    state.entries[key] = _CacheEntry(
                        slot_id=-1,
                        state=ExpertCacheSlotState.INVALID,
                        utility=1.0,
                        last_access=now_ts,
                        access_count=1,
                        last_queued_step=-1,
                    )
                    continue

                age = max(now_ts - entry.last_access, 0.0)
                entry.utility = entry.utility * self.config.decay + 1.0 / (1.0 + age)
                entry.last_access = now_ts
                entry.access_count += 1

    def schedule_promotion(self, keys: List[ExpertCacheKey]) -> int:
        """Queue keys for async promotion if not already READY/LOADING.

        Candidates are scored and queued in descending hot-rebound priority.
        """
        queued = 0
        with self._lock:
            self._schedule_step += 1
            now_ts = time.time()
            cooldown_steps = max(int(self.config.promote_cooldown_steps), 0)
            candidates_by_projection: Dict[str, List[Tuple[float, ExpertCacheKey, _CacheEntry]]] = {}

            for key in keys:
                state = self._states.get(key.projection)
                if state is None:
                    continue

                entry = state.entries.get(key)
                if entry is None:
                    continue

                if entry.state == ExpertCacheSlotState.READY:
                    continue
                if key in state.queued:
                    continue
                if entry.access_count < self.config.promote_min_hits:
                    continue
                if cooldown_steps > 0 and entry.last_queued_step >= 0:
                    if (self._schedule_step - entry.last_queued_step) < cooldown_steps:
                        self._dropped_promotions += 1
                        self._dropped_promotions_by_cooldown += 1
                        continue

                priority = self._promotion_priority(entry, now_ts)
                candidates_by_projection.setdefault(key.projection, []).append((priority, key, entry))

            queue_hwm = self._get_queue_high_watermark()
            for projection, candidates in candidates_by_projection.items():
                state = self._states.get(projection)
                if state is None:
                    continue
                # Hot rebound first.
                candidates.sort(key=lambda x: x[0], reverse=True)
                for _, key, entry in candidates:
                    if len(state.promotion_queue) >= queue_hwm:
                        self._dropped_promotions += 1
                        self._dropped_promotions_by_queue += 1
                        continue

                    entry.state = ExpertCacheSlotState.LOADING
                    entry.last_queued_step = self._schedule_step
                    state.promotion_queue.append(key)
                    state.queued.add(key)
                    queued += 1

        return queued

    def evict_one(self, projection: str) -> Optional[int]:
        """Evict one READY slot using LFU+LRU composite score."""
        state = self._states.get(projection)
        if state is None:
            return None

        candidate_key = None
        candidate_score = None
        now_ts = time.time()

        for key, entry in state.entries.items():
            if entry.state != ExpertCacheSlotState.READY or entry.slot_id < 0:
                continue

            age = max(now_ts - entry.last_access, 0.0)
            # Lower score means lower utility and older usage -> evict first.
            score = entry.utility + 0.05 * entry.access_count - 0.001 * age
            if candidate_score is None or score < candidate_score:
                candidate_key = key
                candidate_score = score

        if candidate_key is None:
            return None

        entry = state.entries[candidate_key]
        slot_id = entry.slot_id
        del state.entries[candidate_key]
        state.slot_to_key.pop(slot_id, None)
        return slot_id

    def apply_completed_promotions(self) -> int:
        """Apply up to max_promote_per_step queued promotions."""
        promoted = 0
        with self._lock:
            for projection, state in self._states.items():
                src_pool = self._source_pools.get(projection)
                if src_pool is None or state.max_slots <= 0:
                    continue

                max_step = max(int(self.config.max_promote_per_step), 0)
                for _ in range(max_step):
                    if not state.promotion_queue:
                        break

                    key = state.promotion_queue.popleft()
                    state.queued.discard(key)

                    entry = state.entries.get(key)
                    if entry is None:
                        continue
                    if entry.state == ExpertCacheSlotState.READY and entry.slot_id >= 0:
                        continue

                    slot_id = entry.slot_id if entry.slot_id >= 0 else None
                    if slot_id is None:
                        if state.free_slots:
                            slot_id = state.free_slots.pop()
                        else:
                            slot_id = self.evict_one(projection)
                            if slot_id is None:
                                entry.state = ExpertCacheSlotState.INVALID
                                self._dropped_promotions += 1
                                self._dropped_promotions_by_no_slot += 1
                                continue

                    src_slot = self._get_source_slot(src_pool, key)
                    if src_slot is None:
                        entry.state = ExpertCacheSlotState.INVALID
                        self._dropped_promotions += 1
                        self._dropped_promotions_by_missing_source += 1
                        continue

                    rank = self._get_adapter_rank(src_pool, key.adapter_idx)
                    rank = max(min(rank, state.max_rank), 0)

                    assert state.a_buffer is not None and state.b_buffer is not None
                    if rank > 0:
                        state.a_buffer[slot_id, :rank].copy_(src_pool.key_buffer[src_slot, :rank], non_blocking=True)
                        state.b_buffer[slot_id, :rank].copy_(src_pool.value_buffer[src_slot, :rank], non_blocking=True)
                    if rank < state.max_rank:
                        state.a_buffer[slot_id, rank:].zero_()
                        state.b_buffer[slot_id, rank:].zero_()

                    entry.slot_id = slot_id
                    entry.state = ExpertCacheSlotState.READY
                    entry.last_access = time.time()
                    state.slot_to_key[slot_id] = key
                    promoted += 1

        return promoted

    def _get_adapter_rank(self, pool, adapter_idx: int) -> int:
        if hasattr(pool, "a_rank") and len(pool.a_rank) > adapter_idx:
            return int(pool.a_rank[adapter_idx].item())
        return int(pool.max_rank)

    def _get_source_slot(self, pool, key: ExpertCacheKey) -> Optional[int]:
        if key.adapter_idx < 0 or key.adapter_idx >= len(pool.a_start):
            return None

        layer_offset = key.layer_id
        if getattr(pool, "num_experts", 1) > 1:
            layer_offset = key.layer_id * pool.num_experts + key.expert_id

        src_slot = int(pool.a_start[key.adapter_idx].item()) + int(layer_offset)
        if src_slot < 0 or src_slot >= pool.key_buffer.shape[0]:
            return None
        return src_slot
