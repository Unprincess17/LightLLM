"""
S-LoRA Batched LoRA Dispatch for Qwen3-VL-MoE

This module provides S-LoRA style batched LoRA computation for:
- Vision-Language Adapter: vl.q_proj, vl.k_proj, vl.v_proj, vl.o_proj, vl.linear_fc1, vl.linear_fc2
- Attention: self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj
- MoE MLP: moe.gate_proj, moe.up_proj, moe.down_proj
- Language Model Head: moe.lm_head

Key Features:
- Mixed adapter batches: different requests in same batch can use different adapters
- req_bins tracking: maps each request to its adapter index
- dispatch_bgmv kernel: efficient batched LoRA computation

Debugging:
- Set LIGHTLLM_LOGGING=DEBUG to see detailed LoRA dispatch logs
"""
from __future__ import annotations
import torch
import os
import logging
import queue
import time
import threading
from collections import OrderedDict, deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Optional, Any, List, NamedTuple, Set, Tuple

from lightllm.server.core.objs.lora_compute_config import LoRAComputeConfig
from lightllm.server.lora.expert_cache import ExpertCacheKey, MoEExpertCacheManager
from lightllm.utils.nvtx_utils import NvtxAnnotate

# Configure logging using global env var
_LOG_LEVEL = os.environ.get("LIGHTLLM_LOGGING", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL, logging.INFO)
logger = logging.getLogger("lightllm.lora.dispatch")
logger.setLevel(_LOG_LEVEL)

MISS_POLICY_CPU_FIRST = "cpu_first"
MISS_POLICY_LOAD_THEN_RUN = "load_then_run"
MISS_POLICY_NO_CPU_PATH = "no_cpu_path"
MISS_POLICY_NO_DEFERRED_SYNC = "no_deferred_sync"
OVERLAP_MODE_FULL = "full"
OVERLAP_MODE_NO_OVERLAP = "no_overlap"

# Try to import dispatch_bgmv kernel, fall back to naive implementation
try:
    from lightllm._kernels.lora.bgmv import (
        dispatch_bgmv,
        batch_lora_get_qkv,
        batch_lora_get_o,
        batch_lora_get_mlp,
        batch_lora_get_vl,
        bgmv_debug_bounds_enabled,
    )
    BGMV_AVAILABLE = True
except ImportError:
    BGMV_AVAILABLE = False

    def bgmv_debug_bounds_enabled() -> bool:  # type: ignore[misc]
        return False

    # Fallback: naive per-request computation


def _bgmv_trace_kwargs(pool, projection: str, *, compact: bool = False) -> dict:
    """Forward ``projection`` / optional ``a_rank`` into BGMV entrypoints."""
    if not BGMV_AVAILABLE:
        return {}
    kw: dict = {"projection": projection}
    if compact or not bgmv_debug_bounds_enabled():
        return kw
    rank = getattr(pool, "a_rank", None)
    if rank is not None and rank.numel() > 0:
        kw["a_rank"] = rank
    return kw

# AVX CPU kernels: import symbols only — JIT compile is deferred (see _touch_*_avx_flags)
# so importing this module does not block on torch cpp_extension file locks.
AVX_AVAILABLE = None  # resolved on first _touch_lora_avx_flags(); bool if tests pre-set
MOE_AVX_AVAILABLE = None  # resolved on first _touch_moe_avx_flags(); bool if tests pre-set
_lora_cpu_kernel_import_ok = False
_moe_cpu_kernel_import_ok = False
_lora_avx_flags_touched = False
_moe_avx_flags_touched = False

try:
    from lightllm._kernels.lora.lora_cpu_kernel import (
        batch_lora_avx,
        lora_down_avx,
        lora_up_avx,
        ensure_kernel_loaded as _ensure_lora_cpu_kernel_loaded,
        is_available as _lora_cpu_is_available,
    )

    _lora_cpu_kernel_import_ok = True
except ImportError:
    pass

try:
    from lightllm._kernels.lora.moe_lora_cpu_kernel import (
        moe_batch_lora_avx,
        moe_batch_lora_gate_avx,
        moe_batch_lora_up_avx,
        moe_batch_lora_down_avx,
        ensure_kernel_loaded as _ensure_moe_cpu_kernel_loaded,
        is_available as _moe_cpu_is_available,
    )

    _moe_cpu_kernel_import_ok = True
except Exception:
    pass


def _touch_lora_avx_flags() -> None:
    """Resolve AVX_AVAILABLE once; honors pre-set bool (e.g. tests) without probing."""
    global AVX_AVAILABLE, _lora_avx_flags_touched

    if _lora_avx_flags_touched:
        return
    _lora_avx_flags_touched = True
    if AVX_AVAILABLE is not None:
        return
    if not _lora_cpu_kernel_import_ok:
        AVX_AVAILABLE = False
        return
    try:
        _ensure_lora_cpu_kernel_loaded()
        AVX_AVAILABLE = bool(_lora_cpu_is_available())
        if AVX_AVAILABLE:
            logger.info("AVX-512 BF16 CPU kernel available")
    except Exception:
        AVX_AVAILABLE = False


def _touch_moe_avx_flags() -> None:
    """Resolve MOE_AVX_AVAILABLE once; honors pre-set bool (e.g. tests) without probing."""
    global MOE_AVX_AVAILABLE, _moe_avx_flags_touched

    if _moe_avx_flags_touched:
        return
    _moe_avx_flags_touched = True
    if MOE_AVX_AVAILABLE is not None:
        return
    if not _moe_cpu_kernel_import_ok:
        MOE_AVX_AVAILABLE = False
        return
    try:
        _ensure_moe_cpu_kernel_loaded()
        MOE_AVX_AVAILABLE = bool(_moe_cpu_is_available())
        if MOE_AVX_AVAILABLE:
            logger.info("MoE-specific AVX-512 BF16 CPU kernel available")
    except Exception:
        MOE_AVX_AVAILABLE = False


def _colora_resolved_cpu_kernel_mode() -> str:
    """Return ``naive`` or ``avx`` (default) from ``COLORA_CPU_KERNEL_MODE``."""
    raw = os.environ.get("COLORA_CPU_KERNEL_MODE", "").strip().lower()
    if raw == "naive":
        return "naive"
    return "avx"


def _naive_moe_lora_gate(
    batch_input: torch.Tensor, A: torch.Tensor, scaling: float
) -> torch.Tensor:
    _ = scaling
    return torch.matmul(batch_input, A.transpose(0, 1))


def _naive_moe_lora_stage2(
    intermediate: torch.Tensor, B: torch.Tensor, scaling: float
) -> torch.Tensor:
    return torch.matmul(intermediate, B) * scaling


def is_moe_cpu_kernel_available() -> bool:
    """Expose MoE kernel readiness for backend startup checks."""
    _touch_moe_avx_flags()
    return bool(MOE_AVX_AVAILABLE)


class SpecJobKey(NamedTuple):
    layer_id: int
    decode_step_id: int
    op_kind: str
    adapter_bin: int
    expert_id: int
    row_group_sig: Tuple[Tuple[int, int], ...]


class SpecJobHandle:
    __slots__ = (
        "key",
        "future",
        "submitted_at",
        "retired_at",
        "bound_at",
        "retire_reason",
        "stale",
    )

    def __init__(self, key: SpecJobKey, future: Any, submitted_at: Optional[float] = None):
        self.key = key
        self.future = future
        self.submitted_at = time.perf_counter() if submitted_at is None else float(submitted_at)
        self.retired_at: Optional[float] = None
        self.bound_at: Optional[float] = None
        self.retire_reason: Optional[str] = None
        self.stale = False


class SpecSubmitOutcome(NamedTuple):
    status: str
    reason: str
    handle: Optional[SpecJobHandle]


class SpecBindOutcome(NamedTuple):
    status: str
    reason: str
    result: Optional[Any]


class JointObjectKey(NamedTuple):
    layer_id: int
    adapter_bin: int
    expert_id: int


class TemporalPrefetchJobKey(NamedTuple):
    layer_id: int
    decode_step_id: int
    adapter_bin: int
    expert_id: int


@dataclass(frozen=True)
class _JointAccessEvent:
    key: JointObjectKey
    decode_step_id: int


@dataclass
class _JointAccessState:
    last_decode_step_id: int = -1
    ema_interval_steps: Optional[float] = None
    interval_sample_count: int = 0

@dataclass
class ColoraCompletionTask:
    """Task for COLoRA request-level completion of one paused MoE layer."""
    req_obj: 'InferReq'
    layer_id: int
    hidden_after_attention: torch.Tensor
    partial_ffn_output: torch.Tensor
    cold_expert_ids: List[int]
    cold_routing_weights: List[float]
    adapter_bin: int
    layer_weight: Any


@dataclass
class PrefetchedProjectionWeights:
    a_buffer: torch.Tensor
    b_buffer: torch.Tensor
    rank: int
    scaling: float


@dataclass
class _HitIndex:
    """Projection-invariant hit-side indexing tensors (Phase 4 memoization).

    Reused across Gate/Up/Down of the same ``(batch_size, key_expert)`` when
    the set of ready hit adapters is identical. ``slot_ids`` is projection-
    specific and is *not* cached here; callers derive it from
    ``hit_unique_cpu_tensor`` via a cheap Python list walk.
    """

    hit_rows: torch.Tensor              # GPU LongTensor
    hit_pos: torch.Tensor               # GPU LongTensor, valid_pos[hit_rows]
    hit_bins: torch.Tensor              # GPU LongTensor, valid_bins[hit_rows]
    hit_unique_adapters: torch.Tensor   # GPU LongTensor
    hit_inverse: torch.Tensor           # GPU LongTensor
    hit_unique_cpu_tensor: torch.Tensor  # CPU LongTensor (Phase 1 sibling)
    hit_unique_cpu_list: List[int]


@dataclass
class MoEHybridSharedPrepareContext:
    """Projection-invariant hybrid-prepare metadata reused by Gate/Up/Down."""

    batch_size: int
    key_expert: int
    valid_pos: torch.Tensor
    valid_bins: torch.Tensor
    adapter_ids_cpu: List[int]
    # Phase 4: memoize per hit-adapter-set projection-invariant indexing.
    hit_index_cache: Dict[frozenset, _HitIndex] = field(default_factory=dict)
    # Phase 5: CPU mirror of valid_bins (only populated when hit_indexing == "cpu"),
    # amortized across Gate/Up/Down of the same expert call.
    valid_bins_cpu: Optional[torch.Tensor] = None


@dataclass
class _MoEHybridPhaseState:
    output: torch.Tensor
    manager: MoEExpertCacheManager
    layer_id: int
    buffer_layer_id: int
    projection: str
    key_expert: int
    decode_context: Optional[Tuple[int, int, int]]
    miss_policy: str = MISS_POLICY_CPU_FIRST
    shared_prepare_ctx: Optional[MoEHybridSharedPrepareContext] = None
    valid_pos: Optional[torch.Tensor] = None
    valid_bins: Optional[torch.Tensor] = None
    keys: List[ExpertCacheKey] = field(default_factory=list)
    miss_keys: List[ExpertCacheKey] = field(default_factory=list)
    ready_slots: Dict[ExpertCacheKey, int] = field(default_factory=dict)
    hit_mask: Optional[torch.Tensor] = None
    miss_mask: Optional[torch.Tensor] = None
    miss_pos: Optional[torch.Tensor] = None
    miss_bins: Optional[torch.Tensor] = None
    miss_input: Optional[torch.Tensor] = None
    miss_future: Optional[Any] = None
    # Phase 5 (CPU indexing): pre-computed CPU tensors stashed in BuildHitMissMasks
    # and consumed by _hybrid_run_gpu_hit to avoid GPU nonzero/unique syncs.
    hit_rows_cpu: Optional[torch.Tensor] = None
    hit_inverse_cpu: Optional[torch.Tensor] = None
    hit_unique_cpu_tensor: Optional[torch.Tensor] = None
    hit_unique_cpu_list: Optional[List[int]] = None
    async_overlap_used: bool = False
    gpu_compute_time: float = 0.0
    cpu_compute_time: float = 0.0
    cpu_queue_wait_time: float = 0.0
    cpu_queue_admit_wait_time: float = 0.0
    cpu_join_stall_time: float = 0.0
    d2h_bytes: float = 0.0
    h2d_bytes: float = 0.0
    fallback_degrade_count: int = 0
    cpu_async_submitted: int = 0
    cpu_inline_executed: int = 0
    moe_kernel_calls: int = 0
    moe_kernel_tokens: int = 0
    blocking_promotion_time: float = 0.0
    blocking_promotion_bytes: float = 0.0
    blocking_promotion_count: int = 0
    # P2 microbench component timings (seconds)
    pack_time: float = 0.0
    d2h_activation_time: float = 0.0
    h2d_residual_time: float = 0.0
    merge_time: float = 0.0
    admit_time: float = 0.0


@dataclass
class _MoEHybridMissTicket:
    state: _MoEHybridPhaseState
    input_tensor: torch.Tensor
    pool: Any


@dataclass
class _TemporalProjectionBuffers:
    a_buffer: torch.Tensor
    b_buffer: torch.Tensor
    max_rank: int


@dataclass
class _TemporalHotCacheSlot:
    generation: int = 0
    key: Optional[TemporalPrefetchJobKey] = None
    state: str = "invalid"
    refcount: int = 0
    ranks: Dict[str, int] = field(default_factory=dict)
    scalings: Dict[str, float] = field(default_factory=dict)
    used: bool = False


@dataclass
class TemporalPrefetchJobHandle:
    key: TemporalPrefetchJobKey
    future: Any
    slot_id: int
    generation: int
    submitted_at: float = field(default_factory=time.perf_counter)
    stale: bool = False
    used: bool = False


class JointAccessIntervalTracker:
    def __init__(self, ema_alpha: float, max_queue_size: int = 4096):
        alpha = float(ema_alpha)
        if alpha <= 0.0 or alpha > 1.0:
            alpha = 0.5
        self.ema_alpha = alpha
        self._events: "queue.Queue[_JointAccessEvent]" = queue.Queue(maxsize=max(int(max_queue_size), 1))
        self._lock = threading.Lock()
        self._states: Dict[JointObjectKey, _JointAccessState] = {}
        self._dropped_events = 0
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._run,
            name="colora_interval_tracker",
            daemon=True,
        )
        self._worker.start()

    def submit_access(self, key: JointObjectKey, decode_step_id: int) -> bool:
        try:
            self._events.put_nowait(_JointAccessEvent(key=key, decode_step_id=int(decode_step_id)))
            return True
        except queue.Full:
            with self._lock:
                self._dropped_events += 1
            return False

    def get_ema_interval_steps(self, key: JointObjectKey) -> Optional[float]:
        with self._lock:
            state = self._states.get(key)
            if state is None or state.interval_sample_count <= 0 or state.ema_interval_steps is None:
                return None
            return float(state.ema_interval_steps)

    def get_dropped_events(self) -> int:
        with self._lock:
            return int(self._dropped_events)

    def close(self) -> None:
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop_event.is_set() or not self._events.empty():
            try:
                event = self._events.get(timeout=0.05)
            except queue.Empty:
                continue
            self._apply_event(event)

    def _apply_event(self, event: _JointAccessEvent) -> None:
        with self._lock:
            state = self._states.get(event.key)
            if state is None:
                self._states[event.key] = _JointAccessState(last_decode_step_id=int(event.decode_step_id))
                return

            if state.last_decode_step_id < 0:
                state.last_decode_step_id = int(event.decode_step_id)
                return
            if int(event.decode_step_id) < int(state.last_decode_step_id):
                return

            gap = int(event.decode_step_id) - int(state.last_decode_step_id)
            if state.ema_interval_steps is None or state.interval_sample_count <= 0:
                state.ema_interval_steps = float(gap)
            else:
                state.ema_interval_steps = (
                    self.ema_alpha * float(gap)
                    + (1.0 - self.ema_alpha) * float(state.ema_interval_steps)
                )
            state.interval_sample_count += 1
            state.last_decode_step_id = int(event.decode_step_id)


class TemporalHotCacheHandle:
    def __init__(self, cache: "TemporalPrefetchHotCache", slot_id: int, generation: int):
        self._cache = cache
        self.slot_id = int(slot_id)
        self.generation = int(generation)
        self._released = False

    def get_projection(self, projection: str) -> Optional[PrefetchedProjectionWeights]:
        return self._cache.get_projection_weights(self.slot_id, self.generation, projection)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._cache.release(self.slot_id, self.generation)


class TemporalPrefetchHotCache:
    def __init__(self, slots: int, lora_mem_pool: Optional[Any]):
        self.slot_count = max(int(slots), 0)
        self.enabled = bool(self.slot_count > 0 and lora_mem_pool is not None)
        self._lock = threading.Lock()
        self._next_slot = 0
        self._slot_overwrite_count = 0
        self._slots: List[_TemporalHotCacheSlot] = [_TemporalHotCacheSlot() for _ in range(self.slot_count)]
        self._key_to_slot: Dict[TemporalPrefetchJobKey, Tuple[int, int]] = {}
        self._buffers: Dict[str, _TemporalProjectionBuffers] = {}

        if not self.enabled:
            return

        projection_pools = {
            "gate": getattr(lora_mem_pool, "moe_gate_pool", None),
            "up": getattr(lora_mem_pool, "moe_up_pool", None),
            "down": getattr(lora_mem_pool, "moe_down_pool", None),
        }
        if any(pool is None for pool in projection_pools.values()):
            self.enabled = False
            return

        for projection, pool in projection_pools.items():
            dtype = pool.key_buffer.dtype
            self._buffers[projection] = _TemporalProjectionBuffers(
                a_buffer=torch.empty(
                    (self.slot_count, int(pool.max_rank), int(pool.key_buffer.shape[2])),
                    dtype=dtype,
                    device="cpu",
                ),
                b_buffer=torch.empty(
                    (self.slot_count, int(pool.max_rank), int(pool.value_buffer.shape[2])),
                    dtype=dtype,
                    device="cpu",
                ),
                max_rank=int(pool.max_rank),
            )

    def reserve_slot(self, key: TemporalPrefetchJobKey) -> Optional[Tuple[int, int]]:
        if not self.enabled:
            return None

        with self._lock:
            for slot_offset in range(self.slot_count):
                slot_id = (self._next_slot + slot_offset) % self.slot_count
                slot = self._slots[slot_id]
                if slot.refcount > 0:
                    continue
                if slot.key is not None and slot.state in ("ready", "filling"):
                    self._slot_overwrite_count += 1
                self._invalidate_slot_locked(slot_id)
                slot.generation += 1
                slot.key = key
                slot.state = "filling"
                slot.refcount = 0
                slot.ranks = {}
                slot.scalings = {}
                slot.used = False
                self._key_to_slot[key] = (slot_id, slot.generation)
                self._next_slot = (slot_id + 1) % max(self.slot_count, 1)
                return int(slot_id), int(slot.generation)
        return None

    def publish_ready(
        self,
        slot_id: int,
        generation: int,
        key: TemporalPrefetchJobKey,
        ranks: Dict[str, int],
        scalings: Dict[str, float],
    ) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            if slot_id < 0 or slot_id >= self.slot_count:
                return False
            slot = self._slots[slot_id]
            if slot.generation != int(generation) or slot.key != key:
                return False
            slot.ranks = dict(ranks)
            slot.scalings = dict(scalings)
            slot.state = "ready"
            return True

    def invalidate_slot(self, slot_id: int, generation: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            if slot_id < 0 or slot_id >= self.slot_count:
                return
            slot = self._slots[slot_id]
            if slot.generation != int(generation) or slot.refcount > 0:
                return
            self._invalidate_slot_locked(slot_id)

    def invalidate_older_than(self, min_decode_step_id: int) -> int:
        if not self.enabled:
            return 0
        invalidated = 0
        with self._lock:
            for slot_id, slot in enumerate(self._slots):
                if slot.key is None or slot.refcount > 0:
                    continue
                if int(slot.key.decode_step_id) >= int(min_decode_step_id):
                    continue
                self._invalidate_slot_locked(slot_id)
                invalidated += 1
        return invalidated

    def has_key(self, key: TemporalPrefetchJobKey) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            slot_ref = self._key_to_slot.get(key)
            if slot_ref is None:
                return False
            slot_id, generation = slot_ref
            slot = self._slots[slot_id]
            return slot.generation == int(generation) and slot.key == key and slot.state in ("filling", "ready")

    def get_status(self, key: TemporalPrefetchJobKey) -> str:
        if not self.enabled:
            return "missing"
        with self._lock:
            slot_ref = self._key_to_slot.get(key)
            if slot_ref is None:
                return "missing"
            slot_id, generation = slot_ref
            slot = self._slots[slot_id]
            if slot.generation != int(generation) or slot.key != key:
                return "missing"
            return str(slot.state)

    def acquire(self, key: TemporalPrefetchJobKey) -> Optional[TemporalHotCacheHandle]:
        if not self.enabled:
            return None
        with self._lock:
            slot_ref = self._key_to_slot.get(key)
            if slot_ref is None:
                return None
            slot_id, generation = slot_ref
            slot = self._slots[slot_id]
            if slot.generation != int(generation) or slot.key != key or slot.state != "ready":
                return None
            slot.refcount += 1
            slot.used = True
            return TemporalHotCacheHandle(self, slot_id, generation)

    def get_projection_weights(
        self,
        slot_id: int,
        generation: int,
        projection: str,
    ) -> Optional[PrefetchedProjectionWeights]:
        with self._lock:
            if slot_id < 0 or slot_id >= self.slot_count:
                return None
            slot = self._slots[slot_id]
            if slot.generation != int(generation) or slot.state != "ready":
                return None
            rank = int(slot.ranks.get(projection, 0))
            if rank <= 0:
                return None
            scaling = float(slot.scalings.get(projection, 1.0))
            buffers = self._buffers.get(projection)
            if buffers is None:
                return None
            return PrefetchedProjectionWeights(
                a_buffer=buffers.a_buffer[slot_id],
                b_buffer=buffers.b_buffer[slot_id],
                rank=rank,
                scaling=scaling,
            )

    def release(self, slot_id: int, generation: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            if slot_id < 0 or slot_id >= self.slot_count:
                return
            slot = self._slots[slot_id]
            if slot.generation != int(generation):
                return
            slot.refcount = max(int(slot.refcount) - 1, 0)

    def get_slot_overwrite_count(self) -> int:
        with self._lock:
            return int(self._slot_overwrite_count)

    def count_unused_ready_for_step(self, decode_step_id: int) -> int:
        if not self.enabled:
            return 0
        with self._lock:
            count = 0
            for slot in self._slots:
                if (
                    slot.key is not None
                    and slot.state == "ready"
                    and int(slot.key.decode_step_id) == int(decode_step_id)
                    and not slot.used
                ):
                    count += 1
            return count

    def _invalidate_slot_locked(self, slot_id: int) -> None:
        slot = self._slots[slot_id]
        if slot.key is not None:
            self._key_to_slot.pop(slot.key, None)
        slot.key = None
        slot.state = "invalid"
        slot.ranks = {}
        slot.scalings = {}
        slot.used = False



class Qwen3VLMoELoRADispatcher:
    """
    S-LoRA Batched LoRA Dispatcher for Qwen3-VL-MoE.

    Supports mixed adapter batches where different requests use different adapters.
    The key insight is to use req_bins to track which adapter each request uses,
    then apply LoRA in a batched manner using the dispatch_bgmv kernel.

    Supported LoRA Targets:
    - Vision: vl_q_proj, vl_k_proj, vl_v_proj, vl_o_proj, vl_linear_fc1, vl_linear_fc2
    - Attention: attn_q_proj, attn_k_proj, attn_v_proj, attn_o_proj
    - MoE MLP: moe_gate_proj, moe_up_proj, moe_down_proj
    - LM Head: moe_lm_head
    """

    def __init__(
        self,
        num_layers: int,
        # Attention LoRA ranks (0 = disabled)
        q_lora_rank: int = 0,
        k_lora_rank: int = 0,
        v_lora_rank: int = 0,
        o_lora_rank: int = 0,
        # MoE MLP LoRA ranks
        gate_lora_rank: int = 0,
        up_lora_rank: int = 0,
        down_lora_rank: int = 0,
        # Vision adapter LoRA ranks
        vl_q_rank: int = 0,
        vl_k_rank: int = 0,
        vl_v_rank: int = 0,
        vl_o_rank: int = 0,
        vl_fc1_rank: int = 0,
        vl_fc2_rank: int = 0,
        # Common settings
        lora_alpha: float = 1.0,
        lora_compute_config: Optional[LoRAComputeConfig] = None,
        # COLoRA async CPU miss fallback controls
        colora_async_fallback: bool = True,
        colora_cpu_workers: int = 4,
        colora_cpu_queue_depth: int = 256,
        colora_cpu_batch_timeout_us: int = 50,
        colora_deferred_promotion_delta_steps: int = 4,
        colora_promotion_ema_alpha: float = 0.5,
        colora_overlap_mode: str = OVERLAP_MODE_FULL,
        colora_temporal_prefetch: bool = False,
        colora_temporal_hot_cache_slots: int = 64,
        # COLaRA request-level skip-and-reinsert
        colora_request_skip: bool = True,
        colora_max_continuations: int = 8,
        # Phase 5: hit-side indexing path selector ("gpu" or "cpu").
        colora_hit_indexing: str = "gpu",
        metric_client: Optional[Any] = None,
    ):
        self.num_layers = num_layers
        self.lora_compute_config = lora_compute_config or LoRAComputeConfig()
        self.lora_alpha = lora_alpha

        # Attention ranks
        self.q_lora_rank = q_lora_rank
        self.k_lora_rank = k_lora_rank
        self.v_lora_rank = v_lora_rank
        self.o_lora_rank = o_lora_rank

        # MLP ranks
        self.gate_lora_rank = gate_lora_rank
        self.up_lora_rank = up_lora_rank
        self.down_lora_rank = down_lora_rank

        # Vision adapter ranks
        self.vl_q_rank = vl_q_rank
        self.vl_k_rank = vl_k_rank
        self.vl_v_rank = vl_v_rank
        self.vl_o_rank = vl_o_rank
        self.vl_fc1_rank = vl_fc1_rank
        self.vl_fc2_rank = vl_fc2_rank

        # Calculate scaling factors
        self.q_scaling = lora_alpha / q_lora_rank if q_lora_rank > 0 else 1.0
        self.k_scaling = lora_alpha / k_lora_rank if k_lora_rank > 0 else 1.0
        self.v_scaling = lora_alpha / v_lora_rank if v_lora_rank > 0 else 1.0
        self.o_scaling = lora_alpha / o_lora_rank if o_lora_rank > 0 else 1.0

        # S-LoRA mode state
        self.lora_mem_pool = None
        self.req_bins = None
        self.use_batched_mode = False
        self.expert_cache_manager: Optional[MoEExpertCacheManager] = None

        # CPU Storage + GPU Compute scratchpad buffers (compact, per-batch)
        self.gpu_scratchpad_a = None
        self.gpu_scratchpad_b = None
        self.transfer_stream = None
        # Phase 2: lazily-built GPU mirror of pool.a_scaling, keyed by id(pool).
        # Value: (gpu_tensor, source_cpu_data_ptr, source_numel, source_device_id).
        # Invalidated when the CPU tensor is replaced (e.g. adapter load/unload).
        self._pool_scaling_gpu_cache: Dict[int, Tuple[torch.Tensor, int, int, int]] = {}
        # Phase 3: lazily-grown GPU scratch for BGMV's per-adapter index tensors.
        # temp_a_start is the identity-arange, temp_a_len is all-ones (int32).
        # Both only depend on active_count and grow monotonically.
        self._bgmv_temp_arange_gpu: Optional[torch.Tensor] = None
        self._bgmv_temp_ones_gpu: Optional[torch.Tensor] = None
        self.colora_async_fallback = bool(colora_async_fallback)
        self.colora_cpu_workers = max(int(colora_cpu_workers), 1)
        self.colora_cpu_queue_depth = max(int(colora_cpu_queue_depth), 1)
        self.colora_cpu_batch_timeout_us = max(int(colora_cpu_batch_timeout_us), 0)
        self.colora_overlap_mode = str(colora_overlap_mode)
        # COLaRA request-level skip-and-reinsert
        self.colora_request_skip = bool(colora_request_skip)
        self.colora_max_continuations = max(int(colora_max_continuations), 1)
        # Phase 5: hit-indexing path ("gpu" or "cpu"). Default stays "gpu" until
        # the regression test confirms byte-identical outputs in both paths.
        _hit_indexing = str(colora_hit_indexing).lower()
        if _hit_indexing not in ("gpu", "cpu"):
            logger.warning(
                "[COLoRA] invalid colora_hit_indexing=%r, falling back to 'gpu'", colora_hit_indexing
            )
            _hit_indexing = "gpu"
        self.colora_hit_indexing = _hit_indexing
        self.metric_client = metric_client
        self._cpu_executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_executor: Optional[ThreadPoolExecutor] = None
        self._cpu_queue_lock = threading.Lock()
        self._cpu_inflight = 0
        self._cpu_group_plan_cache: "OrderedDict[Tuple[int, bytes], Tuple[Tuple[int, Tuple[int, ...]], ...]]" = OrderedDict()
        self._cpu_group_plan_cache_cap = 256
        self._thread_local = threading.local()
        self._deferred_promotion_delta_steps = max(int(colora_deferred_promotion_delta_steps), 0)
        self._promotion_interval_tracker = JointAccessIntervalTracker(
            ema_alpha=float(colora_promotion_ema_alpha),
            max_queue_size=max(self.colora_cpu_queue_depth * 4, 1024),
        )
        self._temporal_prefetch_enabled = bool(colora_temporal_prefetch)
        self._temporal_hot_cache_slots = max(int(colora_temporal_hot_cache_slots), 0)
        self._temporal_hot_cache: Optional[TemporalPrefetchHotCache] = None
        self._prefetch_lock = threading.Lock()
        self._prefetch_current_step_id: Optional[int] = None
        self._prefetch_active_jobs: Dict[TemporalPrefetchJobKey, TemporalPrefetchJobHandle] = {}
        self._prefetch_active_job_keys_by_step: Dict[int, Set[TemporalPrefetchJobKey]] = {}
        self._spec_lock = threading.Lock()
        self._spec_current_step_id: Optional[int] = None
        self._spec_active_jobs: Dict[SpecJobKey, SpecJobHandle] = {}
        self._spec_active_job_keys_by_step: Dict[int, Set[SpecJobKey]] = {}
        self._spec_retired_jobs: Deque[SpecJobHandle] = deque()
        self._spec_reserved_job_keys: Set[SpecJobKey] = set()
        self._pending_background_stats = {
            "promotion_admitted": 0,
            "promotion_reject_delta": 0,
            "promotion_reject_no_ema": 0,
            "tracker_queue_drop": 0,
            "prefetch_submitted": 0,
            "prefetch_stale": 0,
            "prefetch_false_positives": 0,
        }

        # Last-call COLoRA stats for decode observability.
        self._last_colora_stats = {
            "colora_hit_tokens": 0,
            "colora_miss_tokens": 0,
            "promotion_queue_depth": 0,
            "cache_hit_rate": 0.0,
            "cache_capacity_slots": 0,
            "cache_resident_slots": 0,
            "cache_free_slots": 0,
            "cache_evictions_total": 0,
            "gate_capacity_slots": 0,
            "gate_resident_slots": 0,
            "gate_free_slots": 0,
            "gate_evictions_total": 0,
            "up_capacity_slots": 0,
            "up_resident_slots": 0,
            "up_free_slots": 0,
            "up_evictions_total": 0,
            "down_capacity_slots": 0,
            "down_resident_slots": 0,
            "down_free_slots": 0,
            "down_evictions_total": 0,
            "cpu_compute_time": 0.0,
            "gpu_compute_time": 0.0,
            "cpu_queue_wait_time": 0.0,
            "cpu_queue_admit_wait_time": 0.0,
            "cpu_join_stall_time": 0.0,
            "d2h_bytes": 0.0,
            "h2d_bytes": 0.0,
            "weight_h2d_bytes": 0.0,
            "weight_h2d_time": 0.0,
            "overlap_ratio": 0.0,
            "fallback_degrade_count": 0,
            "cpu_async_submitted": 0,
            "cpu_inline_executed": 0,
            "blocking_promotion_count": 0,
            "miss_policy": MISS_POLICY_CPU_FIRST,
            "overlap_mode": self.colora_overlap_mode,
            "cpu_queue_depth": 0,
            "promotion_drop_total": 0,
            "promotion_drop_queue_high_watermark": 0,
            "promotion_drop_cooldown": 0,
            "promotion_admitted": 0,
            "promotion_reject_delta": 0,
            "promotion_reject_no_ema": 0,
            "tracker_queue_drop": 0,
            "moe_kernel_calls": 0,
            "moe_kernel_tokens": 0,
            "prefetch_submitted": 0,
            "prefetch_ready_hits": 0,
            "prefetch_not_ready": 0,
            "prefetch_stale": 0,
            "prefetch_false_positives": 0,
            "prefetch_slot_overwrite": 0,
            "pack_time": 0.0,
            "d2h_activation_time": 0.0,
            "h2d_residual_time": 0.0,
            "merge_time": 0.0,
            "admit_time": 0.0,
        }

        # Check if any LoRA is enabled
        self.has_any_lora = any([
            q_lora_rank > 0, k_lora_rank > 0, v_lora_rank > 0, o_lora_rank > 0,
            gate_lora_rank > 0, up_lora_rank > 0, down_lora_rank > 0,
            vl_q_rank > 0, vl_k_rank > 0, vl_v_rank > 0, vl_o_rank > 0,
            vl_fc1_rank > 0, vl_fc2_rank > 0
        ])

        moe_mode = self.lora_compute_config.get_compute_device("moe")
        if moe_mode in ("cpu", "hybrid"):
            self._require_moe_cpu_kernel(mode=f"moe_compute={moe_mode}")

        # Completion queue for COLaRA request-level continuation
        # CPU worker threads push completed tasks here, main infer loop collects
        from queue import SimpleQueue
        self.colora_completion_queue: SimpleQueue[ColoraCompletionTask] = SimpleQueue()

    def init_batched_mode(
        self,
        lora_mem_pool,
        req_bins: torch.Tensor,
        expert_cache_manager: Optional[MoEExpertCacheManager] = None,
    ):
        """
        Initialize batched mode with LoRA memory pool and req_bins.

        Args:
            lora_mem_pool: LoRAMemPool containing all adapter weights
            req_bins: Tensor mapping request index -> adapter index [batch_size]

        Debug:
            Logs batched mode initialization with batch size and adapter info
        """
        self.lora_mem_pool = lora_mem_pool
        self.req_bins = req_bins
        self.use_batched_mode = True
        self.expert_cache_manager = expert_cache_manager
        self._reset_colora_stats()
        self._reset_pending_background_stats()
        if self._temporal_prefetch_enabled and self._should_use_hybrid_moe_compute():
            self._temporal_hot_cache = TemporalPrefetchHotCache(
                slots=self._temporal_hot_cache_slots,
                lora_mem_pool=lora_mem_pool,
            )
        else:
            self._temporal_hot_cache = None

        # batch_size = req_bins.shape[0] if req_bins is not None else 0
        # unique_adapters = len(torch.unique(req_bins)) if req_bins is not None else 0

    def use_single_adapter_mode(self):
        """Switch back to single adapter mode (original behavior)."""
        logger.debug(f"[LoRA Dispatch] Switching to single adapter mode")
        self.cleanup_speculation_state()
        self.cleanup_temporal_prefetch_state()
        self.use_batched_mode = False
        self.lora_mem_pool = None
        self.req_bins = None
        self.expert_cache_manager = None
        self._temporal_hot_cache = None
        self._reset_colora_stats()
        self._reset_pending_background_stats()

    # =====================================================================
    # S-LoRA Batched Methods (FIXED)
    # =====================================================================

    def _get_output_buffer(self, input_tensor, pool):
        """Helper to create output buffer with correct dimension (Crucial for GQA/MLP)"""
        # Use actual buffer dimension to ensure correct shape for copy operations
        return torch.zeros(
            input_tensor.shape[0],
            pool.value_buffer.shape[2],  # Use actual B buffer dimension
            dtype=input_tensor.dtype,
            device=input_tensor.device
        )

    def _should_use_cpu_compute(self, component: str) -> bool:
        """Check if a component should use CPU computation."""
        if self.lora_compute_config is None:
            return False
        return self.lora_compute_config.should_compute_on_cpu(component)

    def _should_use_cpu_storage(self, component: str) -> bool:
        """Check if component uses CPU storage (requires transfer to GPU for compute)."""
        if self.lora_compute_config is None:
            return False
        return self.lora_compute_config.get_storage_device(component) == "cpu"

    def _should_use_hybrid_moe_compute(self) -> bool:
        """Check whether COLoRA hybrid mode is enabled for MoE."""
        if self.lora_compute_config is None:
            return False
        return self.lora_compute_config.should_compute_hybrid("moe")

    def _require_moe_cpu_kernel(self, mode: str) -> None:
        _touch_moe_avx_flags()
        if not MOE_AVX_AVAILABLE:
            raise RuntimeError(
                f"MoE-specific CPU kernel is required for strict {mode} MoE path "
                "(moe_compute=cpu|hybrid), but it is unavailable."
            )

    def _select_moe_stage2_kernel(self, projection: str):
        proj = projection.lower()
        if proj in ("gate", "up"):
            return moe_batch_lora_up_avx
        if proj == "down":
            return moe_batch_lora_down_avx
        raise ValueError(f"Unsupported MoE projection '{projection}', expected gate|up|down")

    def _reset_pending_background_stats(self) -> None:
        self._pending_background_stats = {
            "promotion_admitted": 0,
            "promotion_reject_delta": 0,
            "promotion_reject_no_ema": 0,
            "tracker_queue_drop": 0,
            "prefetch_submitted": 0,
            "prefetch_stale": 0,
            "prefetch_false_positives": 0,
        }

    def _record_background_stat(self, key: str, value: int = 1) -> None:
        if key not in self._pending_background_stats:
            return
        self._pending_background_stats[key] += int(value)

    def _flush_pending_background_stats(self) -> None:
        for key, value in self._pending_background_stats.items():
            if key in self._last_colora_stats:
                self._last_colora_stats[key] += int(value)
        self._reset_pending_background_stats()

    def _get_or_create_prefetch_executor(self) -> ThreadPoolExecutor:
        if self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="colora_prefetch",
            )
        return self._prefetch_executor

    def begin_decode_joint_context(self, decode_step_id: int, layer_id: int, expert_id: int) -> None:
        self._thread_local.decode_joint_context = (
            int(decode_step_id),
            int(layer_id),
            int(expert_id),
        )

    def end_decode_joint_context(self) -> None:
        if hasattr(self._thread_local, "decode_joint_context"):
            del self._thread_local.decode_joint_context

    def _get_current_decode_joint_context(self) -> Optional[Tuple[int, int, int]]:
        return getattr(self._thread_local, "decode_joint_context", None)

    def _build_temporal_prefetch_job_key(
        self,
        decode_step_id: int,
        layer_id: int,
        adapter_bin: int,
        expert_id: int,
    ) -> TemporalPrefetchJobKey:
        return TemporalPrefetchJobKey(
            layer_id=int(layer_id),
            decode_step_id=int(decode_step_id),
            adapter_bin=int(adapter_bin),
            expert_id=int(expert_id),
        )

    def _build_joint_object_key(self, layer_id: int, adapter_bin: int, expert_id: int) -> JointObjectKey:
        return JointObjectKey(
            layer_id=int(layer_id),
            adapter_bin=int(adapter_bin),
            expert_id=int(expert_id),
        )

    def _get_pool_rank_and_scaling(self, pool, adapter_idx: int) -> Tuple[int, float]:
        if hasattr(pool, "a_rank") and len(pool.a_rank) > adapter_idx:
            rank = int(pool.a_rank[adapter_idx].item())
        else:
            rank = int(pool.max_rank)
        scaling = float(pool.a_scaling[adapter_idx].item())
        return rank, scaling

    def _get_pool_source_slot(self, pool, layer_id: int, expert_id: int, adapter_idx: int) -> Optional[int]:
        if adapter_idx < 0 or adapter_idx >= len(pool.a_start):
            return None

        layer_offset = int(layer_id)
        if getattr(pool, "num_experts", 1) > 1:
            layer_offset = int(layer_id) * int(pool.num_experts) + int(expert_id)

        src_slot = int(pool.a_start[adapter_idx].item()) + int(layer_offset)
        if src_slot < 0 or src_slot >= int(pool.key_buffer.shape[0]):
            return None
        return src_slot

    def _is_joint_projection_ready(self, projection: str, layer_id: int, adapter_bin: int, expert_id: int) -> bool:
        manager = self.expert_cache_manager
        if manager is None:
            return False
        key = ExpertCacheKey(
            projection=str(projection),
            adapter_idx=int(adapter_bin),
            layer_id=int(layer_id),
            expert_id=int(expert_id),
        )
        ready_slots = manager.peek_ready_slots([key])
        return key in ready_slots

    def _is_joint_ready(self, layer_id: int, adapter_bin: int, expert_id: int) -> bool:
        return (
            self._is_joint_projection_ready("gate", layer_id, adapter_bin, expert_id)
            and self._is_joint_projection_ready("up", layer_id, adapter_bin, expert_id)
            and self._is_joint_projection_ready("down", layer_id, adapter_bin, expert_id)
        )

    def note_decode_joint_access(
        self,
        decode_step_id: int,
        layer_id: int,
        expert_id: int,
        adapter_bins: List[int],
    ) -> None:
        manager = self.expert_cache_manager
        if manager is None or not self._should_use_hybrid_moe_compute():
            return
        miss_policy = str(getattr(getattr(manager, "config", None), "miss_policy", MISS_POLICY_CPU_FIRST))

        unique_adapter_bins = sorted({int(adapter_bin) for adapter_bin in adapter_bins if int(adapter_bin) >= 0})
        for adapter_bin in unique_adapter_bins:
            joint_key = self._build_joint_object_key(layer_id, adapter_bin, expert_id)
            promotion_keys = [
                ExpertCacheKey("gate", adapter_bin, int(layer_id), int(expert_id)),
                ExpertCacheKey("up", adapter_bin, int(layer_id), int(expert_id)),
                ExpertCacheKey("down", adapter_bin, int(layer_id), int(expert_id)),
            ]
            manager.record_access(promotion_keys)
            if miss_policy in (MISS_POLICY_NO_CPU_PATH, MISS_POLICY_NO_DEFERRED_SYNC):
                continue
            if not self._promotion_interval_tracker.submit_access(joint_key, int(decode_step_id)):
                self._record_background_stat("tracker_queue_drop", 1)
            if self._deferred_promotion_delta_steps <= 0:
                continue
            if self._is_joint_ready(layer_id, adapter_bin, expert_id):
                continue
            ema_interval = self._promotion_interval_tracker.get_ema_interval_steps(joint_key)
            if ema_interval is None:
                self._record_background_stat("promotion_reject_no_ema", 1)
                continue
            if float(ema_interval) > float(self._deferred_promotion_delta_steps):
                self._record_background_stat("promotion_reject_delta", 1)
                continue
            queued = manager.schedule_promotion(promotion_keys)
            if queued > 0:
                self._record_background_stat("promotion_admitted", 1)

    def begin_temporal_prefetch_step(self, decode_step_id: int) -> Dict[str, int]:
        step_id = int(decode_step_id)
        retired = 0
        with self._prefetch_lock:
            stale_steps = [active_step for active_step in self._prefetch_active_job_keys_by_step.keys() if active_step != step_id]
            for active_step in stale_steps:
                retired += self._retire_prefetch_step_locked(active_step, mark_stale=True)
            self._prefetch_current_step_id = step_id
            self._prefetch_active_job_keys_by_step.setdefault(step_id, set())
        stale_from_cache = 0
        if self._temporal_hot_cache is not None:
            stale_from_cache = self._temporal_hot_cache.invalidate_older_than(step_id)
        if stale_from_cache > 0:
            self._record_background_stat("prefetch_stale", stale_from_cache)
        return {"retired_active": int(retired), "retired_ready": int(stale_from_cache)}

    def finalize_temporal_prefetch_step_nonblocking(self, decode_step_id: int) -> Dict[str, int]:
        step_id = int(decode_step_id)
        retired = 0
        false_positives = 0
        with self._prefetch_lock:
            retired = self._retire_prefetch_step_locked(step_id, mark_stale=True)
            if self._prefetch_current_step_id == step_id:
                self._prefetch_current_step_id = None
        if self._temporal_hot_cache is not None:
            false_positives = self._temporal_hot_cache.count_unused_ready_for_step(step_id)
        if retired > 0:
            self._record_background_stat("prefetch_stale", retired)
        if false_positives > 0:
            self._record_background_stat("prefetch_false_positives", false_positives)
        return {"retired_active": int(retired), "unused_ready": int(false_positives)}

    def cleanup_temporal_prefetch_state(self) -> Dict[str, int]:
        retired = 0
        with self._prefetch_lock:
            active_steps = tuple(self._prefetch_active_job_keys_by_step.keys())
            for step_id in active_steps:
                retired += self._retire_prefetch_step_locked(step_id, mark_stale=True)
            self._prefetch_current_step_id = None
            self._prefetch_active_jobs.clear()
            self._prefetch_active_job_keys_by_step.clear()
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=False, cancel_futures=False)
            self._prefetch_executor = None
        if self._temporal_hot_cache is not None:
            retired += self._temporal_hot_cache.invalidate_older_than(1 << 30)
        return {"retired": int(retired)}

    def maybe_submit_temporal_prefetch_job(
        self,
        decode_step_id: int,
        layer_id: int,
        adapter_bin: int,
        expert_id: int,
    ) -> bool:
        if (
            not self._temporal_prefetch_enabled
            or not self._should_use_hybrid_moe_compute()
            or self._temporal_hot_cache is None
            or self.lora_mem_pool is None
        ):
            return False
        if int(adapter_bin) < 0 or int(expert_id) < 0:
            return False
        if self._is_joint_ready(layer_id, adapter_bin, expert_id):
            return False

        key = self._build_temporal_prefetch_job_key(decode_step_id, layer_id, adapter_bin, expert_id)
        with self._prefetch_lock:
            if self._prefetch_current_step_id is None or int(self._prefetch_current_step_id) != int(decode_step_id):
                return False
            if key in self._prefetch_active_jobs or self._temporal_hot_cache.has_key(key):
                return False

        slot_ref = self._temporal_hot_cache.reserve_slot(key)
        if slot_ref is None:
            return False
        slot_id, generation = slot_ref

        def _worker() -> Tuple[Dict[str, int], Dict[str, float]]:
            pools = {
                "gate": self.lora_mem_pool.moe_gate_pool,
                "up": self.lora_mem_pool.moe_up_pool,
                "down": self.lora_mem_pool.moe_down_pool,
            }
            ranks: Dict[str, int] = {}
            scalings: Dict[str, float] = {}
            for projection, pool in pools.items():
                src_slot = self._get_pool_source_slot(pool, layer_id, expert_id, adapter_bin)
                if src_slot is None:
                    raise ValueError(f"missing source slot for {projection} layer={layer_id} expert={expert_id} adapter={adapter_bin}")
                rank, scaling = self._get_pool_rank_and_scaling(pool, adapter_bin)
                ranks[projection] = int(rank)
                scalings[projection] = float(scaling)
                buffers = self._temporal_hot_cache._buffers[projection]
                if rank > 0:
                    buffers.a_buffer[slot_id, :rank].copy_(pool.key_buffer[src_slot, :rank], non_blocking=False)
                    buffers.b_buffer[slot_id, :rank].copy_(pool.value_buffer[src_slot, :rank], non_blocking=False)
                if rank < buffers.max_rank:
                    buffers.a_buffer[slot_id, rank:].zero_()
                    buffers.b_buffer[slot_id, rank:].zero_()
            return ranks, scalings

        future = self._get_or_create_prefetch_executor().submit(_worker)
        handle = TemporalPrefetchJobHandle(
            key=key,
            future=future,
            slot_id=int(slot_id),
            generation=int(generation),
        )

        def _on_complete(done_future) -> None:
            try:
                ranks, scalings = done_future.result()
            except Exception:
                self._temporal_hot_cache.invalidate_slot(handle.slot_id, handle.generation)
                with self._prefetch_lock:
                    active = self._prefetch_active_jobs.pop(key, None)
                    if active is not None:
                        step_keys = self._prefetch_active_job_keys_by_step.get(int(key.decode_step_id))
                        if step_keys is not None:
                            step_keys.discard(key)
                            if not step_keys:
                                self._prefetch_active_job_keys_by_step.pop(int(key.decode_step_id), None)
                return

            publish = False
            with self._prefetch_lock:
                active = self._prefetch_active_jobs.get(key)
                publish = active is not None and not active.stale
                if active is not None:
                    self._prefetch_active_jobs.pop(key, None)
                    step_keys = self._prefetch_active_job_keys_by_step.get(int(key.decode_step_id))
                    if step_keys is not None:
                        step_keys.discard(key)
                        if not step_keys:
                            self._prefetch_active_job_keys_by_step.pop(int(key.decode_step_id), None)
            if publish:
                self._temporal_hot_cache.publish_ready(handle.slot_id, handle.generation, key, ranks, scalings)
            else:
                self._temporal_hot_cache.invalidate_slot(handle.slot_id, handle.generation)

        with self._prefetch_lock:
            self._prefetch_active_jobs[key] = handle
            self._prefetch_active_job_keys_by_step.setdefault(int(key.decode_step_id), set()).add(key)
        future.add_done_callback(_on_complete)
        self._record_background_stat("prefetch_submitted", 1)
        return True

    def _retire_prefetch_step_locked(self, decode_step_id: int, mark_stale: bool) -> int:
        retired = 0
        keys = tuple(self._prefetch_active_job_keys_by_step.get(int(decode_step_id), ()))
        for key in keys:
            handle = self._prefetch_active_jobs.get(key)
            if handle is None:
                continue
            if mark_stale:
                handle.stale = True
            self._prefetch_active_jobs.pop(key, None)
            retired += 1
        if int(decode_step_id) in self._prefetch_active_job_keys_by_step:
            self._prefetch_active_job_keys_by_step.pop(int(decode_step_id), None)
        return retired

    def _maybe_acquire_prefetched_projection(
        self,
        projection: str,
        adapter_idx: int,
        temporal_prefetch_context: Optional[Tuple[int, int, int]],
    ) -> Tuple[Optional[PrefetchedProjectionWeights], Optional[TemporalHotCacheHandle], str]:
        if self._temporal_hot_cache is None or temporal_prefetch_context is None:
            return None, None, "disabled"
        decode_step_id, layer_id, expert_id = temporal_prefetch_context
        key = self._build_temporal_prefetch_job_key(
            decode_step_id=decode_step_id,
            layer_id=layer_id,
            adapter_bin=adapter_idx,
            expert_id=expert_id,
        )
        handle = self._temporal_hot_cache.acquire(key)
        if handle is not None:
            weights = handle.get_projection(projection)
            if weights is not None:
                return weights, handle, "ready"
            handle.release()
            return None, None, "missing"
        status = self._temporal_hot_cache.get_status(key)
        return None, None, status

    def _strict_moe_cpu_batch_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        pool,
        req_bins: Optional[torch.Tensor],
        projection: str,
        adapter_group_plan: Optional[Tuple[Tuple[int, Tuple[int, ...]], ...]] = None,
        return_to_original_device: bool = True,
        temporal_prefetch_context: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """Strict MoE CPU fallback: AVX MoE kernels or PyTorch reference (``naive``).
        
        Returns: (output, kernel_calls, kernel_tokens)
        
        Note: Timing is tracked internally in self._last_colora_stats:
            - d2h_activation_time: input D2H transfer (if input was on CUDA)
            - h2d_residual_time: output H2D transfer (if returning to CUDA)
            Both happen OUTSIDE cpu_compute_time measurement in the caller.
        """
        cpu_kernel_mode = _colora_resolved_cpu_kernel_mode()
        if cpu_kernel_mode != "naive":
            self._require_moe_cpu_kernel(mode=projection)

        if req_bins is None:
            req_bins = self.req_bins
        if req_bins is None:
            return self._get_output_buffer(input_tensor, pool), 0, 0

        proj_lower = projection.lower()
        nvtx_root = f"MoE_CPUStrictMoELoRA/{proj_lower}/L{int(layer_id)}/kern_{cpu_kernel_mode}"
        with NvtxAnnotate(nvtx_root):
            original_device = input_tensor.device
            original_dtype = input_tensor.dtype

            with NvtxAnnotate(f"{nvtx_root}/HostTensorPrep"):
                _d2h_t0 = time.perf_counter()
                compute_input = input_tensor
                if compute_input.device.type != "cpu":
                    compute_input = compute_input.to(device="cpu", non_blocking=True)
                # For naive mode: convert directly to float32 upfront (faster CPU compute)
                if cpu_kernel_mode == "naive":
                    if compute_input.dtype != torch.float32:
                        compute_input = compute_input.to(dtype=torch.float32)
                else:
                    if compute_input.dtype != torch.bfloat16:
                        compute_input = compute_input.to(dtype=torch.bfloat16)
                if input_tensor.device.type != "cpu":
                    self._last_colora_stats["d2h_activation_time"] += max(time.perf_counter() - _d2h_t0, 0.0)

            batch_size = compute_input.shape[0]
            output_dim = pool.value_buffer.shape[2]
            # Use float32 for naive mode output (faster matmul, converts once at H2D)
            output_dtype = torch.float32 if cpu_kernel_mode == "naive" else torch.bfloat16
            output = torch.zeros(batch_size, output_dim, dtype=output_dtype, device="cpu")

            if len(req_bins) > batch_size:
                req_bins = req_bins[:batch_size]

            if adapter_group_plan is not None:
                adapter_groups = adapter_group_plan
            else:
                adapter_to_indices: Dict[int, List[int]] = {}
                for i, bin_idx in enumerate(req_bins):
                    idx = int(bin_idx.item())
                    adapter_to_indices.setdefault(idx, []).append(i)
                adapter_groups = tuple((adapter_idx, tuple(indices)) for adapter_idx, indices in adapter_to_indices.items())

            if cpu_kernel_mode == "naive":
                gate_kernel = _naive_moe_lora_gate
                if proj_lower in ("gate", "up", "down"):
                    stage2_kernel = _naive_moe_lora_stage2
                else:
                    raise ValueError(
                        f"Unsupported MoE projection '{projection}', expected gate|up|down"
                    )
            else:
                gate_kernel = moe_batch_lora_gate_avx
                stage2_kernel = self._select_moe_stage2_kernel(projection)
            kernel_calls = 0
            kernel_tokens = 0

            for adapter_idx, req_indices_tuple in adapter_groups:
                if adapter_idx < 0:
                    continue
                req_indices = list(req_indices_tuple)
                if len(req_indices) == 0:
                    continue

                hot_weights = None
                hot_handle = None
                hot_status = "disabled"
                if temporal_prefetch_context is not None:
                    hot_weights, hot_handle, hot_status = self._maybe_acquire_prefetched_projection(
                        projection=projection,
                        adapter_idx=int(adapter_idx),
                        temporal_prefetch_context=temporal_prefetch_context,
                    )

                adapter_nvtx = f"{nvtx_root}/Adapter_bin{int(adapter_idx)}/n{len(req_indices)}"
                with NvtxAnnotate(adapter_nvtx):
                    try:
                        if hot_weights is not None:
                            A = hot_weights.a_buffer[: hot_weights.rank]
                            B = hot_weights.b_buffer[: hot_weights.rank]
                            a_scaling = float(hot_weights.scaling)
                            self._last_colora_stats["prefetch_ready_hits"] += 1
                        else:
                            if hot_status == "filling":
                                self._last_colora_stats["prefetch_not_ready"] += 1
                            a_start = int(pool.a_start[adapter_idx].item())
                            a_len = int(pool.a_len[adapter_idx].item())
                            a_scaling = float(pool.a_scaling[adapter_idx].item())
                            if hasattr(pool, "a_rank") and len(pool.a_rank) > adapter_idx:
                                a_rank = int(pool.a_rank[adapter_idx].item())
                            else:
                                a_rank = int(a_len)

                            loc = a_start + layer_id
                            if loc >= a_start + a_len:
                                continue

                            A = pool.key_buffer[loc, :a_rank]
                            B = pool.value_buffer[loc, :a_rank]

                        with NvtxAnnotate(f"{adapter_nvtx}/WeightHostPrep"):
                            # For naive mode, convert directly to float32 for faster matmul
                            # (bf16 CPU matmul is very slow without AVX-512 BF16 support)
                            target_dtype = torch.float32 if cpu_kernel_mode == "naive" else torch.bfloat16
                            if A.dtype != target_dtype:
                                A = A.to(dtype=target_dtype)
                            if B.dtype != target_dtype:
                                B = B.to(dtype=target_dtype)
                            if not A.is_contiguous():
                                A = A.contiguous()
                            if not B.is_contiguous():
                                B = B.contiguous()

                            batch_input = compute_input[req_indices]
                            if not batch_input.is_contiguous():
                                batch_input = batch_input.contiguous()

                        # Use fused kernel for small batches to save kernel launch overhead
                        # Fusing gate + up gives ~1.6x speedup for single token case
                        n_tokens = len(req_indices)
                        a_rank = A.shape[0]
                        if cpu_kernel_mode != "naive" and n_tokens <= 2 and a_rank <= 128 and proj_lower in ("gate", "up"):
                            with NvtxAnnotate(f"{adapter_nvtx}/Fused_{cpu_kernel_mode}_{proj_lower}_n{n_tokens}_r{a_rank}"):
                                batch_output = moe_batch_lora_avx(batch_input, A, B, a_scaling)
                            kernel_calls += 1
                        else:
                            # Stage-1: x @ A^T (naive matmul or AVX gate path)
                            with NvtxAnnotate(f"{adapter_nvtx}/Stage1_{cpu_kernel_mode}_gate"):
                                intermediate = gate_kernel(batch_input, A, 1.0)
                            # Stage-2: intermediate @ B, projection-specific kernel.
                            with NvtxAnnotate(f"{adapter_nvtx}/Stage2_{cpu_kernel_mode}_{proj_lower}"):
                                batch_output = stage2_kernel(intermediate, B, scaling=a_scaling)
                            kernel_calls += 2
                        output[req_indices] = batch_output
                        kernel_tokens += int(len(req_indices))
                    finally:
                        if hot_handle is not None:
                            hot_handle.release()

            # H2D transfer happens AFTER kernel computation (tracked separately)
            with NvtxAnnotate(f"{nvtx_root}/ReturnToOriginalDevice"):
                if return_to_original_device and (original_device.type != "cpu" or original_dtype != torch.bfloat16):
                    _h2d_t0 = time.perf_counter()
                    # For naive mode: convert to bf16 on CPU first to halve H2D bandwidth
                    if cpu_kernel_mode == "naive" and output.dtype != torch.bfloat16:
                        output = output.to(dtype=torch.bfloat16, non_blocking=True)
                    output = output.to(device=original_device, dtype=original_dtype, non_blocking=True)
                    if original_device.type == "cuda":
                        torch.cuda.synchronize()
                    self._last_colora_stats["h2d_residual_time"] += max(time.perf_counter() - _h2d_t0, 0.0)

            return output, kernel_calls, kernel_tokens

    def _reset_colora_stats(self) -> None:
        manager = getattr(self, "expert_cache_manager", None)
        default_miss_policy = str(getattr(getattr(manager, "config", None), "miss_policy", MISS_POLICY_CPU_FIRST))
        self._last_colora_stats = {
            "colora_hit_tokens": 0,
            "colora_miss_tokens": 0,
            "promotion_queue_depth": 0,
            "cache_hit_rate": 0.0,
            "cache_capacity_slots": 0,
            "cache_resident_slots": 0,
            "cache_free_slots": 0,
            "cache_evictions_total": 0,
            "gate_capacity_slots": 0,
            "gate_resident_slots": 0,
            "gate_free_slots": 0,
            "gate_evictions_total": 0,
            "up_capacity_slots": 0,
            "up_resident_slots": 0,
            "up_free_slots": 0,
            "up_evictions_total": 0,
            "down_capacity_slots": 0,
            "down_resident_slots": 0,
            "down_free_slots": 0,
            "down_evictions_total": 0,
            "cpu_compute_time": 0.0,
            "gpu_compute_time": 0.0,
            "cpu_queue_wait_time": 0.0,
            "cpu_queue_admit_wait_time": 0.0,
            "cpu_join_stall_time": 0.0,
            "d2h_bytes": 0.0,
            "h2d_bytes": 0.0,
            "weight_h2d_bytes": 0.0,
            "weight_h2d_time": 0.0,
            "overlap_ratio": 0.0,
            "fallback_degrade_count": 0,
            "cpu_async_submitted": 0,
            "cpu_inline_executed": 0,
            "blocking_promotion_count": 0,
            "miss_policy": default_miss_policy,
            "overlap_mode": self.colora_overlap_mode,
            "cpu_queue_depth": 0,
            "promotion_drop_total": 0,
            "promotion_drop_queue_high_watermark": 0,
            "promotion_drop_cooldown": 0,
            "promotion_admitted": 0,
            "promotion_reject_delta": 0,
            "promotion_reject_no_ema": 0,
            "tracker_queue_drop": 0,
            "moe_kernel_calls": 0,
            "moe_kernel_tokens": 0,
            "prefetch_submitted": 0,
            "prefetch_ready_hits": 0,
            "prefetch_not_ready": 0,
            "prefetch_stale": 0,
            "prefetch_false_positives": 0,
            "prefetch_slot_overwrite": 0,
            "pack_time": 0.0,
            "d2h_activation_time": 0.0,
            "h2d_residual_time": 0.0,
            "merge_time": 0.0,
            "admit_time": 0.0,
        }

    def _should_use_async_cpu_fallback(self) -> bool:
        return self.colora_async_fallback and self.colora_overlap_mode != OVERLAP_MODE_NO_OVERLAP

    def _has_cpu_miss_overlap_opportunity(
        self,
        state: _MoEHybridPhaseState,
        input_tensor: torch.Tensor,
    ) -> bool:
        if not self._should_use_async_cpu_fallback():
            return False
        if input_tensor.device.type != "cuda":
            return False
        if not state.ready_slots:
            return False
        if not BGMV_AVAILABLE:
            return False
        cache_a, cache_b = state.manager.get_projection_buffers(state.projection)
        return cache_a is not None and cache_b is not None

    def _get_or_create_cpu_executor(self) -> ThreadPoolExecutor:
        if self._cpu_executor is None:
            self._cpu_executor = ThreadPoolExecutor(
                max_workers=self.colora_cpu_workers,
                thread_name_prefix="colora_cpu",
            )
        return self._cpu_executor

    def _reserve_async_queue_slot(self, reserve_slots: int = 0) -> bool:
        reserve_slots = max(int(reserve_slots), 0)
        usable_depth = max(self.colora_cpu_queue_depth - reserve_slots, 0)
        if usable_depth <= 0:
            return False

        timeout_s = float(self.colora_cpu_batch_timeout_us) / 1_000_000.0
        deadline = time.perf_counter() + timeout_s
        while True:
            with self._cpu_queue_lock:
                if self._cpu_inflight < usable_depth:
                    self._cpu_inflight += 1
                    return True
            if timeout_s <= 0.0 or time.perf_counter() >= deadline:
                return False
            time.sleep(min(0.00001, max(deadline - time.perf_counter(), 0.0)))

    def _release_async_queue_slot(self) -> None:
        with self._cpu_queue_lock:
            self._cpu_inflight = max(self._cpu_inflight - 1, 0)

    def _get_cpu_queue_depth(self) -> int:
        with self._cpu_queue_lock:
            return int(self._cpu_inflight)

    def _get_moe_buffer_layer_id(self, pool, layer_id: int, expert_id: Optional[int]) -> int:
        if expert_id is None:
            return int(layer_id)
        return int(layer_id) * int(getattr(pool, "num_experts", 1)) + int(expert_id)

    def _is_joint_gate_up_ready(self, layer_id: int, adapter_bin: int, expert_id: int) -> bool:
        manager = self.expert_cache_manager
        if manager is None:
            return False

        gate_key = ExpertCacheKey(
            projection="gate",
            adapter_idx=int(adapter_bin),
            layer_id=int(layer_id),
            expert_id=int(expert_id),
        )
        up_key = ExpertCacheKey(
            projection="up",
            adapter_idx=int(adapter_bin),
            layer_id=int(layer_id),
            expert_id=int(expert_id),
        )
        peek_ready = getattr(manager, "peek_ready_slots", None)
        if callable(peek_ready):
            ready_slots = peek_ready([gate_key, up_key])
        else:
            ready_slots = manager.lookup_many([gate_key, up_key])
        return gate_key in ready_slots and up_key in ready_slots

    def maybe_submit_fused_gate_up_spec_job(
        self,
        key: SpecJobKey,
        input_tensor: torch.Tensor,
        req_bins: torch.Tensor,
        row_indices: torch.Tensor,
    ) -> SpecSubmitOutcome:
        if key.op_kind != "gate_up":
            raise ValueError(f"maybe_submit_fused_gate_up_spec_job expects op_kind='gate_up', got '{key.op_kind}'")

        if self.lora_mem_pool is None:
            return SpecSubmitOutcome(status="skipped", reason="no_lora_mem_pool", handle=None)

        gate_pool = getattr(self.lora_mem_pool, "moe_gate_pool", None)
        up_pool = getattr(self.lora_mem_pool, "moe_up_pool", None)
        if gate_pool is None or up_pool is None:
            return SpecSubmitOutcome(status="skipped", reason="missing_gate_up_pool", handle=None)
        if input_tensor is None or req_bins is None or row_indices is None:
            return SpecSubmitOutcome(status="skipped", reason="missing_inputs", handle=None)
        if row_indices.numel() <= 0:
            return SpecSubmitOutcome(status="skipped", reason="empty_rows", handle=None)
        if int(key.adapter_bin) < 0 or int(key.expert_id) < 0:
            return SpecSubmitOutcome(status="skipped", reason="invalid_joint_key", handle=None)
        if not self._should_use_hybrid_moe_compute():
            return SpecSubmitOutcome(status="skipped", reason="not_hybrid_moe", handle=None)
        if self.expert_cache_manager is None:
            return SpecSubmitOutcome(status="skipped", reason="no_expert_cache_manager", handle=None)

        row_indices_input = row_indices.to(device=input_tensor.device, dtype=torch.long).contiguous()
        row_indices_bins = row_indices.to(device=req_bins.device, dtype=torch.long).contiguous()
        if int(row_indices_input.max().item()) >= int(input_tensor.shape[0]):
            return SpecSubmitOutcome(status="skipped", reason="row_index_oob_input", handle=None)
        if int(row_indices_bins.max().item()) >= int(req_bins.shape[0]):
            return SpecSubmitOutcome(status="skipped", reason="row_index_oob_bins", handle=None)

        with self._spec_lock:
            current_step_id = self._spec_current_step_id
            if current_step_id is None or int(key.decode_step_id) != int(current_step_id):
                return SpecSubmitOutcome(status="rejected", reason="inactive_step", handle=None)

            existing = self._spec_active_jobs.get(key)
            if existing is not None:
                return SpecSubmitOutcome(status="skipped", reason="duplicate_active_job", handle=existing)
            if key in self._spec_reserved_job_keys:
                return SpecSubmitOutcome(status="skipped", reason="duplicate_reserved_job", handle=None)

        if self._is_joint_gate_up_ready(layer_id=key.layer_id, adapter_bin=key.adapter_bin, expert_id=key.expert_id):
            return SpecSubmitOutcome(status="skipped", reason="joint_gate_up_ready", handle=None)

        gate_buffer_layer_id = self._get_moe_buffer_layer_id(gate_pool, key.layer_id, key.expert_id)
        up_buffer_layer_id = self._get_moe_buffer_layer_id(up_pool, key.layer_id, key.expert_id)

        def _submit():
            selected_input = input_tensor.index_select(0, row_indices_input).contiguous()
            selected_bins = req_bins.index_select(0, row_indices_bins).contiguous()
            group_plan = self._build_cpu_group_plan(selected_bins)

            def _worker():
                gate_out, _, _ = self._strict_moe_cpu_batch_lora(
                    selected_input,
                    gate_buffer_layer_id,
                    gate_pool,
                    selected_bins,
                    projection="gate",
                    adapter_group_plan=group_plan,
                    return_to_original_device=False,
                )
                up_out, _, _ = self._strict_moe_cpu_batch_lora(
                    selected_input,
                    up_buffer_layer_id,
                    up_pool,
                    selected_bins,
                    projection="up",
                    adapter_group_plan=group_plan,
                    return_to_original_device=False,
                )
                return gate_out, up_out

            return self._get_or_create_cpu_executor().submit(_worker)

        handle = self.try_admit_spec_job(key, _submit)
        if handle is None:
            return SpecSubmitOutcome(status="rejected", reason="queue_full", handle=None)
        return SpecSubmitOutcome(status="submitted", reason="admitted", handle=handle)

    def _remove_spec_handle_locked(self, key: SpecJobKey) -> Optional[SpecJobHandle]:
        handle = self._spec_active_jobs.pop(key, None)
        if handle is None:
            return None

        step_keys = self._spec_active_job_keys_by_step.get(int(key.decode_step_id))
        if step_keys is not None:
            step_keys.discard(key)
            if not step_keys:
                self._spec_active_job_keys_by_step.pop(int(key.decode_step_id), None)
        return handle

    def _retire_spec_handle_locked(
        self,
        handle: SpecJobHandle,
        reason: str,
        cancel_future: bool = True,
    ) -> None:
        if handle.retired_at is not None:
            return

        self._remove_spec_handle_locked(handle.key)
        handle.stale = True
        handle.retire_reason = str(reason)
        handle.retired_at = time.perf_counter()

        future = handle.future
        if cancel_future and future is not None and hasattr(future, "cancel"):
            try:
                future.cancel()
            except Exception:
                pass

        self._spec_retired_jobs.append(handle)

    def _retire_step_handles_locked(self, step_id: int, reason: str, cancel_future: bool) -> int:
        retired = 0
        keys = tuple(self._spec_active_job_keys_by_step.get(int(step_id), ()))
        for key in keys:
            handle = self._spec_active_jobs.get(key)
            if handle is None:
                continue
            self._retire_spec_handle_locked(handle, reason=reason, cancel_future=cancel_future)
            retired += 1

        if self._spec_current_step_id == int(step_id):
            self._spec_current_step_id = None
        return retired

    def begin_spec_step(self, decode_step_id: int) -> Dict[str, int]:
        retired = 0
        step_id = int(decode_step_id)
        with self._spec_lock:
            stale_steps = [active_step for active_step in self._spec_active_job_keys_by_step.keys() if active_step != step_id]
            for active_step in stale_steps:
                keys = tuple(self._spec_active_job_keys_by_step.get(active_step, ()))
                for key in keys:
                    handle = self._spec_active_jobs.get(key)
                    if handle is None:
                        continue
                    self._retire_spec_handle_locked(handle, reason="step_advanced")
                    retired += 1

            self._spec_current_step_id = step_id
            self._spec_active_job_keys_by_step.setdefault(step_id, set())

        reap_stats = self.reap_retired()
        return {
            "retired_active": int(retired),
            "reaped_retired": int(reap_stats["reaped_retired"]),
            "pending_retired": int(reap_stats["pending_retired"]),
        }

    def end_spec_step(self, decode_step_id: int) -> Dict[str, int]:
        step_id = int(decode_step_id)
        with self._spec_lock:
            retired = self._retire_step_handles_locked(step_id, reason="step_end", cancel_future=True)

        reap_stats = self.reap_retired()
        return {
            "retired_active": int(retired),
            "reaped_retired": int(reap_stats["reaped_retired"]),
            "pending_retired": int(reap_stats["pending_retired"]),
        }

    def finalize_spec_step_nonblocking(self, decode_step_id: int) -> Dict[str, int]:
        step_id = int(decode_step_id)
        with self._spec_lock:
            retired = self._retire_step_handles_locked(step_id, reason="step_finalize", cancel_future=False)

        reap_stats = self.reap_retired()
        return {
            "retired_active": int(retired),
            "reaped_retired": int(reap_stats["reaped_retired"]),
            "pending_retired": int(reap_stats["pending_retired"]),
        }

    def reap_retired(self) -> Dict[str, int]:
        reaped = 0
        pending: Deque[SpecJobHandle] = deque()
        with self._spec_lock:
            while self._spec_retired_jobs:
                handle = self._spec_retired_jobs.popleft()
                future = handle.future
                if future is None:
                    reaped += 1
                    continue

                is_done = True
                if hasattr(future, "done"):
                    try:
                        is_done = bool(future.done())
                    except Exception:
                        is_done = True

                if not is_done:
                    pending.append(handle)
                    continue

                if hasattr(future, "result"):
                    try:
                        future.result(timeout=0)
                    except TypeError:
                        try:
                            future.result()
                        except Exception:
                            pass
                    except CancelledError:
                        pass
                    except Exception:
                        pass
                reaped += 1

            self._spec_retired_jobs = pending
            pending_count = len(self._spec_retired_jobs)

        return {
            "reaped_retired": int(reaped),
            "pending_retired": int(pending_count),
        }

    def try_admit_spec_job(
        self,
        key: SpecJobKey,
        submit_fn: Callable[[], Any],
    ) -> Optional[SpecJobHandle]:
        if not callable(submit_fn):
            raise TypeError("submit_fn must be callable")

        self.reap_retired()

        reserved = False
        with self._spec_lock:
            current_step_id = self._spec_current_step_id
            if current_step_id is None or int(key.decode_step_id) != int(current_step_id):
                stale = self._spec_active_jobs.get(key)
                if stale is not None:
                    self._retire_spec_handle_locked(stale, reason="admit_step_mismatch")
                return None

            existing = self._spec_active_jobs.get(key)
            if existing is not None:
                return existing
            if key in self._spec_reserved_job_keys:
                return None

            if not self._reserve_async_queue_slot(reserve_slots=1):
                return None

            self._spec_reserved_job_keys.add(key)
            reserved = True

        try:
            future = submit_fn()
            if future is None:
                raise RuntimeError("submit_fn returned None for speculative job")
            if not hasattr(future, "add_done_callback"):
                raise TypeError("speculative job future must support add_done_callback")
            future.add_done_callback(lambda _f: self._release_async_queue_slot())
        except Exception:
            with self._spec_lock:
                if reserved:
                    self._spec_reserved_job_keys.discard(key)
            self._release_async_queue_slot()
            raise

        handle = SpecJobHandle(key=key, future=future)
        with self._spec_lock:
            self._spec_reserved_job_keys.discard(key)
            current_step_id = self._spec_current_step_id
            if current_step_id is None or int(key.decode_step_id) != int(current_step_id):
                self._retire_spec_handle_locked(handle, reason="submit_step_mismatch")
                return None

            existing = self._spec_active_jobs.get(key)
            if existing is not None:
                self._retire_spec_handle_locked(handle, reason="submit_duplicate")
                return existing

            self._spec_active_jobs[key] = handle
            self._spec_active_job_keys_by_step.setdefault(int(key.decode_step_id), set()).add(key)
            return handle

    def try_bind_gate_up_job_with_status(self, key: SpecJobKey) -> SpecBindOutcome:
        if key.op_kind != "gate_up":
            raise ValueError(f"try_bind_gate_up_job expects op_kind='gate_up', got '{key.op_kind}'")

        self.reap_retired()

        with self._spec_lock:
            current_step_id = self._spec_current_step_id
            if current_step_id is None or int(key.decode_step_id) != int(current_step_id):
                stale = self._spec_active_jobs.get(key)
                if stale is not None:
                    self._retire_spec_handle_locked(stale, reason="bind_step_mismatch")
                return SpecBindOutcome(status="stale", reason="bind_step_mismatch", result=None)

            handle = self._spec_active_jobs.get(key)
            if handle is None:
                return SpecBindOutcome(status="missing", reason="bind_missing", result=None)

            future = handle.future
            is_done = False
            if hasattr(future, "done"):
                try:
                    is_done = bool(future.done())
                except Exception:
                    is_done = False

            if not is_done:
                self._retire_spec_handle_locked(handle, reason="bind_not_ready")
                return SpecBindOutcome(status="not_ready", reason="bind_not_ready", result=None)

        try:
            if hasattr(future, "result"):
                try:
                    result = future.result(timeout=0)
                except TypeError:
                    result = future.result()
            else:
                result = future
        except CancelledError:
            with self._spec_lock:
                active = self._spec_active_jobs.get(key)
                if active is not None:
                    self._retire_spec_handle_locked(active, reason="bind_cancelled")
            return SpecBindOutcome(status="stale", reason="bind_cancelled", result=None)
        except Exception:
            with self._spec_lock:
                active = self._spec_active_jobs.get(key)
                if active is not None:
                    self._retire_spec_handle_locked(active, reason="bind_failed")
            return SpecBindOutcome(status="failed", reason="bind_failed", result=None)

        with self._spec_lock:
            active = self._remove_spec_handle_locked(key)
            if active is not None:
                active.bound_at = time.perf_counter()
        return SpecBindOutcome(status="bound", reason="bind_success", result=result)

    def try_bind_gate_up_job(self, key: SpecJobKey) -> Optional[Any]:
        outcome = self.try_bind_gate_up_job_with_status(key)
        if outcome.status == "bound":
            return outcome.result
        return None

    def retire_unbound_gate_up_jobs(
        self,
        layer_id: int,
        decode_step_id: int,
        keep_keys: Optional[Set[SpecJobKey]] = None,
    ) -> Dict[str, int]:
        keep = set(keep_keys or ())
        retired = 0
        with self._spec_lock:
            keys = tuple(self._spec_active_job_keys_by_step.get(int(decode_step_id), ()))
            for key in keys:
                if int(key.layer_id) != int(layer_id):
                    continue
                if key.op_kind != "gate_up":
                    continue
                if key in keep:
                    continue
                handle = self._spec_active_jobs.get(key)
                if handle is None:
                    continue
                self._retire_spec_handle_locked(handle, reason="bind_unmatched_after_actual", cancel_future=True)
                retired += 1

        reap_stats = self.reap_retired()
        return {
            "retired_active": int(retired),
            "reaped_retired": int(reap_stats["reaped_retired"]),
            "pending_retired": int(reap_stats["pending_retired"]),
        }

    def cleanup_speculation_state(self) -> Dict[str, int]:
        retired_active = 0
        with self._spec_lock:
            active_handles = tuple(self._spec_active_jobs.values())
            for handle in active_handles:
                self._retire_spec_handle_locked(handle, reason="cleanup")
                retired_active += 1

            cleared_reserved = len(self._spec_reserved_job_keys)
            self._spec_reserved_job_keys.clear()
            self._spec_current_step_id = None
            self._spec_active_jobs.clear()
            self._spec_active_job_keys_by_step.clear()

        reap_stats = self.reap_retired()

        with self._spec_lock:
            pending_retired = len(self._spec_retired_jobs)

        return {
            "retired_active": int(retired_active),
            "reaped_retired": int(reap_stats["reaped_retired"]),
            "pending_retired": int(pending_retired),
            "cleared_retired": 0,
            "cleared_reserved": int(cleared_reserved),
        }

    def _build_cpu_group_plan(self, req_bins: torch.Tensor) -> Tuple[Tuple[int, Tuple[int, ...]], ...]:
        if req_bins.device.type != "cpu":
            bins_cpu = req_bins.detach().to(device="cpu", dtype=torch.int32)
        else:
            bins_cpu = req_bins.detach().to(dtype=torch.int32)
        bins_cpu = bins_cpu.contiguous()
        key = (int(bins_cpu.numel()), bins_cpu.numpy().tobytes())

        plan = self._cpu_group_plan_cache.get(key)
        if plan is not None:
            self._cpu_group_plan_cache.move_to_end(key)
            return plan

        adapter_to_indices: Dict[int, List[int]] = {}
        for i, bin_idx in enumerate(bins_cpu.tolist()):
            adapter_to_indices.setdefault(int(bin_idx), []).append(i)
        plan = tuple((adapter_idx, tuple(indices)) for adapter_idx, indices in adapter_to_indices.items())

        self._cpu_group_plan_cache[key] = plan
        self._cpu_group_plan_cache.move_to_end(key)
        if len(self._cpu_group_plan_cache) > self._cpu_group_plan_cache_cap:
            self._cpu_group_plan_cache.popitem(last=False)
        return plan

    def _run_cpu_miss_job(
        self,
        miss_input: torch.Tensor,
        buffer_layer_id: int,
        pool,
        miss_bins: torch.Tensor,
        projection: str,
    ) -> Tuple[torch.Tensor, float, float]:
        enqueue_ts = time.perf_counter()
        if not self._reserve_async_queue_slot():
            miss_plan = self._build_cpu_group_plan(miss_bins)
            out, _, _ = self._strict_moe_cpu_batch_lora(
                miss_input,
                buffer_layer_id,
                pool,
                miss_bins,
                projection=projection,
                adapter_group_plan=miss_plan,
            )
            return out, 0.0, time.perf_counter() - enqueue_ts

        def _worker():
            try:
                start_ts = time.perf_counter()
                queue_wait = max(start_ts - enqueue_ts, 0.0)
                t0 = time.perf_counter()
                plan = self._build_cpu_group_plan(miss_bins)
                out, _, _ = self._strict_moe_cpu_batch_lora(
                    miss_input,
                    buffer_layer_id,
                    pool,
                    miss_bins,
                    projection=projection,
                    adapter_group_plan=plan,
                )
                return out, queue_wait, time.perf_counter() - t0
            finally:
                self._release_async_queue_slot()

        future = self._get_or_create_cpu_executor().submit(_worker)
        return future.result()

    def pop_colora_stats(self) -> Dict[str, float]:
        self._flush_pending_background_stats()
        if self._temporal_hot_cache is not None:
            self._last_colora_stats["prefetch_slot_overwrite"] = int(
                self._temporal_hot_cache.get_slot_overwrite_count()
            )
        self._append_cache_observability_stats(self._last_colora_stats)
        stats = dict(self._last_colora_stats)
        self._publish_cache_observability_metrics(stats)
        self._reset_colora_stats()
        return stats

    def _append_cache_observability_stats(self, stats: Dict[str, float]) -> None:
        manager = self.expert_cache_manager
        if manager is None or not hasattr(manager, "get_cache_observability_stats"):
            return

        cache_stats = manager.get_cache_observability_stats()
        total = cache_stats.get("total", {})
        by_projection = cache_stats.get("by_projection", {})

        stats["cache_capacity_slots"] = int(total.get("capacity_slots", 0))
        stats["cache_resident_slots"] = int(total.get("resident_slots", 0))
        stats["cache_free_slots"] = int(total.get("free_slots", 0))
        stats["cache_evictions_total"] = int(total.get("evictions_total", 0))

        for projection in ("gate", "up", "down"):
            projection_stats = by_projection.get(projection, {})
            stats[f"{projection}_capacity_slots"] = int(projection_stats.get("capacity_slots", 0))
            stats[f"{projection}_resident_slots"] = int(projection_stats.get("resident_slots", 0))
            stats[f"{projection}_free_slots"] = int(projection_stats.get("free_slots", 0))
            stats[f"{projection}_evictions_total"] = int(projection_stats.get("evictions_total", 0))

    def _publish_cache_observability_metrics(self, stats: Dict[str, float]) -> None:
        metric_client = self.metric_client
        if metric_client is None:
            return

        gauge_values = {
            "lightllm_colora_cache_capacity_slots": stats.get("cache_capacity_slots", 0),
            "lightllm_colora_cache_resident_slots": stats.get("cache_resident_slots", 0),
            "lightllm_colora_cache_free_slots": stats.get("cache_free_slots", 0),
            "lightllm_colora_cache_evictions_total": stats.get("cache_evictions_total", 0),
            "lightllm_colora_cache_capacity_slots_gate": stats.get("gate_capacity_slots", 0),
            "lightllm_colora_cache_resident_slots_gate": stats.get("gate_resident_slots", 0),
            "lightllm_colora_cache_free_slots_gate": stats.get("gate_free_slots", 0),
            "lightllm_colora_cache_evictions_total_gate": stats.get("gate_evictions_total", 0),
            "lightllm_colora_cache_capacity_slots_up": stats.get("up_capacity_slots", 0),
            "lightllm_colora_cache_resident_slots_up": stats.get("up_resident_slots", 0),
            "lightllm_colora_cache_free_slots_up": stats.get("up_free_slots", 0),
            "lightllm_colora_cache_evictions_total_up": stats.get("up_evictions_total", 0),
            "lightllm_colora_cache_capacity_slots_down": stats.get("down_capacity_slots", 0),
            "lightllm_colora_cache_resident_slots_down": stats.get("down_resident_slots", 0),
            "lightllm_colora_cache_free_slots_down": stats.get("down_free_slots", 0),
            "lightllm_colora_cache_evictions_total_down": stats.get("down_evictions_total", 0),
        }
        for gauge_name, value in gauge_values.items():
            try:
                metric_client.gauge_set(gauge_name, float(value))
            except Exception as e:
                logger.debug("[COLoRA] Failed to update gauge %s: %s", gauge_name, e)

    def build_moe_hybrid_shared_prepare_context(
        self,
        batch_size: int,
        bins: torch.Tensor,
        expert_id: Optional[int],
    ) -> Optional[MoEHybridSharedPrepareContext]:
        """Build projection-invariant token/bin metadata once per expert call."""
        if bins is None:
            return None
        local_bins = bins
        if len(local_bins) > int(batch_size):
            local_bins = local_bins[: int(batch_size)]
        valid_mask = local_bins >= 0
        if not torch.any(valid_mask):
            return None
        valid_pos = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        valid_bins = local_bins.index_select(0, valid_pos).long()
        # Convert once, then reuse across Gate/Up/Down.
        adapter_ids_cpu = torch.unique(valid_bins).detach().to(device="cpu", dtype=torch.long).tolist()
        key_expert = int(expert_id) if expert_id is not None else 0
        # Phase 5: when CPU indexing is enabled, D2H valid_bins once here and
        # reuse across BuildHitMissMasks + HitIndicesAndGroups.
        valid_bins_cpu: Optional[torch.Tensor] = None
        if self.colora_hit_indexing == "cpu":
            valid_bins_cpu = valid_bins.detach().to(device="cpu", dtype=torch.long)
        return MoEHybridSharedPrepareContext(
            batch_size=int(batch_size),
            key_expert=key_expert,
            valid_pos=valid_pos,
            valid_bins=valid_bins,
            adapter_ids_cpu=adapter_ids_cpu,
            valid_bins_cpu=valid_bins_cpu,
        )

    def _batch_apply_moe_lora_hybrid(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        buffer_layer_id: int,
        pool,
        bins: torch.Tensor,
        projection: str,
        expert_id: Optional[int],
        hybrid_prepare_ctx: Optional[MoEHybridSharedPrepareContext] = None,
    ) -> torch.Tensor:
        """COLoRA hybrid path: GPU cache hit + CPU miss fallback + async promotion."""
        self._reset_colora_stats()
        self._flush_pending_background_stats()
        decode_context = self._get_current_decode_joint_context()

        manager = self.expert_cache_manager
        if manager is None:
            output = self._get_output_buffer(input_tensor, pool)
            valid_miss_tokens = int((bins >= 0).sum().item()) if bins is not None else 0
            miss_plan = self._build_cpu_group_plan(bins)
            miss_out, kernel_calls, kernel_tokens = self._strict_moe_cpu_batch_lora(
                input_tensor,
                buffer_layer_id,
                pool,
                bins,
                projection=projection,
                adapter_group_plan=miss_plan,
                temporal_prefetch_context=decode_context,
            )
            self._last_colora_stats.update(
                {
                    "colora_hit_tokens": 0,
                    "colora_miss_tokens": valid_miss_tokens,
                    "moe_kernel_calls": int(kernel_calls),
                    "moe_kernel_tokens": int(kernel_tokens),
                }
            )
            return miss_out

        state = self._hybrid_prepare_through_masks(
            input_tensor=input_tensor,
            layer_id=layer_id,
            buffer_layer_id=buffer_layer_id,
            pool=pool,
            bins=bins,
            projection=projection,
            expert_id=expert_id,
            hybrid_prepare_ctx=hybrid_prepare_ctx,
            manager=manager,
            decode_context=decode_context,
        )
        if state is None:
            return self._get_output_buffer(input_tensor, pool)

        self._hybrid_try_submit_cpu_miss(state=state, input_tensor=input_tensor, pool=pool, worker_owned_d2h=True)
        self._hybrid_run_gpu_hit(state=state, input_tensor=input_tensor, pool=pool)
        self._hybrid_finalize_miss(state=state, input_tensor=input_tensor, pool=pool)
        self._hybrid_finalize_stats(state)
        return state.output

    def _hybrid_prepare_through_masks(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        buffer_layer_id: int,
        pool,
        bins: torch.Tensor,
        projection: str,
        expert_id: Optional[int],
        hybrid_prepare_ctx: Optional[MoEHybridSharedPrepareContext],
        manager: MoEExpertCacheManager,
        decode_context: Optional[Tuple[int, int, int]],
    ) -> Optional[_MoEHybridPhaseState]:
        state = _MoEHybridPhaseState(
            output=self._get_output_buffer(input_tensor, pool),
            manager=manager,
            layer_id=int(layer_id),
            buffer_layer_id=int(buffer_layer_id),
            projection=str(projection),
            key_expert=int(expert_id) if expert_id is not None else 0,
            decode_context=decode_context,
            shared_prepare_ctx=hybrid_prepare_ctx,
        )

        with NvtxAnnotate("COLoRA_Hybrid_Prepare"):
            with NvtxAnnotate("COLoRA_ApplyCompletedPromotions"):
                manager.apply_completed_promotions()

            with NvtxAnnotate("COLoRA_BuildValidTokenView"):
                if (
                    state.shared_prepare_ctx is None
                    or int(state.shared_prepare_ctx.batch_size) != int(input_tensor.shape[0])
                    or int(state.shared_prepare_ctx.key_expert) != int(state.key_expert)
                ):
                    state.shared_prepare_ctx = self.build_moe_hybrid_shared_prepare_context(
                        batch_size=int(input_tensor.shape[0]),
                        bins=bins,
                        expert_id=expert_id,
                    )
                if state.shared_prepare_ctx is None:
                    self._last_colora_stats["promotion_queue_depth"] = manager.get_promotion_queue_depth()
                    self._last_colora_stats["cache_hit_rate"] = manager.get_hit_rate()
                    return None

                state.valid_pos = state.shared_prepare_ctx.valid_pos
                state.valid_bins = state.shared_prepare_ctx.valid_bins

            with NvtxAnnotate("COLoRA_BuildCacheKeys"):
                unique_adapters = state.shared_prepare_ctx.adapter_ids_cpu
                state.keys = [
                    ExpertCacheKey(
                        projection=projection,
                        adapter_idx=int(adapter_idx),
                        layer_id=int(layer_id),
                        expert_id=int(state.key_expert),
                    )
                    for adapter_idx in unique_adapters
                ]

        with NvtxAnnotate("COLoRA_CacheLookupAndPolicy"):
            if decode_context is None:
                manager.record_access(state.keys)
            state.miss_policy = str(getattr(getattr(manager, "config", None), "miss_policy", MISS_POLICY_CPU_FIRST))
            state.ready_slots = manager.lookup_many(state.keys)
            state.miss_keys = [key for key in state.keys if key not in state.ready_slots]
            allow_inline_promotion_schedule = decode_context is None
            if allow_inline_promotion_schedule and state.miss_policy in (MISS_POLICY_CPU_FIRST, MISS_POLICY_LOAD_THEN_RUN):
                manager.schedule_promotion(state.miss_keys)

        if state.miss_keys and state.miss_policy == MISS_POLICY_LOAD_THEN_RUN:
            manager.apply_completed_promotions()
            promoted_now = manager.lookup_many(state.miss_keys)
            if promoted_now:
                state.ready_slots.update(promoted_now)
                state.miss_keys = [key for key in state.miss_keys if key not in promoted_now]
            if state.miss_keys:
                promotion_t0 = time.perf_counter()
                promotion_result = manager.promote_blocking(state.miss_keys)
                state.blocking_promotion_time += max(time.perf_counter() - promotion_t0, 0.0)
                state.blocking_promotion_bytes += float(promotion_result.transferred_bytes)
                state.blocking_promotion_count += int(promotion_result.promoted_count)
                state.ready_slots.update(promotion_result.ready_slots)
                state.miss_keys = [key for key in state.miss_keys if key not in promotion_result.ready_slots]
                if state.miss_keys:
                    raise RuntimeError(f"COLoRA load_then_run left unresolved misses: {state.miss_keys!r}")

        if state.miss_keys and state.miss_policy == MISS_POLICY_NO_CPU_PATH:
            promotion_t0 = time.perf_counter()
            promotion_result = manager.promote_blocking(state.miss_keys)
            state.blocking_promotion_time += max(time.perf_counter() - promotion_t0, 0.0)
            state.blocking_promotion_bytes += float(promotion_result.transferred_bytes)
            state.blocking_promotion_count += int(promotion_result.promoted_count)
            state.ready_slots.update(promotion_result.ready_slots)
            state.miss_keys = [key for key in state.miss_keys if key not in promotion_result.ready_slots]
            if state.miss_keys:
                raise RuntimeError(f"COLoRA no_cpu_path left unresolved misses: {state.miss_keys!r}")

        assert state.valid_bins is not None
        with NvtxAnnotate("COLoRA_BuildHitMissMasks"):
            hit_adapters = {int(key.adapter_idx) for key in state.ready_slots.keys()}
            use_cpu_indexing = (
                self.colora_hit_indexing == "cpu"
                and state.shared_prepare_ctx is not None
                and state.shared_prepare_ctx.valid_bins_cpu is not None
            )
            if use_cpu_indexing:
                # Phase 5: CPU set-membership replaces torch.isin. Pre-compute
                # hit_rows/hit_inverse/hit_unique on the host; upload only the
                # small integer index tensors to GPU later.
                valid_bins_cpu = state.shared_prepare_ctx.valid_bins_cpu
                if hit_adapters:
                    bins_list = valid_bins_cpu.tolist()
                    unique_sorted = sorted(hit_adapters)
                    unique_to_idx: Dict[int, int] = {v: i for i, v in enumerate(unique_sorted)}
                    hit_rows_py: List[int] = []
                    hit_inverse_py: List[int] = []
                    used_ids: List[int] = []
                    used_seen: Dict[int, int] = {}
                    for row_idx, bin_val in enumerate(bins_list):
                        idx_in_sorted = unique_to_idx.get(int(bin_val))
                        if idx_in_sorted is None:
                            continue
                        hit_rows_py.append(row_idx)
                        if int(bin_val) not in used_seen:
                            used_seen[int(bin_val)] = len(used_ids)
                            used_ids.append(int(bin_val))
                        hit_inverse_py.append(used_seen[int(bin_val)])
                    # torch.unique returns sorted unique values; match that ordering
                    # and remap inverse indices so outputs are byte-identical with
                    # the GPU path.
                    final_sorted = sorted(set(used_ids))
                    if final_sorted != used_ids:
                        remap = {old_idx: final_sorted.index(v) for old_idx, v in enumerate(used_ids)}
                        hit_inverse_py = [remap[i] for i in hit_inverse_py]
                    hit_rows_cpu = torch.as_tensor(hit_rows_py, dtype=torch.long)
                    hit_inverse_cpu = torch.as_tensor(hit_inverse_py, dtype=torch.long)
                    hit_unique_cpu_tensor = torch.as_tensor(final_sorted, dtype=torch.long)
                    # Build hit_mask on CPU (bool) and H2D once.
                    hit_mask_cpu = torch.zeros(
                        state.valid_bins.numel(), dtype=torch.bool, device="cpu"
                    )
                    if hit_rows_py:
                        hit_mask_cpu.index_fill_(0, hit_rows_cpu, True)
                    state.hit_mask = hit_mask_cpu.to(state.valid_bins.device, non_blocking=True)
                    state.hit_rows_cpu = hit_rows_cpu
                    state.hit_inverse_cpu = hit_inverse_cpu
                    state.hit_unique_cpu_tensor = hit_unique_cpu_tensor
                    state.hit_unique_cpu_list = final_sorted
                else:
                    state.hit_mask = torch.zeros_like(state.valid_bins, dtype=torch.bool)
            else:
                if hit_adapters:
                    adapter_tensor = torch.tensor(
                        sorted(hit_adapters),
                        dtype=state.valid_bins.dtype,
                        device=state.valid_bins.device,
                    )
                    state.hit_mask = torch.isin(state.valid_bins, adapter_tensor)
                else:
                    state.hit_mask = torch.zeros_like(state.valid_bins, dtype=torch.bool)
            state.miss_mask = ~state.hit_mask
        return state

    def _hybrid_try_submit_cpu_miss(
        self,
        state: _MoEHybridPhaseState,
        input_tensor: torch.Tensor,
        pool,
        worker_owned_d2h: bool,
    ) -> None:
        assert state.miss_mask is not None
        assert state.valid_pos is not None and state.valid_bins is not None
        with NvtxAnnotate("COLoRA_MissPath_CheckAndPrepare"):
            if not state.miss_keys:
                return
            _pack_t0 = time.perf_counter()
            if not state.ready_slots:
                state.miss_pos = state.valid_pos
                state.miss_bins = state.valid_bins
            else:
                miss_rows = torch.nonzero(state.miss_mask, as_tuple=False).squeeze(-1)
                state.miss_pos = state.valid_pos.index_select(0, miss_rows)
                state.miss_bins = state.valid_bins.index_select(0, miss_rows)
            assert state.miss_pos is not None and state.miss_bins is not None
            state.pack_time += max(time.perf_counter() - _pack_t0, 0.0)

            if int(state.miss_pos.numel()) == int(input_tensor.shape[0]):
                state.miss_input = input_tensor

            if input_tensor.device.type == "cuda":
                state.d2h_bytes += float(
                    state.miss_pos.numel() * input_tensor.shape[1] * input_tensor.element_size()
                )
            state.h2d_bytes += float(
                state.miss_pos.numel() * pool.value_buffer.shape[2] * input_tensor.element_size()
            )

            if self._has_cpu_miss_overlap_opportunity(state, input_tensor):
                reserve_t0 = time.perf_counter()
                if self._reserve_async_queue_slot():
                    _admit_elapsed = max(time.perf_counter() - reserve_t0, 0.0)
                    state.cpu_queue_admit_wait_time += _admit_elapsed
                    state.admit_time += _admit_elapsed
                    state.async_overlap_used = True
                    state.cpu_async_submitted += 1
                    enqueue_ts = time.perf_counter()
                    miss_plan = self._build_cpu_group_plan(state.miss_bins)
                    miss_pos = state.miss_pos
                    miss_bins = state.miss_bins
                    buffer_layer_id = int(state.buffer_layer_id)
                    projection = str(state.projection)
                    decode_context = state.decode_context

                    if worker_owned_d2h:
                        miss_input_ref = input_tensor
                    else:
                        miss_input_ref = input_tensor.index_select(0, miss_pos).contiguous()
                        state.miss_input = miss_input_ref

                    def _miss_worker():
                        try:
                            start_ts = time.perf_counter()
                            queue_wait = max(start_ts - enqueue_ts, 0.0)
                            t0 = time.perf_counter()
                            if worker_owned_d2h:
                                local_miss_input = miss_input_ref.index_select(0, miss_pos).contiguous()
                            else:
                                local_miss_input = miss_input_ref
                            miss_out, kernel_calls, kernel_tokens = self._strict_moe_cpu_batch_lora(
                                local_miss_input,
                                buffer_layer_id,
                                pool,
                                miss_bins,
                                projection=projection,
                                adapter_group_plan=miss_plan,
                                temporal_prefetch_context=decode_context,
                            )
                            elapsed = time.perf_counter() - t0
                            # Subtract H2D time that was tracked internally
                            internal_h2d = self._last_colora_stats.get("h2d_residual_time", 0.0)
                            self._last_colora_stats["h2d_residual_time"] = 0.0
                            cpu_t = max(elapsed - internal_h2d, 0.0)
                            return miss_out, queue_wait, cpu_t, kernel_calls, kernel_tokens, internal_h2d
                        finally:
                            self._release_async_queue_slot()

                    state.miss_future = self._get_or_create_cpu_executor().submit(_miss_worker)
                else:
                    state.cpu_queue_admit_wait_time += max(time.perf_counter() - reserve_t0, 0.0)
                    state.fallback_degrade_count += 1

    def _hybrid_run_gpu_hit(self, state: _MoEHybridPhaseState, input_tensor: torch.Tensor, pool) -> None:
        assert state.hit_mask is not None and state.miss_mask is not None
        assert state.valid_pos is not None and state.valid_bins is not None
        with NvtxAnnotate("COLoRA_HitPath_Prepare"):
            if not state.ready_slots:
                return
            if not torch.any(state.hit_mask):
                return
            # Phase 4: look up / populate projection-invariant hit indexing memo.
            shared_ctx = state.shared_prepare_ctx
            hit_adapter_key: Optional[frozenset] = None
            hit_idx: Optional[_HitIndex] = None
            if shared_ctx is not None and state.ready_slots:
                hit_adapter_key = frozenset(int(k.adapter_idx) for k in state.ready_slots)
                hit_idx = shared_ctx.hit_index_cache.get(hit_adapter_key)
            # Phase 5: CPU-computed index tensors (if BuildHitMissMasks populated them).
            cpu_indexing = (
                hit_idx is None
                and state.hit_rows_cpu is not None
                and state.hit_unique_cpu_tensor is not None
                and state.hit_inverse_cpu is not None
            )
            with NvtxAnnotate("COLoRA_HitIndicesAndGroups"):
                if hit_idx is not None:
                    hit_rows = hit_idx.hit_rows
                    hit_pos = hit_idx.hit_pos
                    hit_bins = hit_idx.hit_bins
                    hit_unique_adapters = hit_idx.hit_unique_adapters
                    hit_inverse = hit_idx.hit_inverse
                elif cpu_indexing:
                    # H2D only the small integer index tensors; no GPU sync needed.
                    device = state.valid_bins.device
                    hit_rows = state.hit_rows_cpu.to(device, non_blocking=True)
                    hit_inverse = state.hit_inverse_cpu.to(device, non_blocking=True)
                    hit_unique_adapters = state.hit_unique_cpu_tensor.to(
                        device=device, dtype=state.valid_bins.dtype, non_blocking=True
                    ).long()
                    hit_pos = state.valid_pos.index_select(0, hit_rows)
                    hit_bins = state.valid_bins.index_select(0, hit_rows)
                else:
                    hit_rows = torch.nonzero(state.hit_mask, as_tuple=False).squeeze(-1)
                    hit_pos = state.valid_pos.index_select(0, hit_rows)
                    hit_bins = state.valid_bins.index_select(0, hit_rows)
                    hit_unique_adapters, hit_inverse = torch.unique(hit_bins, return_inverse=True)

            with NvtxAnnotate("COLoRA_ResolveSlotIds"):
                slot_by_adapter = {int(key.adapter_idx): int(slot_id) for key, slot_id in state.ready_slots.items()}
                if hit_idx is not None:
                    hit_unique_cpu_tensor = hit_idx.hit_unique_cpu_tensor
                    hit_unique_cpu = hit_idx.hit_unique_cpu_list
                elif cpu_indexing:
                    hit_unique_cpu_tensor = state.hit_unique_cpu_tensor
                    hit_unique_cpu = state.hit_unique_cpu_list
                else:
                    # Phase 1: single D2H for hit_unique_adapters; reused downstream in KernelSetup.
                    hit_unique_cpu_tensor = hit_unique_adapters.detach().to(device="cpu", dtype=torch.long)
                    hit_unique_cpu = hit_unique_cpu_tensor.tolist()
                slot_ids = [int(slot_by_adapter.get(int(adapter_idx), -1)) for adapter_idx in hit_unique_cpu]

            # Populate memo on miss so Up/Down of the same expert skip the
            # nonzero + unique(return_inverse=True) and the D2H in ResolveSlotIds.
            if hit_idx is None and shared_ctx is not None and hit_adapter_key is not None:
                shared_ctx.hit_index_cache[hit_adapter_key] = _HitIndex(
                    hit_rows=hit_rows,
                    hit_pos=hit_pos,
                    hit_bins=hit_bins,
                    hit_unique_adapters=hit_unique_adapters,
                    hit_inverse=hit_inverse,
                    hit_unique_cpu_tensor=hit_unique_cpu_tensor,
                    hit_unique_cpu_list=hit_unique_cpu,
                )

            cache_a, cache_b = state.manager.get_projection_buffers(state.projection)
            if (
                BGMV_AVAILABLE
                and input_tensor.device.type == "cuda"
                and cache_a is not None
                and cache_b is not None
                and all(slot_id >= 0 for slot_id in slot_ids)
            ):
                with NvtxAnnotate("COLoRA_GPU_Hit_Path"):
                    t0 = time.perf_counter()

                    with NvtxAnnotate("COLoRA_GPU_Hit_InputGather"):
                        hit_input = input_tensor.index_select(0, hit_pos).contiguous()

                    with NvtxAnnotate("COLoRA_GPU_Hit_ScratchpadEnsure"):
                        active_count = hit_unique_adapters.size(0)
                        self._ensure_compact_scratchpad(pool, input_tensor.device, int(active_count))
                        assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None

                    with NvtxAnnotate("COLoRA_GPU_Hit_CacheGather"):
                        slot_tensor = torch.tensor(slot_ids, dtype=torch.long, device=cache_a.device)
                        gathered_a = cache_a.index_select(0, slot_tensor)
                        gathered_b = cache_b.index_select(0, slot_tensor)

                    with NvtxAnnotate("COLoRA_GPU_Hit_CopyToScratchpad"):
                        self.gpu_scratchpad_a[:active_count].copy_(
                            gathered_a.to(input_tensor.device), non_blocking=True
                        )
                        self.gpu_scratchpad_b[:active_count].copy_(
                            gathered_b.to(input_tensor.device), non_blocking=True
                        )

                    with NvtxAnnotate("COLoRA_GPU_Hit_KernelSetup"):
                        hit_output = self._get_output_buffer(hit_input, pool)
                        temp_a_start, temp_a_len = self._get_bgmv_temp_index_buffers(
                            input_tensor.device, int(active_count)
                        )
                        # Phase 2: prefer GPU-resident a_scaling mirror (no H2D).
                        scaling_gpu = self._get_pool_a_scaling_gpu(pool, input_tensor.device)
                        if scaling_gpu is not None:
                            temp_scaling = scaling_gpu.index_select(0, hit_unique_adapters)
                        else:
                            # Phase 1: reuse the hit_unique_cpu_tensor from ResolveSlotIds.
                            temp_scaling = pool.a_scaling.index_select(0, hit_unique_cpu_tensor).to(
                                input_tensor.device
                            )

                    with NvtxAnnotate("COLoRA_GPU_Hit_BGMV"):
                        batch_lora_get_mlp(
                            hit_output,
                            hit_input,
                            self.gpu_scratchpad_a,
                            self.gpu_scratchpad_b,
                            temp_a_start,
                            temp_a_len,
                            temp_scaling,
                            hit_inverse,
                            a_hidden_dim=hit_input.shape[1],
                            b_hidden_dim=hit_output.shape[1],
                            layer_id=0,
                            **_bgmv_trace_kwargs(pool, "colora_gpu_hit/moe", compact=True),
                        )

                    with NvtxAnnotate("COLoRA_GPU_Hit_Writeback"):
                        state.output.index_copy_(0, hit_pos, hit_output)
                    state.gpu_compute_time += time.perf_counter() - t0
            else:
                if state.miss_policy in (MISS_POLICY_NO_CPU_PATH, MISS_POLICY_LOAD_THEN_RUN):
                    raise RuntimeError(
                        f"COLoRA {state.miss_policy} requires GPU cached execution on the promoted hot path; CPU fallback is not allowed."
                    )
                state.hit_mask = torch.zeros_like(state.valid_bins, dtype=torch.bool)
                state.miss_mask = torch.ones_like(state.valid_bins, dtype=torch.bool)
                if state.miss_pos is None:
                    state.miss_pos = state.valid_pos
                    state.miss_bins = state.valid_bins
                if state.miss_future is not None:
                    cancelled = state.miss_future.cancel()
                    if cancelled:
                        self._release_async_queue_slot()
                    state.miss_future = None
                    state.async_overlap_used = False
                    state.fallback_degrade_count += 1

    def _hybrid_finalize_miss(self, state: _MoEHybridPhaseState, input_tensor: torch.Tensor, pool) -> None:
        assert state.miss_mask is not None
        with NvtxAnnotate("COLoRA_MissPath_ExecuteAndCommit"):
            if not state.miss_keys:
                return
            if not torch.any(state.miss_mask):
                return
            assert state.miss_pos is not None and state.miss_bins is not None
            if state.miss_policy in (MISS_POLICY_NO_CPU_PATH, MISS_POLICY_LOAD_THEN_RUN):
                raise RuntimeError(f"COLoRA {state.miss_policy} cannot execute remaining misses on the CPU path.")

            if state.miss_future is not None:
                with NvtxAnnotate("COLoRA_CPU_Miss_Path_AsyncJoin"):
                    join_t0 = time.perf_counter()
                    # Returns: miss_output, queue_wait, cpu_t, kernel_calls, kernel_tokens, internal_h2d
                    miss_output, queue_wait, cpu_t, kernel_calls, kernel_tokens, internal_h2d = state.miss_future.result()
                    state.cpu_join_stall_time += max(time.perf_counter() - join_t0, 0.0)
                    state.h2d_residual_time += internal_h2d
            else:
                with NvtxAnnotate("COLoRA_CPU_Miss_Path"):
                    t0 = time.perf_counter()
                    miss_input = state.miss_input
                    if miss_input is None:
                        _d2h_t0 = time.perf_counter()
                        miss_input = input_tensor.index_select(0, state.miss_pos).contiguous()
                        if input_tensor.device.type == "cuda":
                            state.d2h_activation_time += max(time.perf_counter() - _d2h_t0, 0.0)
                    miss_plan = self._build_cpu_group_plan(state.miss_bins)
                    miss_output, kernel_calls, kernel_tokens = self._strict_moe_cpu_batch_lora(
                        miss_input,
                        state.buffer_layer_id,
                        pool,
                        state.miss_bins,
                        projection=state.projection,
                        adapter_group_plan=miss_plan,
                        temporal_prefetch_context=state.decode_context,
                    )
                    queue_wait = 0.0
                    cpu_t = time.perf_counter() - t0
                    # H2D transfer is tracked separately in h2d_residual_time, not cpu_compute_time
                    if miss_output.device.type == "cuda" and miss_input.device.type == "cpu":
                        # _strict_moe_cpu_batch_lora internally did the H2D and tracked it
                        # We need to: 1) transfer to state, 2) subtract from cpu_t
                        internal_h2d = self._last_colora_stats.get("h2d_residual_time", 0.0)
                        state.h2d_residual_time += internal_h2d
                        cpu_t = max(cpu_t - internal_h2d, 0.0)
                        self._last_colora_stats["h2d_residual_time"] = 0.0
                    state.cpu_inline_executed += 1

            _merge_t0 = time.perf_counter()
            with NvtxAnnotate("COLoRA_MissPath_Writeback"):
                state.output.index_copy_(0, state.miss_pos, miss_output)
            if miss_output.device.type == "cuda":
                torch.cuda.synchronize()
            state.merge_time += max(time.perf_counter() - _merge_t0, 0.0)
            with NvtxAnnotate("COLoRA_MissPath_StatsAccumulate"):
                state.cpu_compute_time += float(cpu_t)
                state.cpu_queue_wait_time += float(queue_wait)
                state.moe_kernel_calls += int(kernel_calls)
                state.moe_kernel_tokens += int(kernel_tokens)

            if state.miss_policy == MISS_POLICY_NO_DEFERRED_SYNC and state.miss_keys:
                with NvtxAnnotate("COLoRA_MissPath_BlockingPromotionSync"):
                    promotion_t0 = time.perf_counter()
                    promotion_result = state.manager.promote_blocking(state.miss_keys)
                    state.blocking_promotion_time += max(time.perf_counter() - promotion_t0, 0.0)
                    state.blocking_promotion_bytes += float(promotion_result.transferred_bytes)
                    state.blocking_promotion_count += int(promotion_result.promoted_count)

    def _hybrid_finalize_stats(self, state: _MoEHybridPhaseState) -> None:
        with NvtxAnnotate("COLoRA_PostCompute_StatsFinalize"):
            overlap_ratio = 0.0
            if state.async_overlap_used and state.cpu_compute_time > 0.0 and state.gpu_compute_time > 0.0:
                overlap = min(state.cpu_compute_time, state.gpu_compute_time)
                overlap_ratio = overlap / max(state.cpu_compute_time + state.gpu_compute_time, 1e-9)
            drop_breakdown = {}
            if hasattr(state.manager, "get_promotion_drop_breakdown"):
                drop_breakdown = state.manager.get_promotion_drop_breakdown()

        assert state.hit_mask is not None and state.miss_mask is not None
        self._last_colora_stats = {
            "colora_hit_tokens": int(state.hit_mask.sum().item()),
            "colora_miss_tokens": int(state.miss_mask.sum().item()),
            "promotion_queue_depth": state.manager.get_promotion_queue_depth(),
            "cache_hit_rate": state.manager.get_hit_rate(),
            "cpu_compute_time": state.cpu_compute_time,
            "gpu_compute_time": state.gpu_compute_time,
            "cpu_queue_wait_time": state.cpu_queue_wait_time,
            "cpu_queue_admit_wait_time": state.cpu_queue_admit_wait_time,
            "cpu_join_stall_time": state.cpu_join_stall_time,
            "d2h_bytes": state.d2h_bytes,
            "h2d_bytes": state.h2d_bytes,
            "weight_h2d_bytes": state.blocking_promotion_bytes,
            "weight_h2d_time": state.blocking_promotion_time,
            "overlap_ratio": overlap_ratio,
            "overlap_down_gemm_ms": 0.0,
            "fallback_degrade_count": int(state.fallback_degrade_count),
            "cpu_async_submitted": int(state.cpu_async_submitted),
            "cpu_inline_executed": int(state.cpu_inline_executed),
            "blocking_promotion_count": int(state.blocking_promotion_count),
            "miss_policy": state.miss_policy,
            "overlap_mode": self.colora_overlap_mode,
            "cpu_queue_depth": self._get_cpu_queue_depth(),
            "promotion_drop_total": int(drop_breakdown.get("total", 0)),
            "promotion_drop_queue_high_watermark": int(drop_breakdown.get("queue_high_watermark", 0)),
            "promotion_drop_cooldown": int(drop_breakdown.get("cooldown", 0)),
            "moe_kernel_calls": int(state.moe_kernel_calls),
            "moe_kernel_tokens": int(state.moe_kernel_tokens),
            "promotion_admitted": int(self._last_colora_stats.get("promotion_admitted", 0)),
            "promotion_reject_delta": int(self._last_colora_stats.get("promotion_reject_delta", 0)),
            "promotion_reject_no_ema": int(self._last_colora_stats.get("promotion_reject_no_ema", 0)),
            "tracker_queue_drop": int(self._last_colora_stats.get("tracker_queue_drop", 0)),
            "prefetch_submitted": int(self._last_colora_stats.get("prefetch_submitted", 0)),
            "prefetch_ready_hits": int(self._last_colora_stats.get("prefetch_ready_hits", 0)),
            "prefetch_not_ready": int(self._last_colora_stats.get("prefetch_not_ready", 0)),
            "prefetch_stale": int(self._last_colora_stats.get("prefetch_stale", 0)),
            "prefetch_false_positives": int(self._last_colora_stats.get("prefetch_false_positives", 0)),
            "prefetch_slot_overwrite": int(
                self._temporal_hot_cache.get_slot_overwrite_count() if self._temporal_hot_cache is not None else 0
            ),
            # P2 microbench component timings
            # Combine state-level and _strict_moe_cpu_batch_lora-level timings
            "pack_time": state.pack_time,
            "d2h_activation_time": state.d2h_activation_time + float(self._last_colora_stats.get("d2h_activation_time", 0.0)),
            "h2d_residual_time": state.h2d_residual_time + float(self._last_colora_stats.get("h2d_residual_time", 0.0)),
            "merge_time": state.merge_time,
            "admit_time": state.admit_time,
        }

    def begin_moe_hybrid_miss_async(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        buffer_layer_id: int,
        pool,
        bins: torch.Tensor,
        projection: str,
        expert_id: Optional[int],
        hybrid_prepare_ctx: Optional[MoEHybridSharedPrepareContext] = None,
    ) -> Optional[_MoEHybridMissTicket]:
        manager = self.expert_cache_manager
        if manager is None:
            return None
        decode_context = self._get_current_decode_joint_context()
        state = self._hybrid_prepare_through_masks(
            input_tensor=input_tensor,
            layer_id=layer_id,
            buffer_layer_id=buffer_layer_id,
            pool=pool,
            bins=bins,
            projection=projection,
            expert_id=expert_id,
            hybrid_prepare_ctx=hybrid_prepare_ctx,
            manager=manager,
            decode_context=decode_context,
        )
        if state is None:
            return None
        self._hybrid_try_submit_cpu_miss(state=state, input_tensor=input_tensor, pool=pool, worker_owned_d2h=True)
        return _MoEHybridMissTicket(state=state, input_tensor=input_tensor, pool=pool)

    def finish_moe_hybrid(self, ticket: Optional[_MoEHybridMissTicket]) -> Optional[torch.Tensor]:
        if ticket is None:
            return None
        self._hybrid_run_gpu_hit(state=ticket.state, input_tensor=ticket.input_tensor, pool=ticket.pool)
        self._hybrid_finalize_miss(state=ticket.state, input_tensor=ticket.input_tensor, pool=ticket.pool)
        self._hybrid_finalize_stats(ticket.state)
        return ticket.state.output

    def begin_moe_down_hybrid_miss_async(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        buffer_layer_id: Optional[int] = None,
        pool=None,
        bins: Optional[torch.Tensor] = None,
        expert_id: Optional[int] = None,
        hybrid_prepare_ctx: Optional[MoEHybridSharedPrepareContext] = None,
    ) -> Optional[_MoEHybridMissTicket]:
        if bins is None:
            return None
        if pool is None:
            pool = getattr(getattr(self, "lora_mem_pool", None), "moe_down_pool", None)
        if pool is None:
            return None
        if buffer_layer_id is None:
            buffer_layer_id = self._get_moe_buffer_layer_id(pool, layer_id, expert_id)
        return self.begin_moe_hybrid_miss_async(
            input_tensor=input_tensor,
            layer_id=layer_id,
            buffer_layer_id=buffer_layer_id,
            pool=pool,
            bins=bins,
            projection="down",
            expert_id=expert_id,
            hybrid_prepare_ctx=hybrid_prepare_ctx,
        )

    def begin_moe_gate_hybrid_miss_async(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        buffer_layer_id: Optional[int] = None,
        pool=None,
        bins: Optional[torch.Tensor] = None,
        expert_id: Optional[int] = None,
        hybrid_prepare_ctx: Optional[MoEHybridSharedPrepareContext] = None,
    ) -> Optional[_MoEHybridMissTicket]:
        if bins is None:
            return None
        if pool is None:
            pool = getattr(getattr(self, "lora_mem_pool", None), "moe_gate_pool", None)
        if pool is None:
            return None
        if buffer_layer_id is None:
            buffer_layer_id = self._get_moe_buffer_layer_id(pool, layer_id, expert_id)
        return self.begin_moe_hybrid_miss_async(
            input_tensor=input_tensor,
            layer_id=layer_id,
            buffer_layer_id=buffer_layer_id,
            pool=pool,
            bins=bins,
            projection="gate",
            expert_id=expert_id,
            hybrid_prepare_ctx=hybrid_prepare_ctx,
        )

    def begin_moe_up_hybrid_miss_async(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        buffer_layer_id: Optional[int] = None,
        pool=None,
        bins: Optional[torch.Tensor] = None,
        expert_id: Optional[int] = None,
        hybrid_prepare_ctx: Optional[MoEHybridSharedPrepareContext] = None,
    ) -> Optional[_MoEHybridMissTicket]:
        if bins is None:
            return None
        if pool is None:
            pool = getattr(getattr(self, "lora_mem_pool", None), "moe_up_pool", None)
        if pool is None:
            return None
        if buffer_layer_id is None:
            buffer_layer_id = self._get_moe_buffer_layer_id(pool, layer_id, expert_id)
        return self.begin_moe_hybrid_miss_async(
            input_tensor=input_tensor,
            layer_id=layer_id,
            buffer_layer_id=buffer_layer_id,
            pool=pool,
            bins=bins,
            projection="up",
            expert_id=expert_id,
            hybrid_prepare_ctx=hybrid_prepare_ctx,
        )

    def finish_moe_down_hybrid(self, ticket: Optional[_MoEHybridMissTicket]) -> Optional[torch.Tensor]:
        return self.finish_moe_hybrid(ticket)

    def finish_moe_gate_hybrid(self, ticket: Optional[_MoEHybridMissTicket]) -> Optional[torch.Tensor]:
        return self.finish_moe_hybrid(ticket)

    def finish_moe_up_hybrid(self, ticket: Optional[_MoEHybridMissTicket]) -> Optional[torch.Tensor]:
        return self.finish_moe_hybrid(ticket)

    def _ensure_compact_scratchpad(self, pool, device, active_count: int):
        """
        Allocate scratchpad for ONLY active adapters (compact, not full pool).
        Avoids OOM with thousands of adapters.

        NOTE: Always reallocate when pool dimensions change (e.g., switching from Q to K pool).
        """
        max_rank = pool.max_rank
        a_hidden = pool.key_buffer.shape[2]
        b_hidden = pool.value_buffer.shape[2]

        # Check if we need to resize or if dimensions changed
        needs_resize = False
        current_a_hidden = self.gpu_scratchpad_a.shape[2] if self.gpu_scratchpad_a is not None else 0
        current_b_hidden = self.gpu_scratchpad_b.shape[2] if self.gpu_scratchpad_b is not None else 0

        if self.gpu_scratchpad_a is None or self.gpu_scratchpad_a.shape[0] < active_count:
            needs_resize = True
        elif current_a_hidden != a_hidden or current_b_hidden != b_hidden:
            # Pool dimensions changed (e.g., Q pool -> K pool), need to reallocate
            needs_resize = True

        if needs_resize:
            # Allocate for active count only
            self.gpu_scratchpad_a = torch.empty(
                (active_count, max_rank, a_hidden),
                dtype=pool.key_buffer.dtype, device=device
            )
            self.gpu_scratchpad_b = torch.empty(
                (active_count, max_rank, b_hidden),
                dtype=pool.value_buffer.dtype, device=device
            )
            # Create dedicated stream for async transfers
            if device.type == "cuda" and self.transfer_stream is None:
                self.transfer_stream = torch.cuda.Stream(priority=0)

        # Phase 3: grow BGMV temp index buffers in lockstep with scratchpad.
        self._ensure_bgmv_temp_index_buffers(device, active_count)

    def _ensure_bgmv_temp_index_buffers(self, device, active_count: int) -> None:
        """Phase 3: lazily allocate / grow int32 arange and ones buffers on ``device``.

        These are consumed as ``temp_a_start`` / ``temp_a_len`` by the BGMV kernel.
        They only depend on ``active_count``, so we reuse views across calls and
        only reallocate when the current capacity is too small or the device moved.
        """
        if active_count <= 0:
            return
        current = self._bgmv_temp_arange_gpu
        need_resize = (
            current is None
            or current.device != device
            or current.numel() < active_count
        )
        if need_resize:
            # Grow geometrically so repeated small bumps do not thrash.
            new_capacity = max(active_count, 1)
            if current is not None and current.numel() > 0:
                new_capacity = max(new_capacity, current.numel() * 2)
            self._bgmv_temp_arange_gpu = torch.arange(
                new_capacity, device=device, dtype=torch.int32
            )
            self._bgmv_temp_ones_gpu = torch.ones(
                new_capacity, device=device, dtype=torch.int32
            )

    def _get_bgmv_temp_index_buffers(
        self, device, active_count: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(temp_a_start, temp_a_len)`` views for the BGMV kernel."""
        self._ensure_bgmv_temp_index_buffers(device, active_count)
        assert self._bgmv_temp_arange_gpu is not None and self._bgmv_temp_ones_gpu is not None
        return (
            self._bgmv_temp_arange_gpu[:active_count],
            self._bgmv_temp_ones_gpu[:active_count],
        )

    def _get_pool_a_scaling_gpu(self, pool, device) -> Optional[torch.Tensor]:
        """Phase 2: return a GPU-resident mirror of ``pool.a_scaling``.

        The mirror is built lazily on first access and rebuilt whenever the CPU
        source tensor is replaced (detected by data_ptr / numel change), which
        covers adapter load/unload (see ``LoRAModulePool.load_adapter_weights``
        and ``unload_adapter``).

        Returns ``None`` if ``device`` is not CUDA or if the source is empty; the
        caller must fall back to the CPU path in that case.
        """
        if device is None or device.type != "cuda":
            return None
        src = getattr(pool, "a_scaling", None)
        if src is None:
            return None
        numel = int(src.numel())
        if numel == 0:
            return None
        key = id(pool)
        device_id = device.index if device.index is not None else torch.cuda.current_device()
        src_ptr = int(src.data_ptr())
        cached = self._pool_scaling_gpu_cache.get(key)
        if cached is not None:
            mirror, cached_ptr, cached_numel, cached_dev = cached
            if (
                cached_ptr == src_ptr
                and cached_numel == numel
                and cached_dev == device_id
                and mirror.device == device
            ):
                return mirror
        mirror = src.detach().to(device=device, non_blocking=True)
        self._pool_scaling_gpu_cache[key] = (mirror, src_ptr, numel, device_id)
        return mirror

    def _transfer_compact_to_gpu(self, pool, layer_id: int, global_adapter_ids: torch.Tensor,
                                  a_dest: torch.Tensor, b_dest: torch.Tensor):
        """
        Transfer ONLY active adapters from CPU pool to compact GPU scratchpad.
        Maps: Global[5, 999] → Local[0, 1]
        """
        # Get CPU slot indices for each active adapter (indices must be on same device as indexed tensor)
        cpu_slots = pool.a_start[global_adapter_ids.cpu()] + layer_id  # [active_count]

        logger.debug(f"[LoRA Transfer] pool={type(pool).__name__}, layer_id={layer_id}, active_count={len(global_adapter_ids)}")
        logger.debug(f"[LoRA Transfer] a_dest[0].shape={a_dest[0].shape if len(a_dest) > 0 else 'empty'}, b_dest[0].shape={b_dest[0].shape if len(b_dest) > 0 else 'empty'}")
        logger.debug(f"[LoRA Transfer] pool.key_buffer.shape={pool.key_buffer.shape}, pool.value_buffer.shape={pool.value_buffer.shape}")

        pool_size = int(pool.key_buffer.shape[0])
        num_meta = int(pool.a_start.shape[0])
        for i in range(len(global_adapter_ids)):
            gid = int(global_adapter_ids[i].item())
            if gid < 0 or gid >= num_meta:
                raise RuntimeError(
                    f"[LoRA Transfer] global adapter id {gid} out of metadata range [0, {num_meta}) "
                    f"(pool={type(pool).__name__})"
                )

        # Async transfer using dedicated stream
        if self.transfer_stream is not None:
            with torch.cuda.stream(self.transfer_stream):
                for i in range(len(global_adapter_ids)):
                    adapter_id = global_adapter_ids[i].item()
                    if adapter_id < 0:
                        continue
                    slot = cpu_slots[i].item()
                    if slot < 0 or slot >= pool_size:
                        raise RuntimeError(
                            f"[LoRA Transfer] computed slot {slot} out of buffer bounds [0, {pool_size}) "
                            f"(adapter_id={adapter_id}, layer_id={layer_id})"
                        )
                    logger.debug(f"[LoRA Transfer] i={i}, slot={slot}, key_buffer[{slot}].shape={pool.key_buffer[slot].shape}, value_buffer[{slot}].shape={pool.value_buffer[slot].shape}")
                    a_dest[i].copy_(pool.key_buffer[slot], non_blocking=True)
                    b_dest[i].copy_(pool.value_buffer[slot], non_blocking=True)
            self.transfer_stream.synchronize()  # Sync for baseline correctness
        else:
            # Sync fallback (no CUDA stream available)
            for i in range(len(global_adapter_ids)):
                adapter_id = global_adapter_ids[i].item()
                if adapter_id < 0:
                    continue
                slot = cpu_slots[i].item()
                if slot < 0 or slot >= pool_size:
                    raise RuntimeError(
                        f"[LoRA Transfer] computed slot {slot} out of buffer bounds [0, {pool_size}) "
                        f"(adapter_id={adapter_id}, layer_id={layer_id})"
                    )
                a_dest[i].copy_(pool.key_buffer[slot])
                b_dest[i].copy_(pool.value_buffer[slot])

    def batch_apply_q_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply q_proj LoRA to batch with different adapters."""
        if self.lora_mem_pool is None or self.lora_mem_pool.attn_q_pool is None:
            return torch.zeros_like(input_tensor)

        pool = self.lora_mem_pool.attn_q_pool
        bins = req_bins if req_bins is not None else self.req_bins
        if bins is None:
            return torch.zeros_like(input_tensor)

        # Use CPU compute if configured
        if self._should_use_cpu_compute("attn"):
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

        if BGMV_AVAILABLE:
            output = self._get_output_buffer(input_tensor, pool)

            # Compact dispatcher: CPU storage + GPU compute
            if self._should_use_cpu_storage("attn"):
                # 1. Identify ONLY the adapters needed for this batch
                unique_adapters, inverse_indices = torch.unique(bins, return_inverse=True)
                active_count = unique_adapters.size(0)

                # 2. Allocate SMALL scratchpad (only for active adapters)
                self._ensure_compact_scratchpad(pool, input_tensor.device, active_count)

                # 3. Transfer ONLY active adapters to packed scratchpad
                assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None
                self._transfer_compact_to_gpu(
                    pool, layer_id, unique_adapters,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b
                )

                # 4. Create TEMPORARY compact metadata for kernel
                temp_a_start, temp_a_len = self._get_bgmv_temp_index_buffers(
                    input_tensor.device, int(active_count)
                )
                temp_scaling = pool.a_scaling[unique_adapters.long().cpu()].to(input_tensor.device).to(input_tensor.device)

                # 5. Launch kernel with REMAPPED indices
                assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None
                batch_lora_get_qkv(
                    output, input_tensor,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b,
                    temp_a_start, temp_a_len, temp_scaling,
                    inverse_indices,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=0,
                    **_bgmv_trace_kwargs(pool, "batch_apply_q_lora/attn_q/compact", compact=True),
                )
            else:
                # Standard GPU path
                batch_lora_get_qkv(
                    output, input_tensor,
                    pool.key_buffer,
                    pool.value_buffer,
                    pool.a_start,
                    pool.a_len,
                    pool.a_scaling,
                    bins,
                    layer_id=layer_id,
                    **_bgmv_trace_kwargs(pool, "batch_apply_q_lora/attn_q"),
                )
            return output
        else:
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

    def batch_apply_k_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply k_proj LoRA (GQA Aware). Output dim < Input dim."""
        if self.lora_mem_pool is None or self.lora_mem_pool.attn_k_pool is None:
            return torch.zeros(input_tensor.shape[0], 0, device=input_tensor.device)

        pool = self.lora_mem_pool.attn_k_pool
        
        bins = req_bins if req_bins is not None else self.req_bins
        if bins is None:
            return torch.zeros(input_tensor.shape[0], pool.value_buffer.shape[2], dtype=input_tensor.dtype, device=input_tensor.device)

        output = self._get_output_buffer(input_tensor, pool)

        # Use CPU compute if configured
        if self._should_use_cpu_compute("attn"):
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

        if BGMV_AVAILABLE:
            # Compact dispatcher: CPU storage + GPU compute
            if self._should_use_cpu_storage("attn"):
                unique_adapters, inverse_indices = torch.unique(bins, return_inverse=True)
                active_count = unique_adapters.size(0)
                self._ensure_compact_scratchpad(pool, input_tensor.device, active_count)
                assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None
                self._transfer_compact_to_gpu(
                    pool, layer_id, unique_adapters,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b
                )
                temp_a_start, temp_a_len = self._get_bgmv_temp_index_buffers(
                    input_tensor.device, int(active_count)
                )
                temp_scaling = pool.a_scaling[unique_adapters.long().cpu()].to(input_tensor.device)
                batch_lora_get_qkv(
                    output, input_tensor,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b,
                    temp_a_start, temp_a_len, temp_scaling,
                    inverse_indices,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=0,
                    **_bgmv_trace_kwargs(pool, "batch_apply_k_lora/attn_k/compact", compact=True),
                )
            else:
                batch_lora_get_qkv(
                    output, input_tensor,
                    pool.key_buffer, pool.value_buffer,
                    pool.a_start, pool.a_len, pool.a_scaling, bins,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=layer_id,
                    **_bgmv_trace_kwargs(pool, "batch_apply_k_lora/attn_k"),
                )
            return output
        else:
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

    def batch_apply_v_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply v_proj LoRA (GQA Aware)."""
        if self.lora_mem_pool is None or self.lora_mem_pool.attn_v_pool is None:
            return torch.zeros_like(input_tensor)

        pool = self.lora_mem_pool.attn_v_pool
        bins = req_bins if req_bins is not None else self.req_bins
        if bins is None:
            return torch.zeros_like(input_tensor)

        output = self._get_output_buffer(input_tensor, pool)

        # Use CPU compute if configured
        if self._should_use_cpu_compute("attn"):
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

        if BGMV_AVAILABLE:
            # Compact dispatcher: CPU storage + GPU compute
            if self._should_use_cpu_storage("attn"):
                unique_adapters, inverse_indices = torch.unique(bins, return_inverse=True)
                active_count = unique_adapters.size(0)
                self._ensure_compact_scratchpad(pool, input_tensor.device, active_count)
                assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None
                self._transfer_compact_to_gpu(
                    pool, layer_id, unique_adapters,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b
                )
                temp_a_start, temp_a_len = self._get_bgmv_temp_index_buffers(
                    input_tensor.device, int(active_count)
                )
                temp_scaling = pool.a_scaling[unique_adapters.long().cpu()].to(input_tensor.device)
                batch_lora_get_qkv(
                    output, input_tensor,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b,
                    temp_a_start, temp_a_len, temp_scaling,
                    inverse_indices,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=0,
                    **_bgmv_trace_kwargs(pool, "batch_apply_v_lora/attn_v/compact", compact=True),
                )
            else:
                batch_lora_get_qkv(
                    output, input_tensor,
                    pool.key_buffer, pool.value_buffer,
                    pool.a_start, pool.a_len, pool.a_scaling, bins,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=layer_id,
                    **_bgmv_trace_kwargs(pool, "batch_apply_v_lora/attn_v"),
                )
            return output
        else:
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

    def batch_apply_o_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply o_proj LoRA to batch with different adapters."""
        if self.lora_mem_pool is None or self.lora_mem_pool.attn_o_pool is None:
            return torch.zeros_like(input_tensor)

        pool = self.lora_mem_pool.attn_o_pool
        bins = req_bins if req_bins is not None else self.req_bins
        if bins is None:
            return torch.zeros_like(input_tensor)

        # Use CPU compute if configured
        if self._should_use_cpu_compute("attn"):
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

        if BGMV_AVAILABLE:
            output = torch.zeros_like(input_tensor)
            # Compact dispatcher: CPU storage + GPU compute
            if self._should_use_cpu_storage("attn"):
                unique_adapters, inverse_indices = torch.unique(bins, return_inverse=True)
                active_count = unique_adapters.size(0)
                self._ensure_compact_scratchpad(pool, input_tensor.device, active_count)
                assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None
                self._transfer_compact_to_gpu(
                    pool, layer_id, unique_adapters,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b
                )
                temp_a_start, temp_a_len = self._get_bgmv_temp_index_buffers(
                    input_tensor.device, int(active_count)
                )
                temp_scaling = pool.a_scaling[unique_adapters.long().cpu()].to(input_tensor.device)
                batch_lora_get_o(
                    output,
                    input_tensor,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b,
                    temp_a_start, temp_a_len, temp_scaling,
                    inverse_indices,
                    layer_id=0,
                    **_bgmv_trace_kwargs(pool, "batch_apply_o_lora/attn_o/compact", compact=True),
                )
            else:
                batch_lora_get_o(
                    output,
                    input_tensor,
                    pool.key_buffer,
                    pool.value_buffer,
                    pool.a_start,
                    pool.a_len,
                    pool.a_scaling,
                    bins,
                    layer_id=layer_id,
                    **_bgmv_trace_kwargs(pool, "batch_apply_o_lora/attn_o"),
                )
            return output
        else:
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

    @NvtxAnnotate("batch_apply_gate_lora")
    def batch_apply_gate_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None,
        expert_id: Optional[int] = None,
        hybrid_prepare_ctx: Optional[MoEHybridSharedPrepareContext] = None,
    ) -> torch.Tensor:
        """Apply gate_proj LoRA to batch with different adapters.

        Args:
            input_tensor: Input tensor [batch, hidden]
            layer_id: Layer index
            req_bins: Request to adapter mapping
            expert_id: For MoE, the LOCAL expert index to apply LoRA for.
                       In EP mode, this is the local index (0 to num_local_experts-1).
                       If None, uses base layer_id (for weight loading).
        """
        if self.lora_mem_pool is None or self.lora_mem_pool.moe_gate_pool is None:
            return torch.zeros_like(input_tensor)

        pool = self.lora_mem_pool.moe_gate_pool
        bins = req_bins if req_bins is not None else self.req_bins
        if bins is None:
            return torch.zeros_like(input_tensor)

        # Calculate buffer index with expert dimension
        # expert_id is now always LOCAL (after translation in transformer_layer_infer.py)
        if expert_id is not None:
            # Use pool's actual num_experts which matches local expert count
            buffer_layer_id = layer_id * pool.num_experts + expert_id
        else:
            buffer_layer_id = layer_id

        if self._should_use_hybrid_moe_compute():
            return self._batch_apply_moe_lora_hybrid(
                input_tensor=input_tensor,
                layer_id=layer_id,
                buffer_layer_id=buffer_layer_id,
                pool=pool,
                bins=bins,
                projection="gate",
                expert_id=expert_id,
                hybrid_prepare_ctx=hybrid_prepare_ctx,
            )

        # Use CPU compute if configured
        if self._should_use_cpu_compute("moe"):
            with NvtxAnnotate("batch_apply_gate_lora_cpu"):
                out, _, _ = self._strict_moe_cpu_batch_lora(
                    input_tensor,
                    buffer_layer_id,
                    pool,
                    bins,
                    projection="gate",
                )
                return out

        if BGMV_AVAILABLE:
            output = self._get_output_buffer(input_tensor, pool)
            # Compact dispatcher: CPU storage + GPU compute
            if self._should_use_cpu_storage("moe"):
                with NvtxAnnotate("batch_apply_gate_lora_gpu_compact"):
                    unique_adapters, inverse_indices = torch.unique(bins, return_inverse=True)
                    active_count = unique_adapters.size(0)
                    self._ensure_compact_scratchpad(pool, input_tensor.device, active_count)
                    assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None
                    
                    with NvtxAnnotate("batch_apply_gate_lora_data_movement"):
                        self._transfer_compact_to_gpu(
                            pool, buffer_layer_id, unique_adapters,
                            self.gpu_scratchpad_a, self.gpu_scratchpad_b
                        )
                        temp_a_start, temp_a_len = self._get_bgmv_temp_index_buffers(
                            input_tensor.device, int(active_count)
                        )
                        temp_scaling = pool.a_scaling[unique_adapters.long().cpu()].to(input_tensor.device)

                    batch_lora_get_mlp(
                        output,
                        input_tensor,
                        self.gpu_scratchpad_a, self.gpu_scratchpad_b,
                        temp_a_start, temp_a_len, temp_scaling,
                        inverse_indices,
                        a_hidden_dim=input_tensor.shape[1],
                        b_hidden_dim=output.shape[1],
                        layer_id=0,
                        **_bgmv_trace_kwargs(pool, "batch_apply_gate_lora/moe_gate/compact", compact=True),
                    )
            else:
                with NvtxAnnotate("batch_apply_gate_lora_gpu"):
                    batch_lora_get_mlp(
                        output,
                        input_tensor,
                        pool.key_buffer,
                        pool.value_buffer,
                        pool.a_start,
                        pool.a_len,
                        pool.a_scaling,
                        bins,
                        a_hidden_dim=input_tensor.shape[1],
                        b_hidden_dim=output.shape[1],
                        layer_id=buffer_layer_id,
                        **_bgmv_trace_kwargs(pool, "batch_apply_gate_lora/moe_gate"),
                    )
            return output
        else:
            with NvtxAnnotate("batch_apply_gate_lora_cpu"):
                return self._naive_batch_lora(input_tensor, buffer_layer_id, pool, bins)

    @NvtxAnnotate("batch_apply_up_lora")
    def batch_apply_up_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None,
        expert_id: Optional[int] = None,
        hybrid_prepare_ctx: Optional[MoEHybridSharedPrepareContext] = None,
    ) -> torch.Tensor:
        """Apply up_proj LoRA to batch with different adapters.

        Args:
            input_tensor: Input tensor [batch, hidden]
            layer_id: Layer index
            req_bins: Request to adapter mapping
            expert_id: For MoE, the expert index to apply LoRA for.
                       If None, uses base layer_id (for weight loading).
        """
        if self.lora_mem_pool is None or self.lora_mem_pool.moe_up_pool is None:
            return torch.zeros_like(input_tensor)

        pool = self.lora_mem_pool.moe_up_pool
        bins = req_bins if req_bins is not None else self.req_bins
        if bins is None:
            return torch.zeros_like(input_tensor)

        # [MODIFIED] Calculate buffer index with expert dimension
        if expert_id is not None:
            buffer_layer_id = layer_id * pool.num_experts + expert_id
        else:
            buffer_layer_id = layer_id

        if self._should_use_hybrid_moe_compute():
            return self._batch_apply_moe_lora_hybrid(
                input_tensor=input_tensor,
                layer_id=layer_id,
                buffer_layer_id=buffer_layer_id,
                pool=pool,
                bins=bins,
                projection="up",
                expert_id=expert_id,
                hybrid_prepare_ctx=hybrid_prepare_ctx,
            )

        # Use CPU compute if configured
        if self._should_use_cpu_compute("moe"):
            out, _, _ = self._strict_moe_cpu_batch_lora(
                input_tensor,
                buffer_layer_id,
                pool,
                bins,
                projection="up",
            )
            return out

        if BGMV_AVAILABLE:
            output = self._get_output_buffer(input_tensor, pool)
            # Compact dispatcher: CPU storage + GPU compute
            if self._should_use_cpu_storage("moe"):
                unique_adapters, inverse_indices = torch.unique(bins, return_inverse=True)
                active_count = unique_adapters.size(0)
                self._ensure_compact_scratchpad(pool, input_tensor.device, active_count)
                assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None
                self._transfer_compact_to_gpu(
                    pool, buffer_layer_id, unique_adapters,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b
                )
                temp_a_start, temp_a_len = self._get_bgmv_temp_index_buffers(
                    input_tensor.device, int(active_count)
                )
                temp_scaling = pool.a_scaling[unique_adapters.long().cpu()].to(input_tensor.device)
                batch_lora_get_mlp(
                    output,
                    input_tensor,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b,
                    temp_a_start, temp_a_len, temp_scaling,
                    inverse_indices,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=0,
                    **_bgmv_trace_kwargs(pool, "batch_apply_up_lora/moe_up/compact", compact=True),
                )
            else:
                batch_lora_get_mlp(
                    output,
                    input_tensor,
                    pool.key_buffer,
                    pool.value_buffer,
                    pool.a_start,
                    pool.a_len,
                    pool.a_scaling,
                    bins,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=buffer_layer_id,
                    **_bgmv_trace_kwargs(pool, "batch_apply_up_lora/moe_up"),
                )
            return output
        else:
            return self._naive_batch_lora(input_tensor, buffer_layer_id, pool, bins)

    @NvtxAnnotate("batch_apply_down_lora")
    def batch_apply_down_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None,
        expert_id: Optional[int] = None,
        hybrid_prepare_ctx: Optional[MoEHybridSharedPrepareContext] = None,
    ) -> torch.Tensor:
        """Apply down_proj LoRA. Input: Intermediate, Output: Hidden.

        Args:
            input_tensor: Input tensor [batch, hidden]
            layer_id: Layer index
            req_bins: Request to adapter mapping
            expert_id: For MoE, the expert index to apply LoRA for.
                       If None, uses base layer_id (for weight loading).
        """
        if self.lora_mem_pool is None or self.lora_mem_pool.moe_down_pool is None:
            return torch.zeros_like(input_tensor)

        pool = self.lora_mem_pool.moe_down_pool
        bins = req_bins if req_bins is not None else self.req_bins
        if bins is None:
            return torch.zeros_like(input_tensor)

        # [MODIFIED] Calculate buffer index with expert dimension
        if expert_id is not None:
            buffer_layer_id = layer_id * pool.num_experts + expert_id
        else:
            buffer_layer_id = layer_id

        if self._should_use_hybrid_moe_compute():
            return self._batch_apply_moe_lora_hybrid(
                input_tensor=input_tensor,
                layer_id=layer_id,
                buffer_layer_id=buffer_layer_id,
                pool=pool,
                bins=bins,
                projection="down",
                expert_id=expert_id,
                hybrid_prepare_ctx=hybrid_prepare_ctx,
            )

        output = self._get_output_buffer(input_tensor, pool)

        # Use CPU compute if configured
        if self._should_use_cpu_compute("moe"):
            out, _, _ = self._strict_moe_cpu_batch_lora(
                input_tensor,
                buffer_layer_id,
                pool,
                bins,
                projection="down",
            )
            return out

        if BGMV_AVAILABLE:
            # Compact dispatcher: CPU storage + GPU compute
            if self._should_use_cpu_storage("moe"):
                unique_adapters, inverse_indices = torch.unique(bins, return_inverse=True)
                active_count = unique_adapters.size(0)
                self._ensure_compact_scratchpad(pool, input_tensor.device, active_count)
                assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None
                self._transfer_compact_to_gpu(
                    pool, buffer_layer_id, unique_adapters,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b
                )
                temp_a_start, temp_a_len = self._get_bgmv_temp_index_buffers(
                    input_tensor.device, int(active_count)
                )
                temp_scaling = pool.a_scaling[unique_adapters.long().cpu()].to(input_tensor.device)
                batch_lora_get_mlp(
                    output, input_tensor,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b,
                    temp_a_start, temp_a_len, temp_scaling,
                    inverse_indices,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=0,
                    **_bgmv_trace_kwargs(pool, "batch_apply_down_lora/moe_down/compact", compact=True),
                )
            else:
                batch_lora_get_mlp(
                    output, input_tensor,
                    pool.key_buffer, pool.value_buffer,
                    pool.a_start, pool.a_len, pool.a_scaling, bins,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=buffer_layer_id,
                    **_bgmv_trace_kwargs(pool, "batch_apply_down_lora/moe_down"),
                )
            return output
        else:
            return self._naive_batch_lora(input_tensor, buffer_layer_id, pool, bins)

    def batch_apply_vl_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        target_type: str,
        req_bins: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply Vision-Language adapter LoRA to batch."""
        if self.lora_mem_pool is None:
            return torch.zeros_like(input_tensor)

        pool_map = {
            "vl_q": self.lora_mem_pool.vl_q_pool,
            "vl_k": self.lora_mem_pool.vl_k_pool,
            "vl_v": self.lora_mem_pool.vl_v_pool,
            "vl_o": self.lora_mem_pool.vl_o_pool,
            "vl_fc1": self.lora_mem_pool.vl_fc1_pool,
            "vl_fc2": self.lora_mem_pool.vl_fc2_pool,
        }

        pool = pool_map.get(target_type)
        if pool is None:
            return torch.zeros_like(input_tensor)

        bins = req_bins if req_bins is not None else self.req_bins
        if bins is None:
            return torch.zeros_like(input_tensor)

        output = self._get_output_buffer(input_tensor, pool)

        # Use CPU compute if configured
        if self._should_use_cpu_compute("vl"):
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

        if BGMV_AVAILABLE:
            # Compact dispatcher: CPU storage + GPU compute
            if self._should_use_cpu_storage("vl"):
                unique_adapters, inverse_indices = torch.unique(bins, return_inverse=True)
                active_count = unique_adapters.size(0)
                self._ensure_compact_scratchpad(pool, input_tensor.device, active_count)
                assert self.gpu_scratchpad_a is not None and self.gpu_scratchpad_b is not None
                self._transfer_compact_to_gpu(
                    pool, layer_id, unique_adapters,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b
                )
                temp_a_start, temp_a_len = self._get_bgmv_temp_index_buffers(
                    input_tensor.device, int(active_count)
                )
                temp_scaling = pool.a_scaling[unique_adapters.long().cpu()].to(input_tensor.device)
                batch_lora_get_vl(
                    output, input_tensor,
                    self.gpu_scratchpad_a, self.gpu_scratchpad_b,
                    temp_a_start, temp_a_len, temp_scaling,
                    inverse_indices,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=0,
                    **_bgmv_trace_kwargs(pool, f"batch_apply_vl_lora/{target_type}/compact", compact=True),
                )
            else:
                batch_lora_get_vl(
                    output, input_tensor,
                    pool.key_buffer, pool.value_buffer,
                    pool.a_start, pool.a_len, pool.a_scaling, bins,
                    a_hidden_dim=input_tensor.shape[1],
                    b_hidden_dim=output.shape[1],
                    layer_id=layer_id,
                    **_bgmv_trace_kwargs(pool, f"batch_apply_vl_lora/{target_type}"),
                )
            return output
        else:
            return self._naive_batch_lora(input_tensor, layer_id, pool, bins)

    # =====================================================================
    # Fallback Naive Implementation (when BGMV kernel unavailable)
    # =====================================================================

    @NvtxAnnotate
    def _naive_batch_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        pool,
        req_bins: Optional[torch.Tensor] = None,
        force_cpu: bool = False,
        adapter_group_plan: Optional[Tuple[Tuple[int, Tuple[int, ...]], ...]] = None,
    ) -> torch.Tensor:
        """
        Naive per-request LoRA computation (fallback when BGMV unavailable).

        Args:
            input_tensor: Input tensor [batch, hidden]
            layer_id: Layer index
            pool: Module pool
            req_bins: Request to adapter mapping (optional, uses self.req_bins if None)

        This is slower but correctness-preserving.
        Uses AVX-512 BF16 kernel when available on CPU.
        """
        _touch_lora_avx_flags()
        # Ensure req_bins is available
        if req_bins is None:
            req_bins = self.req_bins
        if req_bins is None:
            return torch.zeros(input_tensor.shape[0], pool.value_buffer.shape[2], dtype=input_tensor.dtype, device=input_tensor.device)

        original_device = input_tensor.device
        original_dtype = input_tensor.dtype

        compute_input = input_tensor
        if force_cpu and compute_input.device.type != "cpu":
            compute_input = compute_input.to("cpu", non_blocking=True)
        if force_cpu and AVX_AVAILABLE and compute_input.dtype != torch.bfloat16:
            compute_input = compute_input.to(dtype=torch.bfloat16)

        batch_size = compute_input.shape[0]
        # Use pool's B dimension (handles GQA where K/V output != input)
        output_dim = pool.value_buffer.shape[2]
        output = torch.zeros(batch_size, output_dim, dtype=compute_input.dtype, device=compute_input.device)

        # Truncate req_bins to match batch_size (handles decode phase with fewer requests)
        if len(req_bins) > batch_size:
            req_bins = req_bins[:batch_size]

        # Group requests by adapter (optionally reusing cached grouping plan).
        if adapter_group_plan is not None:
            adapter_groups = adapter_group_plan
        else:
            adapter_to_indices = {}
            for i, bin_idx in enumerate(req_bins):
                bin_idx = bin_idx.item()
                if bin_idx not in adapter_to_indices:
                    adapter_to_indices[bin_idx] = []
                adapter_to_indices[bin_idx].append(i)
            adapter_groups = tuple((adapter_idx, tuple(indices)) for adapter_idx, indices in adapter_to_indices.items())

        # Determine if we should use AVX kernel
        use_avx = (AVX_AVAILABLE and compute_input.device.type == 'cpu' and compute_input.dtype == torch.bfloat16)

        # Process each adapter group
        for adapter_idx, req_indices in adapter_groups:
            if adapter_idx < 0:
                continue  # Skip requests with no adapter
            req_indices = list(req_indices)

            # Get adapter metadata
            a_start = pool.a_start[adapter_idx].item()
            a_len = pool.a_len[adapter_idx].item()
            a_scaling = pool.a_scaling[adapter_idx].item()
            if hasattr(pool, "a_rank") and len(pool.a_rank) > adapter_idx:
                a_rank = int(pool.a_rank[adapter_idx].item())
            else:
                # Backward compatibility for older pool metadata.
                a_rank = int(a_len)

            # Get A and B matrices for this layer
            loc = a_start + layer_id
            if loc >= a_start + a_len:
                continue

            A = pool.key_buffer[loc, :a_rank]  # [rank, hidden]
            B = pool.value_buffer[loc, :a_rank]  # [rank, hidden]

            # Compute LoRA for each request in this group
            if use_avx:
                # Use AVX-512 BF16 kernel for batched computation
                batch_input = compute_input[req_indices]  # [n, hidden]

                # Convert to bfloat16 if needed
                if A.dtype != torch.bfloat16:
                    A = A.to(dtype=torch.bfloat16)
                    B = B.to(dtype=torch.bfloat16)

                # Ensure contiguous layout
                if not batch_input.is_contiguous():
                    batch_input = batch_input.contiguous()
                if not A.is_contiguous():
                    A = A.contiguous()
                if not B.is_contiguous():
                    B = B.contiguous()

                # Call AVX kernel for batched LoRA
                batch_output = batch_lora_avx(batch_input, A, B, a_scaling)  # [n, output_dim]
                output[req_indices] = batch_output
            else:
                # Fallback: PyTorch matmul per request
                # Move to input device and dtype if pool is on different device
                if A.device != compute_input.device or A.dtype != compute_input.dtype:
                    A = A.to(dtype=compute_input.dtype, device=compute_input.device)
                    B = B.to(dtype=compute_input.dtype, device=compute_input.device)

                for req_idx in req_indices:
                    x = compute_input[req_idx]  # [hidden]
                    # LoRA: x @ A @ B * scaling
                    # A stored as [rank, hidden], need A.T for [hidden, rank]
                    # B stored as [rank, hidden], need B for [hidden, rank]
                    # x @ A.T: [hidden] @ [hidden, rank] = [rank]
                    intermediate = torch.matmul(x, A.T)  # [rank]
                    # intermediate @ B: [rank] @ [hidden, rank] = [hidden]
                    lora_out = torch.matmul(intermediate, B) * a_scaling
                    output[req_idx] = lora_out

        if force_cpu and (original_device.type != "cpu" or output.dtype != original_dtype):
            output = output.to(device=original_device, dtype=original_dtype, non_blocking=True)
        return output

    # =====================================================================
    # Utility Methods
    # =====================================================================

    def get_attn_qkv_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply all attention LoRA (q, k, v, o) in batched mode.

        Returns dict with q_lora, k_lora, v_lora, o_lora tensors.
        """
        return {
            "q_lora": self.batch_apply_q_lora(input_tensor, layer_id, req_bins),
            "k_lora": self.batch_apply_k_lora(input_tensor, layer_id, req_bins),
            "v_lora": self.batch_apply_v_lora(input_tensor, layer_id, req_bins),
        }

    def get_mlp_lora(
        self,
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply all MLP LoRA (gate, up, down) in batched mode.

        Returns dict with gate_lora, up_lora, down_lora tensors.
        """
        return {
            "gate_lora": self.batch_apply_gate_lora(input_tensor, layer_id, req_bins),
            "up_lora": self.batch_apply_up_lora(input_tensor, layer_id, req_bins),
            "down_lora": self.batch_apply_down_lora(input_tensor, layer_id, req_bins),
        }


def create_vl_moe_lora_dispatcher(
    num_layers: int,
    lora_rank: int = 64,
    lora_alpha: float = 1.0,
    lora_compute_config: Optional[LoRAComputeConfig] = None,
    colora_async_fallback: bool = True,
    colora_cpu_workers: int = 4,
    colora_cpu_queue_depth: int = 256,
    colora_cpu_batch_timeout_us: int = 50,
    colora_deferred_promotion_delta_steps: int = 4,
    colora_promotion_ema_alpha: float = 0.5,
    colora_overlap_mode: str = OVERLAP_MODE_FULL,
    colora_temporal_prefetch: bool = False,
    colora_temporal_hot_cache_slots: int = 64,
    colora_request_skip: bool = True,
    colora_max_continuations: int = 8,
    colora_hit_indexing: str = "gpu",
    metric_client: Optional[Any] = None,
) -> Qwen3VLMoELoRADispatcher:
    """
    Factory function to create a VL-MoE LoRA dispatcher with S-LoRA batched mode.

    All modules use the same global lora_rank.

    Args:
        num_layers: Number of transformer layers
        lora_rank: Global LoRA rank for all modules (q, k, v, o, gate, up, down, vl_*)
        lora_alpha: LoRA alpha scaling factor
        lora_compute_config: Configuration for compute location per component

    Returns:
        Qwen3VLMoELoRADispatcher instance configured for batched inference
    """
    return Qwen3VLMoELoRADispatcher(
        num_layers=num_layers,
        q_lora_rank=lora_rank,
        k_lora_rank=lora_rank,
        v_lora_rank=lora_rank,
        o_lora_rank=lora_rank,
        gate_lora_rank=lora_rank,
        up_lora_rank=lora_rank,
        down_lora_rank=lora_rank,
        vl_q_rank=lora_rank,
        vl_k_rank=lora_rank,
        vl_v_rank=lora_rank,
        vl_o_rank=lora_rank,
        vl_fc1_rank=lora_rank,
        vl_fc2_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_compute_config=lora_compute_config,
        colora_async_fallback=colora_async_fallback,
        colora_cpu_workers=colora_cpu_workers,
        colora_cpu_queue_depth=colora_cpu_queue_depth,
        colora_cpu_batch_timeout_us=colora_cpu_batch_timeout_us,
        colora_deferred_promotion_delta_steps=colora_deferred_promotion_delta_steps,
        colora_promotion_ema_alpha=colora_promotion_ema_alpha,
        colora_overlap_mode=colora_overlap_mode,
        colora_temporal_prefetch=colora_temporal_prefetch,
        colora_temporal_hot_cache_slots=colora_temporal_hot_cache_slots,
        colora_request_skip=colora_request_skip,
        colora_max_continuations=colora_max_continuations,
        colora_hit_indexing=colora_hit_indexing,
        metric_client=metric_client,
    )


def load_lora_adapter(*args, **kwargs):
    """Lazy import wrapper to avoid heavyweight model imports at module import time."""
    from lightllm.models.qwen3_vl_moe.layer_weights.lora_layer_weight import load_moe_lora_adapter

    return load_moe_lora_adapter(*args, **kwargs)


__all__ = [
    "SpecJobHandle",
    "SpecSubmitOutcome",
    "SpecBindOutcome",
    "SpecJobKey",
    "Qwen3VLMoELoRADispatcher",
    "create_vl_moe_lora_dispatcher",
    "load_lora_adapter",
]
