#!/usr/bin/env python3
"""Fixed ablation variant registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


REPLAY_POLICY_SELECTIVE = "selective_decode_admission"
REPLAY_POLICY_ALWAYS_ADMIT = "always_admit"

SERVICE_MODEL_EXECUTION_FIRST = "execution_first"
SERVICE_MODEL_BLOCKING_PROMOTION_FIRST = "blocking_promotion_first"
SERVICE_MODEL_EXECUTION_FIRST_SYNC_PROMOTION = "execution_first_sync_promotion"

OVERLAP_POLICY_CALIBRATED = "calibrated"
OVERLAP_POLICY_DISABLED = "disabled"


@dataclass(frozen=True)
class AblationVariant:
    variant_id: str
    label: str
    suite_tags: tuple[str, ...]
    trace_condition: str
    replay_policy: str
    service_model: str
    overlap_policy: str
    deferred_promotion_delta_steps: int
    temporal_prefetch: bool
    notes: str


VARIANTS: Dict[str, AblationVariant] = {
    "colora_full": AblationVariant(
        variant_id="colora_full",
        label="COLoRA-Full",
        suite_tags=("core", "all"),
        trace_condition="joint_corr",
        replay_policy=REPLAY_POLICY_SELECTIVE,
        service_model=SERVICE_MODEL_EXECUTION_FIRST,
        overlap_policy=OVERLAP_POLICY_CALIBRATED,
        deferred_promotion_delta_steps=4,
        temporal_prefetch=True,
        notes="Execution-first activation path with deferred promotion and temporal prefetch.",
    ),
    "no_cpu_path": AblationVariant(
        variant_id="no_cpu_path",
        label="No-CPU-Path",
        suite_tags=("core", "all"),
        trace_condition="joint_corr",
        replay_policy=REPLAY_POLICY_ALWAYS_ADMIT,
        service_model=SERVICE_MODEL_BLOCKING_PROMOTION_FIRST,
        overlap_policy=OVERLAP_POLICY_DISABLED,
        deferred_promotion_delta_steps=0,
        temporal_prefetch=False,
        notes="Blocking promotion-first baseline with synchronous weight transfer on every miss.",
    ),
    "no_overlap": AblationVariant(
        variant_id="no_overlap",
        label="No-Overlap",
        suite_tags=("core", "all"),
        trace_condition="joint_corr",
        replay_policy=REPLAY_POLICY_SELECTIVE,
        service_model=SERVICE_MODEL_EXECUTION_FIRST,
        overlap_policy=OVERLAP_POLICY_DISABLED,
        deferred_promotion_delta_steps=4,
        temporal_prefetch=True,
        notes="Execution-first path with calibrated overlap windows disabled.",
    ),
    "no_deferred_sync": AblationVariant(
        variant_id="no_deferred_sync",
        label="No-Deferred-Sync",
        suite_tags=("background", "all"),
        trace_condition="joint_corr",
        replay_policy=REPLAY_POLICY_ALWAYS_ADMIT,
        service_model=SERVICE_MODEL_EXECUTION_FIRST_SYNC_PROMOTION,
        overlap_policy=OVERLAP_POLICY_CALIBRATED,
        deferred_promotion_delta_steps=0,
        temporal_prefetch=False,
        notes="Execution-first miss service plus synchronous immediate promotion.",
    ),
    "no_deferred_never": AblationVariant(
        variant_id="no_deferred_never",
        label="No-Deferred-Never",
        suite_tags=("background", "all"),
        trace_condition="joint_corr",
        replay_policy=REPLAY_POLICY_SELECTIVE,
        service_model=SERVICE_MODEL_EXECUTION_FIRST,
        overlap_policy=OVERLAP_POLICY_CALIBRATED,
        deferred_promotion_delta_steps=0,
        temporal_prefetch=False,
        notes="Execution-first path with decode misses never admitted to the hot cache.",
    ),
    "no_prefetch": AblationVariant(
        variant_id="no_prefetch",
        label="No-Prefetch",
        suite_tags=("background", "all"),
        trace_condition="joint_corr",
        replay_policy=REPLAY_POLICY_SELECTIVE,
        service_model=SERVICE_MODEL_EXECUTION_FIRST,
        overlap_policy=OVERLAP_POLICY_CALIBRATED,
        deferred_promotion_delta_steps=4,
        temporal_prefetch=False,
        notes="Execution-first deferred-promotion path with temporal prefetch disabled.",
    ),
    "expert_only": AblationVariant(
        variant_id="expert_only",
        label="Expert-Only",
        suite_tags=("granularity", "all"),
        trace_condition="expert_only",
        replay_policy=REPLAY_POLICY_SELECTIVE,
        service_model=SERVICE_MODEL_EXECUTION_FIRST,
        overlap_policy=OVERLAP_POLICY_CALIBRATED,
        deferred_promotion_delta_steps=4,
        temporal_prefetch=True,
        notes="Execution-first baseline rerun on expert-only cache objects.",
    ),
}


SUITE_ORDER = ("core", "background", "granularity", "all")


def resolve_variants(
    suite_id: str,
    requested_variants: Optional[Sequence[str]] = None,
) -> List[AblationVariant]:
    if suite_id not in SUITE_ORDER:
        raise ValueError(f"unsupported suite_id={suite_id!r}; allowed={SUITE_ORDER}")
    if requested_variants:
        missing = [variant_id for variant_id in requested_variants if variant_id not in VARIANTS]
        if missing:
            raise ValueError(f"unknown ablation variants: {missing}")
        variant_ids = list(requested_variants)
    else:
        variant_ids = [variant_id for variant_id, variant in VARIANTS.items() if suite_id in variant.suite_tags]
    return [VARIANTS[variant_id] for variant_id in variant_ids]


def parse_variant_csv(raw_value: Optional[str]) -> Optional[List[str]]:
    if raw_value is None:
        return None
    values = [token.strip() for token in str(raw_value).split(",") if token.strip()]
    return values or None
