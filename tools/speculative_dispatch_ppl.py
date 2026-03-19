#!/usr/bin/env python3
"""Standalone teacher-forced PPL probe for routed-expert stale activation swaps.

This script is intended for offline validation of speculative dispatch ideas on
HF MoE checkpoints. For Qwen3-VL-MoE, it runs the model in text-only mode and
patches the text decoder MoE block so routed experts consume stale
layer-(L-1) activations when the current layer's routed expert set for token t
matches token (t-1) at the same layer.
"""

from __future__ import annotations

import argparse
import json
import math
import types
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import requests
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer


DEFAULT_MODEL_PATH = (
    "/home/shufan/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-30B-A3B-Instruct/"
    "snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
)
DEFAULT_WIKITEXT_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/"
    "word_language_model/data/wikitext-2/valid.txt"
)
SUPPORTED_MODEL_TYPES = {"qwen3_vl_moe", "qwen3_moe", "qwen2_moe"}
SUPPORTED_BLOCK_CLASS_NAMES = {
    "Qwen3VLMoeTextSparseMoeBlock",
    "Qwen3MoeSparseMoeBlock",
    "Qwen2MoeSparseMoeBlock",
}


@dataclass
class PatchStats:
    candidate_token_layers: int = 0
    matched_token_layers: int = 0
    swapped_token_layers: int = 0
    any_overlap_token_layers: int = 0
    overlap_expert_count: int = 0
    patched_layer_calls: int = 0

    def matched_rate(self) -> float:
        if self.candidate_token_layers == 0:
            return 0.0
        return self.matched_token_layers / self.candidate_token_layers

    def swapped_rate(self) -> float:
        if self.candidate_token_layers == 0:
            return 0.0
        return self.swapped_token_layers / self.candidate_token_layers

    def any_overlap_rate(self) -> float:
        if self.candidate_token_layers == 0:
            return 0.0
        return self.any_overlap_token_layers / self.candidate_token_layers

    def average_overlap(self) -> float:
        if self.candidate_token_layers == 0:
            return 0.0
        return self.overlap_expert_count / self.candidate_token_layers


@dataclass
class EvalResult:
    avg_nll: float
    ppl: float
    predicted_tokens: int
    stats: Optional[PatchStats] = None
    debug_report: Optional[dict] = None


@dataclass
class RoutingTransitionStats:
    transition_count: int = 0
    same_set_count: int = 0
    ordered_count: int = 0
    top1_count: int = 0
    any_overlap_count: int = 0
    overlap_expert_count: int = 0
    prefix_same_set_counts: dict[int, int] = field(default_factory=dict)

    def same_set_rate(self) -> float:
        if self.transition_count == 0:
            return 0.0
        return self.same_set_count / self.transition_count

    def ordered_rate(self) -> float:
        if self.transition_count == 0:
            return 0.0
        return self.ordered_count / self.transition_count

    def top1_rate(self) -> float:
        if self.transition_count == 0:
            return 0.0
        return self.top1_count / self.transition_count

    def any_overlap_rate(self) -> float:
        if self.transition_count == 0:
            return 0.0
        return self.any_overlap_count / self.transition_count

    def average_overlap(self) -> float:
        if self.transition_count == 0:
            return 0.0
        return self.overlap_expert_count / self.transition_count

    def mean_hit_rate(self, top_k: int) -> float:
        if self.transition_count == 0 or top_k <= 0:
            return 0.0
        return self.overlap_expert_count / (self.transition_count * top_k)

    def to_dict(self, top_k: int) -> dict:
        payload = {
            "transition_count": self.transition_count,
            "same_set_count": self.same_set_count,
            "ordered_count": self.ordered_count,
            "top1_count": self.top1_count,
            "any_overlap_count": self.any_overlap_count,
            "overlap_expert_count": self.overlap_expert_count,
            "same_set_rate": self.same_set_rate(),
            "ordered_rate": self.ordered_rate(),
            "top1_rate": self.top1_rate(),
            "any_overlap_rate": self.any_overlap_rate(),
            "average_overlap": self.average_overlap(),
            "mean_hit_rate": self.mean_hit_rate(top_k),
        }
        for prefix_k, count in sorted(self.prefix_same_set_counts.items()):
            key = f"prefix_set_match_k{prefix_k}"
            payload[key] = count
            payload[f"{key}_rate"] = 0.0 if self.transition_count == 0 else count / self.transition_count
        return payload


@dataclass
class RouterTraceAnalysis:
    trace_path: str
    phase: str
    event_count: int
    top_k: int
    same_layer_prev_token: dict
    adjacent_layer_same_token: dict
    same_layer_prev_token_per_layer: list[dict]
    adjacent_layer_same_token_per_layer: list[dict]


class RoutedExpertStalePatch:
    """Patch sparse MoE blocks to swap routed-expert inputs with stale states."""

    def __init__(
        self,
        model: torch.nn.Module,
        match_rule: str = "same_set",
        debug_sample_limit: int = 0,
    ) -> None:
        if match_rule != "same_set":
            raise ValueError(f"Unsupported match_rule={match_rule!r}; only 'same_set' is implemented.")

        self.model = model
        self.match_rule = match_rule
        self.debug_sample_limit = debug_sample_limit
        self.enabled = False
        self._patched_blocks: list[torch.nn.Module] = []
        self._forward_pre_hook_handle = None
        self._stats = PatchStats()
        self._shared_state = self._init_shared_state()
        self._window_router_history: dict[int, dict[str, object]] = {}
        self._debug_prefix_ks: list[int] = []
        self._layer_debug: dict[int, dict[str, int | float]] = {}
        self._debug_samples: list[dict] = []
        self.install()

    def _init_shared_state(self) -> dict[str, object]:
        shared_state = getattr(self.model, "_spec_dispatch_shared_state", None)
        if shared_state is None:
            shared_state = {
                "prev_hidden_flat": None,
                "prev_layer_idx": None,
                "window_idx": None,
                "window_begin": None,
                "window_end": None,
            }
            setattr(self.model, "_spec_dispatch_shared_state", shared_state)
        else:
            shared_state["prev_hidden_flat"] = None
            shared_state["prev_layer_idx"] = None
            shared_state["window_idx"] = None
            shared_state["window_begin"] = None
            shared_state["window_end"] = None
        return shared_state

    def install(self) -> None:
        layers = list(iter_sparse_moe_layers(self.model))
        if not layers:
            raise RuntimeError(
                "No supported sparse MoE layers found. "
                f"Supported classes: {sorted(SUPPORTED_BLOCK_CLASS_NAMES)}."
            )

        for layer_idx, block in layers:
            if hasattr(block, "_spec_dispatch_original_forward"):
                continue

            block._spec_dispatch_original_forward = block.forward
            block._spec_dispatch_layer_idx = layer_idx
            block_name = block.__class__.__name__
            if block_name == "Qwen3VLMoeTextSparseMoeBlock":
                patched = self._make_qwen3_vl_forward(block)
            elif block_name in {"Qwen3MoeSparseMoeBlock", "Qwen2MoeSparseMoeBlock"}:
                patched = self._make_qwen_sparse_forward(block)
            else:
                raise RuntimeError(f"Unsupported sparse MoE block class: {block_name}")

            block.forward = types.MethodType(patched, block)
            self._patched_blocks.append(block)

        if self._forward_pre_hook_handle is None:
            self._forward_pre_hook_handle = self.model.register_forward_pre_hook(
                self._forward_pre_hook,
                with_kwargs=True,
            )

    def remove(self) -> None:
        for block in self._patched_blocks:
            if hasattr(block, "_spec_dispatch_original_forward"):
                block.forward = block._spec_dispatch_original_forward
                delattr(block, "_spec_dispatch_original_forward")
            if hasattr(block, "_spec_dispatch_layer_idx"):
                delattr(block, "_spec_dispatch_layer_idx")
        self._patched_blocks.clear()
        if self._forward_pre_hook_handle is not None:
            self._forward_pre_hook_handle.remove()
            self._forward_pre_hook_handle = None
        if hasattr(self.model, "_spec_dispatch_shared_state"):
            delattr(self.model, "_spec_dispatch_shared_state")

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def reset_window_state(self) -> None:
        self._shared_state["prev_hidden_flat"] = None
        self._shared_state["prev_layer_idx"] = None

    def _forward_pre_hook(self, module, args, kwargs) -> None:
        del module, args, kwargs
        self.reset_window_state()

    def reset_run_stats(self) -> None:
        self._stats = PatchStats()
        self._layer_debug = {}
        self._debug_samples = []
        self._window_router_history = {}

    def set_window_context(self, window_idx: int, begin: int, end: int) -> None:
        self._shared_state["window_idx"] = window_idx
        self._shared_state["window_begin"] = begin
        self._shared_state["window_end"] = end

    def stats(self) -> PatchStats:
        return PatchStats(
            candidate_token_layers=self._stats.candidate_token_layers,
            matched_token_layers=self._stats.matched_token_layers,
            swapped_token_layers=self._stats.swapped_token_layers,
            any_overlap_token_layers=self._stats.any_overlap_token_layers,
            overlap_expert_count=self._stats.overlap_expert_count,
            patched_layer_calls=self._stats.patched_layer_calls,
        )

    def debug_report(self) -> dict:
        return {
            "aggregate": {
                "candidate_token_layers": self._stats.candidate_token_layers,
                "matched_token_layers": self._stats.matched_token_layers,
                "swapped_token_layers": self._stats.swapped_token_layers,
                "any_overlap_token_layers": self._stats.any_overlap_token_layers,
                "overlap_expert_count": self._stats.overlap_expert_count,
                "patched_layer_calls": self._stats.patched_layer_calls,
                "matched_rate": self._stats.matched_rate(),
                "swapped_rate": self._stats.swapped_rate(),
                "any_overlap_rate": self._stats.any_overlap_rate(),
                "average_overlap": self._stats.average_overlap(),
            },
            "per_layer": [
                self._finalize_layer_debug(layer_idx, stats)
                for layer_idx, stats in sorted(self._layer_debug.items())
            ],
            "samples": list(self._debug_samples),
        }

    def _make_qwen3_vl_forward(self, block: torch.nn.Module):
        original_forward = block.forward

        def patched_forward(module: torch.nn.Module, hidden_states: torch.Tensor):
            if not self.enabled:
                return original_forward(hidden_states)

            layer_idx = int(module._spec_dispatch_layer_idx)
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_flat = hidden_states.reshape(-1, hidden_dim)

            router_logits = module.gate(hidden_flat)
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
            routing_weights, router_indices = torch.topk(routing_weights, module.top_k, dim=-1)
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(hidden_flat.dtype)
            router_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)

            effective_hidden_flat = self._maybe_swap(
                layer_idx=layer_idx,
                hidden_flat=hidden_flat,
                router_indices=router_indices,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
            effective_hidden_states = effective_hidden_flat.reshape(batch_size, sequence_length, hidden_dim)
            routed_out = module.experts(effective_hidden_states, router_weights, router_indices)
            self._update_previous(
                layer_idx=layer_idx,
                hidden_flat=hidden_flat,
                router_indices=router_indices,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
            return routed_out, router_logits

        return patched_forward

    def _make_qwen_sparse_forward(self, block: torch.nn.Module):
        original_forward = block.forward

        def patched_forward(module: torch.nn.Module, hidden_states: torch.Tensor):
            if not self.enabled:
                return original_forward(hidden_states)

            layer_idx = int(module._spec_dispatch_layer_idx)
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_flat = hidden_states.view(-1, hidden_dim)

            router_logits = module.gate(hidden_flat)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, router_indices = torch.topk(routing_weights, module.top_k, dim=-1)
            if getattr(module, "norm_topk_prob", True):
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(hidden_flat.dtype)

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=hidden_flat.dtype,
                device=hidden_flat.device,
            )
            expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=module.num_experts).permute(2, 1, 0)
            effective_hidden_flat = self._maybe_swap(
                layer_idx=layer_idx,
                hidden_flat=hidden_flat,
                router_indices=router_indices,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )

            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)
            for expert_idx_tensor in expert_hit:
                expert_idx = int(expert_idx_tensor.item())
                expert_layer = module.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                current_state = effective_hidden_flat[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_flat.dtype))

            if hasattr(module, "shared_expert") and hasattr(module, "shared_expert_gate"):
                shared_expert_output = module.shared_expert(hidden_flat)
                shared_expert_output = torch.sigmoid(module.shared_expert_gate(hidden_flat)) * shared_expert_output
                final_hidden_states = final_hidden_states + shared_expert_output

            self._update_previous(
                layer_idx=layer_idx,
                hidden_flat=hidden_flat,
                router_indices=router_indices,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states, router_logits

        return patched_forward

    def _maybe_swap(
        self,
        layer_idx: int,
        hidden_flat: torch.Tensor,
        router_indices: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        self._stats.patched_layer_calls += 1
        prev_hidden_flat = self._shared_state["prev_hidden_flat"]
        prev_layer_idx = self._shared_state["prev_layer_idx"]
        if (
            layer_idx == 0
            or prev_hidden_flat is None
            or prev_layer_idx != layer_idx - 1
            or prev_hidden_flat.shape != hidden_flat.shape
        ):
            return hidden_flat

        if prev_hidden_flat.device != hidden_flat.device:
            prev_hidden_flat = prev_hidden_flat.to(hidden_flat.device, non_blocking=True)
        (
            candidate_mask,
            previous_token_router_indices,
            match_mask,
            ordered_match_mask,
            top1_match_mask,
            overlap_counts,
        ) = self._compute_time_locality_stats(
            layer_idx=layer_idx,
            router_indices=router_indices,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )

        candidate_count = int(candidate_mask.sum().item())
        if candidate_count == 0:
            return hidden_flat
        matched_count = int(match_mask.sum().item())
        any_overlap_count = int((overlap_counts > 0).sum().item())
        overlap_expert_count = int(overlap_counts.sum().item())
        self._stats.candidate_token_layers += candidate_count
        self._stats.matched_token_layers += matched_count
        self._stats.swapped_token_layers += matched_count
        self._stats.any_overlap_token_layers += any_overlap_count
        self._stats.overlap_expert_count += overlap_expert_count
        self._record_layer_debug(
            layer_idx=layer_idx,
            batch_size=batch_size,
            sequence_length=sequence_length,
            candidate_mask=candidate_mask,
            current_router_indices=router_indices,
            previous_router_indices=previous_token_router_indices,
            match_mask=match_mask,
            ordered_match_mask=ordered_match_mask,
            top1_match_mask=top1_match_mask,
            overlap_counts=overlap_counts,
        )

        if matched_count == 0:
            return hidden_flat
        return torch.where(match_mask[:, None], prev_hidden_flat, hidden_flat)

    def _update_previous(
        self,
        layer_idx: int,
        hidden_flat: torch.Tensor,
        router_indices: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        self._shared_state["prev_layer_idx"] = layer_idx
        self._shared_state["prev_hidden_flat"] = hidden_flat.detach().clone()
        window_begin = self._shared_state.get("window_begin")
        window_end = self._shared_state.get("window_end")
        if window_begin is not None and window_end is not None:
            self._window_router_history[layer_idx] = {
                "window_begin": int(window_begin),
                "window_end": int(window_end),
                "router_indices": router_indices.detach().reshape(batch_size, sequence_length, -1).cpu(),
            }

    def _compute_time_locality_stats(
        self,
        layer_idx: int,
        router_indices: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        top_k = int(router_indices.shape[-1])
        router_by_token = router_indices.reshape(batch_size, sequence_length, top_k)
        previous_router_by_token = torch.empty_like(router_by_token)
        candidate_by_token = torch.zeros(
            (batch_size, sequence_length),
            dtype=torch.bool,
            device=router_indices.device,
        )

        if sequence_length > 1:
            previous_router_by_token[:, 1:, :] = router_by_token[:, :-1, :]
            candidate_by_token[:, 1:] = True

        boundary_router = self._lookup_previous_token_router(
            layer_idx=layer_idx,
            batch_size=batch_size,
            top_k=top_k,
            device=router_indices.device,
        )
        if boundary_router is not None:
            previous_router_by_token[:, 0, :] = boundary_router
            candidate_by_token[:, 0] = True
        elif sequence_length > 0:
            previous_router_by_token[:, 0, :] = router_by_token[:, 0, :]

        candidate_mask = candidate_by_token.reshape(-1)
        previous_router_flat = previous_router_by_token.reshape(-1, top_k)
        match_mask = torch.zeros_like(candidate_mask)
        ordered_match_mask = torch.zeros_like(candidate_mask)
        top1_match_mask = torch.zeros_like(candidate_mask)
        overlap_counts = torch.zeros(candidate_mask.shape[0], dtype=torch.int64, device=router_indices.device)
        if candidate_mask.any():
            current_candidates = router_indices[candidate_mask]
            previous_candidates = previous_router_flat[candidate_mask]
            match_mask[candidate_mask] = self._same_set_mask(current_candidates, previous_candidates)
            ordered_match_mask[candidate_mask] = torch.eq(current_candidates, previous_candidates).all(dim=-1)
            top1_match_mask[candidate_mask] = torch.eq(current_candidates[:, 0], previous_candidates[:, 0])
            overlap_counts[candidate_mask] = self._overlap_counts(current_candidates, previous_candidates)

        return (
            candidate_mask,
            previous_router_flat,
            match_mask,
            ordered_match_mask,
            top1_match_mask,
            overlap_counts,
        )

    def _lookup_previous_token_router(
        self,
        layer_idx: int,
        batch_size: int,
        top_k: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        window_begin = self._shared_state.get("window_begin")
        if window_begin is None or int(window_begin) <= 0:
            return None

        history = self._window_router_history.get(layer_idx)
        if history is None:
            return None

        history_begin = int(history["window_begin"])
        history_end = int(history["window_end"])
        previous_global_token = int(window_begin) - 1
        if not (history_begin <= previous_global_token < history_end):
            return None

        history_router = history["router_indices"]
        if not isinstance(history_router, torch.Tensor):
            return None
        if history_router.dim() != 3 or history_router.shape[0] != batch_size or history_router.shape[2] != top_k:
            return None

        previous_offset = previous_global_token - history_begin
        return history_router[:, previous_offset, :].to(device=device, non_blocking=True)

    @staticmethod
    def _same_set_mask(current_router_indices: torch.Tensor, previous_router_indices: torch.Tensor) -> torch.Tensor:
        sorted_current = torch.sort(current_router_indices, dim=-1).values
        sorted_previous = torch.sort(previous_router_indices, dim=-1).values
        return torch.eq(sorted_current, sorted_previous).all(dim=-1)

    @staticmethod
    def _overlap_counts(current_router_indices: torch.Tensor, previous_router_indices: torch.Tensor) -> torch.Tensor:
        pairwise_equal = current_router_indices[:, :, None] == previous_router_indices[:, None, :]
        return pairwise_equal.any(dim=-1).sum(dim=-1)

    def _record_layer_debug(
        self,
        layer_idx: int,
        batch_size: int,
        sequence_length: int,
        candidate_mask: torch.Tensor,
        current_router_indices: torch.Tensor,
        previous_router_indices: torch.Tensor,
        match_mask: torch.Tensor,
        ordered_match_mask: torch.Tensor,
        top1_match_mask: torch.Tensor,
        overlap_counts: torch.Tensor,
    ) -> None:
        layer_stats = self._layer_debug.setdefault(
            layer_idx,
            {
                "layer_idx": layer_idx,
                "candidate": 0,
                "same_set_match": 0,
                "ordered_match": 0,
                "top1_match": 0,
                "any_overlap": 0,
                "overlap_sum": 0,
            },
        )
        candidate_count = int(candidate_mask.sum().item())
        layer_stats["candidate"] += candidate_count
        layer_stats["same_set_match"] += int(match_mask.sum().item())
        layer_stats["ordered_match"] += int(ordered_match_mask.sum().item())
        layer_stats["top1_match"] += int(top1_match_mask.sum().item())
        layer_stats["any_overlap"] += int((overlap_counts > 0).sum().item())
        layer_stats["overlap_sum"] += int(overlap_counts.sum().item())

        top_k = int(current_router_indices.shape[1])
        if not self._debug_prefix_ks:
            self._debug_prefix_ks = [k for k in (1, 2, 4, 8) if k <= top_k]
        current_candidates = current_router_indices[candidate_mask]
        previous_candidates = previous_router_indices[candidate_mask]
        for k in self._debug_prefix_ks:
            key = f"prefix_set_match_k{k}"
            layer_stats.setdefault(key, 0)
            if candidate_count == 0:
                prefix_match_count = 0
            elif k == top_k:
                prefix_match_count = int(match_mask.sum().item())
            else:
                prefix_match_count = int(
                    self._same_set_mask(
                        current_candidates[:, :k],
                        previous_candidates[:, :k],
                    ).sum().item()
                )
            layer_stats[key] += prefix_match_count

        if self.debug_sample_limit <= 0 or len(self._debug_samples) >= self.debug_sample_limit:
            return

        mismatch_indices = torch.where(candidate_mask & (overlap_counts > 0) & (~match_mask))[0]
        if mismatch_indices.numel() == 0:
            return

        window_begin = self._shared_state.get("window_begin")
        window_end = self._shared_state.get("window_end")
        window_idx = self._shared_state.get("window_idx")
        for token_idx_tensor in mismatch_indices:
            if len(self._debug_samples) >= self.debug_sample_limit:
                break
            token_idx = int(token_idx_tensor.item())
            batch_idx = token_idx // sequence_length
            token_offset = token_idx % sequence_length
            global_token_idx = None if window_begin is None else int(window_begin) + token_offset
            current = current_router_indices[token_idx].detach().cpu()
            previous = previous_router_indices[token_idx].detach().cpu()
            self._debug_samples.append(
                {
                    "window_idx": window_idx,
                    "window_begin": window_begin,
                    "window_end": window_end,
                    "layer_idx": layer_idx,
                    "stale_source_layer_idx": layer_idx - 1,
                    "batch_idx": batch_idx,
                    "token_offset": token_offset,
                    "global_token_idx": global_token_idx,
                    "previous_global_token_idx": None if global_token_idx is None else global_token_idx - 1,
                    "current_router_indices": current.tolist(),
                    "previous_token_router_indices": previous.tolist(),
                    "sorted_current_router_indices": torch.sort(current).values.tolist(),
                    "sorted_previous_token_router_indices": torch.sort(previous).values.tolist(),
                    "overlap_count": int(overlap_counts[token_idx].item()),
                    "ordered_match": bool(ordered_match_mask[token_idx].item()),
                    "top1_match": bool(top1_match_mask[token_idx].item()),
                }
            )

    @staticmethod
    def _finalize_layer_debug(layer_idx: int, stats: dict[str, int | float]) -> dict:
        candidate = int(stats["candidate"])
        finalized = {
            "layer_idx": layer_idx,
            "candidate": candidate,
            "same_set_match": int(stats["same_set_match"]),
            "ordered_match": int(stats["ordered_match"]),
            "top1_match": int(stats["top1_match"]),
            "any_overlap": int(stats["any_overlap"]),
            "overlap_sum": int(stats["overlap_sum"]),
            "same_set_match_rate": 0.0 if candidate == 0 else float(stats["same_set_match"]) / candidate,
            "ordered_match_rate": 0.0 if candidate == 0 else float(stats["ordered_match"]) / candidate,
            "top1_match_rate": 0.0 if candidate == 0 else float(stats["top1_match"]) / candidate,
            "any_overlap_rate": 0.0 if candidate == 0 else float(stats["any_overlap"]) / candidate,
            "average_overlap": 0.0 if candidate == 0 else float(stats["overlap_sum"]) / candidate,
        }
        for key, value in stats.items():
            if not key.startswith("prefix_set_match_k"):
                continue
            finalized[key] = int(value)
            finalized[f"{key}_rate"] = 0.0 if candidate == 0 else float(value) / candidate
        return finalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone PPL probe for speculative routed-expert stale activation swaps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="HF model path or repo id.")
    parser.add_argument(
        "--router_trace_path",
        type=str,
        default=None,
        help="Optional canonical router_trace.jsonl path used to reconcile trace locality with the online PPL metrics.",
    )
    parser.add_argument(
        "--router_trace_phase",
        type=str,
        default="decode",
        help="Phase to analyze from router_trace.jsonl.",
    )
    parser.add_argument(
        "--trace_only",
        action="store_true",
        help="Skip model loading and only analyze router_trace.jsonl.",
    )
    parser.add_argument("--text_path", type=str, default=None, help="Optional local plain-text corpus path.")
    parser.add_argument(
        "--wikitext_url",
        type=str,
        default=DEFAULT_WIKITEXT_URL,
        help="Fallback URL used when --text_path is not set.",
    )
    parser.add_argument(
        "--max_eval_tokens",
        type=int,
        default=8192,
        help="Truncate evaluation after this many tokenizer output tokens. Use <=0 for all tokens.",
    )
    parser.add_argument("--seq_len", type=int, default=512, help="Maximum input length per scoring window.")
    parser.add_argument("--stride", type=int, default=512, help="Step size between scoring windows.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="from_pretrained device_map. Use 'none' to disable HF dispatching.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        help="Optional attention implementation override.",
    )
    parser.add_argument(
        "--match_rule",
        type=str,
        default="same_set",
        choices=["same_set"],
        help="Routing match criterion between token t and token t-1 at the same layer.",
    )
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict HF loading to local files only.",
    )
    parser.add_argument(
        "--http_timeout_s",
        type=float,
        default=30.0,
        help="HTTP timeout when fetching the fallback text corpus.",
    )
    parser.add_argument(
        "--debug_sample_limit",
        type=int,
        default=0,
        help="Record up to this many mismatched overlap samples for debugging.",
    )
    parser.add_argument(
        "--debug_match_report_path",
        type=str,
        default=None,
        help="Optional JSON path to dump per-layer routing match diagnostics.",
    )
    parser.add_argument(
        "--debug_print_layer_stats",
        action="store_true",
        help="Print one line of same-layer previous-token routing statistics per layer.",
    )
    parser.add_argument(
        "--debug_print_trace_layer_stats",
        action="store_true",
        help="Print per-layer router-trace statistics for same-layer previous-token and adjacent-layer same-token metrics.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-window progress.")
    args = parser.parse_args()

    if args.trace_only and args.router_trace_path is None:
        raise ValueError("--trace_only requires --router_trace_path.")
    if args.seq_len < 2:
        raise ValueError(f"--seq_len must be >= 2, got {args.seq_len}")
    if args.stride < 1:
        raise ValueError(f"--stride must be >= 1, got {args.stride}")
    if args.stride > args.seq_len:
        raise ValueError(
            f"--stride must be <= --seq_len for gap-free scoring, got stride={args.stride}, seq_len={args.seq_len}"
        )
    return args


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def load_model(model_path: str, dtype_name: str, device_map: str, attn_implementation: str, local_files_only: bool):
    config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
    if config.model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type={config.model_type!r}. "
            f"Supported types: {sorted(SUPPORTED_MODEL_TYPES)}."
        )

    load_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": resolve_torch_dtype(dtype_name),
        "local_files_only": local_files_only,
    }
    if device_map != "none":
        load_kwargs["device_map"] = device_map
    if attn_implementation != "auto":
        load_kwargs["attn_implementation"] = attn_implementation

    if config.model_type == "qwen3_vl_moe":
        from transformers.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    elif config.model_type == "qwen3_moe":
        from transformers.models.qwen3_moe import Qwen3MoeForCausalLM

        model = Qwen3MoeForCausalLM.from_pretrained(model_path, **load_kwargs)
    elif config.model_type == "qwen2_moe":
        from transformers.models.qwen2_moe import Qwen2MoeForCausalLM

        model = Qwen2MoeForCausalLM.from_pretrained(model_path, **load_kwargs)
    else:
        raise AssertionError(f"Unexpected model_type={config.model_type!r}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only, use_fast=True)
    model.eval()
    return model, tokenizer, config


def get_input_device(model: torch.nn.Module) -> torch.device:
    embedding_weight = model.get_input_embeddings().weight
    return embedding_weight.device


def read_eval_text(text_path: Optional[str], wikitext_url: str, timeout_s: float) -> str:
    if text_path is not None:
        return Path(text_path).read_text(encoding="utf-8")

    response = requests.get(wikitext_url, timeout=timeout_s)
    response.raise_for_status()
    return response.text


def tokenize_eval_text(tokenizer, text: str, max_eval_tokens: int) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if max_eval_tokens > 0:
        input_ids = input_ids[:, :max_eval_tokens]
    if input_ids.shape[1] < 2:
        raise ValueError("Need at least 2 tokens to compute causal LM perplexity.")
    return input_ids


def build_eval_windows(total_length: int, seq_len: int, stride: int) -> list[tuple[int, int, int, int]]:
    windows: list[tuple[int, int, int, int]] = []
    previous_end = 0
    for step in range(0, total_length, stride):
        end = min(step + stride, total_length)
        begin = max(end - seq_len, 0)
        target_len = end - previous_end
        if target_len <= 0:
            continue
        windows.append((len(windows), begin, end, target_len))
        previous_end = end
        if end == total_length:
            break
    return windows


def _topk_prefixes(top_k: int) -> list[int]:
    return [prefix_k for prefix_k in (1, 2, 4, 8) if prefix_k <= top_k]


def _sorted_prefix_tuple(experts: list[int], prefix_k: int) -> tuple[int, ...]:
    return tuple(sorted(experts[:prefix_k]))


def update_routing_transition_stats(
    stats: RoutingTransitionStats,
    current_experts: list[int],
    previous_experts: list[int],
    prefix_ks: list[int],
) -> None:
    current_tuple = tuple(int(expert_id) for expert_id in current_experts)
    previous_tuple = tuple(int(expert_id) for expert_id in previous_experts)
    current_set = set(current_tuple)
    previous_set = set(previous_tuple)
    overlap = len(current_set & previous_set)

    stats.transition_count += 1
    stats.same_set_count += int(tuple(sorted(current_tuple)) == tuple(sorted(previous_tuple)))
    stats.ordered_count += int(current_tuple == previous_tuple)
    stats.top1_count += int(bool(current_tuple) and bool(previous_tuple) and current_tuple[0] == previous_tuple[0])
    stats.any_overlap_count += int(overlap > 0)
    stats.overlap_expert_count += overlap
    for prefix_k in prefix_ks:
        stats.prefix_same_set_counts.setdefault(prefix_k, 0)
        stats.prefix_same_set_counts[prefix_k] += int(
            _sorted_prefix_tuple(current_experts, prefix_k) == _sorted_prefix_tuple(previous_experts, prefix_k)
        )


def finalize_routing_transition_map(
    layer_stats: dict[int, RoutingTransitionStats],
    top_k: int,
) -> list[dict]:
    finalized_rows = []
    for layer_idx in sorted(layer_stats):
        stats = layer_stats[layer_idx]
        row = {"layer_idx": layer_idx}
        row.update(stats.to_dict(top_k))
        finalized_rows.append(row)
    return finalized_rows


def analyze_router_trace(trace_path: str | Path, phase: str = "decode") -> RouterTraceAnalysis:
    trace_path = Path(trace_path)
    if not trace_path.exists():
        raise FileNotFoundError(f"router trace not found: {trace_path}")

    event_count = 0
    top_k: Optional[int] = None
    same_layer_prev_stats = RoutingTransitionStats()
    adjacent_layer_stats = RoutingTransitionStats()
    same_layer_prev_per_layer: dict[int, RoutingTransitionStats] = defaultdict(RoutingTransitionStats)
    adjacent_layer_per_layer: dict[int, RoutingTransitionStats] = defaultdict(RoutingTransitionStats)

    previous_token_by_req_layer: dict[tuple[int, int], tuple[int, list[int]]] = {}
    current_token_layers_by_req: dict[int, tuple[int, dict[int, list[int]]]] = {}

    with trace_path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{trace_path}:{line_num} invalid JSON: {exc}") from exc

            if str(record.get("phase")) != phase:
                continue

            req_idx = int(record["req_idx"])
            layer_id = int(record["layer_id"])
            token_pos = int(record["token_pos"])
            experts = [int(expert_id) for expert_id in record["topk_experts"]]
            if top_k is None:
                top_k = len(experts)
            elif len(experts) != top_k:
                raise ValueError(
                    f"inconsistent top-k width at {trace_path}:{line_num}: expected {top_k}, got {len(experts)}"
                )
            prefix_ks = _topk_prefixes(top_k)
            event_count += 1

            previous_key = (req_idx, layer_id)
            previous_token = previous_token_by_req_layer.get(previous_key)
            if previous_token is not None:
                _, previous_experts = previous_token
                update_routing_transition_stats(
                    same_layer_prev_stats,
                    current_experts=experts,
                    previous_experts=previous_experts,
                    prefix_ks=prefix_ks,
                )
                update_routing_transition_stats(
                    same_layer_prev_per_layer[layer_id],
                    current_experts=experts,
                    previous_experts=previous_experts,
                    prefix_ks=prefix_ks,
                )
            previous_token_by_req_layer[previous_key] = (token_pos, experts)

            active_token = current_token_layers_by_req.get(req_idx)
            if active_token is None or active_token[0] != token_pos:
                active_layers: dict[int, list[int]] = {}
                current_token_layers_by_req[req_idx] = (token_pos, active_layers)
            else:
                active_layers = active_token[1]

            previous_layer_experts = active_layers.get(layer_id - 1)
            if previous_layer_experts is not None:
                update_routing_transition_stats(
                    adjacent_layer_stats,
                    current_experts=experts,
                    previous_experts=previous_layer_experts,
                    prefix_ks=prefix_ks,
                )
                update_routing_transition_stats(
                    adjacent_layer_per_layer[layer_id],
                    current_experts=experts,
                    previous_experts=previous_layer_experts,
                    prefix_ks=prefix_ks,
                )
            active_layers[layer_id] = experts

    if top_k is None:
        top_k = 0

    return RouterTraceAnalysis(
        trace_path=str(trace_path),
        phase=phase,
        event_count=event_count,
        top_k=top_k,
        same_layer_prev_token=same_layer_prev_stats.to_dict(top_k),
        adjacent_layer_same_token=adjacent_layer_stats.to_dict(top_k),
        same_layer_prev_token_per_layer=finalize_routing_transition_map(same_layer_prev_per_layer, top_k),
        adjacent_layer_same_token_per_layer=finalize_routing_transition_map(adjacent_layer_per_layer, top_k),
    )


def evaluate_ppl(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    seq_len: int,
    stride: int,
    mode_name: str,
    patch: Optional[RoutedExpertStalePatch],
    enable_patch: bool,
    verbose: bool,
) -> EvalResult:
    windows = build_eval_windows(token_ids.shape[1], seq_len, stride)
    input_device = get_input_device(model)
    total_nll = torch.zeros((), dtype=torch.float64)
    total_predicted_tokens = 0

    if patch is not None:
        patch.reset_window_state()
        patch.reset_run_stats()
        if enable_patch:
            patch.enable()
        else:
            patch.disable()

    with torch.inference_mode():
        for window_idx, begin, end, target_len in windows:
            if patch is not None:
                patch.reset_window_state()
                patch.set_window_context(window_idx=window_idx, begin=begin, end=end)

            input_ids = token_ids[:, begin:end].to(input_device)
            attention_mask = torch.ones_like(input_ids, device=input_device)
            labels = input_ids.clone()
            if target_len < labels.shape[1]:
                labels[:, :-target_len] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                output_router_logits=False,
            )
            valid_tokens = int((labels[:, 1:] != -100).sum().item())
            if valid_tokens == 0:
                continue

            total_nll += outputs.loss.detach().to(torch.float64).cpu() * valid_tokens
            total_predicted_tokens += valid_tokens

            if verbose:
                running_avg_nll = (total_nll / max(total_predicted_tokens, 1)).item()
                print(
                    f"[{mode_name}] window {window_idx + 1}/{len(windows)} "
                    f"tokens={begin}:{end} valid={valid_tokens} avg_nll={running_avg_nll:.6f}"
                )

    if total_predicted_tokens == 0:
        raise RuntimeError("No valid target tokens were scored; check seq_len/stride/text length.")

    avg_nll = (total_nll / total_predicted_tokens).item()
    ppl = math.exp(avg_nll)
    stats = patch.stats() if patch is not None and enable_patch else None
    debug_report = patch.debug_report() if patch is not None and enable_patch else None
    return EvalResult(
        avg_nll=avg_nll,
        ppl=ppl,
        predicted_tokens=total_predicted_tokens,
        stats=stats,
        debug_report=debug_report,
    )


def iter_sparse_moe_layers(model: torch.nn.Module) -> Iterable[tuple[int, torch.nn.Module]]:
    model_type = model.config.model_type
    if model_type == "qwen3_vl_moe":
        layers = model.model.language_model.layers
    elif model_type in {"qwen3_moe", "qwen2_moe"}:
        layers = model.model.layers
    else:
        return []

    for layer_idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        if mlp.__class__.__name__ in SUPPORTED_BLOCK_CLASS_NAMES:
            yield layer_idx, mlp


def format_rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.2f}% ({numerator}/{denominator})"


def print_layer_debug_stats(debug_report: dict) -> None:
    for layer_stats in debug_report.get("per_layer", []):
        layer_idx = layer_stats["layer_idx"]
        candidate = layer_stats["candidate"]
        same_set_rate = 100.0 * layer_stats["same_set_match_rate"]
        ordered_rate = 100.0 * layer_stats["ordered_match_rate"]
        top1_rate = 100.0 * layer_stats["top1_match_rate"]
        any_overlap_rate = 100.0 * layer_stats["any_overlap_rate"]
        avg_overlap = layer_stats["average_overlap"]
        prefix_parts = []
        for key in sorted(k for k in layer_stats if k.startswith("prefix_set_match_k") and not k.endswith("_rate")):
            rate = 100.0 * layer_stats[f"{key}_rate"]
            prefix_parts.append(f"{key}={rate:.2f}%")
        prefix_summary = ", ".join(prefix_parts)
        print(
            f"Layer {layer_idx}: candidates={candidate} "
            f"same_set={same_set_rate:.2f}% ordered={ordered_rate:.2f}% "
            f"top1={top1_rate:.2f}% any_overlap={any_overlap_rate:.2f}% avg_overlap={avg_overlap:.4f}"
        )
        if prefix_summary:
            print(f"  Prefix set match rates: {prefix_summary}")


def print_trace_section(title: str, stats: dict, top_k: int) -> None:
    transitions = int(stats["transition_count"])
    print(f"{title}:")
    print(f"  Transitions: {transitions}")
    print(f"  Same-Set Rate: {100.0 * stats['same_set_rate']:.2f}%")
    print(f"  Ordered Rate: {100.0 * stats['ordered_rate']:.2f}%")
    print(f"  Top-1 Rate: {100.0 * stats['top1_rate']:.2f}%")
    print(f"  Any-Overlap Rate: {100.0 * stats['any_overlap_rate']:.2f}%")
    print(f"  Average Expert Overlap: {stats['average_overlap']:.4f}")
    if top_k > 0:
        print(f"  Mean Overlap Hit Rate: {100.0 * stats['mean_hit_rate']:.2f}%")
    prefix_parts = []
    for key in sorted(k for k in stats if k.startswith("prefix_set_match_k") and not k.endswith("_rate")):
        prefix_parts.append(f"{key}={100.0 * stats[f'{key}_rate']:.2f}%")
    if prefix_parts:
        print(f"  Prefix set match rates: {', '.join(prefix_parts)}")


def print_trace_layer_stats(title: str, per_layer_rows: list[dict]) -> None:
    print(title)
    for row in per_layer_rows:
        layer_idx = row["layer_idx"]
        transitions = row["transition_count"]
        same_set_rate = 100.0 * row["same_set_rate"]
        ordered_rate = 100.0 * row["ordered_rate"]
        top1_rate = 100.0 * row["top1_rate"]
        any_overlap_rate = 100.0 * row["any_overlap_rate"]
        avg_overlap = row["average_overlap"]
        mean_hit_rate = 100.0 * row["mean_hit_rate"]
        prefix_parts = []
        for key in sorted(k for k in row if k.startswith("prefix_set_match_k") and not k.endswith("_rate")):
            prefix_parts.append(f"{key}={100.0 * row[f'{key}_rate']:.2f}%")
        prefix_summary = ", ".join(prefix_parts)
        print(
            f"  Layer {layer_idx}: transitions={transitions} "
            f"same_set={same_set_rate:.2f}% ordered={ordered_rate:.2f}% "
            f"top1={top1_rate:.2f}% any_overlap={any_overlap_rate:.2f}% "
            f"avg_overlap={avg_overlap:.4f} mean_hit={mean_hit_rate:.2f}%"
        )
        if prefix_summary:
            print(f"    Prefix set match rates: {prefix_summary}")


def print_router_trace_analysis(analysis: RouterTraceAnalysis, print_layer_stats: bool) -> None:
    print(f"Router Trace: {analysis.trace_path}")
    print(f"Trace Phase: {analysis.phase}")
    print(f"Trace Events: {analysis.event_count}")
    print(f"Trace Top-K: {analysis.top_k}")
    print_trace_section("Same-Layer Previous-Token", analysis.same_layer_prev_token, analysis.top_k)
    print_trace_section("Adjacent-Layer Same-Token", analysis.adjacent_layer_same_token, analysis.top_k)
    if print_layer_stats:
        print_trace_layer_stats(
            "Trace Same-Layer Previous-Token Per-Layer Stats:",
            analysis.same_layer_prev_token_per_layer,
        )
        print_trace_layer_stats(
            "Trace Adjacent-Layer Same-Token Per-Layer Stats:",
            analysis.adjacent_layer_same_token_per_layer,
        )


def main() -> None:
    args = parse_args()
    trace_analysis = None
    if args.router_trace_path is not None:
        trace_analysis = analyze_router_trace(args.router_trace_path, phase=args.router_trace_phase)
        if args.trace_only:
            print_router_trace_analysis(trace_analysis, print_layer_stats=args.debug_print_trace_layer_stats)
            return

    model, tokenizer, config = load_model(
        model_path=args.model_path,
        dtype_name=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )

    text = read_eval_text(args.text_path, args.wikitext_url, args.http_timeout_s)
    token_ids = tokenize_eval_text(tokenizer, text, args.max_eval_tokens)
    patch = RoutedExpertStalePatch(
        model=model,
        match_rule=args.match_rule,
        debug_sample_limit=args.debug_sample_limit,
    )

    clean = evaluate_ppl(
        model=model,
        token_ids=token_ids,
        seq_len=args.seq_len,
        stride=args.stride,
        mode_name="clean",
        patch=patch,
        enable_patch=False,
        verbose=args.verbose,
    )
    speculative = evaluate_ppl(
        model=model,
        token_ids=token_ids,
        seq_len=args.seq_len,
        stride=args.stride,
        mode_name="speculative",
        patch=patch,
        enable_patch=True,
        verbose=args.verbose,
    )

    stats = speculative.stats or PatchStats()
    print(f"Model Path: {args.model_path}")
    print(f"Model Type: {config.model_type}")
    print(f"Evaluated Tokens: {token_ids.shape[1]}")
    print(f"Predicted Tokens: {clean.predicted_tokens}")
    print(f"Seq Len: {args.seq_len}")
    print(f"Stride: {args.stride}")
    print(f"Device Map: {args.device_map}")
    print(f"DType: {args.dtype}")
    print(f"Clean PPL: {clean.ppl:.6f}")
    print(f"Speculative PPL: {speculative.ppl:.6f}")
    print(f"Delta PPL: {speculative.ppl - clean.ppl:+.6f}")
    print(f"Matched-Token Rate: {format_rate(stats.matched_token_layers, stats.candidate_token_layers)}")
    print(f"Swapped-Token Rate: {format_rate(stats.swapped_token_layers, stats.candidate_token_layers)}")
    print(f"Any-Overlap Token Rate: {format_rate(stats.any_overlap_token_layers, stats.candidate_token_layers)}")
    print(f"Average Expert Overlap: {stats.average_overlap():.4f}")
    if stats.matched_token_layers == 0 and stats.any_overlap_token_layers > 0:
        print(
            "Warning: consecutive tokens at the same layer shared some routed experts but never matched on the full "
            "top-k set. Compare this rate against the trace's Same-Layer Previous-Token statistics."
        )
    if args.debug_print_layer_stats and speculative.debug_report is not None:
        print_layer_debug_stats(speculative.debug_report)
    if args.debug_match_report_path is not None and speculative.debug_report is not None:
        report_path = Path(args.debug_match_report_path)
        report_path.write_text(json.dumps(speculative.debug_report, indent=2), encoding="utf-8")
        print(f"Debug Match Report: {report_path}")
    if trace_analysis is not None:
        print_router_trace_analysis(trace_analysis, print_layer_stats=args.debug_print_trace_layer_stats)


if __name__ == "__main__":
    main()
