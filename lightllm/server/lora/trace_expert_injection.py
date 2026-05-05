from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

_INJECTION_BIAS = 1e9


class TraceExpertInjection:
    def __init__(self, num_experts: int):
        self._num_experts = max(int(num_experts), 1)
        self._index: Dict[Tuple[int, int, int], List[int]] = {}

    @property
    def num_experts(self) -> int:
        return self._num_experts

    def load_router_trace(self, path: str) -> int:
        records = _load_jsonl(path)
        router_events = _parse_router_events(records)
        self._index.clear()
        for event in router_events:
            key = (int(event["layer_id"]), int(event["req_idx"]), int(event["token_pos"]))
            self._index[key] = [int(eid) for eid in event.get("topk_experts", [])]
        logger.info(
            "TraceExpertInjection loaded %d unique (layer, req, pos) keys from %d events",
            len(self._index),
            len(router_events),
        )
        return len(router_events)

    def get_experts(self, layer_id: int, req_idx: int, token_pos: int) -> Optional[List[int]]:
        return self._index.get((int(layer_id), int(req_idx), int(token_pos)))

    def bias_router_logits(
        self,
        router_logits: torch.Tensor,
        layer_id: int,
        req_indices: Sequence[int],
        token_positions: Sequence[int],
        top_k: int,
    ) -> torch.Tensor:
        num_tokens, num_experts = router_logits.shape
        assert num_experts == self._num_experts, (
            f"num_experts mismatch: injection={self._num_experts}, logits={num_experts}"
        )
        assert len(req_indices) == num_tokens, (
            f"req_indices length {len(req_indices)} != num_tokens {num_tokens}"
        )
        assert len(token_positions) == num_tokens, (
            f"token_positions length {len(token_positions)} != num_tokens {num_tokens}"
        )

        masked = router_logits.clone()
        k = max(int(top_k), 1)

        for token_idx in range(num_tokens):
            key = (int(layer_id), int(req_indices[token_idx]), int(token_positions[token_idx]))
            trace_experts = self._index.get(key)
            if trace_experts is None or len(trace_experts) == 0:
                continue

            chosen = [int(eid) for eid in trace_experts if 0 <= int(eid) < self._num_experts]
            chosen = list(dict.fromkeys(chosen))

            if len(chosen) > k:
                chosen = chosen[:k]
            elif len(chosen) < k:
                surplus = list(dict.fromkeys(
                    eid for eid in range(self._num_experts) if eid not in chosen
                ))
                needed = k - len(chosen)
                chosen = chosen + surplus[:needed]

            mask = torch.full((num_experts,), -float("inf"), device=router_logits.device, dtype=router_logits.dtype)
            for eid in chosen:
                mask[eid] = 0.0

            masked[token_idx] = router_logits[token_idx] + mask

        return masked

    def verify_trace_coverage(
        self,
        req_indices: Sequence[int],
        token_positions: Sequence[int],
        layers: Sequence[int],
    ) -> Dict[str, int]:
        total = 0
        covered = 0
        for layer_id in layers:
            for token_idx in range(len(req_indices)):
                key = (int(layer_id), int(req_indices[token_idx]), int(token_positions[token_idx]))
                total += 1
                if key in self._index:
                    covered += 1
        return {"total": total, "covered": covered, "missing": total - covered}


def _load_jsonl(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_num} is not a JSON object")
            records.append(payload)
    return records


def _parse_router_events(records: List[dict]) -> List[dict]:
    events = []
    for record in records:
        if record.get("event") not in (None, "router_trace"):
            continue
        if "req_idx" not in record or "topk_experts" not in record:
            continue
        events.append(
            {
                "arrival_idx": int(record.get("arrival_idx", len(events))),
                "req_idx": int(record["req_idx"]),
                "phase": str(record.get("phase", "decode")),
                "layer_id": int(record.get("layer_id", 0)),
                "token_pos": int(record.get("token_pos", 0)),
                "topk_experts": [int(expert_id) for expert_id in record.get("topk_experts", [])],
                "topk_weights": [float(weight) for weight in record.get("topk_weights", [])],
            }
        )
    events.sort(key=lambda item: (item["arrival_idx"], item["layer_id"], item["token_pos"]))
    return events
