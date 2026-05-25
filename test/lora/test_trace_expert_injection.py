import json
from pathlib import Path

import torch
import pytest

from lightllm.server.lora.trace_expert_injection import TraceExpertInjection


NUM_EXPERTS = 128
TOP_K = 8


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _build_trace_rows():
    rows = []
    for req_idx in range(4):
        for layer_id in range(24):
            for token_pos in range(5):
                seed = req_idx * 1000 + layer_id * 50 + token_pos
                expert_set = [(seed + rank * 13) % NUM_EXPERTS for rank in range(TOP_K)]
                rows.append({
                    "event": "router_trace",
                    "arrival_idx": len(rows),
                    "req_idx": req_idx,
                    "phase": "decode",
                    "layer_id": layer_id,
                    "token_pos": token_pos,
                    "topk_experts": expert_set,
                    "topk_weights": [1.0 / TOP_K] * TOP_K,
                })
    return rows


def _select_topk_pytorch(router_logits, top_k):
    probs = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(probs, k=top_k, dim=-1)
    return topk_weights, topk_ids


class TestTraceExpertInjection:
    @pytest.fixture
    def trace_path(self, tmp_path):
        path = tmp_path / "router_trace.jsonl"
        _write_jsonl(path, _build_trace_rows())
        return str(path)

    @pytest.fixture
    def injection(self, trace_path):
        inj = TraceExpertInjection(NUM_EXPERTS)
        inj.load_router_trace(trace_path)
        return inj

    def test_load_and_lookup(self, injection):
        experts = injection.get_experts(layer_id=0, req_idx=0, token_pos=0)
        assert experts is not None
        assert len(experts) == TOP_K

    def test_missing_key_returns_none(self, injection):
        assert injection.get_experts(layer_id=99, req_idx=99, token_pos=99) is None

    def test_bias_router_logits_forces_correct_experts(self, injection):
        num_tokens = 8
        router_logits = torch.randn(num_tokens, NUM_EXPERTS)

        req_indices = [0, 1, 2, 3, 0, 1, 2, 3]
        token_positions = [0, 0, 0, 0, 1, 1, 1, 1]

        biased = injection.bias_router_logits(
            router_logits=router_logits,
            layer_id=0,
            req_indices=req_indices,
            token_positions=token_positions,
            top_k=TOP_K,
        )

        topk_weights, topk_ids = _select_topk_pytorch(biased, TOP_K)

        for token_idx in range(num_tokens):
            expected = set(injection.get_experts(
                layer_id=0,
                req_idx=req_indices[token_idx],
                token_pos=token_positions[token_idx],
            ))
            actual = set(topk_ids[token_idx].tolist())
            assert expected == actual, (
                f"token {token_idx}: expected={expected}, actual={actual}"
            )

    def test_bias_across_multiple_layers(self, injection):
        num_tokens = 6
        router_logits = torch.randn(num_tokens, NUM_EXPERTS)
        req_indices = [0, 1, 2, 0, 1, 2]
        token_positions = [0, 0, 0, 0, 0, 0]

        for layer_id in [0, 5, 10, 23]:
            biased = injection.bias_router_logits(
                router_logits=router_logits,
                layer_id=layer_id,
                req_indices=req_indices,
                token_positions=token_positions,
                top_k=TOP_K,
            )
            topk_weights, topk_ids = _select_topk_pytorch(biased, TOP_K)
            for token_idx in range(num_tokens):
                expected = set(injection.get_experts(
                    layer_id=layer_id,
                    req_idx=req_indices[token_idx],
                    token_pos=token_positions[token_idx],
                ))
                actual = set(topk_ids[token_idx].tolist())
                assert expected == actual, (
                    f"layer={layer_id} token={token_idx}: expected={expected}, actual={actual}"
                )

    def test_token_not_in_trace_still_selects_topk(self, injection):
        num_tokens = 4
        router_logits = torch.randn(num_tokens, NUM_EXPERTS)

        req_indices = [99, 99, 99, 99]
        token_positions = [99, 99, 99, 99]

        biased = injection.bias_router_logits(
            router_logits=router_logits,
            layer_id=0,
            req_indices=req_indices,
            token_positions=token_positions,
            top_k=TOP_K,
        )

        assert torch.equal(biased, router_logits)

        topk_weights, topk_ids = _select_topk_pytorch(biased, TOP_K)
        assert topk_ids.shape == (num_tokens, TOP_K)

    def test_verify_trace_coverage(self, injection):
        req_indices = [0, 1, 99, 0]
        token_positions = [0, 0, 0, 1]
        layers = list(range(24))

        coverage = injection.verify_trace_coverage(req_indices, token_positions, layers)
        assert coverage["total"] == len(req_indices) * len(layers)
        assert coverage["covered"] >= 2 * len(layers)
        assert coverage["covered"] + coverage["missing"] == coverage["total"]

    def test_injection_idempotent_on_empty_trace(self, tmp_path):
        empty_path = tmp_path / "empty_trace.jsonl"
        _write_jsonl(empty_path, [])

        inj = TraceExpertInjection(NUM_EXPERTS)
        n = inj.load_router_trace(str(empty_path))
        assert n == 0

        router_logits = torch.randn(2, NUM_EXPERTS)
        biased = inj.bias_router_logits(
            router_logits=router_logits,
            layer_id=0,
            req_indices=[0, 1],
            token_positions=[0, 0],
            top_k=TOP_K,
        )
        assert torch.equal(biased, router_logits)

    def test_partial_trace_coverage_mixed_batch(self, injection):
        num_tokens = 4
        router_logits = torch.randn(num_tokens, NUM_EXPERTS)

        req_indices = [0, 99, 1, 99]
        token_positions = [0, 99, 0, 99]

        biased = injection.bias_router_logits(
            router_logits=router_logits,
            layer_id=0,
            req_indices=req_indices,
            token_positions=token_positions,
            top_k=TOP_K,
        )

        topk_weights, topk_ids = _select_topk_pytorch(biased, TOP_K)

        for token_idx in [0, 2]:
            expected = set(injection.get_experts(
                layer_id=0,
                req_idx=req_indices[token_idx],
                token_pos=token_positions[token_idx],
            ))
            actual = set(topk_ids[token_idx].tolist())
            assert expected == actual, f"token {token_idx}: expected={expected}, actual={actual}"

        for token_idx in [1, 3]:
            actual = set(topk_ids[token_idx].tolist())
            assert len(actual) == TOP_K, f"token {token_idx}: should have {TOP_K} experts"
