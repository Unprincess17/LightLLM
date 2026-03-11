#!/usr/bin/env python3
"""Build a deterministic ShareGPT-derived request corpus for router tracing."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

try:
    import ijson
except ImportError:  # pragma: no cover - fallback path
    ijson = None

from transformers import AutoTokenizer

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from common import (
    bucket_histogram,
    load_global_config,
    load_seed_config,
    numeric_summary,
    stable_hash_int,
    stage_output_dir,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a fixed ShareGPT request corpus")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--seeds", type=str, default=None, help="Path to configs/seeds.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Case-study run id")
    parser.add_argument("--output_dir", type=str, default=None, help="Override stage output directory")
    parser.add_argument("--sharegpt_path", type=str, default=None, help="ShareGPT V3 JSON path")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer or model path")
    parser.add_argument("--max_requests", type=int, default=None)
    parser.add_argument("--max_history_turns", type=int, default=None)
    parser.add_argument("--min_prompt_tokens", type=int, default=None)
    parser.add_argument("--max_prompt_tokens", type=int, default=None)
    parser.add_argument("--min_decode_tokens", type=int, default=None)
    parser.add_argument("--max_decode_tokens", type=int, default=None)
    return parser.parse_args()


def to_openai_role(raw_role: Optional[str]) -> str:
    token = (raw_role or "user").strip().lower()
    if token in {"system"}:
        return "system"
    if token in {"human", "user"}:
        return "user"
    return "assistant"


def iter_sharegpt_records(path: Path) -> Iterator[dict]:
    if ijson is not None:
        with path.open("r", encoding="utf-8") as handle:
            yield from ijson.items(handle, "item")
        return

    with path.open("r", encoding="utf-8") as handle:  # pragma: no cover - fallback path
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"expected top-level list in {path}")
    for record in payload:
        if isinstance(record, dict):
            yield record


def extract_sharegpt_example(record: dict, max_history_turns: int) -> Optional[Tuple[List[dict], str, str, Optional[str]]]:
    conversations = record.get("conversations", [])
    if not isinstance(conversations, list) or not conversations:
        return None

    last_assistant_idx = -1
    for idx in range(len(conversations) - 1, -1, -1):
        turn = conversations[idx]
        role = to_openai_role(turn.get("from") or turn.get("role"))
        if role == "assistant":
            last_assistant_idx = idx
            break

    if last_assistant_idx <= 0:
        return None

    completion_text = conversations[last_assistant_idx].get("value") or conversations[last_assistant_idx].get("content") or ""
    if not completion_text:
        return None

    start_idx = max(0, last_assistant_idx - max_history_turns)
    context_turns = conversations[start_idx:last_assistant_idx]
    messages: List[dict] = []
    for turn in context_turns:
        content = turn.get("value") or turn.get("content") or ""
        if not content:
            continue
        messages.append(
            {
                "role": to_openai_role(turn.get("from") or turn.get("role")),
                "content": content,
            }
        )

    if not messages:
        return None

    fallback_source = stable_hash_int(json.dumps(record, sort_keys=True, ensure_ascii=True), seed=0)
    source_id = str(record.get("id") or record.get("sharegpt_id") or record.get("conversation_id") or f"record_{fallback_source}")
    split_tag = record.get("split") or record.get("split_tag")
    split_value = None if split_tag is None else str(split_tag)
    return messages, completion_text, source_id, split_value


def render_prompt(tokenizer, messages: List[dict]) -> Tuple[str, str]:
    try:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return str(prompt_text), "chat_template"
    except Exception:
        rendered = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        return rendered + "\nassistant:", "fallback"


def token_length(tokenizer, text: str) -> int:
    tokenized = tokenizer(text, add_special_tokens=False)
    return len(tokenized.input_ids)


def main() -> None:
    args = parse_args()
    config = load_global_config(args.config)
    seeds = load_seed_config(args.seeds)

    case_config = config.get("case_study", {})
    prompt_config = case_config.get("prompt_corpus", {})
    paths_config = config.get("paths", {})

    sharegpt_path = Path(args.sharegpt_path or paths_config["sharegpt_v3"])
    tokenizer_path = args.tokenizer_path or paths_config["model_dir"]
    max_requests = int(args.max_requests or prompt_config.get("max_requests", 512))
    max_history_turns = int(args.max_history_turns or prompt_config.get("max_history_turns", 6))
    min_prompt_tokens = int(args.min_prompt_tokens or prompt_config.get("min_prompt_tokens", 32))
    max_prompt_tokens = int(args.max_prompt_tokens or prompt_config.get("max_prompt_tokens", 4096))
    min_decode_tokens = int(args.min_decode_tokens or prompt_config.get("min_decode_tokens", 16))
    max_decode_tokens = int(args.max_decode_tokens or prompt_config.get("max_decode_tokens", 256))
    bucket_size = int(prompt_config.get("prompt_hist_bucket_size", 128))
    prompt_seed = int(seeds.get("prompt_corpus_seed", seeds.get("global_seed", 7)))

    output_dir = stage_output_dir("prompt_corpus", config, args.run_id, args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)

    requests: List[dict] = []
    prompt_lengths: List[int] = []
    decode_lengths: List[int] = []
    dropped = Counter()
    scanned_records = 0

    for record in iter_sharegpt_records(sharegpt_path):
        scanned_records += 1
        extracted = extract_sharegpt_example(record, max_history_turns=max_history_turns)
        if extracted is None:
            dropped["invalid_conversation"] += 1
            continue

        messages, completion_text, source_id, split_tag = extracted
        prompt_text, render_mode = render_prompt(tokenizer, messages)
        prompt_len_tokens = token_length(tokenizer, prompt_text)
        completion_len_tokens = token_length(tokenizer, completion_text)
        target_decode_len = min(completion_len_tokens, max_decode_tokens)

        if prompt_len_tokens < min_prompt_tokens:
            dropped["prompt_too_short"] += 1
            continue
        if prompt_len_tokens > max_prompt_tokens:
            dropped["prompt_too_long"] += 1
            continue
        if completion_len_tokens < min_decode_tokens:
            dropped["decode_too_short"] += 1
            continue
        if target_decode_len > max_decode_tokens:
            dropped["decode_too_long"] += 1
            continue

        req_idx = len(requests)
        requests.append(
            {
                "req_idx": req_idx,
                "prompt_text": prompt_text,
                "prompt_len_tokens": prompt_len_tokens,
                "target_decode_len": target_decode_len,
                "source_id": source_id,
                "split_tag": split_tag,
                "messages": messages,
                "render_mode": render_mode,
            }
        )
        prompt_lengths.append(prompt_len_tokens)
        decode_lengths.append(target_decode_len)

        if len(requests) >= max_requests:
            break

    if not requests:
        raise RuntimeError("no requests passed the configured filters")

    fixed_requests_path = output_dir / "fixed_requests.jsonl"
    prompt_stats_path = output_dir / "prompt_stats.json"
    prompt_hist_path = output_dir / "prompt_length_hist.csv"
    request_manifest_path = output_dir / "request_manifest.csv"

    write_jsonl(fixed_requests_path, requests)
    write_json(
        prompt_stats_path,
        {
            "run_id": case_config.get("default_run_id") if args.run_id is None else args.run_id,
            "sharegpt_path": str(sharegpt_path),
            "tokenizer_path": str(tokenizer_path),
            "prompt_seed": prompt_seed,
            "records_scanned": scanned_records,
            "accepted_requests": len(requests),
            "dropped_counts": dict(dropped),
            "prompt_len_tokens": numeric_summary(prompt_lengths),
            "target_decode_len": numeric_summary(decode_lengths),
            "filters": {
                "max_requests": max_requests,
                "max_history_turns": max_history_turns,
                "min_prompt_tokens": min_prompt_tokens,
                "max_prompt_tokens": max_prompt_tokens,
                "min_decode_tokens": min_decode_tokens,
                "max_decode_tokens": max_decode_tokens,
            },
        },
    )
    write_csv(prompt_hist_path, ["bucket_start", "bucket_end", "count"], bucket_histogram(prompt_lengths, bucket_size))
    write_csv(
        request_manifest_path,
        ["req_idx", "source_id", "prompt_len_tokens", "target_decode_len", "split_tag", "render_mode"],
        (
            {
                "req_idx": record["req_idx"],
                "source_id": record["source_id"],
                "prompt_len_tokens": record["prompt_len_tokens"],
                "target_decode_len": record["target_decode_len"],
                "split_tag": record["split_tag"],
                "render_mode": record["render_mode"],
            }
            for record in requests
        ),
    )

    print(f"wrote {len(requests)} requests to {fixed_requests_path}")


if __name__ == "__main__":
    main()
