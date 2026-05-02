#!/usr/bin/env python3
"""
MoE LoRA API Test Script

This script tests MoE LoRA serving with different configurations:
1. Merged mode - LoRA weights merged into base model
2. Detached mode - LoRA computed in parallel with base model
3. All-GPU - LoRA weights on GPU

Usage:
    # Start server first (see start_server.sh)
    # Then run this script to test

    python test_moe_lora_api.py --help
"""

import argparse
import json
import time
import requests
from typing import Optional, Dict, List, Union
import os
import base64
import mimetypes
import concurrent.futures
import math
import random
from collections import Counter



DEFAULT_URL = "http://localhost:8040"
DEFAULT_MODEL = "Qwen3-VL-30B-A3B-Instruct"
DEFAULT_NUM_REQUESTS = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Test MoE LoRA API")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="API server URL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--prompt", type=str, default="Describe the image", help="Input prompt")
    parser.add_argument(
        "--prompt_namespace",
        type=str,
        default="Req",
        help="Prefix namespace used to make prompts distinct across benchmark phases",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1,
        help="Max completion tokens to generate",
    )
    parser.add_argument(
        "--decode_target_tokens",
        type=int,
        default=None,
        help=(
            "Target decode-phase tokens per request. If set, this overrides --max_tokens "
            "using max_tokens = decode_target_tokens + 1."
        ),
    )
    parser.add_argument(
        "--ignore_eos",
        dest="ignore_eos",
        action="store_true",
        help="Ignore EOS so generation continues until max_tokens",
    )
    parser.add_argument(
        "--no_ignore_eos",
        dest="ignore_eos",
        action="store_false",
        help="Stop when EOS is generated",
    )
    parser.set_defaults(ignore_eos=True)
    parser.add_argument("--adapter_id", type=str, default="lora_dummy", help="Adapter ID for LoRA switching")
    parser.add_argument(
        "--adapter_ids",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Multiple adapter IDs (space/comma separated), "
            "e.g. --adapter_ids lora_dummy_0 lora_dummy_1 or "
            "--adapter_ids lora_dummy_0,lora_dummy_1"
        ),
    )
    parser.add_argument(
        "--poisson_lambda",
        type=float,
        default=3.0,
        help="Lambda used to sample adapter IDs from Poisson distribution in batched requests",
    )
    parser.add_argument(
        "--poisson_seed",
        type=int,
        default=42,
        help="Random seed for Poisson adapter sampling",
    )
    parser.add_argument(
        "--max_concurrent_requests",
        type=int,
        default=None,
        help=(
            "Cap the number of concurrent in-flight requests. "
            "Defaults to len(prompts) (no cap)."
        ),
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=3.0,
        help=(
            "Target requests per second. Requests are staggered at 1/rps "
            "intervals. Set to 0 for immediate (burst) dispatch."
        ),
    )
    parser.add_argument("--mode", type=str, default="detached", choices=["merged", "detached"], help="LoRA mode")
    parser.add_argument("--num_requests", type=int, default=DEFAULT_NUM_REQUESTS, help="Number of requests to send")
    parser.add_argument(
        "--adapter_trace_path",
        type=str,
        default=None,
        help=(
            "Optional JSONL path with explicit per-request adapter assignments. "
            "Rows are consumed in arrival order and should include adapter_id."
        ),
    )
    parser.add_argument(
        "--print_per_request",
        action="store_true",
        help="Print one performance line for every request in concurrent batch mode",
    )
    parser.add_argument(
        "--top_k_slowest",
        type=int,
        default=10,
        help="Print top-K slowest successful requests (0 to disable)",
    )
    parser.add_argument(
        "--per_request_log_path",
        type=str,
        default=None,
        help="Optional JSONL path for per-request metrics",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        choices=["warmup", "measurement"],
        help="Phase label written into each per-request metrics row",
    )
    parser.add_argument("--vision", action="store_true", help="Use image input")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def _is_base_model_token(token: str) -> bool:
    return token.lower() in {"default", "none", "null", "base"}


def parse_adapter_pool(single_adapter_id: str, adapter_ids: Optional[List[str]]) -> List[Optional[str]]:
    """Parse adapter IDs from CLI into a deduplicated adapter pool."""
    raw_tokens: List[str] = adapter_ids if adapter_ids else [single_adapter_id]
    parsed: List[Optional[str]] = []

    for raw in raw_tokens:
        for token in raw.split(","):
            normalized = token.strip()
            if not normalized:
                continue
            if _is_base_model_token(normalized):
                parsed.append(None)
            else:
                parsed.append(normalized)

    if not parsed:
        return [None]

    deduped: List[Optional[str]] = []
    seen = set()
    for adapter_id in parsed:
        if adapter_id in seen:
            continue
        seen.add(adapter_id)
        deduped.append(adapter_id)
    return deduped


def load_explicit_adapter_trace(trace_path: str, limit: Optional[int] = None) -> List[Optional[str]]:
    """Load adapter IDs from a JSONL trace ordered by arrival_idx, then req_idx."""
    rows = []
    with open(trace_path, "r", encoding="utf-8") as trace_file:
        for line_num, line in enumerate(trace_file, start=1):
            line = line.strip()
            if not line:
                continue

            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"adapter trace row {line_num} is not a JSON object")
            if "adapter_id" not in payload:
                raise ValueError(f"adapter trace row {line_num} is missing adapter_id")

            arrival_idx = int(payload.get("arrival_idx", line_num - 1))
            req_idx = int(payload.get("req_idx", line_num - 1))
            adapter_token = payload.get("adapter_id")
            adapter_id = None if adapter_token is None or _is_base_model_token(str(adapter_token)) else str(adapter_token)
            rows.append((arrival_idx, req_idx, adapter_id))

    rows.sort(key=lambda item: (item[0], item[1]))
    adapter_ids = [adapter_id for _arrival_idx, _req_idx, adapter_id in rows]
    if limit is not None:
        return adapter_ids[:limit]
    return adapter_ids


def _sample_poisson_value(poisson_lambda: float, rng: random.Random) -> int:
    """Sample one Poisson random value using Knuth's algorithm."""
    if poisson_lambda <= 0:
        return 0

    threshold = math.exp(-poisson_lambda)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def build_poisson_adapter_ids(
    num_requests: int,
    adapter_pool: List[Optional[str]],
    poisson_lambda: float,
    poisson_seed: int,
) -> List[Optional[str]]:
    """
    Build per-request adapter IDs by sampling adapter indices from a Poisson distribution.

    Sampled index k is clipped into [0, len(adapter_pool)-1].
    """
    if num_requests <= 0:
        return []
    if poisson_lambda < 0:
        raise ValueError(f"poisson_lambda must be >= 0, got {poisson_lambda}")
    if not adapter_pool:
        return [None] * num_requests
    if len(adapter_pool) == 1:
        return [adapter_pool[0]] * num_requests

    max_idx = len(adapter_pool) - 1
    rng = random.Random(poisson_seed)
    sampled_ids: List[Optional[str]] = []
    for _ in range(num_requests):
        sampled_idx = _sample_poisson_value(poisson_lambda, rng)
        sampled_ids.append(adapter_pool[min(sampled_idx, max_idx)])
    return sampled_ids


def _format_adapter_id(adapter_id: Optional[str]) -> str:
    return adapter_id if adapter_id is not None else "base_model"


def _percentile(values: List[float], pct: float) -> float:
    """Compute percentile with linear interpolation."""
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 1:
        return max(values)

    ordered = sorted(values)
    position = (len(ordered) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]

    lower_value = ordered[lower]
    upper_value = ordered[upper]
    weight = position - lower
    return lower_value + (upper_value - lower_value) * weight


def build_request_prompts(prompt: str, num_requests: int, prompt_namespace: str = "Req") -> List[str]:
    normalized_namespace = str(prompt_namespace).strip() or "Req"
    return [f"[{normalized_namespace}-{i}] {prompt}" for i in range(max(int(num_requests), 0))]


def resolve_effective_max_tokens(max_tokens: int, decode_target_tokens: Optional[int]) -> int:
    """Resolve runtime max_tokens, optionally targeting decode-phase token count."""
    if decode_target_tokens is not None:
        if decode_target_tokens < 1:
            raise ValueError(f"decode_target_tokens must be >= 1, got {decode_target_tokens}")
        return decode_target_tokens + 1
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
    return max_tokens


from requests.adapters import HTTPAdapter

class MoELoRAPIClient:
    """Client for testing MoE LoRA API with high concurrency support."""

    def __init__(self, url: str, model: str, max_concurrent_requests: int = 100):
        self.url = url.rstrip("/")
        self.model = model
        self.session = requests.Session()
        
        # Configure the connection pool to handle large concurrent batches
        adapter = HTTPAdapter(
            pool_connections=max_concurrent_requests, 
            pool_maxsize=max_concurrent_requests
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        adapter_id: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        image_path: Optional[str] = None,
        ignore_eos: bool = True,
    ) -> Dict:
        """Send a generation request with optional image support."""
        
        # 1. Prepare the content structure
        # If there is an image, content must be a list of dictionaries
        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Detect MIME type (image/jpeg, image/png, etc.)
            mime_type, _ = mimetypes.guess_type(image_path)
            mime_type = mime_type or "image/jpeg"

            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Multimodal content format
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ]
        else:
            # Standard text-only content
            content = prompt

        # 2. Build the request body
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "ignore_eos": ignore_eos,
        }

        # Add LoRA adapter ID if provided
        if adapter_id:
            body["adapters"] = [adapter_id]

        # 3. Execute the request
        try:
            resp = self.session.post(
                f"{self.url}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )

            if resp.status_code == 200:
                return resp.json()
            else:
                return {
                    "error": resp.text, 
                    "status_code": resp.status_code
                }
                
        except Exception as e:
            return {"error": str(e)}

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        adapter_id: Optional[str] = None,
        ignore_eos: bool = True,
    ):
        """Stream generation request."""
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "ignore_eos": ignore_eos,
        }

        if adapter_id:
            body["adapters"] = [adapter_id]

        try:
            resp = self.session.post(
                f"{self.url}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=120,
            )

            if resp.status_code == 200:
                for line in resp.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            data = line[6:]
                            if data != "[DONE]":
                                yield json.loads(data)
            else:
                yield {"error": resp.text, "status_code": resp.status_code}
        except Exception as e:
            yield {"error": str(e)}


def test_basic_generation(
    client: MoELoRAPIClient, prompt: str, max_tokens: int, verbose: bool, ignore_eos: bool
):
    """Test basic generation without LoRA."""
    print(f"\n=== Basic Generation (no LoRA) ===")
    print(f"Prompt: {prompt[:50]}...")

    start_time = time.time()
    result = client.generate(prompt, max_tokens=max_tokens, ignore_eos=ignore_eos)
    elapsed = time.time() - start_time

    if "error" in result:
        print(f"Error: {result['error']}")
        return None

    output = result["choices"][0]["message"]["content"]
    if verbose:
        print(f"Output: {output}")
    else:
        print(f"Output (first 100 chars): {output[:100]}...")

    print(f"Tokens generated: {result['usage']['completion_tokens']}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Speed: {result['usage']['completion_tokens'] / elapsed:.2f} tokens/s")

    return result


def test_lora_generation(
    client: MoELoRAPIClient,
    prompt: str,
    max_tokens: int,
    adapter_id: str,
    verbose: bool,
    ignore_eos: bool,
):
    """Test generation with LoRA.
    """
    print(f"\n=== LoRA Generation, adapter={adapter_id}) ===")
    print(f"Prompt: {prompt[:50]}...")

    start_time = time.time()
    result = client.generate(
        prompt,
        max_tokens=max_tokens,
        adapter_id=adapter_id,
        temperature=0.7,
        ignore_eos=ignore_eos,
    )
    elapsed = time.time() - start_time

    if "error" in result:
        print(f"Error: {result['error']}")
        return None

    output = result["choices"][0]["message"]["content"]
    if verbose:
        print(f"Output: {output}")
    else:
        print(f"Output (first 100 chars): {output[:100]}...")

    print(f"Tokens generated: {result['usage']['completion_tokens']}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Speed: {result['usage']['completion_tokens'] / elapsed:.2f} tokens/s")

    return result


def test_streaming(
    client: MoELoRAPIClient,
    prompt: str,
    max_tokens: int,
    adapter_id: Optional[str],
    verbose: bool,
    ignore_eos: bool,
):
    """Test streaming generation."""
    print(f"\n=== Streaming Generation ===")
    print(f"Prompt: {prompt[:50]}...")

    start_time = time.time()
    token_count = 0

    for chunk in client.stream_generate(
        prompt,
        max_tokens=max_tokens,
        adapter_id=adapter_id,
        ignore_eos=ignore_eos,
    ):
        if "error" in chunk:
            print(f"Error: {chunk['error']}")
            return None

        if "choices" in chunk:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                token_count += 1
                if verbose:
                    print(content, end="", flush=True)

    elapsed = time.time() - start_time
    print()  # Newline after streaming
    print(f"Tokens generated: {token_count}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Speed: {token_count / elapsed:.2f} tokens/s")

    return {"token_count": token_count, "elapsed": elapsed}



def test_batch_generation(
    client: MoELoRAPIClient,
    prompts: List[str],
    max_tokens: int,
    adapter_ids: Optional[Union[str, List[Optional[str]]]],
    verbose: bool,
    ignore_eos: bool,
    print_per_request: bool,
    top_k_slowest: int,
    per_request_log_path: Optional[str],
    phase: Optional[str] = None,
    max_concurrent_requests: Optional[int] = None,
    rps: float = 3.0,
):
    """Test batch generation with multiple concurrent prompts.

    ``adapter_ids`` can be:
    - ``None``: no adapter for all requests.
    - ``str``: same adapter for all requests.
    - ``List[Optional[str]]``: per-request adapter IDs (must match ``prompts`` length).
    """
    print(f"\n=== Concurrent Batch Generation ({len(prompts)} requests, rps={rps}) ===")

    if not prompts:
        print("No prompts provided. Skipping batch generation.")
        return []

    if adapter_ids is None:
        request_adapter_ids = [None] * len(prompts)
    elif isinstance(adapter_ids, str):
        request_adapter_ids = [adapter_ids] * len(prompts)
    elif isinstance(adapter_ids, list):
        if len(adapter_ids) != len(prompts):
            raise ValueError(
                f"adapter_ids length ({len(adapter_ids)}) must equal prompts length ({len(prompts)})."
            )
        request_adapter_ids = adapter_ids
    else:
        raise TypeError(
            "adapter_ids must be None, a string adapter ID, or a list of optional adapter IDs."
        )

    start_time = time.perf_counter()
    results: List[Dict] = [{} for _ in prompts]
    request_metrics: List[Optional[Dict]] = [None] * len(prompts)

    # Wrapper function for the executor
    def fetch(prompt_text: str, req_adapter_id: Optional[str]) -> Dict:
        req_start = time.perf_counter()
        result = client.generate(
            prompt_text,
            max_tokens=max_tokens,
            adapter_id=req_adapter_id,
            ignore_eos=ignore_eos,
        )
        req_end = time.perf_counter()
        return {
            "result": result,
            "req_start": req_start,
            "req_end": req_end,
        }

    # Dispatch all requests concurrently
    workers = min(len(prompts), max_concurrent_requests) if max_concurrent_requests else len(prompts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit tasks and store future-to-prompt mapping
        future_to_req = {}
        inter_arrival = 1.0 / rps if rps > 0 else 0.0
        for idx, (prompt_text, req_adapter_id) in enumerate(zip(prompts, request_adapter_ids)):
            if idx > 0 and inter_arrival > 0:
                time.sleep(inter_arrival)
            future = executor.submit(fetch, prompt_text, req_adapter_id)
            future_to_req[future] = idx
        
        # As each request completes, collect the results
        for future in concurrent.futures.as_completed(future_to_req):
            idx = future_to_req[future]
            try:
                payload = future.result()
                result = payload["result"]
                results[idx] = result
                req_start = payload["req_start"]
                req_end = payload["req_end"]
                latency = req_end - req_start

                completion_tokens = 0
                if "error" not in result:
                    completion_tokens = result.get("usage", {}).get("completion_tokens", 0)

                request_metrics[idx] = {
                    "request_id": f"req_{idx:06d}",
                    "index": idx,
                    "adapter_id": _format_adapter_id(request_adapter_ids[idx]),
                    "phase": phase,
                    "status": "error" if "error" in result else "ok",
                    "latency_s": latency,
                    "start_offset_s": req_start - start_time,
                    "finish_offset_s": req_end - start_time,
                    "completion_tokens": completion_tokens,
                    "tpot_mean_us": (latency * 1e6 / completion_tokens) if completion_tokens > 0 and latency > 0 else None,
                    "token_throughput": (completion_tokens / latency) if latency > 0 else 0.0,
                    "error": result.get("error") if "error" in result else None,
                }
            except Exception as exc:
                results[idx] = {"error": str(exc)}
                failed_time = time.perf_counter()
                request_metrics[idx] = {
                    "request_id": f"req_{idx:06d}",
                    "index": idx,
                    "adapter_id": _format_adapter_id(request_adapter_ids[idx]),
                    "phase": phase,
                    "status": "error",
                    "latency_s": None,
                    "start_offset_s": 0.0,
                    "finish_offset_s": failed_time - start_time,
                    "completion_tokens": 0,
                    "tpot_mean_us": None,
                    "token_throughput": 0.0,
                    "error": str(exc),
                }

    elapsed = time.perf_counter() - start_time
    
    # Calculate performance metrics
    total_tokens = sum(
        r.get("usage", {}).get("completion_tokens", 0) if "error" not in r else 0
        for r in results
    )

    success_count = sum(1 for r in results if "error" not in r)
    print(f"Successful requests: {success_count}/{len(prompts)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Overall System Throughput: {total_tokens / elapsed:.2f} tokens/s")

    successful_metrics = [
        m for m in request_metrics
        if m is not None and m["status"] == "ok"
    ]
    if successful_metrics:
        latencies = [m["latency_s"] for m in successful_metrics]
        min_latency = min(latencies)
        max_latency = max(latencies)
        starvation_ratio = max_latency / min_latency if min_latency > 0 else float("inf")
        print(
            "Per-request latency (success): "
            f"p50={_percentile(latencies, 0.50):.4f}s, "
            f"p95={_percentile(latencies, 0.95):.4f}s, "
            f"p99={_percentile(latencies, 0.99):.4f}s, "
            f"min={min_latency:.4f}s, max={max_latency:.4f}s, "
            f"max/min={starvation_ratio:.2f}x"
        )

    if top_k_slowest > 0 and successful_metrics:
        slowest = sorted(successful_metrics, key=lambda m: m["latency_s"], reverse=True)[:top_k_slowest]
        print(f"\nTop {len(slowest)} slowest successful requests (starvation candidates):")
        for rank, metric in enumerate(slowest, start=1):
            print(
                f"  #{rank:02d} req={metric['index']} "
                f"adapter={metric['adapter_id']} "
                f"latency={metric['latency_s']:.4f}s "
                f"finish@{metric['finish_offset_s']:.4f}s "
                f"tokens={metric['completion_tokens']} "
                f"tok/s={metric['token_throughput']:.2f}"
            )

    if print_per_request:
        print("\nPer-request performance:")
        for metric in request_metrics:
            if metric is None:
                continue
            if metric["status"] == "ok":
                print(
                    f"  req={metric['index']} "
                    f"adapter={metric['adapter_id']} "
                    f"latency={metric['latency_s']:.4f}s "
                    f"start@{metric['start_offset_s']:.4f}s "
                    f"finish@{metric['finish_offset_s']:.4f}s "
                    f"tokens={metric['completion_tokens']} "
                    f"tok/s={metric['token_throughput']:.2f}"
                )
            else:
                print(
                    f"  req={metric['index']} "
                    f"adapter={metric['adapter_id']} "
                    f"status=error "
                    f"latency={metric['latency_s']:.4f}s "
                    f"error={metric['error']}"
                )

    if per_request_log_path:
        with open(per_request_log_path, "w", encoding="utf-8") as log_file:
            for metric in request_metrics:
                if metric is None:
                    continue
                log_file.write(json.dumps(metric, ensure_ascii=True) + "\n")
        print(f"Per-request metrics saved to: {per_request_log_path}")

    if verbose:
        for i, res in enumerate(results):
            if "error" in res:
                print(f"Request {i} failed: {res['error']}")
            else:
                out = res["choices"][0]["message"]["content"]
                print(f"Request {i} output: {out[:50]}...")

    return results


def main():
    args = parse_args()
    effective_max_tokens = resolve_effective_max_tokens(args.max_tokens, args.decode_target_tokens)
    adapter_pool = parse_adapter_pool(args.adapter_id, args.adapter_ids)
    explicit_adapter_trace: Optional[List[Optional[str]]] = None
    effective_num_requests = args.num_requests
    if args.adapter_trace_path:
        explicit_adapter_trace = load_explicit_adapter_trace(args.adapter_trace_path)
        if not explicit_adapter_trace:
            raise ValueError(f"adapter trace is empty: {args.adapter_trace_path}")
        if effective_num_requests == DEFAULT_NUM_REQUESTS:
            effective_num_requests = len(explicit_adapter_trace)
        if effective_num_requests > len(explicit_adapter_trace):
            raise ValueError(
                f"adapter trace has only {len(explicit_adapter_trace)} requests, "
                f"but --num_requests={effective_num_requests}"
            )
        batch_adapter_ids = explicit_adapter_trace[:effective_num_requests]
    else:
        batch_adapter_ids = build_poisson_adapter_ids(
            num_requests=effective_num_requests,
            adapter_pool=adapter_pool,
            poisson_lambda=args.poisson_lambda,
            poisson_seed=args.poisson_seed,
        )
    vision_adapter_id = batch_adapter_ids[0] if batch_adapter_ids else None

    print("=" * 60)
    print("MoE LoRA API Test Script (Concurrent Batching)")
    print("=" * 60)
    print(f"URL: {args.url}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Max tokens (effective): {effective_max_tokens}")
    if args.decode_target_tokens is not None:
        print(f"Decode target tokens: {args.decode_target_tokens}")
    print(f"Prompt namespace: {args.prompt_namespace}")
    print(f"Ignore EOS: {args.ignore_eos}")
    print(
        "Adapter Pool: "
        + ", ".join(_format_adapter_id(adapter_id) for adapter_id in adapter_pool)
    )
    if args.adapter_trace_path:
        print(f"Adapter trace: {args.adapter_trace_path}")
    else:
        print(f"Poisson lambda: {args.poisson_lambda}")
        print(f"Poisson seed: {args.poisson_seed}")
    adapter_counts = Counter(_format_adapter_id(adapter_id) for adapter_id in batch_adapter_ids)
    print(
        "Batch adapter distribution: "
        + ", ".join(f"{adapter}:{count}" for adapter, count in adapter_counts.items())
    )
    print(f"Num requests (Target Batch Size): {effective_num_requests}")
    print(f"Target RPS: {args.rps}")
    print("=" * 60)

    client = MoELoRAPIClient(args.url, args.model)

    # Basic image generation logic remains unchanged
    if args.vision:
        print("\n=== Basic Image Generation ===")
        image_path = "/home/shufan/LightLLM/test/lora/test.png"  
        result = client.generate(
            args.prompt,
            max_tokens=effective_max_tokens,
            adapter_id=vision_adapter_id,
            temperature=0.7,
            image_path=image_path,
            ignore_eos=args.ignore_eos,
        )
        print(f"Image generation result: {result}")

    # Concurrent Batch generation with Cache Evasion
    # if args.num_requests > 1:
    # Prepend a phase-specific unique ID to bypass prompt-cache reuse across phases.
    prompts = build_request_prompts(args.prompt, effective_num_requests, args.prompt_namespace)

    test_batch_generation(
        client,
        prompts,
        max_tokens=effective_max_tokens,
        adapter_ids=batch_adapter_ids,
        verbose=args.verbose,
        ignore_eos=args.ignore_eos,
        print_per_request=args.print_per_request,
        top_k_slowest=args.top_k_slowest,
        per_request_log_path=args.per_request_log_path,
        phase=args.phase,
        max_concurrent_requests=args.max_concurrent_requests,
        rps=args.rps,
    )

    return 0


if __name__ == "__main__":
    exit(main())
