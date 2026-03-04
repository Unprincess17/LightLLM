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


def parse_args():
    parser = argparse.ArgumentParser(description="Test MoE LoRA API")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="API server URL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--prompt", type=str, default="Describe the image", help="Input prompt")
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
    parser.add_argument("--mode", type=str, default="detached", choices=["merged", "detached"], help="LoRA mode")
    parser.add_argument("--num_requests", type=int, default=1, help="Number of requests to send")
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
):
    """Test batch generation with multiple concurrent prompts.

    ``adapter_ids`` can be:
    - ``None``: no adapter for all requests.
    - ``str``: same adapter for all requests.
    - ``List[Optional[str]]``: per-request adapter IDs (must match ``prompts`` length).
    """
    print(f"\n=== Concurrent Batch Generation ({len(prompts)} requests) ===")

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

    start_time = time.time()
    results: List[Dict] = [{} for _ in prompts]

    # Wrapper function for the executor
    def fetch(prompt_text: str, req_adapter_id: Optional[str]) -> Dict:
        return client.generate(
            prompt_text,
            max_tokens=max_tokens,
            adapter_id=req_adapter_id,
            ignore_eos=ignore_eos,
        )

    # Dispatch all requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        # Submit tasks and store future-to-prompt mapping
        future_to_req = {
            executor.submit(fetch, prompt_text, req_adapter_id): idx
            for idx, (prompt_text, req_adapter_id) in enumerate(zip(prompts, request_adapter_ids))
        }
        
        # As each request completes, collect the results
        for future in concurrent.futures.as_completed(future_to_req):
            idx = future_to_req[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as exc:
                results[idx] = {"error": str(exc)}

    elapsed = time.time() - start_time
    
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
    batch_adapter_ids = build_poisson_adapter_ids(
        num_requests=args.num_requests,
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
    print(f"Ignore EOS: {args.ignore_eos}")
    print(
        "Adapter Pool: "
        + ", ".join(_format_adapter_id(adapter_id) for adapter_id in adapter_pool)
    )
    print(f"Poisson lambda: {args.poisson_lambda}")
    print(f"Poisson seed: {args.poisson_seed}")
    adapter_counts = Counter(_format_adapter_id(adapter_id) for adapter_id in batch_adapter_ids)
    print(
        "Batch adapter distribution: "
        + ", ".join(f"{adapter}:{count}" for adapter, count in adapter_counts.items())
    )
    print(f"Num requests (Target Batch Size): {args.num_requests}")
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
    # Prepend a unique ID to each prompt to bypass the RadixAttention prefix cache
    prompts = [f"[Req-{i}] {args.prompt}" for i in range(args.num_requests)]
    
    test_batch_generation(
        client,
        prompts,
        max_tokens=effective_max_tokens,
        adapter_ids=batch_adapter_ids,
        verbose=args.verbose,
        ignore_eos=args.ignore_eos,
    )

    return 0


if __name__ == "__main__":
    exit(main())
