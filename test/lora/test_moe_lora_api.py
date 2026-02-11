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
from typing import Optional, Dict, List
import os
import base64
import mimetypes

DEFAULT_URL = "http://localhost:8040"
DEFAULT_MODEL = "Qwen3-VL-30B-A3B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description="Test MoE LoRA API")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="API server URL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--prompt", type=str, default="Describe the image", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=1, help="Max tokens to generate")
    parser.add_argument("--adapter_id", type=str, default="lora_dummy", help="Adapter ID for LoRA switching")
    parser.add_argument("--mode", type=str, default="detached", choices=["merged", "detached"], help="LoRA mode")
    parser.add_argument("--num_requests", type=int, default=1, help="Number of requests to send")
    parser.add_argument("--vision", action="store_true", help="Use image input")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


class MoELoRAPIClient:
    """Client for testing MoE LoRA API."""

    def __init__(self, url: str, model: str):
        self.url = url.rstrip("/")
        self.model = model
        self.session = requests.Session()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        adapter_id: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        image_path: Optional[str] = None,
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
    ):
        """Stream generation request."""
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
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


def test_basic_generation(client: MoELoRAPIClient, prompt: str, max_tokens: int, verbose: bool):
    """Test basic generation without LoRA."""
    print(f"\n=== Basic Generation (no LoRA) ===")
    print(f"Prompt: {prompt[:50]}...")

    start_time = time.time()
    result = client.generate(prompt, max_tokens=max_tokens)
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
):
    """Test streaming generation."""
    print(f"\n=== Streaming Generation ===")
    print(f"Prompt: {prompt[:50]}...")

    start_time = time.time()
    token_count = 0

    for chunk in client.stream_generate(prompt, max_tokens=max_tokens, adapter_id=adapter_id):
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
    adapter_id: Optional[str],
    verbose: bool,
):
    """Test batch generation with multiple prompts."""
    print(f"\n=== Batch Generation ({len(prompts)} prompts) ===")

    start_time = time.time()
    results = []

    for i, prompt in enumerate(prompts):
        result = client.generate(prompt, max_tokens=max_tokens, adapter_id=adapter_id)
        results.append(result)

    elapsed = time.time() - start_time
    total_tokens = sum(
        r.get("usage", {}).get("completion_tokens", 0) if "error" not in r else 0
        for r in results
    )

    success_count = sum(1 for r in results if "error" not in r)
    print(f"Successful requests: {success_count}/{len(prompts)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Speed: {total_tokens / elapsed:.2f} tokens/s")

    return results


def main():
    args = parse_args()

    print("=" * 60)
    print("MoE LoRA API Test Script")
    print("=" * 60)
    print(f"URL: {args.url}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Adapter ID: {args.adapter_id}")
    print(f"Num requests: {args.num_requests}")
    print("=" * 60)

    client = MoELoRAPIClient(args.url, args.model)

    # # Health check
    # if not test_health_check(client):
    #     print("\nServer is not healthy. Please start the server first.")
    #     print("See: bash test/lora/start_server.sh")
    #     return 1

    # # Get model config
    # test_model_config(client)

    # Test prompts
    test_prompts = [
        "Hello, I am a language model. My name is",
        "The capital of France is",
        "Explain quantum computing in simple terms:",
    ]

    # Basic generation (no LoRA)
    # test_basic_generation(client, args.prompt, args.max_tokens, args.verbose)

    # Basic image generation
    if args.vision:
        print("\n=== Basic Image Generation ===")
        image_path = "/home/shufan/LightLLM/test/lora/test.png"  # Replace with your test image path
        result = client.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            adapter_id=args.adapter_id if args.adapter_id != "default" else None,
            temperature=0.7,
            image_path= image_path,
        )
        print(f"Image generation result: {result}")

    # Test with LoRA (if adapter_id provided)
    if args.adapter_id:
        test_lora_generation(
            client,
            args.prompt,
            args.max_tokens,
            args.adapter_id,
            args.verbose,
        )


    # Batch generation
    if args.num_requests > 1:
        prompts = [args.prompt] * args.num_requests
        test_batch_generation(
            client,
            prompts,
            args.max_tokens,
            args.adapter_id if args.adapter_id != "default" else None,
            args.verbose,
        )


    return 0


if __name__ == "__main__":
    exit(main())
