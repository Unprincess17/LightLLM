import pytest
from unittest.mock import MagicMock, patch

from test_moe_lora_api import test_batch_generation as batch_generation, MoELoRAPIClient


class FakeClient:
    def __init__(self):
        self.max_concurrent = 0
        self.current = 0

    def generate(self, prompt, max_tokens, adapter_id=None, ignore_eos=False):
        self.current += 1
        self.max_concurrent = max(self.max_concurrent, self.current)
        # Tiny delay so concurrency actually matters
        import time
        time.sleep(0.05)
        self.current -= 1
        return {"usage": {"completion_tokens": 1}}


def test_concurrent_batch_generation_respects_max_concurrent():
    """When max_concurrent_requests < len(prompts), actual concurrency is capped."""
    client = FakeClient()
    prompts = ["p"] * 20

    batch_generation(
        client=client,
        prompts=prompts,
        max_tokens=1,
        adapter_ids=None,
        verbose=False,
        ignore_eos=True,
        print_per_request=False,
        top_k_slowest=0,
        per_request_log_path=None,
        max_concurrent_requests=5,
    )

    assert client.max_concurrent <= 5, f"Expected <=5 concurrent, got {client.max_concurrent}"


def test_concurrent_batch_generation_defaults_to_all():
    """When max_concurrent_requests is None, all requests run concurrently."""
    client = FakeClient()
    prompts = ["p"] * 20

    batch_generation(
        client=client,
        prompts=prompts,
        max_tokens=1,
        adapter_ids=None,
        verbose=False,
        ignore_eos=True,
        print_per_request=False,
        top_k_slowest=0,
        per_request_log_path=None,
        max_concurrent_requests=None,
    )

    assert client.max_concurrent == 20
