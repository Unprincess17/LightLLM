from lightllm.server.core.objs.req import Req


class DummyShmArray:
    def __init__(self):
        self.detach_calls = 0
        self.destroy_calls = 0

    def detach(self):
        self.detach_calls += 1

    def destroy(self):
        self.destroy_calls += 1


def test_detach_local_prompt_logprob_shm_clears_handles():
    req = Req()
    prompt = DummyShmArray()
    logprobs = DummyShmArray()
    req.shm_prompt_ids = prompt
    req.shm_logprobs = logprobs
    req._cache_prompt_metadata = {"cached": True}

    req.detach_local_prompt_logprob_shm()

    assert prompt.detach_calls == 1
    assert logprobs.detach_calls == 1
    assert prompt.destroy_calls == 0
    assert logprobs.destroy_calls == 0
    assert req.shm_prompt_ids is None
    assert req.shm_logprobs is None
    assert req._cache_prompt_metadata is None


def test_destroy_owned_prompt_logprob_shm_uses_destroy():
    req = Req()
    prompt = DummyShmArray()
    logprobs = DummyShmArray()
    req.shm_prompt_ids = prompt
    req.shm_logprobs = logprobs

    req.destroy_owned_prompt_logprob_shm()

    assert prompt.destroy_calls == 1
    assert logprobs.destroy_calls == 1
    assert prompt.detach_calls == 0
    assert logprobs.detach_calls == 0
    assert req.shm_prompt_ids is None
    assert req.shm_logprobs is None
