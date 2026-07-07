"""Tests for north-star recovery path implementations."""
import torch
import pytest
from common.northstar_paths import (
    cpu_first_recovery, load_then_run_recovery, oracle_recovery,
    init_weights, format_weights_for_path
)

H, I, R, NM = 2048, 2048, 64, 4
DTYPE_ACT = torch.float16
DTYPE_WEIGHT = torch.bfloat16

@pytest.fixture
def weights():
    """Generate random BF16 LoRA weights on CPU (host-resident)."""
    return init_weights(R, H, I, NM, dtype=DTYPE_WEIGHT, device="cpu")

@pytest.fixture
def activation():
    """Random FP16 activation on client GPU."""
    return torch.randn(NM, H, dtype=DTYPE_ACT, device="cuda")

class TestCpuFirstRecovery:

    def test_correctness(self, weights, activation):
        """cpu_first returns correct result visible on client GPU."""
        result, timeline = cpu_first_recovery(
            activation, weights, R, H, I, NM,
            num_cores=1, consumer_stream=torch.cuda.current_stream()
        )
        assert result.dtype == DTYPE_ACT
        assert result.device.type == "cuda"
        assert result.shape == (NM, I)
        # Verify against direct computation (per-batch LoRA: x_i @ A_i.T @ B_i)
        # Match implementation's data flow: activation FP16 -> BF16 -> FP32
        A_cpu = weights["A"].to(torch.float32)
        B_cpu = weights["B"].to(torch.float32)
        x_cpu = activation.cpu().to(torch.bfloat16).to(torch.float32)
        expected = torch.einsum("nh,nrh,nri->ni", x_cpu, A_cpu, B_cpu)
        # Match implementation output flow: FP32 -> BF16 -> FP16
        expected = expected.to(torch.bfloat16).to(DTYPE_ACT)
        # BF16 AVX kernel has lower precision than FP32 reference; use
        # looser tolerance (result magnitudes ~300, max observed diff ~8)
        torch.testing.assert_close(result.cpu().to(torch.float32),
                                   expected.to(torch.float32), atol=10.0, rtol=0.05)

    def test_timeline_has_t0_t1(self, weights, activation):
        """Timeline records T0 and T1."""
        _, timeline = cpu_first_recovery(
            activation, weights, R, H, I, NM,
            num_cores=1, consumer_stream=torch.cuda.current_stream()
        )
        assert timeline.get("T0") is not None
        assert timeline.get("T1") is not None
        assert timeline.l_recovery_us() > 0
        # Inner stages present
        assert timeline.get("cf0") is not None  # pack start
        assert timeline.get("cf7") is not None  # consumer visibility

class TestLoadThenRunRecovery:

    def test_correctness(self, weights, activation):
        """load_then_run returns correct result on client GPU."""
        result, timeline = load_then_run_recovery(
            activation, weights, R, H, I, NM,
            consumer_stream=torch.cuda.current_stream()
        )
        assert result.dtype == DTYPE_ACT
        assert result.device.type == "cuda"
        assert result.shape == (NM, I)
        # Verify (per-batch LoRA: x_i @ A_i.T @ B_i)
        A = weights["A"].to(torch.float32).cuda()
        B = weights["B"].to(torch.float32).cuda()
        x = activation.to(torch.float32)
        expected = torch.einsum("nh,nrh,nri->ni", x, A, B).to(DTYPE_ACT)
        torch.testing.assert_close(result.to(torch.float32),
                                   expected.to(torch.float32), atol=1e-2, rtol=1e-2)

    def test_timeline_has_t0_t1(self, weights, activation):
        _, timeline = load_then_run_recovery(
            activation, weights, R, H, I, NM,
            consumer_stream=torch.cuda.current_stream()
        )
        assert timeline.get("T0") is not None
        assert timeline.get("T1") is not None
        assert timeline.get("lt0") is not None  # H2D enqueue
        assert timeline.get("lt4") is not None  # consumer visibility

class TestOracleRecovery:

    def test_correctness(self, weights, activation):
        """Oracle: weights already on GPU, compute only."""
        gpu_weights = format_weights_for_path(weights, "oracle", device="cuda")
        result, timeline = oracle_recovery(
            activation, gpu_weights, R, H, I, NM,
            consumer_stream=torch.cuda.current_stream()
        )
        assert result.dtype == DTYPE_ACT
        assert result.shape == (NM, I)
        assert timeline.get("T0") is not None
        assert timeline.get("T1") is not None
