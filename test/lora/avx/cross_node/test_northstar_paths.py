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


from common.forced_cold import ForcedColdWeightPool

class TestForcedColdPool:

    def test_no_reuse_within_window(self):
        """Each request gets a distinct weight pair; no reuse until pool exhausted."""
        pool = ForcedColdWeightPool(R=64, H=2048, I=2048, pool_size=100,
                                    dtype=torch.bfloat16, device="cpu", seed=42)
        w0 = pool.get(0)
        w1 = pool.get(1)
        assert not torch.equal(w0["A"], w1["A"])
        assert not torch.equal(w0["B"], w1["B"])

    def test_oracle_exempt(self):
        """Oracle gets weights on GPU (warm); pool still provides cold weights for others."""
        pool = ForcedColdWeightPool(R=64, H=2048, I=2048, pool_size=50,
                                    dtype=torch.bfloat16, device="cpu", seed=42)
        # Oracle path: weights on GPU (warm) -- not from the cold pool
        oracle_weights = pool.get_oracle_weights(0, device="cuda")
        assert oracle_weights["A"].device.type == "cuda"
        # Cold path: weights on CPU (cold on GPU)
        cold_weights = pool.get(0)
        assert cold_weights["A"].device.type == "cpu"

    def test_pool_size_exceeds_measurement_window(self):
        """Pool must be large enough that no weight is reused in the window."""
        pool = ForcedColdWeightPool(R=64, H=2048, I=2048, pool_size=10,
                                    dtype=torch.bfloat16, device="cpu", seed=42)
        # Requesting index 10 should raise (exceeds pool)
        with pytest.raises(IndexError):
            pool.get(10)


from common.path_randomization import randomize_path_order, paired_block_order

class TestPathRandomization:

    def test_randomize_path_order(self):
        """Path order is randomized within a paired block."""
        paths = ["cpu_first", "load_then_run", "remote_improved", "oracle"]
        order1 = randomize_path_order(paths, seed=42)
        order2 = randomize_path_order(paths, seed=42)
        order3 = randomize_path_order(paths, seed=43)
        # Same seed = same order (reproducible)
        assert order1 == order2
        # Different seed = likely different order
        assert order1 != order3 or len(paths) <= 2
        # All paths present
        assert set(order1) == set(paths)

    def test_paired_block_order(self):
        """Paired block: same path order across all cells in one trial."""
        cells = [(16, 1), (16, 8), (64, 1), (64, 8), (256, 8)]
        paths = ["cpu_first", "load_then_run", "remote_improved", "oracle"]
        block = paired_block_order(cells, paths, seed=42)
        # All cells get the same path order within a block
        first_order = block[cells[0]]
        for cell in cells:
            assert block[cell] == first_order
        # All paths present
        assert set(first_order) == set(paths)

