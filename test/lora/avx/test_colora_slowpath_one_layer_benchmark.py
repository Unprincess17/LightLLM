import importlib.util
from pathlib import Path


_BENCH_PATH = (
    Path(__file__).resolve().parent / "benchmark_colora_slowpath_one_layer.py"
)
_BENCH_SPEC = importlib.util.spec_from_file_location("colora_one_layer_bench", _BENCH_PATH)
assert _BENCH_SPEC is not None and _BENCH_SPEC.loader is not None
bench = importlib.util.module_from_spec(_BENCH_SPEC)
_BENCH_SPEC.loader.exec_module(bench)


def test_phase_order_covers_figure_flow():
    phase_order = [phase.name for phase in bench.build_phase_plan()]
    assert phase_order == [
        "COLoRA_Hybrid_Prepare",
        "COLoRA_CacheLookupAndPolicy",
        "COLoRA_BuildHitMissMasks",
        "COLoRA_HitPath_Prepare",
        "COLoRA_GPU_Hit_Path",
        "COLoRA_MissPath_CheckAndPrepare",
        "COLoRA_MissPath_ExecuteAndCommit",
        "COLoRA_PostCompute_StatsFinalize",
    ]


def test_build_bins_hit_rate_split():
    bins = bench.build_bins_for_hit_rate(
        batch_size=10,
        hit_rate=0.6,
        hit_adapter_id=0,
        miss_adapter_id=1,
        device="cpu",
    )
    assert int((bins == 0).sum().item()) == 6
    assert int((bins == 1).sum().item()) == 4


def test_measurement_policy_from_hit_rate():
    assert bench.resolve_measurement_policy(0.0) == {
        "prepromote_hit_adapter": False,
        "disable_measurement_promotion": True,
    }
    assert bench.resolve_measurement_policy(1.0) == {
        "prepromote_hit_adapter": True,
        "disable_measurement_promotion": False,
    }
    assert bench.resolve_measurement_policy(0.5) == {
        "prepromote_hit_adapter": True,
        "disable_measurement_promotion": True,
    }
