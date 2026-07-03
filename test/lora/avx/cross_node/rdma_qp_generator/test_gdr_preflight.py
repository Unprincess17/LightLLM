"""Pytest for preflight_gdr.py — verifies that GPUDirect RDMA prerequisites
exist on this machine."""
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _run_preflight() -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, str(_HERE / "preflight_gdr.py")],
        capture_output=True, text=True,
    )
    return result.returncode, result.stdout + result.stderr


def test_preflight_runs_without_crash():
    rc, output = _run_preflight()
    # 0 = all OK, 1 = at least one check failed (still a valid run)
    assert rc in (0, 1), f"unexpected returncode {rc}: {output}"


def test_preflight_checks_all_three_items():
    _, output = _run_preflight()
    assert "nvidia_peermem" in output
    assert "ibv_devinfo" in output
    assert "cuda" in output.lower()
