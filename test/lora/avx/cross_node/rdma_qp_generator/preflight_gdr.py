"""Verify GPUDirect RDMA prerequisites:
  1. nvidia_peermem kernel module is loaded
  2. ibv_devinfo finds at least one InfiniBand device
  3. CUDA runtime is available

Exit code 0 if all checks pass, 1 if any fail."""
from __future__ import annotations

import shutil
import subprocess
import sys


def check_nvidia_peermem() -> bool:
    """Returns True if nvidia_peermem (or nv_peer_mem) kernel module is loaded."""
    try:
        out = subprocess.run(
            ["lsmod"], capture_output=True, text=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  [FAIL] lsmod not available")
        return False
    if "nvidia_peermem" in out or "nv_peer_mem" in out:
        print("  [OK]   nvidia_peermem (or nv_peer_mem) loaded")
        return True
    print("  [FAIL] nvidia_peermem NOT loaded "
          "(run: sudo modprobe nvidia_peermem)")
    return False


def check_ibv_devinfo() -> bool:
    """Returns True if ibv_devinfo lists at least one IB device."""
    if shutil.which("ibv_devinfo") is None:
        print("  [FAIL] ibv_devinfo not in PATH (install rdma-core)")
        return False
    try:
        out = subprocess.run(
            ["ibv_devinfo"], capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        print(f"  [FAIL] ibv_devinfo failed: {exc.stderr}")
        return False
    if "hca_id" in out.lower() or "transport:" in out.lower():
        print("  [OK]   ibv_devinfo found IB device")
        return True
    print("  [FAIL] ibv_devinfo produced no devices")
    return False


def check_cuda() -> bool:
    """Returns True if CUDA runtime can be imported via PyTorch."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  [FAIL] torch not installed; cannot check CUDA")
        return False
    if torch.cuda.is_available():
        print(f"  [OK]   CUDA available "
              f"({torch.cuda.device_count()} device(s), "
              f"{torch.cuda.get_device_name(0)})")
        return True
    print("  [FAIL] CUDA not available via torch.cuda")
    return False


def main() -> int:
    print("GPUDirect RDMA pre-flight checks:")
    results = [
        check_nvidia_peermem(),
        check_ibv_devinfo(),
        check_cuda(),
    ]
    if all(results):
        print("\nAll checks passed.")
        return 0
    print("\nSome checks failed. GPUDirect RDMA will not work.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
