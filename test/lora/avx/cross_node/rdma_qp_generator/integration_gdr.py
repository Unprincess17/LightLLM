# test/lora/avx/cross_node/rdma_qp_generator/integration_gdr.py
"""Two-node integration test for GPUDirectTransport.

Usage:
  Server (UM251): python integration_gdr.py server 10.10.1.3
  Client (UM253): python integration_gdr.py client 10.10.1.1 10.10.1.3

The client fills its GPU buffer with a known pattern, RDMA WRITEs into
the server's GPU buffer, then RDMA READs back and verifies the round-trip.
"""
import sys
import time
import torch
from gpudirect_transport import GPUDirectTransport

BUF_BYTES = 1 << 20      # 1 MB
TENSOR_BYTES = 1 << 16   # 64 KB tensor inside the buffer


def make_pattern(nbytes: int, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, 255, (nbytes,), dtype=torch.uint8, device="cuda")


def gpu_ptr_to_tensor(ptr: int, nbytes: int) -> torch.Tensor:
    """Wrap a raw CUDA pointer as a uint8 torch tensor."""
    return torch.cuda.ByteTensor(nbytes)  # placeholder; see step note


def main():
    role = sys.argv[1]
    if role == "server":
        bind_ip = sys.argv[2]
        t = GPUDirectTransport(
            local_ip=bind_ip,
            remote_ip="",  # server doesn't initiate
            gpu_buffer_bytes=BUF_BYTES,
        )
        print("[server] start_server, waiting for client...", flush=True)
        t.start_server()
        print("[server] connected. Sleeping 30s for client to write/read.",
              flush=True)
        time.sleep(30)
        t.close()
        print("[server] done", flush=True)

    elif role == "client":
        local_ip = sys.argv[2]
        remote_ip = sys.argv[3]
        t = GPUDirectTransport(
            local_ip=local_ip,
            remote_ip=remote_ip,
            gpu_buffer_bytes=BUF_BYTES,
        )
        print("[client] start_client...", flush=True)
        t.start_client()
        print("[client] connected", flush=True)

        # Fill local GPU buffer using cudaMemcpy via torch
        pattern = make_pattern(TENSOR_BYTES)
        # Copy pattern bytes into the GPUDirect buffer via cudaMemcpy.
        # The GPUDirect buffer is at t.gpu_ptr(); use torch's
        # cuda runtime to copy via a temporary tensor + manual copy.
        from cuda import cudart   # `pip install cuda-python`
        err, = cudart.cudaMemcpy(
            t.gpu_ptr(), pattern.data_ptr(), TENSOR_BYTES,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
        )
        assert err == 0, f"cudaMemcpy failed: {err}"

        # RDMA WRITE to remote
        print("[client] write_to_remote", flush=True)
        t.write_to_remote(0, 0, TENSOR_BYTES)

        # Zero local buffer; then RDMA READ from remote
        err, = cudart.cudaMemset(t.gpu_ptr(), 0, TENSOR_BYTES)
        assert err == 0

        print("[client] read_from_remote", flush=True)
        t.read_from_remote(0, 0, TENSOR_BYTES)

        # Verify
        verify = torch.empty(TENSOR_BYTES, dtype=torch.uint8, device="cuda")
        err, = cudart.cudaMemcpy(
            verify.data_ptr(), t.gpu_ptr(), TENSOR_BYTES,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
        )
        assert err == 0
        if torch.equal(verify, pattern):
            print("[client] OK: round-trip matches", flush=True)
        else:
            ndiff = (verify != pattern).sum().item()
            print(f"[client] FAIL: {ndiff} bytes differ", flush=True)
            sys.exit(1)
        t.close()

    else:
        print(f"unknown role: {role}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()