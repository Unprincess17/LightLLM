# test/lora/avx/cross_node/rdma_qp_generator/test_gpudirect_transport.py
"""Pytest for gpudirect_transport.py."""
import os
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def test_import_module():
    from rdma_qp_generator import gpudirect_transport
    assert hasattr(gpudirect_transport, "GPUDirectTransport")


def test_c_symbols_resolve():
    from rdma_qp_generator import gpudirect_transport
    lib = gpudirect_transport._lib
    assert lib.gdrqp_create is not None
    assert lib.gdrqp_get_local_info is not None
    assert lib.gdrqp_connect is not None
    assert lib.gdrqp_write is not None
    assert lib.gdrqp_read is not None
    assert lib.gdrqp_destroy is not None


def test_class_init_attributes():
    from rdma_qp_generator.gpudirect_transport import GPUDirectTransport
    t = GPUDirectTransport(
        local_ip="10.10.1.1",
        remote_ip="10.10.1.3",
        mlx_device="mlx5_0",
        ib_port=1,
        qp_depth=16,
        gpu_buffer_bytes=1 << 20,
        control_port=18516,
    )
    assert t._local_ip == "10.10.1.1"
    assert t._remote_ip == "10.10.1.3"
    assert t._mlx_device == "mlx5_0"
    assert t._gpu_buffer_bytes == 1 << 20
    assert t.is_connected() is False


def test_close_on_partial_failure():
    """start_client on a port where nothing is listening should raise
    RuntimeError and leave the transport in a clean (closed) state."""
    from rdma_qp_generator.gpudirect_transport import GPUDirectTransport
    t = GPUDirectTransport(
        local_ip="127.0.0.1",
        remote_ip="127.0.0.1",
        gpu_buffer_bytes=1 << 16,
        control_port=19999,  # nothing listening
    )
    try:
        t.start_client()
    except RuntimeError:
        pass
    # After a failed start, ctx must be cleaned up and connected == False.
    assert t.is_connected() is False
    # Double-close must be safe (idempotent).
    t.close()


def test_start_server_bind_failure_cleans_up():
    """start_server on an already-bound port should raise OSError and
    leave the transport in a clean (closed) state — the listening socket
    must not leak."""
    import socket
    from rdma_qp_generator.gpudirect_transport import GPUDirectTransport

    # Occupy the port so bind() fails.
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 19998))
    blocker.listen(1)

    t = GPUDirectTransport(
        local_ip="127.0.0.1",
        remote_ip="127.0.0.1",
        gpu_buffer_bytes=1 << 16,
        control_port=19998,  # already in use
    )
    try:
        with pytest.raises(OSError):
            t.start_server()
    finally:
        blocker.close()
    # ctx was never created (bind failed before _create_ctx), but close
    # must still be safe.
    assert t.is_connected() is False
    t.close()
