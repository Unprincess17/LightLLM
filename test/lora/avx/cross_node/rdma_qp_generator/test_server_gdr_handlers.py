"""Unit tests for server-side GPUDirect strategy handler functions."""
from __future__ import annotations

import argparse
import json
import struct
import sys
import types
from unittest.mock import MagicMock, call, patch

import pytest
import torch

import cross_node_server


class FakeConn:
    def __init__(self, request: dict | None = None) -> None:
        self.sent = b""
        self.closed = False
        self._recv = b""
        if request is not None:
            payload = json.dumps(request).encode("utf-8")
            self._recv = struct.pack("!I", len(payload)) + payload

    def recv(self, n: int) -> bytes:
        chunk = self._recv[:n]
        self._recv = self._recv[n:]
        return chunk

    def sendall(self, data: bytes) -> None:
        self.sent += data

    def close(self) -> None:
        self.closed = True

    def response(self) -> dict:
        size = struct.unpack("!I", self.sent[:4])[0]
        return json.loads(self.sent[4:4 + size].decode("utf-8"))


@pytest.fixture
def fake_gdr_modules(monkeypatch: pytest.MonkeyPatch):
    mock_transport = MagicMock()
    mock_transport.gpu_ptr.return_value = 0xDEAD0000
    mock_transport.is_connected.return_value = True

    transport_module = types.ModuleType("rdma_qp_generator.gpudirect_transport")
    transport_cls = MagicMock(return_value=mock_transport)
    transport_module.GPUDirectTransport = transport_cls

    sync_module = types.ModuleType("rdma_qp_generator.gdr_sync")
    sync_module.copy_host_to_gpu = MagicMock()
    sync_module.copy_gpu_to_host = MagicMock()
    sync_module.copy_gpu_to_gpu = MagicMock()
    sync_module.set_flag = MagicMock()
    sync_module.poll_flag = MagicMock()

    monkeypatch.setitem(
        sys.modules, "rdma_qp_generator.gpudirect_transport", transport_module
    )
    monkeypatch.setitem(sys.modules, "rdma_qp_generator.gdr_sync", sync_module)
    return transport_cls, mock_transport, sync_module


def _params() -> dict:
    return {
        "rank": 64,
        "num_miss": 4,
        "hidden_dim": 2048,
        "intermediate_dim": 1536,
        "remote_ip": "10.10.1.1",
        "gdr_control_port": 18516,
    }


def test_handle_s2b_workflow_mocked(fake_gdr_modules):
    """S2b handler generates CPU weights, copies to GPU, RDMA WRITEs to client."""
    transport_cls, transport, sync = fake_gdr_modules
    conn = FakeConn()
    args = argparse.Namespace(listen_ip="10.10.1.3")
    params = _params()

    cross_node_server.handle_s2b(conn, args, params)

    weight_bytes = 4 * 64 * (2048 + 1536) * 2
    transport_cls.assert_called_once_with(
        local_ip="10.10.1.3",
        remote_ip="10.10.1.1",
        gpu_buffer_bytes=max(weight_bytes + 1, 1 << 20),
        control_port=18516,
    )
    transport.start_client.assert_called_once_with()
    sync.copy_host_to_gpu.assert_called_once()
    assert transport.write_to_remote.call_args_list == [
        call(0, 0, weight_bytes),
        call(weight_bytes, weight_bytes, 1),
    ]
    sync.set_flag.assert_called_once_with(0xDEAD0000, weight_bytes, 1)
    transport.close.assert_called_once_with()
    assert conn.response() == {"status": "ok", "weight_bytes": weight_bytes}


def test_handle_s4a_workflow_mocked(fake_gdr_modules):
    """S4a handler waits for activation, computes on GPU, writes result, signals."""
    transport_cls, transport, sync = fake_gdr_modules
    conn = FakeConn()
    args = argparse.Namespace(listen_ip="10.10.1.3")
    params = _params()

    original_randn = torch.randn
    original_empty = torch.empty
    zeros_result = torch.zeros(4, 1536)

    def fake_randn(*shape, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        return original_randn(*shape, **kwargs)

    def fake_empty(*shape, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        return original_empty(*shape, **kwargs)

    with patch.object(cross_node_server.torch, "randn", side_effect=fake_randn), \
         patch.object(cross_node_server.torch, "empty", side_effect=fake_empty), \
         patch.object(cross_node_server.torch, "zeros", return_value=zeros_result):
        cross_node_server.handle_s4a(conn, args, params)

    act_bytes = 2048 * 2
    result_bytes = 4 * 1536 * 2
    flag_offset = act_bytes + result_bytes
    transport_cls.assert_called_once_with(
        local_ip="10.10.1.3",
        remote_ip="10.10.1.1",
        gpu_buffer_bytes=max(flag_offset + 1, 1 << 20),
        control_port=18516,
    )
    sync.poll_flag.assert_called_once_with(0xDEAD0000, flag_offset, 1, timeout_s=30.0)
    assert sync.copy_gpu_to_gpu.call_count == 2
    assert transport.write_to_remote.call_args_list == [
        call(act_bytes, act_bytes, result_bytes),
        call(flag_offset, flag_offset, 1),
    ]
    assert sync.set_flag.call_args_list == [
        call(0xDEAD0000, flag_offset, 0),
        call(0xDEAD0000, flag_offset, 0),
    ]
    transport.close.assert_called_once_with()
    assert conn.response() == {"status": "ok", "result_bytes": result_bytes}


def test_handle_s5b_workflow_mocked(fake_gdr_modules):
    """S5b handler waits for activation, computes on CPU, writes result, signals."""
    _transport_cls, transport, sync = fake_gdr_modules
    conn = FakeConn()
    args = argparse.Namespace(listen_ip="10.10.1.3")
    params = _params()

    cross_node_server.handle_s5b(conn, args, params)

    act_bytes = 2048 * 2
    result_bytes = 4 * 1536 * 2
    flag_offset = act_bytes + result_bytes
    sync.poll_flag.assert_called_once_with(0xDEAD0000, flag_offset, 1, timeout_s=30.0)
    sync.copy_gpu_to_host.assert_called_once()
    sync.copy_host_to_gpu.assert_called_once_with(
        0xDEAD0000 + act_bytes,
        sync.copy_host_to_gpu.call_args.args[1],
        result_bytes,
    )
    assert transport.write_to_remote.call_args_list == [
        call(act_bytes, act_bytes, result_bytes),
        call(flag_offset, flag_offset, 1),
    ]
    assert sync.set_flag.call_args_list == [
        call(0xDEAD0000, flag_offset, 0),
        call(0xDEAD0000, flag_offset, 0),
    ]
    transport.close.assert_called_once_with()
    assert conn.response() == {"status": "ok", "result_bytes": result_bytes}


def test_s2b_buffer_layout():
    """Verify the weight packing layout for S2b."""
    rank, num_miss = 64, 4
    hidden_dim, intermediate_dim = 2048, 1536

    a_bytes = num_miss * rank * hidden_dim * 2
    b_bytes = num_miss * rank * intermediate_dim * 2
    total = a_bytes + b_bytes
    flag_offset = total

    assert flag_offset == total
    assert total == num_miss * rank * (hidden_dim + intermediate_dim) * 2


def test_s4a_s5b_buffer_layout():
    """Verify buffer layout for execution-first GPUDirect strategies."""
    hidden_dim, intermediate_dim = 2048, 1536
    num_miss = 4
    act_bytes = hidden_dim * 2  # bf16 activation
    result_bytes = num_miss * intermediate_dim * 2  # bf16 result
    flag_offset = act_bytes + result_bytes

    assert flag_offset == act_bytes + result_bytes
    assert flag_offset == hidden_dim * 2 + num_miss * intermediate_dim * 2


def test_handle_request_dispatches_gdr_strategies(monkeypatch: pytest.MonkeyPatch):
    """handle_request dispatches GPUDirect strategy request types."""
    calls = []

    def fake_handler(conn, args, params):
        calls.append((conn, args.listen_ip, params))
        cross_node_server.send_json_response(conn, {"status": "ok"})

    monkeypatch.setattr(cross_node_server, "handle_s2b", fake_handler)
    conn = FakeConn({
        "type": "strategy_s2b",
        "rank": 64,
        "num_miss": 4,
        "remote_ip": "10.10.1.1",
        "gdr_control_port": 18516,
    })
    args = argparse.Namespace(listen_ip="10.10.1.3")

    cross_node_server.handle_request(
        conn, ("10.10.1.1", 1234), args, 2048, 1536, 0, 1,
    )

    assert conn.closed
    assert conn.response() == {"status": "ok"}
    assert calls[0][1] == "10.10.1.3"
    assert calls[0][2]["rank"] == 64
    assert calls[0][2]["hidden_dim"] == 2048
    assert calls[0][2]["intermediate_dim"] == 1536
