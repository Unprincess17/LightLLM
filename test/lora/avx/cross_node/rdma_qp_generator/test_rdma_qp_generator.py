"""Unit tests for RDMA QP generator — config, CFFI import, handshake helpers."""
import os
import sys
import unittest

# Ensure the cross_node dir is on path
_HERE = os.path.dirname(os.path.abspath(__file__))
_CROSS_NODE = os.path.dirname(_HERE)
if _CROSS_NODE not in sys.path:
    sys.path.insert(0, _CROSS_NODE)

from ep_traffic_generator import EPTrafficConfig


class TestVerbsQPTrafficConfig(unittest.TestCase):
    def test_verbs_qp_server_command_is_not_used(self):
        """verbs_qp mode handles server lifecycle via Python, not server_cmd."""
        config = EPTrafficConfig(
            local_ip="10.10.1.1",
            remote_ip="10.10.1.3",
            mode="verbs_qp",
        )
        self.assertEqual(config.mode, "verbs_qp")
        self.assertEqual(config.server_cmd, ["true"])

    def test_verbs_qp_client_command_is_not_used(self):
        """client_cmd returns something reasonable for verbs_qp mode."""
        config = EPTrafficConfig(
            local_ip="10.10.1.1",
            remote_ip="10.10.1.3",
            mode="verbs_qp",
        )
        cmd = config.client_cmd(50)
        self.assertIsInstance(cmd, list)

    def test_default_qp_params(self):
        config = EPTrafficConfig(
            local_ip="10.10.1.1",
            remote_ip="10.10.1.3",
            mode="verbs_qp",
        )
        self.assertEqual(config.num_qps, 16)
        self.assertEqual(config.qp_depth, 128)
        self.assertEqual(config.msg_bytes, 65536)

    def test_custom_qp_params(self):
        config = EPTrafficConfig(
            local_ip="10.10.1.1",
            remote_ip="10.10.1.3",
            mode="verbs_qp",
            num_qps=32,
            qp_depth=256,
            msg_bytes=131072,
        )
        self.assertEqual(config.num_qps, 32)
        self.assertEqual(config.qp_depth, 256)
        self.assertEqual(config.msg_bytes, 131072)

    def test_verbs_qp_remote_cleanup_cmd(self):
        config = EPTrafficConfig(
            local_ip="10.10.1.1",
            remote_ip="10.10.1.3",
            mode="verbs_qp",
        )
        self.assertIn("rdma_qp_pressure", config.remote_cleanup_cmd)
        self.assertIn("_run_server", config.remote_cleanup_cmd)

    def test_ib_write_bw_mode_unchanged(self):
        """Non-verbs_qp modes should still work as before."""
        config = EPTrafficConfig(
            local_ip="10.10.1.1",
            remote_ip="10.10.1.3",
            mode="ib_write_bw",
        )
        self.assertEqual(config.mode, "ib_write_bw")
        self.assertIn("ib_write_bw", config.server_cmd)


class TestRDMATrafficGeneratorImport(unittest.TestCase):
    def test_can_import_rdma_qp_pressure(self):
        """librdmaqpgen.so loads and symbols resolve."""
        from rdma_qp_generator import rdma_qp_pressure
        self.assertTrue(hasattr(rdma_qp_pressure, 'RDMATrafficGenerator'))
        self.assertTrue(hasattr(rdma_qp_pressure, '_lib'))

    def test_c_symbols_are_callable(self):
        """Verify key C functions are accessible via CFFI."""
        from rdma_qp_generator import rdma_qp_pressure
        lib = rdma_qp_pressure._lib
        self.assertIsNotNone(lib.rdmaqp_create)
        self.assertIsNotNone(lib.rdmaqp_get_local_info)
        self.assertIsNotNone(lib.rdmaqp_connect)
        self.assertIsNotNone(lib.rdmaqp_start_burst_loop)
        self.assertIsNotNone(lib.rdmaqp_stop)
        self.assertIsNotNone(lib.rdmaqp_bytes_sent)
        self.assertIsNotNone(lib.rdmaqp_destroy)

    def test_default_constants(self):
        from rdma_qp_generator import rdma_qp_pressure
        self.assertEqual(rdma_qp_pressure.LINK_CAPACITY_GBPS, 200)
        self.assertEqual(rdma_qp_pressure.DEFAULT_NUM_QPS, 16)
        self.assertEqual(rdma_qp_pressure.DEFAULT_QP_DEPTH, 128)
        self.assertEqual(rdma_qp_pressure.DEFAULT_MSG_BYTES, 65536)


class TestHandshakeHelpers(unittest.TestCase):
    def test_send_recv_json_roundtrip(self):
        """Test _send_json / _recv_json round-trip via socketpair."""
        import socket
        from rdma_qp_generator.rdma_qp_pressure import _send_json, _recv_json

        a, b = socket.socketpair()
        try:
            data = {"qpn_base": 1234, "lid": 5, "gid": "fe80::1", "mr_addr": 0xDEAD, "mr_rkey": 42}
            _send_json(a, data)
            received = _recv_json(b)
            self.assertEqual(received, data)
        finally:
            a.close()
            b.close()

    def test_recv_exact(self):
        import socket
        from rdma_qp_generator.rdma_qp_pressure import _recv_exact

        a, b = socket.socketpair()
        try:
            a.sendall(b"hello world")
            result = _recv_exact(b, 11)
            self.assertEqual(result, b"hello world")
        finally:
            a.close()
            b.close()


if __name__ == "__main__":
    unittest.main()
