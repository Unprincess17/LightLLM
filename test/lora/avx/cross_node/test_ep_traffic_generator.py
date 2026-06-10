import unittest

from ep_traffic_generator import EPTrafficConfig


class TestEPAllToAllTrafficConfig(unittest.TestCase):
    def test_alltoall_server_command_uses_python_generator_with_counter_options(self):
        config = EPTrafficConfig(
            local_ip="10.10.1.1",
            remote_ip="10.10.1.3",
            mlx_device="mlx5_0",
            mode="alltoall",
            ib_port=1,
            counter_names=("port_xmit_data", "port_rcv_data"),
        )

        cmd = config.server_cmd

        self.assertIn("python", cmd[0])
        self.assertTrue(any(part.endswith("ep_alltoall_traffic.py") for part in cmd))
        self.assertIn("--role", cmd)
        self.assertIn("server", cmd)
        self.assertIn("--bind-ip", cmd)
        self.assertIn("10.10.1.3", cmd)
        self.assertIn("--mlx-device", cmd)
        self.assertIn("mlx5_0", cmd)
        self.assertIn("--ib-port", cmd)
        self.assertIn("1", cmd)
        self.assertIn("--counters", cmd)
        self.assertIn("port_xmit_data,port_rcv_data", cmd)

    def test_alltoall_remote_start_uses_separate_cleanup_and_start_commands(self):
        config = EPTrafficConfig(
            local_ip="10.10.1.1",
            remote_ip="10.10.1.3",
            mode="alltoall",
        )

        commands = config.remote_start_cmds

        self.assertEqual(2, len(commands))
        self.assertIn("pkill", commands[0])
        self.assertNotIn("nohup", commands[0])
        self.assertIn("nohup", commands[1])
        self.assertNotIn("pkill", commands[1])

    def test_alltoall_cleanup_command_kills_python_generator_processes(self):
        config = EPTrafficConfig(
            local_ip="10.10.1.1",
            remote_ip="10.10.1.3",
            mode="alltoall",
        )

        cmd = config.remote_cleanup_cmd

        self.assertIn("alltoall_traffic.py", cmd)
        self.assertIn("pkill", cmd)
        self.assertNotIn("ib_write_bw || true", cmd)

    def test_alltoall_client_command_scales_target_bandwidth_by_ep_percentage(self):
        config = EPTrafficConfig(
            local_ip="10.10.1.1",
            remote_ip="10.10.1.3",
            mode="alltoall",
            link_capacity_gbps=200,
            alltoall_streams=16,
        )

        cmd = config.client_cmd(75)

        self.assertTrue(any(part.endswith("ep_alltoall_traffic.py") for part in cmd))
        self.assertIn("--role", cmd)
        self.assertIn("client", cmd)
        self.assertIn("--peer-ip", cmd)
        self.assertIn("10.10.1.3", cmd)
        self.assertIn("--target-gbps", cmd)
        self.assertIn("150", cmd)
        self.assertIn("--streams", cmd)
        self.assertIn("16", cmd)
        self.assertIn("--validate-counters", cmd)


if __name__ == "__main__":
    unittest.main()
