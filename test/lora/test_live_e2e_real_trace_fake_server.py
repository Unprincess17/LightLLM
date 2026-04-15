import json
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        # keep test output clean
        return

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(content_length)

        body = {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"completion_tokens": 2, "prompt_tokens": 8, "total_tokens": 10},
        }
        payload = json.dumps(body, ensure_ascii=True).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def test_adapter_trace_replay_with_fake_server_non_default_port(tmp_path: Path):
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _FakeOpenAIHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    trace_path = tmp_path / "adapter_trace.jsonl"
    per_request_log_path = tmp_path / "per_request_metrics.jsonl"
    trace_rows = [
        {"arrival_idx": 2, "req_idx": 0, "adapter_id": "lora_dummy_2"},
        {"arrival_idx": 0, "req_idx": 0, "adapter_id": "lora_dummy_0"},
        {"arrival_idx": 1, "req_idx": 0, "adapter_id": "lora_dummy_1"},
    ]
    _write_jsonl(trace_path, trace_rows)

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "test/lora/test_moe_lora_api.py"
    cmd = [
        sys.executable,
        str(script),
        "--url",
        f"http://127.0.0.1:{port}",
        "--adapter_trace_path",
        str(trace_path),
        "--num_requests",
        "3",
        "--per_request_log_path",
        str(per_request_log_path),
        "--top_k_slowest",
        "0",
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert per_request_log_path.exists()

    metrics = []
    with per_request_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))

    assert len(metrics) == 3
    metrics_by_idx = sorted(metrics, key=lambda m: m["index"])
    # test_moe_lora_api sorts by (arrival_idx, req_idx) before issuing requests.
    assert [m["adapter_id"] for m in metrics_by_idx] == [
        "lora_dummy_0",
        "lora_dummy_1",
        "lora_dummy_2",
    ]
    assert all(m["status"] == "ok" for m in metrics_by_idx)
