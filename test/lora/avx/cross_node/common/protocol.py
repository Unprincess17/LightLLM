"""Length-prefixed JSON message framing with request IDs.

Replaces the copy-pasted _recv_exact/send_json/recv_json helpers in
bench_first_miss.py, bench_splitting.py, and bench_capacity.py.
"""
import json
import struct
import threading
from typing import Any, BinaryIO

_id_counter = 0
_id_lock = threading.Lock()


def new_request_id() -> int:
    """Return a process-unique, monotonically increasing request ID."""
    global _id_counter
    with _id_lock:
        _id_counter += 1
        return _id_counter


def send_message(stream: BinaryIO, msg: dict) -> None:
    """Write a 4-byte big-endian length prefix followed by UTF-8 JSON."""
    data = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    stream.write(struct.pack(">I", len(data)))
    stream.write(data)
    stream.flush()


def recv_message(stream: BinaryIO) -> dict:
    """Read one length-prefixed JSON message. Raises EOFError on truncation."""
    header = _recv_exact(stream, 4)
    if len(header) < 4:
        raise EOFError("stream closed during header read")
    (length,) = struct.unpack(">I", header)
    body = _recv_exact(stream, length)
    if len(body) < length:
        raise EOFError(f"stream closed during body read: got {len(body)}/{length}")
    return json.loads(body.decode("utf-8"))


def _recv_exact(stream: BinaryIO, n: int) -> bytes:
    """Read exactly n bytes from a stream; may return fewer on EOF."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
