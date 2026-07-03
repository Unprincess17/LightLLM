import struct
import io
import pytest
from common.protocol import send_message, recv_message, new_request_id


def test_round_trip_simple():
    buf = io.BytesIO()
    send_message(buf, {"type": "s4a_pooled", "nm": 8, "req_id": 1})
    buf.seek(0)
    msg = recv_message(buf)
    assert msg == {"type": "s4a_pooled", "nm": 8, "req_id": 1}


def test_round_trip_with_binary():
    buf = io.BytesIO()
    payload = {"type": "setup", "req_id": 2, "sizes": [2048, 64]}
    send_message(buf, payload)
    buf.seek(0)
    assert recv_message(buf) == payload


def test_request_id_unique():
    ids = {new_request_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_partial_read_raises():
    buf = io.BytesIO(struct.pack(">I", 100) + b'{"a": 1}')  # claims 100 bytes, gives 8
    with pytest.raises(EOFError):
        recv_message(buf)
