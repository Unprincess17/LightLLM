"""Test persistent multiplexed TCP transport with request IDs and out-of-order responses."""
import socket
import threading
import time
import pytest
from common.transport import PersistentTransport


def test_round_trip_single_request():
    """Send one request, get one response, matched by request ID."""
    srv, cli = socket.socketpair()
    server_tp = PersistentTransport(srv)
    client_tp = PersistentTransport(cli)

    def server_handler():
        req = server_tp.recv_request()
        server_tp.send_response(req["req_id"], {"result": "ok", "echo_nm": req["nm"]})

    t = threading.Thread(target=server_handler, daemon=True)
    t.start()

    resp = client_tp.request({"type": "s4a_pooled", "nm": 8, "req_id": 1})
    assert resp["echo_nm"] == 8
    t.join(timeout=2)


def test_out_of_order_responses():
    """Send two requests; server responds to second first. Client matches by req_id."""
    srv, cli = socket.socketpair()
    server_tp = PersistentTransport(srv)
    client_tp = PersistentTransport(cli)

    def server_handler():
        req1 = server_tp.recv_request()
        req2 = server_tp.recv_request()
        server_tp.send_response(req2["req_id"], {"order": 2})
        server_tp.send_response(req1["req_id"], {"order": 1})

    t = threading.Thread(target=server_handler, daemon=True)
    t.start()

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(client_tp.request, {"type": "s4a", "req_id": 100})
        f2 = pool.submit(client_tp.request, {"type": "s4a", "req_id": 200})
        r1 = f1.result(timeout=2)
        r2 = f2.result(timeout=2)
    assert r1["order"] == 1
    assert r2["order"] == 2
    t.join(timeout=2)
