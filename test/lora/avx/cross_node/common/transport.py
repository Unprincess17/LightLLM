"""Persistent multiplexed TCP transport.

One TCP connection carries multiple outstanding requests with request IDs.
Responses may complete out of order. The client matches responses to requests
by req_id via a pending-futures dict.

Per spec "Persistent TCP transport requirement": B1-B3 use framed multiplexing
with request IDs; responses may complete out of order. This is a hard
requirement, not an implementation detail.
"""
import socket
import threading
import queue
from concurrent.futures import Future
from common.protocol import send_message, recv_message, new_request_id


class PersistentTransport:
    """Wraps a connected socket with request-ID multiplexing.

    One reader thread dispatches incoming responses to waiting requesters.
    Messages that do not match a pending future (i.e. requests arriving on
    the server side) are placed on an internal queue for ``recv_request``.
    Thread-safe for concurrent ``request`` calls.
    """

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._lock = threading.Lock()        # guards _pending dict
        self._send_lock = threading.Lock()   # serialises writes to _wf
        self._pending: dict = {}
        self._recv_queue: queue.Queue = queue.Queue()

        # Create buffered file objects once.  Calling makefile() per-send
        # would create a new BufferedWriter each time; when it is GC'd
        # (immediately in CPython refcounting) __del__ calls close(),
        # which closes the underlying socket.
        self._rf = sock.makefile("rb")
        self._wf = sock.makefile("wb")

        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def request(self, msg: dict, timeout: float = 30.0) -> dict:
        """Send a request and block until the matching response arrives."""
        req_id = msg.get("req_id")
        if req_id is None:
            req_id = new_request_id()
            msg["req_id"] = req_id
        fut = Future()
        with self._lock:
            self._pending[req_id] = fut
        with self._send_lock:
            send_message(self._wf, msg)
        return fut.result(timeout=timeout)

    def recv_request(self) -> dict:
        """Server-side: block until one request arrives.

        The reader thread is the sole consumer of the socket's read stream;
        messages without a matching pending future are queued here.
        """
        msg = self._recv_queue.get()
        if msg is None:               # sentinel from _read_loop shutdown
            raise ConnectionError("transport closed")
        return msg

    def send_response(self, req_id: int, response: dict) -> None:
        """Server-side: send a response tagged with req_id."""
        response["req_id"] = req_id
        with self._send_lock:
            send_message(self._wf, response)

    def close(self) -> None:
        """Close the underlying socket and fail all pending futures."""
        try:
            self._sock.close()
        except OSError:
            pass

    def _read_loop(self) -> None:
        """Reader thread: dispatch incoming messages to waiting futures
        or to the recv_request queue."""
        try:
            while True:
                msg = recv_message(self._rf)
                req_id = msg.get("req_id")
                with self._lock:
                    fut = self._pending.pop(req_id, None)
                if fut is not None and not fut.done():
                    fut.set_result(msg)
                else:
                    # No matching pending future — route to recv_request().
                    self._recv_queue.put(msg)
        except (EOFError, OSError):
            pass
        finally:
            with self._lock:
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(
                            ConnectionError("transport closed"))
            # Unblock any recv_request() waiting on the queue.
            self._recv_queue.put(None)
