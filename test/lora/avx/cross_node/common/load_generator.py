"""True open-loop load generator per spec section "True open-loop generator".

Arrivals are pre-generated into a trace. The runner dispatches at scheduled
times INDEPENDENTLY of completion. A large client ingress queue absorbs
overload; overflow is flagged, not silently dropped. Drain protocol completes
in-flight requests with a bounded timeout and reports censored count.
"""
import queue
import random
import threading
import time
from dataclasses import dataclass, field, asdict


@dataclass
class TraceEvent:
    arrival_time: float        # seconds from run start
    job_class: str             # "light" or "heavy"
    req_id: int
    nm: int = 1                # number of misses for this request


@dataclass
class Trace:
    events: list
    duration_s: float
    seed: int

    def __len__(self):
        return len(self.events)

    def __iter__(self):
        return iter(self.events)


def generate_poisson_trace(lam: float, duration_s: float, seed: int,
                           classes: list = None, heavy_frac: float = 0.0,
                           nm_options_light: list = None,
                           nm_options_heavy: list = None) -> Trace:
    """Pre-generate a Poisson arrival trace. Class assigned by heavy_frac.

    NM (number of misses) is assigned per event based on job_class:
    - heavy events draw from nm_options_heavy (default 1)
    - light events draw from nm_options_light (default 1)
    """
    classes = classes or ["light"]
    rng = random.Random(seed)
    events = []
    t = 0.0
    req_id = 0
    while t < duration_s:
        t += rng.expovariate(lam)
        if t >= duration_s:
            break
        if "heavy" in classes and rng.random() < heavy_frac:
            cls = "heavy"
            if nm_options_heavy:
                nm = rng.choice(nm_options_heavy)
            else:
                nm = 1
        else:
            cls = "light"
            if nm_options_light:
                nm = rng.choice(nm_options_light)
            else:
                nm = 1
        events.append(TraceEvent(arrival_time=t, job_class=cls,
                                 req_id=req_id, nm=nm))
        req_id += 1
    return Trace(events=events, duration_s=duration_s, seed=seed)


@dataclass
class AdmissionCounters:
    generated: int = 0
    c0_inserted: int = 0
    c1_dispatched: int = 0
    c2_admitted: int = 0
    completed: int = 0
    rejected: int = 0
    timed_out: int = 0
    unfinished: int = 0
    final_queue_length: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class NoopSink:
    """Minimal sink for generator validation. Sleeps process_time_s then completes."""
    def __init__(self, process_time_s: float = 0.0):
        self.process_time_s = process_time_s

    def submit(self, event: TraceEvent) -> None:
        if self.process_time_s > 0:
            time.sleep(self.process_time_s)


class OpenLoopRunner:
    """Runs a trace against a sink in true open-loop fashion.

    Producer thread: at each event's arrival_time, inserts into a bounded
    ingress queue (counted as c0). NEVER blocks the producer on completion.
    Consumer threads: pull from ingress queue, call sink.submit(), mark complete.

    ingress_capacity must be >= len(trace) in primary capacity mode so overload
    is exhibited (queue growth), not hidden by rejection.
    """
    def __init__(self, trace: Trace, sink, ingress_capacity: int,
                 n_consumers: int = 1, drain_timeout_s: float = 10.0):
        self.trace = trace
        self.sink = sink
        self.ingress = queue.Queue(maxsize=ingress_capacity)
        self.n_consumers = n_consumers
        self.drain_timeout_s = drain_timeout_s
        self.counters = AdmissionCounters()
        self._lock = threading.Lock()

    def run(self) -> AdmissionCounters:
        start = time.perf_counter()
        stop_event = threading.Event()
        consumers = []
        for _ in range(self.n_consumers):
            t = threading.Thread(target=self._consumer, args=(stop_event,), daemon=True)
            t.start()
            consumers.append(t)

        # Producer
        for ev in self.trace.events:
            with self._lock:
                self.counters.generated += 1
            target = start + ev.arrival_time
            now = time.perf_counter()
            if now < target:
                time.sleep(target - now)
            try:
                self.ingress.put_nowait(ev)
                with self._lock:
                    self.counters.c0_inserted += 1
            except queue.Full:
                with self._lock:
                    self.counters.rejected += 1

        # Drain: signal stop, wait for consumers with timeout
        stop_event.set()
        deadline = time.perf_counter() + self.drain_timeout_s
        for t in consumers:
            remaining = max(0.0, deadline - time.perf_counter())
            t.join(timeout=remaining)

        with self._lock:
            self.counters.unfinished = self.ingress.qsize()
            self.counters.final_queue_length = self.ingress.qsize()
        return self.counters

    def _consumer(self, stop_event: threading.Event) -> None:
        while True:
            try:
                ev = self.ingress.get(timeout=0.01)
            except queue.Empty:
                if stop_event.is_set() and self.ingress.empty():
                    return
                continue
            with self._lock:
                self.counters.c1_dispatched += 1
                self.counters.c2_admitted += 1
            self.sink.submit(ev)
            with self._lock:
                self.counters.completed += 1


def generate_paired_traces(lam, duration_s, seed, heavy_frac,
                           nm_options_light, nm_options_heavy):
    """Generate two identical traces for paired comparison across paths.

    Both traces have the same arrival times, job classes, and NM assignments.
    Each path receives the same exogenous workload; only the recovery
    mechanism differs.

    Args:
        lam: arrival rate (requests/s)
        duration_s: trace duration
        seed: random seed (same seed = same trace)
        heavy_frac: fraction of heavy requests
        nm_options_light: NM values for light requests
        nm_options_heavy: NM values for heavy requests

    Returns: (trace_a, trace_b) — identical Trace objects
    """
    # Generate once, return two copies
    trace = generate_poisson_trace(
        lam=lam, duration_s=duration_s, seed=seed,
        classes=["light", "heavy"], heavy_frac=heavy_frac,
        nm_options_light=nm_options_light,
        nm_options_heavy=nm_options_heavy
    )
    # Deep copy events for the second trace
    import copy
    trace_a = trace
    trace_b = copy.deepcopy(trace)
    return trace_a, trace_b
