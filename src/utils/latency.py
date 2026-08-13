import time
from contextlib import contextmanager

class LatencyTracker:
    def __init__(self):
        self._timings: dict[str, float] = {}
        self._start = time.perf_counter()

    @contextmanager
    def measure(self, name: str):
        t0 = time.perf_counter()
        yield
        self._timings[name] = (time.perf_counter() - t0) * 1000

    def all(self) -> dict[str, float]:
        total = (time.perf_counter() - self._start) * 1000
        return {k: round(v) for k, v in {**self._timings, "total": total}.items()}
