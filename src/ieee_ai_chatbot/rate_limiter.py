from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, max_requests: int = 30, window_seconds: int = 60) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._records: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._cleanup_counter = 0

    def check(self, key: str) -> tuple[bool, int]:
        now = time.time()
        window_start = now - self._window_seconds

        with self._lock:
            timestamps = self._records.get(key, [])
            timestamps = [t for t in timestamps if t > window_start]

            if len(timestamps) >= self._max_requests:
                remaining = 0
                wait = int(timestamps[0] + self._window_seconds - now) if timestamps else 0
                self._records[key] = timestamps
                return False, wait

            timestamps.append(now)
            self._records[key] = timestamps
            remaining = self._max_requests - len(timestamps)

            self._cleanup_counter += 1
            if self._cleanup_counter % 50 == 0:
                self._cleanup()

        return True, remaining

    def _cleanup(self) -> None:
        now = time.time()
        window_start = now - self._window_seconds
        with self._lock:
            stale_keys = [
                k for k, v in self._records.items()
                if all(t < window_start for t in v)
            ]
            for k in stale_keys:
                del self._records[k]
