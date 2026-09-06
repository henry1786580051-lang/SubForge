"""Bounded, task-local reuse of successful, identical fidelity checks."""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import Future
from typing import Callable


class PositiveVerdictCache:
    """Cache only successful checks, coalescing concurrent identical requests.

    Failures and invalid responses remain retryable. Clearing starts a generation
    so an older in-flight check cannot populate the next document's cache.
    """

    def __init__(self, capacity: int = 256):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._lock = threading.Lock()
        self._generation = 0
        self._passed: OrderedDict[str, None] = OrderedDict()
        self._pending: dict[tuple[int, str], Future[None]] = {}
        self.hits = 0
        self.shared_requests = 0

    def clear(self) -> None:
        with self._lock:
            self._generation += 1
            self._passed.clear()
            self.hits = self.shared_requests = 0

    def validate(self, key: str, check: Callable[[], None]) -> None:
        with self._lock:
            if key in self._passed:
                self._passed.move_to_end(key)
                self.hits += 1
                return
            identity = (self._generation, key)
            pending = self._pending.get(identity)
            owner = pending is None
            if pending is None:
                pending = Future()
                self._pending[identity] = pending
            else:
                self.shared_requests += 1
        if not owner:
            pending.result()
            return
        try:
            check()
        except BaseException as error:
            pending.set_exception(error)
            raise
        else:
            with self._lock:
                if identity[0] == self._generation:
                    self._passed[key] = None
                    while len(self._passed) > self.capacity:
                        self._passed.popitem(last=False)
            pending.set_result(None)
        finally:
            with self._lock:
                self._pending.pop(identity, None)
