"""Small process-local caches for expensive, non-thread-safe ASR models."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator


class SingleEntryModelCache:
    """Keep one model alive and serialize access to its inference runtime."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._key: tuple[Any, ...] | None = None
        self._value: Any = None

    @contextmanager
    def acquire(
        self, key: tuple[Any, ...], loader: Callable[[], Any]
    ) -> Iterator[Any]:
        with self._lock:
            if self._key != key or self._value is None:
                self._value = loader()
                self._key = key
            yield self._value

    def clear(self) -> None:
        with self._lock:
            self._key = None
            self._value = None

    @property
    def key(self) -> tuple[Any, ...] | None:
        with self._lock:
            return self._key
