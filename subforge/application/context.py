"""Interface-neutral task context for long-running application pipelines."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

ProgressCallback = Callable[[int, str, str | None], None]
PreviewCallback = Callable[[list[dict[str, Any]], str | None, str | None], None]
CancelCheck = Callable[[], bool]
CancelRegistration = Callable[[Callable[[], Any]], None]
AttentionRequest = Callable[[dict[str, Any]], bool]
AttentionWait = Callable[[float], str | None]


def _noop_progress(_progress: int, _message: str, _subtitle_file: str | None) -> None:
    return None


def _noop_preview(
    _segments: list[dict[str, Any]],
    _subtitle_file: str | None,
    _message: str | None,
) -> None:
    return None


def _false() -> bool:
    return False


def _noop_callback(_callback: Callable[[], Any]) -> None:
    return None


def _reject_attention(_attention: dict[str, Any]) -> bool:
    return False


def _no_resolution(_timeout: float) -> str | None:
    return None


@dataclass(slots=True)
class PipelineContext:
    """Runtime callbacks supplied by an interface adapter.

    Core application services can report progress and cooperate with
    cancellation without importing FastAPI, Qt, or the global task manager.
    """

    task_id: str = ""
    _progress: ProgressCallback = _noop_progress
    _preview: PreviewCallback = _noop_preview
    _is_cancelled: CancelCheck = _false
    _register_cancel: CancelRegistration = _noop_callback
    _unregister_cancel: CancelRegistration = _noop_callback
    _request_attention: AttentionRequest = _reject_attention
    _wait_attention: AttentionWait = _no_resolution

    def report(
        self,
        progress: int,
        message: str = "",
        *,
        subtitle_file: str | None = None,
    ) -> None:
        self._progress(progress, message, subtitle_file)

    def publish_preview(
        self,
        segments: list[dict[str, Any]],
        *,
        subtitle_file: str | None = None,
        message: str | None = None,
    ) -> None:
        self._preview(segments, subtitle_file, message)

    def is_cancelled(self) -> bool:
        return self._is_cancelled()

    def checkpoint(self) -> None:
        if self.is_cancelled():
            raise asyncio.CancelledError()

    @contextmanager
    def cancellation_scope(self, callback: Callable[[], Any] | None) -> Iterator[None]:
        if not callable(callback):
            yield
            return
        self._register_cancel(callback)
        try:
            yield
        finally:
            self._unregister_cancel(callback)

    def request_attention(self, attention: dict[str, Any]) -> bool:
        return self._request_attention(attention)

    def wait_for_attention(self, timeout: float = 0.5) -> str | None:
        return self._wait_attention(timeout)
