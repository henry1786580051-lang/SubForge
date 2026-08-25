"""Cancelable process isolation for model-library downloads."""

from __future__ import annotations

import multiprocessing
import threading
import time
from typing import Any


def _model_download_worker(
    operation: str,
    kwargs: dict[str, Any],
    result_queue: multiprocessing.Queue,
) -> None:
    try:
        if operation == "huggingface_snapshot":
            from huggingface_hub import snapshot_download

            result = snapshot_download(**kwargs)
        elif operation == "faster_whisper":
            from faster_whisper.utils import download_model

            result = download_model(**kwargs)
        else:
            raise ValueError(f"Unsupported model download operation: {operation}")
        result_queue.put({"ok": True, "result": str(result or "")})
    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def run_cancellable_model_download(
    operation: str,
    kwargs: dict[str, Any],
    cancel_event: threading.Event,
    *,
    poll_interval: float = 0.2,
) -> str:
    """Run a third-party downloader in a process that can be terminated."""
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_model_download_worker,
        args=(operation, kwargs, result_queue),
        name=f"subforge-{operation}",
    )
    process.start()
    try:
        while process.is_alive():
            if cancel_event.is_set():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                raise RuntimeError("Model download cancelled")
            time.sleep(max(0.05, poll_interval))
        process.join(timeout=1)
        try:
            outcome = result_queue.get(timeout=1)
        except Exception as exc:
            raise RuntimeError(
                f"Model download worker exited without a result (exit code {process.exitcode})"
            ) from exc
        if not outcome.get("ok"):
            raise RuntimeError(str(outcome.get("error") or "Model download failed"))
        return str(outcome.get("result") or "")
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()
