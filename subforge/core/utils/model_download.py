"""Cancelable process isolation for model-library downloads."""

from __future__ import annotations

import multiprocessing
import threading
import time
from contextlib import nullcontext
from typing import Any, Callable

from subforge.core.utils.download_progress import DownloadEstimator
from subforge.core.utils.hub_progress import observe_hub_download


def _model_download_worker(
    operation: str,
    kwargs: dict[str, Any],
    result_queue: multiprocessing.Queue,
    shared=None,
) -> None:
    try:
        patterns = kwargs.get("allow_patterns")
        if operation == "faster_whisper":
            patterns = [
                "config.json",
                "preprocessor_config.json",
                "model.bin",
                "tokenizer.json",
                "vocabulary.*",
            ]
        observer = (
            observe_hub_download(shared, patterns, kwargs.get("ignore_patterns"))
            if shared is not None
            else nullcontext()
        )
        with observer:
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
    progress_callback: Callable[[dict], None] | None = None,
) -> str:
    """Run a third-party downloader in a process that can be terminated."""
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    shared = context.Array("d", [0, -1, 0]) if progress_callback else None
    estimator = DownloadEstimator()
    last_report = 0.0
    process = context.Process(
        target=_model_download_worker,
        args=(operation, kwargs, result_queue, shared),
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
            now = time.monotonic()
            if shared is not None and progress_callback and now - last_report >= 1:
                with shared.get_lock():
                    downloaded, total, transferred = shared[:]
                progress_callback(
                    estimator.update(int(downloaded), int(total), int(transferred), now)
                )
                last_report = now
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
