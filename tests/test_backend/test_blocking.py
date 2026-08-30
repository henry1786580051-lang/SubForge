import asyncio
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.core.blocking import run_blocking


def test_cancelled_blocking_work_finishes_before_caller_cleanup():
    started = threading.Event()
    release = threading.Event()
    worker_finished = threading.Event()

    def worker():
        started.set()
        release.wait(timeout=2)
        worker_finished.set()

    async def run():
        task = asyncio.create_task(run_blocking(worker))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert worker_finished.is_set()

    asyncio.run(run())


def test_repeated_cancellation_cannot_release_resources_before_worker_finishes():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    cleaned = threading.Event()
    hooks = []

    def worker():
        started.set()
        release.wait(2)
        finished.set()

    async def caller():
        try:
            await run_blocking(worker, on_cancel=lambda: hooks.append(True))
        finally:
            cleaned.set()

    async def run():
        task = asyncio.create_task(caller())
        try:
            assert await asyncio.to_thread(started.wait, 1)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not cleaned.is_set()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert finished.is_set() and cleaned.is_set()
        assert hooks == [True]

    asyncio.run(run())
