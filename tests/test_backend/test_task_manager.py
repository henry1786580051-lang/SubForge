import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.core.task_manager import TaskManager, TaskStatus


def test_cancelled_task_is_not_overwritten_by_late_completion():
    manager = TaskManager()
    task = manager.create_task("transcribe")

    assert manager.cancel_task(task.id) is True
    assert manager.is_cancelled(task.id) is True
    manager.complete_task(task.id, {"subtitle_file": "/tmp/out.srt"})
    manager.update_progress(task.id, 50, "late progress")

    updated = manager.get_task(task.id)
    assert updated is not None
    assert updated.status == TaskStatus.CANCELLED
    assert updated.result is None
    assert updated.progress == 0


def test_cancel_task_cancels_and_unregisters_running_asyncio_task():
    async def run():
        manager = TaskManager()
        task = manager.create_task("download_model")

        async def never_finishes():
            await asyncio.sleep(60)

        running_task = asyncio.create_task(never_finishes())
        manager.register_running_task(task.id, running_task)

        assert manager.cancel_task(task.id) is True
        await asyncio.sleep(0)

        assert running_task.cancelled()
        manager.unregister_running_task(task.id)
        assert manager.get_task(task.id).status == TaskStatus.CANCELLED

    asyncio.run(run())


def test_cancel_unknown_task_returns_false():
    manager = TaskManager()

    assert manager.cancel_task("missing") is False
    assert manager.is_cancelled("missing") is False
