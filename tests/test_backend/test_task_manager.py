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


def test_cleanup_removes_oldest_terminal_tasks_by_creation_order(monkeypatch):
    manager = TaskManager()
    generated_ids = iter(["ffff-old", "0000-middle", "1111-new"])
    monkeypatch.setattr(
        "app.core.task_manager.uuid.uuid4",
        lambda: next(generated_ids),
    )

    tasks = [manager.create_task("transcribe") for _ in range(3)]
    for task in tasks:
        manager.complete_task(task.id)

    manager.cleanup_old_tasks(keep=2)

    assert manager.get_task("ffff-old") is None
    assert manager.get_task("0000-middle") is not None
    assert manager.get_task("1111-new") is not None


def test_task_progress_carries_preview_segments():
    manager = TaskManager()
    task = manager.create_task("transcribe")
    preview = [
        {
            "id": 1,
            "start": "00:00:00,000",
            "end": "00:00:01,000",
            "text": "Hi",
            "translated": "",
        }
    ]

    manager.update_progress(task.id, 40, "Transcribing", preview_segments=preview)

    updated = manager.get_task(task.id)
    assert updated is not None
    assert updated.preview_segments == preview


def test_progress_is_clamped_and_does_not_move_backwards():
    manager = TaskManager()
    task = manager.create_task("subtitle")

    manager.update_progress(task.id, 70, "Translating")
    manager.update_progress(task.id, 63, "Preparing context")
    assert manager.get_task(task.id).progress == 70

    manager.update_progress(task.id, 500, "Finishing")
    assert manager.get_task(task.id).progress == 100


def test_cancel_task_runs_registered_cleanup_callback():
    manager = TaskManager()
    task = manager.create_task("subtitle")
    calls = []
    manager.register_cancel_callback(task.id, lambda: calls.append(task.id))

    assert manager.cancel_task(task.id) is True
    assert calls == [task.id]
