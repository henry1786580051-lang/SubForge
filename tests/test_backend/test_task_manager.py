import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.core.task_manager import TaskManager, TaskResourceBusyError, TaskStatus


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
    assert updated.preview_revision == 1


def test_attention_request_can_be_resolved_without_ending_task():
    manager = TaskManager()
    task = manager.create_task("transcribe")
    attention = {
        "type": "missing_alignment_models",
        "source_mode": "auto",
        "message": "Waiting for alignment model",
        "models": [{"language": "es"}],
    }

    assert manager.request_attention(task.id, attention) is True
    waiting = manager.get_task(task.id)
    assert waiting is not None
    assert waiting.status == TaskStatus.RUNNING
    assert waiting.attention == attention

    assert manager.resolve_attention(task.id, "continue") is True
    assert manager.wait_for_attention_resolution(task.id, timeout=0) == "continue"
    resumed = manager.get_task(task.id)
    assert resumed is not None
    assert resumed.status == TaskStatus.RUNNING
    assert resumed.attention is None


def test_expired_attention_cannot_be_resolved():
    manager = TaskManager()
    task = manager.create_task("transcribe")
    assert manager.request_attention(
        task.id,
        {"type": "missing_alignment_models", "models": []},
    )

    assert manager.cancel_task(task.id) is True
    assert manager.resolve_attention(task.id, "retry") is False


def test_preview_updates_emit_small_deltas_and_keep_full_snapshot():
    manager = TaskManager()
    task = manager.create_task("subtitle")
    events = []
    manager.add_listener(lambda _task_id, data: events.append(data))
    first = [
        {"id": 1, "text": "one"},
        {"id": 2, "text": "two"},
    ]
    manager.publish_preview(task.id, first)
    manager.publish_preview(task.id, [*first, {"id": 3, "text": "three"}])
    manager.publish_preview(
        task.id,
        [first[0], {"id": 2, "text": "translated"}, {"id": 3, "text": "three"}],
    )

    assert events[0]["preview_delta"] == {
        "mode": "replace",
        "segments": first,
        "total": 2,
    }
    assert events[1]["preview_delta"]["mode"] == "append"
    assert events[1]["preview_delta"]["segments"] == [{"id": 3, "text": "three"}]
    assert events[2]["preview_delta"]["mode"] == "patch"
    assert events[2]["preview_delta"]["segments"] == [{"id": 2, "text": "translated"}]
    assert all("preview_segments" not in event for event in events)
    snapshot = manager.get_task(task.id)
    assert snapshot is not None
    assert len(snapshot.preview_segments or []) == 3
    assert snapshot.preview_revision == 3


def test_preview_patch_only_contains_changed_fields_even_for_large_updates():
    manager = TaskManager()
    task = manager.create_task("subtitle")
    events = []
    manager.add_listener(lambda _task_id, data: events.append(data))
    initial = [
        {
            "id": index,
            "start": f"00:00:{index:02d},000",
            "end": f"00:00:{index:02d},900",
            "text": f"source {index}",
            "translated": f"译文，{index}。",
            "speaker": "",
        }
        for index in range(1, 41)
    ]
    finalized = [
        {**segment, "translated": segment["translated"].replace("，", " ").replace("。", "")}
        for segment in initial
    ]

    manager.publish_preview(task.id, initial)
    manager.publish_preview(task.id, finalized)

    delta = events[-1]["preview_delta"]
    assert delta["mode"] == "patch"
    assert len(delta["segments"]) == len(initial)
    assert delta["segments"][0] == {"id": 1, "translated": "译文 1"}
    assert all(set(segment) == {"id", "translated"} for segment in delta["segments"])


def test_completed_websocket_event_omits_redundant_preview_but_snapshot_keeps_it():
    manager = TaskManager()
    task = manager.create_task("subtitle")
    events = []
    manager.add_listener(lambda _task_id, data: events.append(data))
    segments = [{"id": 1, "text": "source", "translated": "译文"}]
    manager.publish_preview(task.id, segments)
    revision = manager.get_task(task.id).preview_revision

    manager.complete_task(
        task.id,
        {
            "subtitle_file": "/tmp/result.srt",
            "preview_revision": revision,
            "segments": segments,
        },
    )

    assert events[-1]["result"] == {
        "subtitle_file": "/tmp/result.srt",
        "preview_revision": revision,
    }
    snapshot = manager.get_task(task.id)
    assert snapshot is not None
    assert snapshot.result["segments"] == segments
    assert snapshot.preview_segments is None
    assert manager.get_preview_revision(task.id) == revision


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


def test_terminal_task_cannot_be_cancelled_or_overwritten():
    manager = TaskManager()
    task = manager.create_task("subtitle")
    manager.complete_task(task.id, {"subtitle_file": "final.srt"})

    assert manager.cancel_task(task.id) is False
    manager.fail_task(task.id, "late failure")

    completed = manager.get_task(task.id)
    assert completed is not None
    assert completed.status == TaskStatus.COMPLETED
    assert completed.result == {"subtitle_file": "final.srt"}
    assert completed.error is None


@pytest.mark.parametrize("terminal", ["complete", "fail", "cancel"])
def test_resource_lock_blocks_duplicate_task_and_releases_at_terminal_state(terminal):
    manager = TaskManager()
    first = manager.create_task("transcribe", resource_key="transcribe:/video.mp4")

    with pytest.raises(TaskResourceBusyError):
        manager.create_task("transcribe", resource_key="transcribe:/video.mp4")

    if terminal == "complete":
        manager.complete_task(first.id)
    elif terminal == "fail":
        manager.fail_task(first.id, "failed")
    else:
        assert manager.cancel_task(first.id) is True

    replacement = manager.create_task("transcribe", resource_key="transcribe:/video.mp4")
    assert replacement.id != first.id
