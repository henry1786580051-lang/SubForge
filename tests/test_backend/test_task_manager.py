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


def test_cleanup_keeps_cancelled_task_while_worker_is_draining():
    async def run():
        manager = TaskManager()
        task = manager.create_task("subtitle", resource_key="subtitle:/input.srt")

        async def worker():
            await asyncio.sleep(60)

        running = asyncio.create_task(worker())
        manager.register_running_task(task.id, running)
        assert manager.cancel_task(task.id) is True

        manager.cleanup_old_tasks(keep=0)

        assert manager.get_task(task.id) is not None
        with pytest.raises(TaskResourceBusyError):
            manager.create_task("subtitle", resource_key="subtitle:/input.srt")

        await asyncio.sleep(0)
        manager.unregister_running_task(task.id)

    asyncio.run(run())


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


def test_completed_websocket_event_carries_final_preview_snapshot():
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
    assert events[-1]["preview_segments"] == segments
    snapshot = manager.get_task(task.id)
    assert snapshot is not None
    assert snapshot.result["segments"] == segments
    assert snapshot.preview_segments == segments
    assert manager.get_preview_revision(task.id) == revision


def test_failed_websocket_event_carries_recovery_preview_snapshot():
    manager = TaskManager()
    task = manager.create_task("subtitle")
    events = []
    manager.add_listener(lambda _task_id, data: events.append(data))
    segments = [{"id": 1, "text": "source", "translated": "部分译文"}]
    manager.publish_preview(task.id, segments, subtitle_file="/tmp/recovery.srt")

    manager.fail_task(
        task.id,
        "translation failed",
        {"recovery_file": "/tmp/recovery.srt"},
    )

    assert events[-1]["status"] == TaskStatus.FAILED
    assert events[-1]["result"] == {"recovery_file": "/tmp/recovery.srt"}
    assert events[-1]["preview_segments"] == segments


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled"])
def test_terminal_results_are_owned_snapshots(terminal):
    manager = TaskManager()
    task = manager.create_task("subtitle")
    events = []
    manager.add_listener(lambda _id, data: events.append(data))
    segments = [{"id": 1, "text": "source", "translated": "saved"}]
    result = {"segments": segments, "warnings": ["original warning"], "preview_revision": 1}
    if terminal == "completed":
        manager.complete_task(task.id, result)
    elif terminal == "failed":
        manager.fail_task(task.id, "provider failed", result)
    else:
        manager.finalize_cancelled_task(task.id, result, preview_segments=segments)

    segments[0]["translated"] = "late mutation"
    result["warnings"].append("late warning")
    events[-1]["result"]["warnings"].append("listener mutation")
    snapshot = manager.get_task(task.id)
    assert snapshot.result["segments"][0]["translated"] == "saved"
    assert snapshot.result["warnings"] == ["original warning"]
    if terminal != "failed":
        assert snapshot.preview_segments[0]["translated"] == "saved"


@pytest.mark.parametrize("terminal", ["completed", "failed"])
def test_cancel_recovery_cannot_replace_a_different_terminal_state(terminal):
    manager = TaskManager()
    task = manager.create_task("subtitle")
    if terminal == "completed":
        manager.complete_task(task.id, {"subtitle_file": "final.srt"})
    else:
        manager.fail_task(task.id, "original failure", {"recovery_file": "original.srt"})
    expected = manager.get_task(task.id)
    manager.finalize_cancelled_task(
        task.id,
        {"recovery_file": "late.srt"},
        preview_segments=[{"id": 1, "translated": "late"}],
    )
    assert manager.get_task(task.id) == expected


def test_cancel_recovery_notifies_without_releasing_a_draining_worker():
    async def run():
        manager = TaskManager()
        task = manager.create_task("subtitle", resource_key="subtitle:/input.srt")
        events = []
        manager.add_listener(lambda _id, data: events.append(data))
        running = asyncio.create_task(asyncio.sleep(60))
        manager.register_running_task(task.id, running)
        manager.cancel_task(task.id)
        segments = [{"id": 1, "translated": "saved translation"}]
        manager.finalize_cancelled_task(
            task.id, {"recovery_file": "/tmp/recovery.srt"}, preview_segments=segments
        )
        try:
            assert events[-1]["status"] == TaskStatus.CANCELLED
            assert events[-1]["preview_segments"] == segments
            assert events[-1]["result"] == {"recovery_file": "/tmp/recovery.srt"}
            events[-1]["preview_segments"][0]["translated"] = "listener mutation"
            assert manager.get_task(task.id).preview_segments == segments
            with pytest.raises(TaskResourceBusyError):
                manager.create_task("subtitle", resource_key="subtitle:/input.srt")
            with pytest.raises(asyncio.CancelledError):
                await running
        finally:
            manager.unregister_running_task(task.id)
        replacement = manager.create_task("subtitle", resource_key="subtitle:/input.srt")
        assert replacement.id != task.id

    asyncio.run(run())


def test_preview_notification_revision_matches_its_payload_after_interleaving(monkeypatch):
    import threading

    manager = TaskManager()
    task = manager.create_task("subtitle")
    waiting = threading.Event()
    release = threading.Event()
    events = []
    notify = manager._notify_listeners

    def delayed(task_id, **kwargs):
        if threading.current_thread().name == "first-preview":
            waiting.set()
            assert release.wait(2)
        notify(task_id, **kwargs)

    monkeypatch.setattr(manager, "_notify_listeners", delayed)
    manager.add_listener(lambda _id, data: events.append(data))
    first = [{"id": 1, "translated": "old"}]
    latest = [{"id": 1, "translated": "new"}]
    worker = threading.Thread(target=manager.publish_preview, args=(task.id, first), name="first-preview")
    worker.start()
    try:
        assert waiting.wait(2)
        manager.publish_preview(task.id, latest)
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert len(events) == 2
    # Both events observe revision 2; neither may carry revision 1's old text.
    assert all(event["preview_revision"] == 2 for event in events)
    assert all(event["preview_delta"]["segments"] == latest for event in events)


def test_preview_delta_replaces_snapshot_when_a_field_is_removed():
    manager = TaskManager()
    assert manager._build_preview_delta(
        [{"id": 1, "text": "one", "speaker": "A"}],
        [{"id": 1, "text": "one"}],
    ) == {"mode": "replace", "segments": [{"id": 1, "text": "one"}], "total": 1}


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


def test_cancelled_running_task_keeps_resource_until_worker_unregisters():
    async def run():
        manager = TaskManager()
        first = manager.create_task("subtitle", resource_key="subtitle:/input.srt")

        async def worker():
            await asyncio.sleep(60)

        running = asyncio.create_task(worker())
        manager.register_running_task(first.id, running)
        assert manager.cancel_task(first.id) is True

        with pytest.raises(TaskResourceBusyError):
            manager.create_task("subtitle", resource_key="subtitle:/input.srt")

        await asyncio.sleep(0)
        manager.unregister_running_task(first.id)
        replacement = manager.create_task(
            "subtitle", resource_key="subtitle:/input.srt"
        )
        assert replacement.id != first.id

    asyncio.run(run())
