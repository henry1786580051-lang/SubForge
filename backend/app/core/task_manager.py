import asyncio
import logging
import threading
import uuid
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TaskResourceBusyError(RuntimeError):
    """Raised when another active task owns the same processing resource."""


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskInfo(BaseModel):
    id: str
    type: str
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    subtitle_file: str | None = None  # partial results during processing
    preview_segments: list[dict[str, Any]] | None = None
    preview_revision: int = 0
    attention: dict[str, Any] | None = None


class TaskManager:
    """Manages async tasks with WebSocket progress broadcasting."""

    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}
        self._listeners: list[Callable] = []
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._cancel_callbacks: dict[str, list[Callable[[], Any]]] = {}
        self._resource_owners: dict[str, str] = {}
        self._task_resources: dict[str, str] = {}
        self._attention_conditions: dict[str, threading.Condition] = {}
        self._attention_resolutions: dict[str, str] = {}
        self._lock = threading.RLock()

    def create_task(self, task_type: str, resource_key: str | None = None) -> TaskInfo:
        self.cleanup_old_tasks()
        task_id = str(uuid.uuid4())[:12]
        task = TaskInfo(id=task_id, type=task_type)
        with self._lock:
            if resource_key:
                owner_id = self._resource_owners.get(resource_key)
                owner = self._tasks.get(owner_id) if owner_id else None
                if owner and owner.status in {TaskStatus.PENDING, TaskStatus.RUNNING}:
                    raise TaskResourceBusyError(
                        f"Another {owner.type} task is already processing this file"
                    )
                self._resource_owners[resource_key] = task_id
                self._task_resources[task_id] = resource_key
            self._tasks[task_id] = task
            return task.model_copy(deep=True)

    def _release_resource(self, task_id: str) -> None:
        resource_key = self._task_resources.pop(task_id, None)
        if resource_key and self._resource_owners.get(resource_key) == task_id:
            self._resource_owners.pop(resource_key, None)

    def get_task(self, task_id: str) -> TaskInfo | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.model_copy(deep=True) if task else None

    def get_all_tasks(self) -> list[TaskInfo]:
        with self._lock:
            return [task.model_copy(deep=True) for task in self._tasks.values()]

    def get_preview_revision(self, task_id: str) -> int:
        """Read preview revision without copying the full subtitle snapshot."""
        with self._lock:
            task = self._tasks.get(task_id)
            return task.preview_revision if task else 0

    def is_cancelled(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            return bool(task and task.status == TaskStatus.CANCELLED)

    def update_progress(
        self,
        task_id: str,
        progress: int,
        message: str = "",
        subtitle_file: str | None = None,
        preview_segments: list[dict[str, Any]] | None = None,
    ):
        preview_delta = None
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            }:
                return
            task.progress = max(task.progress, min(100, max(0, int(progress))))
            task.message = message
            task.status = TaskStatus.RUNNING
            if subtitle_file is not None:
                task.subtitle_file = subtitle_file
            if preview_segments is not None:
                preview_delta = self._build_preview_delta(
                    task.preview_segments or [], preview_segments
                )
                task.preview_segments = preview_segments
                task.preview_revision += 1
        self._notify_listeners(task_id, preview_delta=preview_delta)

    def publish_preview(
        self,
        task_id: str,
        preview_segments: list[dict[str, Any]],
        *,
        subtitle_file: str | None = None,
        message: str | None = None,
    ) -> None:
        """Publish a preview delta while retaining a full reconnect snapshot."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            }:
                return
            preview_delta = self._build_preview_delta(task.preview_segments or [], preview_segments)
            task.preview_segments = preview_segments
            task.preview_revision += 1
            task.status = TaskStatus.RUNNING
            if subtitle_file is not None:
                task.subtitle_file = subtitle_file
            if message is not None:
                task.message = message
        self._notify_listeners(task_id, preview_delta=preview_delta)

    @staticmethod
    def _build_preview_delta(
        previous: list[dict[str, Any]], current: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not previous:
            return {"mode": "replace", "segments": current, "total": len(current)}

        common_prefix = 0
        shared = min(len(previous), len(current))
        while common_prefix < shared and previous[common_prefix] == current[common_prefix]:
            common_prefix += 1
        if common_prefix == len(previous) and len(current) >= len(previous):
            return {
                "mode": "append",
                "segments": current[common_prefix:],
                "total": len(current),
            }

        if len(previous) == len(current):
            changed = []
            for old, segment in zip(previous, current):
                if old == segment:
                    continue
                if old.get("id") != segment.get("id"):
                    break
                patch = {"id": segment.get("id")}
                patch.update(
                    {
                        key: value
                        for key, value in segment.items()
                        if key != "id" and old.get(key) != value
                    }
                )
                changed.append(patch)
            else:
                return {"mode": "patch", "segments": changed, "total": len(current)}

        return {"mode": "replace", "segments": current, "total": len(current)}

    def complete_task(self, task_id: str, result: dict[str, Any] | None = None):
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            }:
                return
            task.status = TaskStatus.COMPLETED
            task.progress = 100
            task.result = result
            task.preview_segments = None
            task.attention = None
            self._running_tasks.pop(task_id, None)
            self._cancel_callbacks.pop(task_id, None)
            self._attention_conditions.pop(task_id, None)
            self._attention_resolutions.pop(task_id, None)
            self._release_resource(task_id)
        self._notify_listeners(task_id)

    def fail_task(
        self,
        task_id: str,
        error: str,
        result: dict[str, Any] | None = None,
    ):
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            }:
                return
            task.status = TaskStatus.FAILED
            task.error = error
            task.result = result
            task.attention = None
            self._running_tasks.pop(task_id, None)
            self._cancel_callbacks.pop(task_id, None)
            self._attention_conditions.pop(task_id, None)
            self._attention_resolutions.pop(task_id, None)
            self._release_resource(task_id)
        self._notify_listeners(task_id)

    def register_running_task(self, task_id: str, async_task: asyncio.Task):
        """Register an asyncio task for a given task_id so it can be cancelled."""
        with self._lock:
            self._running_tasks[task_id] = async_task

    def unregister_running_task(self, task_id: str):
        """Forget a finished asyncio task without changing user-visible state."""
        with self._lock:
            self._running_tasks.pop(task_id, None)

    def register_cancel_callback(self, task_id: str, callback: Callable[[], Any]) -> None:
        with self._lock:
            if task_id in self._tasks:
                self._cancel_callbacks.setdefault(task_id, []).append(callback)

    def request_attention(self, task_id: str, attention: dict[str, Any]) -> bool:
        """Publish a recoverable user decision without terminating the task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            }:
                return False
            task.attention = attention
            task.message = str(attention.get("message") or "Waiting for user action")
            task.status = TaskStatus.RUNNING
            self._attention_resolutions.pop(task_id, None)
            self._attention_conditions.setdefault(task_id, threading.Condition())
        self._notify_listeners(task_id)
        return True

    def resolve_attention(self, task_id: str, resolution: str) -> bool:
        """Resolve the current attention request and wake its worker thread."""
        with self._lock:
            task = self._tasks.get(task_id)
            condition = self._attention_conditions.get(task_id)
            if not task or not task.attention or condition is None:
                return False
            if task.status not in {TaskStatus.PENDING, TaskStatus.RUNNING}:
                return False
            self._attention_resolutions[task_id] = resolution
            task.attention = None
            task.message = "Resuming transcription..."
        with condition:
            condition.notify_all()
        self._notify_listeners(task_id)
        return True

    def wait_for_attention_resolution(self, task_id: str, timeout: float = 0.5) -> str | None:
        """Wait briefly for a decision while allowing cancellation checks."""
        with self._lock:
            resolution = self._attention_resolutions.pop(task_id, None)
            condition = self._attention_conditions.setdefault(task_id, threading.Condition())
        if resolution is not None:
            return resolution
        with condition:
            condition.wait(timeout=max(0.0, timeout))
        with self._lock:
            return self._attention_resolutions.pop(task_id, None)

    def unregister_cancel_callback(self, task_id: str, callback: Callable[[], Any]) -> None:
        with self._lock:
            callbacks = self._cancel_callbacks.get(task_id)
            if not callbacks:
                return
            self._cancel_callbacks[task_id] = [item for item in callbacks if item != callback]
            if not self._cancel_callbacks[task_id]:
                self._cancel_callbacks.pop(task_id, None)

    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            if task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            }:
                return False
            task.status = TaskStatus.CANCELLED
            task.attention = None
            async_task = self._running_tasks.pop(task_id, None)
            callbacks = self._cancel_callbacks.pop(task_id, [])
            condition = self._attention_conditions.pop(task_id, None)
            self._attention_resolutions.pop(task_id, None)
            self._release_resource(task_id)
        if condition is not None:
            with condition:
                condition.notify_all()
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.exception("Task cancellation callback failed for task %s", task_id)
        if async_task:
            async_task.cancel()
        self._notify_listeners(task_id)
        return True

    def cleanup_old_tasks(self, keep: int = 50):
        """Remove old completed/failed/cancelled tasks, keeping the most recent ones."""
        with self._lock:
            terminal = [
                t
                for t in self._tasks.values()
                if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            ]
            if len(terminal) <= keep:
                return
            # Dict insertion order is chronological. UUID4 text is random and
            # must not be used to decide which tasks are oldest.
            to_remove = terminal[: len(terminal) - keep]
            for t in to_remove:
                self._tasks.pop(t.id, None)
                self._running_tasks.pop(t.id, None)
                self._cancel_callbacks.pop(t.id, None)
                self._attention_conditions.pop(t.id, None)
                self._attention_resolutions.pop(t.id, None)
                self._release_resource(t.id)

    def add_listener(self, callback: Callable):
        with self._lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable):
        with self._lock:
            self._listeners = [listener for listener in self._listeners if listener != callback]

    def _notify_listeners(self, task_id: str, *, preview_delta: dict[str, Any] | None = None):
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            result = task.result
            if (
                task.status == TaskStatus.COMPLETED
                and isinstance(result, dict)
                and "preview_revision" in result
                and "segments" in result
            ):
                task_data = task.model_dump(exclude={"preview_segments", "result"})
                task_data["result"] = {
                    key: value for key, value in result.items() if key != "segments"
                }
            else:
                task_data = task.model_dump(exclude={"preview_segments"})
            if preview_delta is not None:
                task_data["preview_delta"] = preview_delta
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(task_id, task_data)
            except Exception:
                logger.exception("Task listener failed for task %s", task_id)


# Global singleton
task_manager = TaskManager()
