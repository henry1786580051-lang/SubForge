import asyncio
import logging
import threading
import uuid
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


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


class TaskManager:
    """Manages async tasks with WebSocket progress broadcasting."""

    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}
        self._listeners: list[Callable] = []
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._lock = threading.RLock()

    def create_task(self, task_type: str) -> TaskInfo:
        self.cleanup_old_tasks()
        task_id = str(uuid.uuid4())[:12]
        task = TaskInfo(id=task_id, type=task_type)
        with self._lock:
            self._tasks[task_id] = task
            return task.model_copy(deep=True)

    def get_task(self, task_id: str) -> TaskInfo | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.model_copy(deep=True) if task else None

    def get_all_tasks(self) -> list[TaskInfo]:
        with self._lock:
            return [task.model_copy(deep=True) for task in self._tasks.values()]

    def is_cancelled(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            return bool(task and task.status == TaskStatus.CANCELLED)

    def update_progress(
        self, task_id: str, progress: int, message: str = "", subtitle_file: str | None = None
    ):
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            }:
                return
            task.progress = progress
            task.message = message
            task.status = TaskStatus.RUNNING
            if subtitle_file is not None:
                task.subtitle_file = subtitle_file
        self._notify_listeners(task_id)

    def complete_task(self, task_id: str, result: dict[str, Any] | None = None):
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status == TaskStatus.CANCELLED:
                return
            task.status = TaskStatus.COMPLETED
            task.progress = 100
            task.result = result
            self._running_tasks.pop(task_id, None)
        self._notify_listeners(task_id)

    def fail_task(self, task_id: str, error: str):
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status == TaskStatus.CANCELLED:
                return
            task.status = TaskStatus.FAILED
            task.error = error
            self._running_tasks.pop(task_id, None)
        self._notify_listeners(task_id)

    def register_running_task(self, task_id: str, async_task: asyncio.Task):
        """Register an asyncio task for a given task_id so it can be cancelled."""
        with self._lock:
            self._running_tasks[task_id] = async_task

    def unregister_running_task(self, task_id: str):
        """Forget a finished asyncio task without changing user-visible state."""
        with self._lock:
            self._running_tasks.pop(task_id, None)

    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            task.status = TaskStatus.CANCELLED
            async_task = self._running_tasks.pop(task_id, None)
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

    def add_listener(self, callback: Callable):
        with self._lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable):
        with self._lock:
            self._listeners = [listener for listener in self._listeners if listener != callback]

    def _notify_listeners(self, task_id: str):
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task_data = task.model_dump()
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(task_id, task_data)
            except Exception:
                logger.exception("Task listener failed for task %s", task_id)


# Global singleton
task_manager = TaskManager()
