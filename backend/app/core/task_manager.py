import asyncio
import uuid
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel


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


class TaskManager:
    """Manages async tasks with WebSocket progress broadcasting."""

    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}
        self._listeners: list[Callable] = []

    def create_task(self, task_type: str) -> TaskInfo:
        self.cleanup_old_tasks()
        task_id = str(uuid.uuid4())[:12]
        task = TaskInfo(id=task_id, type=task_type)
        self._tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> TaskInfo | None:
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[TaskInfo]:
        return list(self._tasks.values())

    def update_progress(self, task_id: str, progress: int, message: str = ""):
        if task := self._tasks.get(task_id):
            task.progress = progress
            task.message = message
            task.status = TaskStatus.RUNNING
            self._notify_listeners(task_id)

    def complete_task(self, task_id: str, result: dict[str, Any] | None = None):
        if task := self._tasks.get(task_id):
            task.status = TaskStatus.COMPLETED
            task.progress = 100
            task.result = result
            self._notify_listeners(task_id)

    def fail_task(self, task_id: str, error: str):
        if task := self._tasks.get(task_id):
            task.status = TaskStatus.FAILED
            task.error = error
            self._notify_listeners(task_id)

    def cancel_task(self, task_id: str):
        if task := self._tasks.get(task_id):
            task.status = TaskStatus.CANCELLED
            self._notify_listeners(task_id)

    def cleanup_old_tasks(self, keep: int = 50):
        """Remove old completed/failed/cancelled tasks, keeping the most recent ones."""
        terminal = [t for t in self._tasks.values() if t.status in (
            TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED
        )]
        if len(terminal) <= keep:
            return
        # Sort by ID (roughly chronological) and remove oldest
        terminal.sort(key=lambda t: t.id)
        to_remove = terminal[:len(terminal) - keep]
        for t in to_remove:
            self._tasks.pop(t.id, None)

    def add_listener(self, callback: Callable):
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable):
        self._listeners = [l for l in self._listeners if l != callback]

    def _notify_listeners(self, task_id: str):
        task = self._tasks.get(task_id)
        if task:
            for listener in self._listeners:
                try:
                    listener(task_id, task.model_dump())
                except Exception:
                    pass


# Global singleton
task_manager = TaskManager()
