"""FastAPI adapter for interface-neutral application task contexts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from app.core.task_manager import task_manager
from subforge.application import PipelineContext


def create_pipeline_context(task_id: str) -> PipelineContext:
    return PipelineContext(
        task_id=task_id,
        _progress=lambda progress, message, subtitle_file: task_manager.update_progress(
            task_id,
            progress,
            message,
            subtitle_file=subtitle_file,
        ),
        _preview=lambda segments, subtitle_file, message: task_manager.publish_preview(
            task_id,
            segments,
            subtitle_file=subtitle_file,
            message=message,
        ),
        _is_cancelled=lambda: task_manager.is_cancelled(task_id),
        _register_cancel=lambda callback: task_manager.register_cancel_callback(
            task_id, callback
        ),
        _unregister_cancel=lambda callback: task_manager.unregister_cancel_callback(
            task_id, callback
        ),
        _request_attention=lambda attention: task_manager.request_attention(task_id, attention),
        _wait_attention=lambda timeout: task_manager.wait_for_attention_resolution(
            task_id, timeout=timeout
        ),
    )


def schedule_background_task(
    *,
    task_type: str,
    resource_key: str,
    runner: Callable[[str], Coroutine[Any, Any, None]],
    background_tasks: set[asyncio.Task],
) -> str:
    """Create, register, and retain a managed background task consistently."""
    task = task_manager.create_task(task_type, resource_key=resource_key)
    running = asyncio.create_task(runner(task.id))
    task_manager.register_running_task(task.id, running)
    background_tasks.add(running)
    running.add_done_callback(background_tasks.discard)
    running.add_done_callback(lambda _task: task_manager.unregister_running_task(task.id))
    return task.id
