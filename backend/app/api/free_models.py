"""Free hosted-model availability scanner endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from app.api.config import get_llm_provider_runtime_config, get_llm_provider_status
from app.core.task_manager import TaskResourceBusyError, task_manager
from app.services.free_model_scanner import FreeModelScanError, scan_nvidia_models
from app.services.task_runtime import schedule_background_task

router = APIRouter()

_background_tasks: set[asyncio.Task] = set()
_active_task_id: str | None = None
_last_scan: dict | None = None


def _current_task_id() -> str | None:
    global _active_task_id
    if not _active_task_id:
        return None
    task = task_manager.get_task(_active_task_id)
    if task is None or task.status.value in {"completed", "failed", "cancelled"}:
        _active_task_id = None
    return _active_task_id


@router.get("/nvidia")
async def nvidia_scan_status():
    profile = get_llm_provider_status("nvidia")
    return {
        **profile,
        "active_task_id": _current_task_id(),
        "last_scan": _last_scan,
    }


@router.post("/nvidia/scan")
async def start_nvidia_scan():
    global _active_task_id, _last_scan
    try:
        runtime = get_llm_provider_runtime_config("nvidia")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not runtime.api_key:
        raise HTTPException(status_code=400, detail="请先在设置中保存 NVIDIA API Key")

    async def runner(task_id: str) -> None:
        global _active_task_id, _last_scan
        try:
            task_manager.update_progress(task_id, 1, "正在读取 NVIDIA 模型目录")

            def report(done: int, total: int, available: int, busy: int) -> None:
                progress = 5 + round((done / max(1, total)) * 94)
                task_manager.update_progress(
                    task_id,
                    progress,
                    f"已测试 {done}/{total} · 可用 {available} · 繁忙 {busy}",
                )

            result = await scan_nvidia_models(
                api_key=runtime.api_key,
                base_url=runtime.base_url,
                progress=report,
            )
            _last_scan = result
            task_manager.complete_task(task_id, result)
        except asyncio.CancelledError:
            raise
        except FreeModelScanError as exc:
            task_manager.fail_task(task_id, str(exc))
        except Exception as exc:
            task_manager.fail_task(task_id, f"模型扫描失败：{str(exc)[:200]}")
        finally:
            if _active_task_id == task_id:
                _active_task_id = None

    try:
        task_id = schedule_background_task(
            task_type="free-model-scan",
            resource_key="free-model-scan:nvidia",
            runner=runner,
            background_tasks=_background_tasks,
        )
    except TaskResourceBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail="NVIDIA 模型扫描正在进行",
        ) from exc
    _active_task_id = task_id
    return {"task_id": task_id, "status": "started"}
