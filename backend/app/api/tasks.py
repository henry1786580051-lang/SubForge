from fastapi import APIRouter, HTTPException

from app.core.task_manager import task_manager
from app.schemas.tasks import TaskResponse

router = APIRouter()


@router.get("/", response_model=list[TaskResponse])
async def list_tasks():
    task_manager.cleanup_old_tasks(keep=50)
    return [t.model_dump() for t in task_manager.get_all_tasks()]


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.model_dump()


@router.post("/{task_id}/cancel")
async def cancel_task(task_id: str):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if not task_manager.cancel_task(task_id):
        raise HTTPException(status_code=409, detail=f"Task is already {task.status.value}")
    return {"status": "cancelled"}
