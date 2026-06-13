import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.task_manager import task_manager

router = APIRouter()
logger = logging.getLogger(__name__)

_loop: asyncio.AbstractEventLoop | None = None


def set_event_loop(loop: asyncio.AbstractEventLoop):
    """Set the event loop reference for cross-thread coroutine scheduling."""
    global _loop
    _loop = loop


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass

    async def broadcast(self, message: dict):
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for conn in dead:
            self.disconnect(conn)


manager = ConnectionManager()


def on_task_update(task_id: str, task_data: dict):
    """Called by TaskManager when a task updates."""
    if _loop and _loop.is_running():
        try:
            future = asyncio.run_coroutine_threadsafe(
                manager.broadcast({"type": "task_update", "data": task_data}),
                _loop,
            )
            future.add_done_callback(_log_broadcast_failure)
        except RuntimeError:
            logger.debug("Task update dropped because the event loop is closing")


def _log_broadcast_failure(future) -> None:
    try:
        future.result()
    except Exception:
        logger.exception("WebSocket task update broadcast failed")


task_manager.add_listener(on_task_update)


@router.websocket("/tasks")
async def task_websocket(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle client messages (e.g., subscribe to specific task)
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "error": "Invalid JSON message"})
                continue
            if msg.get("type") == "subscribe":
                task_id = msg.get("task_id")
                task = task_manager.get_task(task_id)
                if task:
                    await websocket.send_json(
                        {
                            "type": "task_update",
                            "data": task.model_dump(),
                        }
                    )
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        logger.exception("WebSocket task connection failed")
        manager.disconnect(websocket)
