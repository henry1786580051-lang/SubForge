import asyncio
import json
import logging
from urllib.parse import urlparse

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

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
        self.active_connections: dict[WebSocket, set[str]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = set()

    def disconnect(self, websocket: WebSocket):
        try:
            del self.active_connections[websocket]
        except KeyError:
            pass

    def subscribe(self, websocket: WebSocket, task_id: str) -> None:
        subscriptions = self.active_connections.get(websocket)
        if subscriptions is not None:
            subscriptions.add(task_id)

    async def broadcast(self, task_id: str, message: dict):
        dead = []
        for connection, subscriptions in list(self.active_connections.items()):
            if task_id not in subscriptions:
                continue
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for conn in dead:
            self.disconnect(conn)


manager = ConnectionManager()


def _is_allowed_origin(origin: str | None) -> bool:
    """Allow native clients and frontend pages served from loopback only."""
    if not origin:
        return True
    parsed = urlparse(origin)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }


def on_task_update(task_id: str, task_data: dict):
    """Called by TaskManager when a task updates."""
    if _loop and _loop.is_running():
        try:
            future = asyncio.run_coroutine_threadsafe(
                manager.broadcast(task_id, {"type": "task_update", "data": task_data}),
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
    if not _is_allowed_origin(websocket.headers.get("origin")):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
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
                if not isinstance(task_id, str):
                    await websocket.send_json({"type": "error", "error": "Invalid task ID"})
                    continue
                task = task_manager.get_task(task_id)
                if task:
                    manager.subscribe(websocket, task_id)
                    await websocket.send_json(
                        {
                            "type": "task_update",
                            "data": task.model_dump(),
                        }
                    )
                else:
                    await websocket.send_json({"type": "error", "error": "Task not found"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        logger.exception("WebSocket task connection failed")
        manager.disconnect(websocket)
