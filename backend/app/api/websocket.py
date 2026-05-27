import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.task_manager import task_manager

router = APIRouter()


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
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


def on_task_update(task_id: str, task_data: dict):
    """Called by TaskManager when a task updates."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(
                manager.broadcast({"type": "task_update", "data": task_data})
            )
    except RuntimeError:
        pass


task_manager.add_listener(on_task_update)


@router.websocket("/tasks")
async def task_websocket(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle client messages (e.g., subscribe to specific task)
            msg = json.loads(data)
            if msg.get("type") == "subscribe":
                task_id = msg.get("task_id")
                task = task_manager.get_task(task_id)
                if task:
                    await websocket.send_json({
                        "type": "task_update",
                        "data": task.model_dump(),
                    })
    except (WebSocketDisconnect, Exception):
        manager.disconnect(websocket)
