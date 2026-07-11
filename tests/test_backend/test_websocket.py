import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.api.websocket import ConnectionManager, _is_allowed_origin


class FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def accept(self):
        pass

    async def send_json(self, message):
        self.messages.append(message)


def test_websocket_origin_rejects_remote_web_pages():
    assert _is_allowed_origin("http://127.0.0.1:8000")
    assert _is_allowed_origin("http://localhost:3000")
    assert _is_allowed_origin(None)
    assert not _is_allowed_origin("https://attacker.example")
    assert not _is_allowed_origin("null")


def test_websocket_broadcasts_only_to_task_subscribers():
    async def run():
        manager = ConnectionManager()
        subscribed = FakeWebSocket()
        unrelated = FakeWebSocket()
        await manager.connect(subscribed)
        await manager.connect(unrelated)
        manager.subscribe(subscribed, "task-a")
        manager.subscribe(unrelated, "task-b")

        await manager.broadcast("task-a", {"type": "task_update"})

        assert subscribed.messages == [{"type": "task_update"}]
        assert unrelated.messages == []

    asyncio.run(run())
