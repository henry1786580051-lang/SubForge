import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.local_access import LocalRequestGuard
from app.main import app


@pytest.fixture
def subtitle(tmp_path):
    path = tmp_path / "original.srt"
    path.write_text("1\n00:00:00,000 --> 00:00:01,000\nOriginal\n")
    return path


def save(client, path, headers=None):
    return client.post(
        "/api/subtitles/save", headers=headers,
        json={"file_path": str(path), "segments": [
            {"id": 1, "start": "00:00:00.000", "end": "00:00:01.000", "text": "Edited"}
        ]},
    )


@pytest.mark.parametrize("host", [
    "untrusted.example:8000", "127.0.0.1.attacker.example", "localhost.attacker.example",
    "0.0.0.0:8000", "localhost:bad", "localhost:8000/path", "user@localhost:8000",
])
def test_foreign_or_malformed_host_cannot_write_subtitles(subtitle, host):
    before = subtitle.read_bytes()
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    response = save(client, subtitle, {"host": host, "origin": f"http://{host}"})
    assert response.status_code == 400
    assert subtitle.read_bytes() == before


@pytest.mark.parametrize("headers", [
    {"origin": "https://attacker.example"}, {"origin": "null"},
    {"origin": "http://localhost:9000"}, {"origin": "http://user@localhost:8000"},
    {"origin": "http://127.0.0.1:8000/path"}, {"sec-fetch-site": "cross-site"},
    {"origin": "http://127.0.0.1:8000", "sec-fetch-site": "cross-site"},
])
def test_foreign_browser_origin_cannot_write_subtitles(subtitle, headers):
    before = subtitle.read_bytes()
    response = save(TestClient(app, base_url="http://127.0.0.1:8000"), subtitle, headers)
    assert response.status_code == 403
    assert subtitle.read_bytes() == before


@pytest.mark.parametrize("base", ["http://127.0.0.1:8000", "http://localhost:49321", "http://[::1]:49321"])
def test_same_origin_desktop_and_originless_cli_can_save(subtitle, base):
    # This Starlette TestClient version cannot parse IPv6 transport URLs;
    # the explicit Host still exercises the exact IPv6 authority seen by ASGI.
    client = TestClient(app, base_url="http://127.0.0.1:49321" if "[" in base else base)
    host = base.removeprefix("http://")
    assert save(client, subtitle, {"host": host, "origin": base, "sec-fetch-site": "same-origin"}).status_code == 200
    assert "Edited" in subtitle.read_text(encoding="utf-8-sig")
    assert save(client, subtitle, {"host": host}).status_code == 200


def test_duplicate_host_or_origin_is_rejected(subtitle):
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    before = subtitle.read_bytes()
    for headers in [
        [("host", "127.0.0.1:8000"), ("host", "attacker.example")],
        [("origin", "http://127.0.0.1:8000"), ("origin", "https://attacker.example")],
    ]:
        assert save(client, subtitle, headers).status_code in {400, 403}
        assert subtitle.read_bytes() == before


def guarded_app(dev=False):
    local = FastAPI()
    local.add_middleware(LocalRequestGuard, allow_dev_frontend=dev)

    @local.post("/write")
    async def write():
        return {"ok": True}

    @local.websocket("/socket")
    async def socket(ws: WebSocket):
        await ws.accept()
        await ws.send_json({"ok": True})
        await ws.close()

    return local


def test_source_development_origin_is_allowed_only_when_enabled():
    headers = {"origin": "http://localhost:3000", "sec-fetch-site": "cross-site"}
    for dev, expected in [(False, 403), (True, 200)]:
        client = TestClient(guarded_app(dev), base_url="http://127.0.0.1:49222")
        assert client.post("/write", headers=headers).status_code == expected


@pytest.mark.parametrize("host,origin", [
    ("attacker.example", "http://attacker.example"),
    ("127.0.0.1:8000", "https://attacker.example"),
    ("127.0.0.1:8000", "http://localhost:9000"),
])
def test_websocket_rejects_foreign_host_and_origin(host, origin):
    client = TestClient(guarded_app(), base_url="http://127.0.0.1:8000")
    with pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect("/socket", headers={"host": host, "origin": origin}):
            pytest.fail("Foreign WebSocket must not connect")
    assert error.value.code == 1008


def test_same_origin_websocket_works_on_desktop_fallback_port():
    client = TestClient(guarded_app(), base_url="http://127.0.0.1:49222")
    with client.websocket_connect("ws://127.0.0.1:49222/socket", headers={"origin": "http://127.0.0.1:49222"}) as ws:
        assert ws.receive_json() == {"ok": True}
