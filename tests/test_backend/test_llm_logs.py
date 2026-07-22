import asyncio
import json

from app.api import llm_logs


def test_llm_logs_group_requests_by_task(tmp_path, monkeypatch):
    path = tmp_path / "llm_requests.jsonl"
    entries = [
        {
            "timestamp": "2026-06-13T10:00:00+00:00",
            "task_id": "task-1",
            "file_name": "video.srt",
            "stage": "translate",
            "model": "mimo-v2.5-pro",
            "duration_ms": 1000,
            "tokens": 120,
            "cache_creation_tokens": 64,
        },
        {
            "timestamp": "2026-06-13T10:00:02+00:00",
            "task_id": "task-1",
            "file_name": "video.srt",
            "stage": "translate",
            "model": "mimo-v2.5-pro",
            "duration_ms": 2000,
            "tokens": 180,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(entry) for entry in entries),
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_logs, "_find_log_path", lambda: path)

    result = asyncio.run(llm_logs.get_llm_logs(page=1, page_size=50, search=""))

    assert result["total"] == 1
    group = result["groups"][0]
    assert group["request_count"] == 2
    assert group["duration_ms"] == 3000
    assert group["tokens"] == 300
    assert group["cache_creation_tokens"] == 64
    assert group["entries"] == []

    detail = asyncio.run(llm_logs.get_llm_log_detail("task-1"))
    assert detail["request_count"] == 2
    assert len(detail["entries"]) == 2


def test_llm_logs_skip_malformed_status_and_json(tmp_path, monkeypatch):
    path = tmp_path / "llm_requests.jsonl"
    path.write_text(
        "not json\n"
        + json.dumps({"task_id": "task-2", "timestamp": "2026-01-01", "status": "ok"})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_logs, "_find_log_path", lambda: path)

    result = asyncio.run(llm_logs.get_llm_logs(page=1, page_size=50, search=""))

    assert result["total"] == 1
    assert result["groups"][0]["error_count"] == 0
