import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

import subforge.core.llm.request_logger as request_logger


class _FakeResponse:
    def __init__(self, marker: str):
        self.marker = marker

    def model_dump(self):
        return {"marker": self.marker}


def test_concurrent_requests_keep_their_own_response(tmp_path, monkeypatch):
    log_file = tmp_path / "llm_requests.jsonl"
    monkeypatch.setattr(request_logger, "LLM_LOG_FILE", log_file)
    request_logger._pending_requests.clear()
    barrier = threading.Barrier(2)

    def run_request(marker: str, delay: float):
        request = httpx.Request(
            "POST",
            "https://example.test/v1/chat/completions",
            content=json.dumps({"marker": marker}),
        )
        request_logger._on_request(request)
        request_logger._on_response(httpx.Response(200, request=request))
        barrier.wait()
        time.sleep(delay)
        request_logger.log_llm_response(_FakeResponse(marker))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_request, "first", 0.05),
            executor.submit(run_request, "second", 0),
        ]
        for future in futures:
            future.result()

    entries = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 2
    assert request_logger._pending_requests == {}
    assert {(entry["request"]["marker"], entry["response"]["marker"]) for entry in entries} == {
        ("first", "first"),
        ("second", "second"),
    }


def test_log_extracts_context_model_stage_and_tokens(tmp_path, monkeypatch):
    log_file = tmp_path / "llm_requests.jsonl"
    monkeypatch.setattr(request_logger, "LLM_LOG_FILE", log_file)
    request_logger._pending_requests.clear()
    request = httpx.Request(
        "POST",
        "https://example.test/v1/chat/completions",
        content=json.dumps(
            {
                "model": "mimo-v2.5-pro",
                "messages": [
                    {
                        "role": "user",
                        "content": '{"current_subtitles":{"12":"hello"}}',
                    }
                ],
            }
        ),
    )
    request_logger._on_request(
        request,
        {"task_id": "task-1", "file_name": "video.srt"},
    )
    request_logger._on_response(httpx.Response(200, request=request))

    class _UsageResponse:
        def model_dump(self):
            return {
                "model": "mimo-v2.5-pro",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "completion_tokens_details": {"reasoning_tokens": 20},
                },
            }

    request_logger.log_llm_response(_UsageResponse())
    entry = json.loads(log_file.read_text(encoding="utf-8"))
    assert entry["task_id"] == "task-1"
    assert entry["file_name"] == "video.srt"
    assert entry["stage"] == "translate"
    assert entry["model"] == "mimo-v2.5-pro"
    assert entry["batch"] == "12-12"
    assert entry["tokens"] == 150
    assert entry["reasoning_tokens"] == 20
    assert entry["timestamp"]
