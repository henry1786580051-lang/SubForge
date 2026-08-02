import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

import subforge.core.llm.request_logger as request_logger


def setup_function():
    request_logger.set_llm_log_level("debug")


def teardown_function():
    request_logger.set_llm_log_level("summary")


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
                    "prompt_tokens_details": {"cached_tokens": 38},
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
    assert entry["cached_tokens"] == 38
    assert entry["cache_hit_rate"] == 0.38
    assert entry["reasoning_tokens"] == 20
    assert entry["timestamp"]


def test_context_prompt_is_not_misclassified_as_translation():
    assert (
        request_logger._infer_stage(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Prepare global context for professional subtitle translation. "
                            "Return summary, terminology, and style."
                        ),
                    }
                ]
            }
        )
        == "context"
    )


def test_retry_releases_superseded_pending_request():
    request_logger._pending_requests.clear()
    request_logger._current_request_key.set(None)
    first = httpx.Request(
        "POST",
        "https://example.test/v1/chat/completions",
        content=json.dumps({"model": "test", "messages": []}),
    )
    retry = httpx.Request(
        "POST",
        "https://example.test/v1/chat/completions",
        content=json.dumps({"model": "test", "messages": []}),
    )

    request_logger._on_request(first)
    request_logger._on_response(httpx.Response(429, request=first))
    request_logger._on_request(retry)

    assert id(first) not in request_logger._pending_requests
    assert id(retry) in request_logger._pending_requests
    request_logger._pending_requests.clear()
    request_logger._current_request_key.set(None)


def test_log_extracts_anthropic_cache_usage(tmp_path, monkeypatch):
    log_file = tmp_path / "llm_requests.jsonl"
    monkeypatch.setattr(request_logger, "LLM_LOG_FILE", log_file)
    request_logger._pending_requests.clear()
    request = httpx.Request(
        "POST",
        "https://api.minimaxi.com/anthropic/v1/messages",
        content=json.dumps(
            {
                "model": "MiniMax-M3",
                "system": [{"type": "text", "text": "Translate subtitles"}],
                "messages": [
                    {"role": "user", "content": '{"current_subtitles":{"1":"hello"}}'}
                ],
            }
        ),
    )
    request_logger._on_request(request)
    request_logger._on_response(httpx.Response(200, request=request))

    class _AnthropicResponse:
        def model_dump(self):
            return {
                "model": "MiniMax-M3",
                "usage": {
                    "input_tokens": 500,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 1500,
                    "output_tokens": 200,
                },
            }

    request_logger.log_llm_response(_AnthropicResponse())
    entry = json.loads(log_file.read_text(encoding="utf-8"))
    assert entry["stage"] == "translate"
    assert entry["prompt_tokens"] == 2000
    assert entry["cached_tokens"] == 1500
    assert entry["cache_creation_tokens"] == 0
    assert entry["cache_hit_rate"] == 0.75
    assert entry["completion_tokens"] == 200
    assert entry["tokens"] == 2200
    request_logger._pending_requests.clear()
    request_logger._current_request_key.set(None)


def test_summary_log_omits_prompt_and_response_content(tmp_path, monkeypatch):
    log_file = tmp_path / "llm_requests.jsonl"
    monkeypatch.setattr(request_logger, "LLM_LOG_FILE", log_file)
    request_logger._pending_requests.clear()
    request_logger.set_llm_log_level("summary")
    request = httpx.Request(
        "POST",
        "https://example.test/v1/chat/completions",
        content=json.dumps(
            {
                "model": "MiniMax-M3",
                "messages": [{"role": "user", "content": "private subtitle content"}],
            }
        ),
    )
    request_logger._on_request(request)
    request_logger._on_response(httpx.Response(200, request=request))

    class _SummaryResponse:
        model = "MiniMax-M3"
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    request_logger.log_llm_response(_SummaryResponse())
    entry = json.loads(log_file.read_text(encoding="utf-8"))
    assert entry["log_level"] == "summary"
    assert entry["tokens"] == 15
    assert "request" not in entry
    assert "response" not in entry
    assert "private subtitle content" not in log_file.read_text(encoding="utf-8")
