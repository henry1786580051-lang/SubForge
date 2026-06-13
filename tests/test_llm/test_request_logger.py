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
