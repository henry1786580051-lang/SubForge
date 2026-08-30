import asyncio
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.services.free_model_scanner import (  # noqa: E402
    FreeModelScanError,
    is_chat_model_candidate,
    scan_nvidia_models,
)


def test_chat_candidate_filter_skips_non_generating_models():
    assert is_chat_model_candidate("nvidia/nemotron-3-ultra-550b-a55b") is True
    assert is_chat_model_candidate("nvidia/nv-embedqa-e5-v5") is False
    assert is_chat_model_candidate("nvidia/llama-3.1-nemoguard-8b-content-safety") is False
    assert is_chat_model_candidate("nvidia/parakeet-ctc-1.1b") is False


def test_nvidia_scan_separates_available_busy_and_unavailable_models():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "vendor/available"},
                        {"id": "vendor/busy"},
                        {"id": "vendor/missing"},
                        {"id": "vendor/text-embedding"},
                    ]
                },
            )
        body = request.read().decode("utf-8")
        if '"vendor/available"' in body:
            return httpx.Response(200, json={"choices": []})
        if '"vendor/busy"' in body:
            return httpx.Response(429, json={"detail": "rate limited"})
        return httpx.Response(404, json={"detail": "not found"})

    progress: list[tuple[int, int, int, int]] = []
    result = asyncio.run(
        scan_nvidia_models(
            api_key="test-key",
            base_url="https://integrate.api.nvidia.com/v1",
            concurrency=2,
            progress=lambda *values: progress.append(values),
            transport=httpx.MockTransport(handler),
        )
    )

    assert result["catalog_count"] == 3
    assert result["counts"] == {
        "available": 1,
        "busy": 1,
        "restricted": 0,
        "incompatible": 0,
        "unavailable": 1,
    }
    assert progress[-1] == (3, 3, 1, 1)
    assert len(requests) == 4
    by_id = {item["id"]: item for item in result["results"]}
    assert by_id["vendor/busy"]["reason"] == "rate_limited"
    assert by_id["vendor/busy"]["retryable"] is True
    assert by_id["vendor/missing"]["reason"] == "model_not_deployed"
    assert by_id["vendor/missing"]["retryable"] is False


def test_nvidia_scan_classifies_missing_function_without_leaking_identifiers():
    account_id = "synthetic-account-id"
    function_id = "00000000-0000-4000-8000-000000000000"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "meta/llama2-70b"}]})
        return httpx.Response(
            404,
            json={
                "status": 404,
                "title": "Not Found",
                "detail": (f"Function '{function_id}': Not found for account '{account_id}'"),
            },
        )

    result = asyncio.run(
        scan_nvidia_models(
            api_key="test-key",
            base_url="https://integrate.api.nvidia.com/v1",
            transport=httpx.MockTransport(handler),
        )
    )

    model = result["results"][0]
    assert model["status"] == "unavailable"
    assert model["reason"] == "backend_function_unavailable"
    assert model["retryable"] is False
    assert model["message"] == "NVIDIA 后端实例已下线或当前账号不可用"
    assert account_id not in str(result)
    assert function_id not in str(result)


def test_nvidia_scan_treats_network_failure_as_retryable_busy_state():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "vendor/transient"}]})
        raise httpx.ConnectError("connection failed", request=request)

    result = asyncio.run(
        scan_nvidia_models(
            api_key="test-key",
            base_url="https://integrate.api.nvidia.com/v1",
            transport=httpx.MockTransport(handler),
        )
    )

    model = result["results"][0]
    assert model["status"] == "busy"
    assert model["reason"] == "network_error"
    assert model["retryable"] is True
    assert model["message"] == "网络请求失败，可稍后重试"


def test_nvidia_scan_rejects_invalid_credentials_before_probing_models():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "unauthorized"})

    with pytest.raises(FreeModelScanError, match="API Key"):
        asyncio.run(
            scan_nvidia_models(
                api_key="bad-key",
                base_url="https://integrate.api.nvidia.com/v1",
                transport=httpx.MockTransport(handler),
            )
        )


def test_nvidia_scan_rejects_non_official_base_url():
    with pytest.raises(FreeModelScanError, match="官方 API"):
        asyncio.run(
            scan_nvidia_models(
                api_key="test-key",
                base_url="https://example.test/v1",
            )
        )
