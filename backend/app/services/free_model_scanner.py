"""Low-cost availability probes for free hosted LLM catalogs."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx

NVIDIA_SCAN_CONCURRENCY = 6
NVIDIA_SCAN_MAX_MODELS = 300

_NON_CHAT_MARKERS = (
    "audio2face",
    "content-safety",
    "embed",
    "embedding",
    "/embed-",
    "/e5-",
    "gliner",
    "guardrail",
    "llama-guard",
    "nemoguard",
    "nemo-retriever",
    "nvclip",
    "/ocr",
    "parakeet",
    "re-rank",
    "rerank",
    "retrieval",
    "retriever",
    "safety-guard",
    "stable-diffusion",
    "topic-control",
    "whisper",
)


class FreeModelScanError(RuntimeError):
    """Raised when a provider catalog cannot be scanned safely."""


def is_chat_model_candidate(model_id: str) -> bool:
    """Keep text-generating chat models and skip known specialist endpoints."""
    normalized = str(model_id or "").strip().lower()
    return bool(normalized) and not any(marker in normalized for marker in _NON_CHAT_MARKERS)


def _catalog_model_ids(payload: Any) -> list[str]:
    records = payload.get("data", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return []
    models: list[str] = []
    for item in records:
        model_id = item.get("id", "") if isinstance(item, dict) else str(item)
        clean = str(model_id).strip()
        if clean and len(clean) <= 256 and is_chat_model_candidate(clean):
            models.append(clean)
    return sorted(set(models))[:NVIDIA_SCAN_MAX_MODELS]


def _response_error_text(response: httpx.Response) -> str:
    """Extract upstream error text for classification without exposing it."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:4096].lower()
    if not isinstance(payload, dict):
        return str(payload)[:4096].lower()

    fragments: list[str] = []
    for key in ("status", "title", "detail", "message", "error"):
        value = payload.get(key)
        if isinstance(value, dict):
            fragments.extend(str(item) for item in value.values())
        elif value is not None:
            fragments.append(str(value))
    return " ".join(fragments)[:4096].lower()


def _probe_status(
    status_code: int,
    error_text: str = "",
) -> tuple[str, str, str, bool]:
    """Return public status, reason code, safe message, and retry policy."""
    if status_code == 200:
        return "available", "ok", "响应正常", False
    if status_code == 202:
        return "busy", "request_pending", "请求仍在排队，可稍后重试", True
    if status_code == 429:
        return "busy", "rate_limited", "当前额度限流，可稍后重试", True
    if status_code in {502, 503, 504}:
        return "busy", "provider_busy", "供应商服务暂时不可用，可稍后重试", True
    if status_code in {400, 405, 415, 422}:
        return "incompatible", "chat_probe_incompatible", "不支持标准聊天接口", False
    if status_code == 401:
        return "restricted", "authentication_failed", "API Key 无效或已失效", False
    if status_code == 403:
        return "restricted", "account_restricted", "当前账号无权调用", False
    if status_code == 404:
        if "function" in error_text and "not found for account" in error_text:
            return (
                "unavailable",
                "backend_function_unavailable",
                "NVIDIA 后端实例已下线或当前账号不可用",
                False,
            )
        return "unavailable", "model_not_deployed", "模型未在当前服务中部署", False
    return "unavailable", "unexpected_http_status", f"服务返回 HTTP {status_code}", False


async def _probe_model(
    client: httpx.AsyncClient,
    model_id: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        async with semaphore:
            started = time.monotonic()
            response = await client.post(
                "/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Reply OK"}],
                    "max_tokens": 1,
                    "stream": False,
                },
            )
        status, reason, message, retryable = _probe_status(
            response.status_code,
            _response_error_text(response),
        )
        return {
            "id": model_id,
            "status": status,
            "reason": reason,
            "retryable": retryable,
            "http_status": response.status_code,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "message": message,
        }
    except httpx.TimeoutException:
        return {
            "id": model_id,
            "status": "busy",
            "reason": "request_timeout",
            "retryable": True,
            "http_status": None,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "message": "响应超时，可稍后重试",
        }
    except httpx.HTTPError:
        return {
            "id": model_id,
            "status": "busy",
            "reason": "network_error",
            "retryable": True,
            "http_status": None,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "message": "网络请求失败，可稍后重试",
        }


async def scan_nvidia_models(
    *,
    api_key: str,
    base_url: str,
    concurrency: int = NVIDIA_SCAN_CONCURRENCY,
    progress: Callable[[int, int, int, int], None] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Fetch NVIDIA's live catalog and probe chat models with minimal output."""
    clean_key = str(api_key or "").strip()
    if not clean_key:
        raise FreeModelScanError("请先在设置中保存 NVIDIA API Key")
    clean_base_url = str(base_url or "").rstrip("/")
    if clean_base_url != "https://integrate.api.nvidia.com/v1":
        raise FreeModelScanError("NVIDIA 免费模型扫描只允许使用官方 API 地址")

    timeout = httpx.Timeout(connect=6.0, read=18.0, write=6.0, pool=6.0)
    limits = httpx.Limits(
        max_connections=max(2, concurrency),
        max_keepalive_connections=max(2, concurrency),
    )
    started = time.monotonic()
    async with httpx.AsyncClient(
        base_url=clean_base_url,
        headers={"Authorization": f"Bearer {clean_key}"},
        timeout=timeout,
        limits=limits,
        follow_redirects=False,
        transport=transport,
    ) as client:
        try:
            catalog_response = await client.get("/models")
        except httpx.TimeoutException as exc:
            raise FreeModelScanError("读取 NVIDIA 模型目录超时，请稍后重试") from exc
        except httpx.HTTPError as exc:
            raise FreeModelScanError("无法连接 NVIDIA 模型目录，请检查网络后重试") from exc
        if catalog_response.status_code in {401, 403}:
            raise FreeModelScanError("NVIDIA API Key 无效或无权读取模型目录")
        if catalog_response.status_code != 200:
            raise FreeModelScanError(
                f"NVIDIA 模型目录请求失败：HTTP {catalog_response.status_code}"
            )
        try:
            models = _catalog_model_ids(catalog_response.json())
        except ValueError as exc:
            raise FreeModelScanError("NVIDIA 模型目录返回了无效数据") from exc
        if not models:
            raise FreeModelScanError("NVIDIA 当前没有返回可测试的聊天模型")

        semaphore = asyncio.Semaphore(max(1, min(concurrency, 12)))
        tasks = [
            asyncio.create_task(_probe_model(client, model_id, semaphore)) for model_id in models
        ]
        results: list[dict[str, Any]] = []
        try:
            for completed in asyncio.as_completed(tasks):
                result = await completed
                results.append(result)
                available = sum(item["status"] == "available" for item in results)
                busy = sum(item["status"] == "busy" for item in results)
                if progress is not None:
                    progress(len(results), len(models), available, busy)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    status_order = {
        "available": 0,
        "busy": 1,
        "restricted": 2,
        "incompatible": 3,
        "unavailable": 4,
    }
    results.sort(key=lambda item: (status_order.get(item["status"], 9), item["id"]))
    counts = {status: sum(item["status"] == status for item in results) for status in status_order}
    return {
        "provider": "nvidia",
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "catalog_count": len(models),
        "tested_count": len(results),
        "duration_ms": round((time.monotonic() - started) * 1000),
        "counts": counts,
        "results": results,
    }
