import contextvars
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

from subforge.config import LOG_PATH
from subforge.core.llm.context import get_task_context

LLM_LOG_FILE = LOG_PATH / "llm_requests.jsonl"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
MAX_PENDING_REQUESTS = 1000
LOG_LEVELS = {"summary", "standard", "debug"}

logger = logging.getLogger(__name__)

_log_lock = threading.Lock()
_pending_requests: Dict[int, Dict[str, Any]] = {}  # 暂存请求信息，等待响应后合并
_current_request_key: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "subforge_llm_request_key",
    default=None,
)
_log_level = os.getenv("SUBFORGE_LLM_LOG_LEVEL", "summary").strip().lower()
if _log_level not in LOG_LEVELS:
    _log_level = "summary"


# ==================== 日志写入 ====================


def _rotate_if_needed() -> None:
    """日志文件过大时轮转"""
    if not LLM_LOG_FILE.exists():
        return
    if LLM_LOG_FILE.stat().st_size < MAX_LOG_SIZE:
        return

    backup = LLM_LOG_FILE.with_suffix(".jsonl.old")
    if backup.exists():
        backup.unlink()
    LLM_LOG_FILE.rename(backup)


def _write_log(entry: Dict[str, Any]) -> None:
    """写入日志"""
    try:
        LOG_PATH.mkdir(parents=True, exist_ok=True)
        with _log_lock:
            _rotate_if_needed()
            with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("Failed to write LLM log: %s", e)


def set_llm_log_level(level: str) -> None:
    """Set process-wide LLM log detail without exposing request content by default."""
    normalized = str(level or "summary").strip().lower()
    if normalized not in LOG_LEVELS:
        raise ValueError(f"Unsupported LLM log level: {level}")
    global _log_level
    with _log_lock:
        _log_level = normalized


def _compact_payload(value: Any, *, depth: int = 0) -> Any:
    """Bound diagnostic payload size while preserving its useful structure."""
    if depth >= 5:
        return "<truncated>"
    if isinstance(value, str):
        return value if len(value) <= 2000 else f"{value[:2000]}...<truncated>"
    if isinstance(value, dict):
        items = list(value.items())
        compact = {str(key): _compact_payload(item, depth=depth + 1) for key, item in items[:30]}
        if len(items) > 30:
            compact["<truncated>"] = f"{len(items) - 30} more fields"
        return compact
    if isinstance(value, (list, tuple)):
        compact = [_compact_payload(item, depth=depth + 1) for item in value[:30]]
        if len(value) > 30:
            compact.append(f"<{len(value) - 30} more items>")
        return compact
    return value


def _response_metadata(response: Any, level: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Extract token metadata without serializing the full SDK response in summary mode."""
    model = str(getattr(response, "model", "") or "")
    usage_obj = getattr(response, "usage", None)
    usage: dict[str, Any] = {}
    if usage_obj is not None:
        if hasattr(usage_obj, "model_dump"):
            dumped_usage = usage_obj.model_dump()
            usage = dumped_usage if isinstance(dumped_usage, dict) else {}
        elif isinstance(usage_obj, dict):
            usage = usage_obj

    response_data: dict[str, Any] = {}
    if level != "summary" and response and hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            response_data = dumped if level == "debug" else _compact_payload(dumped)
            model = model or str(dumped.get("model") or "")
            dumped_usage = dumped.get("usage")
            if not usage and isinstance(dumped_usage, dict):
                usage = dumped_usage
    elif (not model or not usage) and response and hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            model = model or str(dumped.get("model") or "")
            dumped_usage = dumped.get("usage")
            if not usage and isinstance(dumped_usage, dict):
                usage = dumped_usage
    return model, usage, response_data


# ==================== HTTPX Hooks ====================


def _infer_stage(request_body: dict) -> str:
    messages = request_body.get("messages") or []
    system = request_body.get("system") or []
    text = " ".join(
        str(item.get("content", item.get("text", "")))
        for item in [*system, *messages]
        if isinstance(item, dict)
    ).lower()
    if "correct the following subtitles" in text or "keep the original language" in text:
        return "optimize"
    if "summary" in text or "terminology" in text or "global context" in text:
        return "context"
    if "current_subtitles" in text or "translate" in text or "target language" in text:
        return "translate"
    if "split" in text or "sentence" in text:
        return "split"
    return "llm"


def _batch_label(request_body: dict) -> str:
    messages = request_body.get("messages") or []
    text = " ".join(str(item.get("content", "")) for item in messages if isinstance(item, dict))
    keys = [int(value) for value in re.findall(r'["\'](\d+)["\']\s*:', text)]
    return f"{min(keys)}-{max(keys)}" if keys else ""


def _on_request(request: httpx.Request, log_context: Optional[dict[str, str]] = None) -> None:
    """请求发送前: 暂存请求信息"""
    if not any(path in str(request.url) for path in ("/chat/completions", "/v1/messages")):
        return

    try:
        request_body = json.loads(request.content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        request_body = {"raw": request.content.decode("utf-8", errors="replace")}

    request_key = id(request)
    previous_key = _current_request_key.get()
    _current_request_key.set(request_key)
    with _log_lock:
        log_level = _log_level
        # OpenAI-compatible clients can retry in the same execution context.
        # Only the last request is paired with the SDK response, so release the
        # superseded attempt instead of retaining it for the process lifetime.
        if previous_key is not None and previous_key != request_key:
            _pending_requests.pop(previous_key, None)
        while len(_pending_requests) >= MAX_PENDING_REQUESTS:
            oldest_key = next(iter(_pending_requests))
            _pending_requests.pop(oldest_key, None)
        _pending_requests[request_key] = {
            "start_time": time.time(),
            "url": str(request.url),
            "request": (
                request_body
                if log_level == "debug"
                else _compact_payload(request_body)
                if log_level == "standard"
                else None
            ),
            "log_level": log_level,
            "context": dict(log_context or {}),
            "stage": _infer_stage(request_body),
            "model": str(request_body.get("model", "")),
            "batch": _batch_label(request_body),
        }


def _on_response(response: httpx.Response) -> None:
    """响应接收后: 记录状态码和耗时"""
    request = response.request
    with _log_lock:
        pending = _pending_requests.get(id(request))
        if not pending:
            return
        pending["status"] = response.status_code
        pending["duration_ms"] = int((time.time() - pending["start_time"]) * 1000)
        pending["completed"] = True


# ==================== 公开 API ====================


def create_logging_http_client(
    log_context: Optional[dict[str, str]] = None,
) -> httpx.Client:
    """创建带日志记录的 HTTPX 客户端"""
    return httpx.Client(
        event_hooks={
            "request": [lambda request: _on_request(request, log_context)],
            "response": [_on_response],
        }
    )


def log_llm_response(response: Any) -> dict[str, Any] | None:
    """记录完整的请求+响应（在 SDK 解析响应后调用）"""
    key = _current_request_key.get()
    if key is None:
        logger.debug("No request context found for LLM response log")
        return None

    with _log_lock:
        pending = _pending_requests.pop(key, None)
    _current_request_key.set(None)
    if pending is None:
        return None

    # HTTP response hooks run when headers arrive. The SDK may still spend most of
    # the request reading and parsing a long generated body, so close the timer here.
    pending["duration_ms"] = int((time.time() - pending["start_time"]) * 1000)

    log_level = pending.get("log_level", "summary")
    response_model, usage, response_data = _response_metadata(response, log_level)

    # 获取任务上下文
    ctx = get_task_context()
    explicit_ctx = pending.get("context", {})
    completion_details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    cache_creation_tokens = int(usage.get("cache_creation_input_tokens") or 0)
    cache_read_tokens = int(usage.get("cache_read_input_tokens") or 0)
    uncached_input_tokens = int(usage.get("input_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    if not prompt_tokens:
        prompt_tokens = cache_creation_tokens + cache_read_tokens + uncached_input_tokens
    cached_tokens = int(prompt_details.get("cached_tokens") or cache_read_tokens)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
    timestamp = datetime.now(timezone.utc).isoformat()

    log_entry = {
        "timestamp": timestamp,
        "time": timestamp,
        "task_id": explicit_ctx.get("task_id") or (ctx.task_id if ctx else ""),
        "file_name": explicit_ctx.get("file_name") or (ctx.file_name if ctx else ""),
        "stage": pending.get("stage") or (ctx.stage if ctx else ""),
        "model": response_model or pending.get("model", ""),
        "batch": pending.get("batch", ""),
        "url": pending.get("url", ""),
        "status": pending.get("status", 0),
        "duration_ms": pending.get("duration_ms", 0),
        "log_level": log_level,
        "tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "cache_hit_rate": round(cached_tokens / prompt_tokens, 4) if prompt_tokens else 0.0,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
    }
    if log_level != "summary":
        log_entry["request"] = pending.get("request", {})
        log_entry["response"] = response_data

    _write_log(log_entry)
    return log_entry


def log_llm_error(error: Exception) -> dict[str, Any] | None:
    """Log and release the exact pending request when the SDK raises."""
    key = _current_request_key.get()
    if key is None:
        return None
    with _log_lock:
        pending = _pending_requests.pop(key, None)
    _current_request_key.set(None)
    if pending is None:
        return None

    ctx = get_task_context()
    explicit_ctx = pending.get("context", {})
    timestamp = datetime.now(timezone.utc).isoformat()
    log_entry = {
        "timestamp": timestamp,
        "time": timestamp,
        "task_id": explicit_ctx.get("task_id") or (ctx.task_id if ctx else ""),
        "file_name": explicit_ctx.get("file_name") or (ctx.file_name if ctx else ""),
        "stage": pending.get("stage") or (ctx.stage if ctx else ""),
        "model": pending.get("model", ""),
        "batch": pending.get("batch", ""),
        "url": pending.get("url", ""),
        "status": pending.get("status", 0),
        "duration_ms": pending.get(
            "duration_ms",
            int((time.time() - pending.get("start_time", time.time())) * 1000),
        ),
        "log_level": pending.get("log_level", "summary"),
        "error": f"{type(error).__name__}: {error}",
    }
    if pending.get("log_level") != "summary":
        log_entry["request"] = pending.get("request", {})
    _write_log(log_entry)
    return log_entry
