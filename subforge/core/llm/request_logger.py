import contextvars
import json
import logging
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

logger = logging.getLogger(__name__)

_log_lock = threading.Lock()
_pending_requests: Dict[int, Dict[str, Any]] = {}  # 暂存请求信息，等待响应后合并
_current_request_key: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "subforge_llm_request_key",
    default=None,
)


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


# ==================== HTTPX Hooks ====================


def _infer_stage(request_body: dict) -> str:
    messages = request_body.get("messages") or []
    text = " ".join(str(item.get("content", "")) for item in messages if isinstance(item, dict)).lower()
    if "correct the following subtitles" in text or "keep the original language" in text:
        return "optimize"
    if "current_subtitles" in text or "translate" in text or "target language" in text:
        return "translate"
    if "summary" in text or "terminology" in text or "global context" in text:
        return "context"
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
    if "/chat/completions" not in str(request.url):
        return

    try:
        request_body = json.loads(request.content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        request_body = {"raw": request.content.decode("utf-8", errors="replace")}

    request_key = id(request)
    previous_key = _current_request_key.get()
    _current_request_key.set(request_key)
    with _log_lock:
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
            "request": request_body,
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


def log_llm_response(response: Any) -> None:
    """记录完整的请求+响应（在 SDK 解析响应后调用）"""
    key = _current_request_key.get()
    if key is None:
        logger.debug("No request context found for LLM response log")
        return

    with _log_lock:
        pending = _pending_requests.pop(key, None)
    _current_request_key.set(None)
    if pending is None:
        return

    # 序列化完整响应体
    response_data = {}
    if response and hasattr(response, "model_dump"):
        response_data = response.model_dump()

    # 获取任务上下文
    ctx = get_task_context()
    explicit_ctx = pending.get("context", {})
    usage = response_data.get("usage") if isinstance(response_data, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    completion_details = usage.get("completion_tokens_details") or {}
    timestamp = datetime.now(timezone.utc).isoformat()

    log_entry = {
        "timestamp": timestamp,
        "time": timestamp,
        "task_id": explicit_ctx.get("task_id") or (ctx.task_id if ctx else ""),
        "file_name": explicit_ctx.get("file_name") or (ctx.file_name if ctx else ""),
        "stage": pending.get("stage") or (ctx.stage if ctx else ""),
        "model": response_data.get("model") or pending.get("model", ""),
        "batch": pending.get("batch", ""),
        "url": pending.get("url", ""),
        "status": pending.get("status", 0),
        "duration_ms": pending.get("duration_ms", 0),
        "request": pending.get("request", {}),
        "response": response_data,
        "tokens": int(usage.get("total_tokens") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
    }

    _write_log(log_entry)


def log_llm_error(error: Exception) -> None:
    """Log and release the exact pending request when the SDK raises."""
    key = _current_request_key.get()
    if key is None:
        return
    with _log_lock:
        pending = _pending_requests.pop(key, None)
    _current_request_key.set(None)
    if pending is None:
        return

    ctx = get_task_context()
    explicit_ctx = pending.get("context", {})
    timestamp = datetime.now(timezone.utc).isoformat()
    _write_log(
        {
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
            "request": pending.get("request", {}),
            "error": f"{type(error).__name__}: {error}",
        }
    )
