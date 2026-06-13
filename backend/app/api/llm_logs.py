import json
from collections import OrderedDict
from pathlib import Path

from fastapi import APIRouter, Query

router = APIRouter()


def _find_log_path() -> Path | None:
    # Try centralized config first
    from subforge.config import LLM_LOG_FILE
    if LLM_LOG_FILE.exists():
        return LLM_LOG_FILE
    # Fallback candidates
    candidates = [
        Path.home() / "SubForge" / "logs" / "llm_requests.jsonl",
        Path.home() / "Desktop" / "Project" / "SubForge" / "AppData" / "logs" / "llm_requests.jsonl",
        Path.home() / "Subtitle" / "logs" / "llm_requests.jsonl",
        Path.home() / "Desktop" / "Project" / "Subtitle" / "AppData" / "logs" / "llm_requests.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


@router.get("/")
async def get_llm_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    search: str = Query("", description="Search in task_id, file_name, model, stage"),
):
    """Read LLM request logs from the JSONL log file."""
    log_path = _find_log_path()
    if not log_path:
        return {"groups": [], "total": 0, "page": page, "pages": 0}

    try:
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    except (OSError, UnicodeDecodeError):
        return {"groups": [], "total": 0, "page": page, "pages": 0}

    entries = []
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            entries.append(entry)
        except json.JSONDecodeError:
            continue

    # Reverse to show newest first
    entries.reverse()

    # Filter by search
    if search:
        search_lower = search.lower()
        entries = [
            e for e in entries
            if search_lower in str(e.get("task_id", "")).lower()
            or search_lower in str(e.get("file_name", "")).lower()
            or search_lower in str(e.get("model", "")).lower()
            or search_lower in str(e.get("stage", "")).lower()
        ]

    groups: OrderedDict[str, dict] = OrderedDict()
    for index, entry in enumerate(entries):
        task_id = str(entry.get("task_id") or "").strip()
        file_name = str(entry.get("file_name") or "").strip()
        timestamp = str(entry.get("timestamp") or entry.get("time") or "")
        group_id = task_id or f"legacy:{file_name}:{timestamp[:16]}:{index}"
        group = groups.setdefault(
            group_id,
            {
                "id": group_id,
                "task_id": task_id,
                "file_name": file_name,
                "started_at": timestamp,
                "ended_at": timestamp,
                "stages": [],
                "models": [],
                "request_count": 0,
                "error_count": 0,
                "duration_ms": 0,
                "tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "entries": [],
            },
        )
        group["entries"].append(entry)
        group["request_count"] += 1
        group["error_count"] += int(bool(entry.get("error") or int(entry.get("status") or 0) >= 400))
        group["duration_ms"] += int(entry.get("duration_ms") or 0)
        for token_key in ("tokens", "prompt_tokens", "completion_tokens", "reasoning_tokens"):
            group[token_key] += int(entry.get(token_key) or 0)
        stage = str(entry.get("stage") or "").strip()
        model = str(entry.get("model") or "").strip()
        if stage and stage not in group["stages"]:
            group["stages"].append(stage)
        if model and model not in group["models"]:
            group["models"].append(model)
        if timestamp:
            if not group["started_at"] or timestamp < group["started_at"]:
                group["started_at"] = timestamp
            if not group["ended_at"] or timestamp > group["ended_at"]:
                group["ended_at"] = timestamp

    grouped_entries = list(groups.values())
    total = len(grouped_entries)
    pages = max(1, (total + page_size - 1) // page_size)
    start = (page - 1) * page_size
    end = start + page_size

    return {
        "groups": grouped_entries[start:end],
        "total": total,
        "page": page,
        "pages": pages,
    }


@router.delete("/")
async def clear_llm_logs():
    """Clear the LLM log file."""
    log_path = _find_log_path()
    if log_path:
        log_path.write_text("", encoding="utf-8")
    return {"status": "ok"}
