import json
from pathlib import Path

from fastapi import APIRouter, Query

router = APIRouter()

# LLM log file path (shared with subforge core)
_LOG_CANDIDATES = [
    Path.home() / "SubForge" / "logs" / "llm_requests.jsonl",
    Path.home() / "Desktop" / "Project" / "SubForge" / "AppData" / "logs" / "llm_requests.jsonl",
]


def _find_log_path() -> Path | None:
    for p in _LOG_CANDIDATES:
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
        return {"logs": [], "total": 0, "page": page, "pages": 0}

    try:
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    except (OSError, UnicodeDecodeError):
        return {"logs": [], "total": 0, "page": page, "pages": 0}

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

    total = len(entries)
    pages = max(1, (total + page_size - 1) // page_size)
    start = (page - 1) * page_size
    end = start + page_size

    return {
        "logs": entries[start:end],
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
