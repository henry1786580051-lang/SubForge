import hashlib
import json
from pathlib import Path
from typing import Iterator

from fastapi import APIRouter, HTTPException, Query

from subforge.core.utils.atomic_write import atomic_write_text

router = APIRouter()


def _find_log_path() -> Path | None:
    from subforge.config import LLM_LOG_FILE

    if LLM_LOG_FILE.exists():
        return LLM_LOG_FILE
    candidates = [
        Path.home() / "SubForge" / "logs" / "llm_requests.jsonl",
        Path.home() / "Desktop" / "Project" / "SubForge" / "AppData" / "logs" / "llm_requests.jsonl",
        Path.home() / "Subtitle" / "logs" / "llm_requests.jsonl",
        Path.home() / "Desktop" / "Project" / "Subtitle" / "AppData" / "logs" / "llm_requests.jsonl",
    ]
    return next((path for path in candidates if path.exists()), None)


def _iter_entries(log_path: Path) -> Iterator[dict]:
    try:
        with log_path.open("r", encoding="utf-8") as log_file:
            for line in log_file:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict):
                    yield entry
    except (OSError, UnicodeDecodeError):
        return


def _group_id(entry: dict) -> str:
    task_id = str(entry.get("task_id") or "").strip()
    if task_id:
        return task_id
    identity = "\x1f".join(
        (
            str(entry.get("file_name") or "").strip(),
            str(entry.get("timestamp") or entry.get("time") or "")[:16],
        )
    )
    return f"legacy:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}"


def _integer(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _is_error(entry: dict) -> bool:
    if entry.get("error"):
        return True
    status = _integer(entry.get("status"))
    return status >= 400


def _new_group(group_id: str, entry: dict, *, include_entries: bool) -> dict:
    timestamp = str(entry.get("timestamp") or entry.get("time") or "")
    return {
        "id": group_id,
        "task_id": str(entry.get("task_id") or "").strip(),
        "file_name": str(entry.get("file_name") or "").strip(),
        "started_at": timestamp,
        "ended_at": timestamp,
        "stages": [],
        "models": [],
        "request_count": 0,
        "error_count": 0,
        "duration_ms": 0,
        "tokens": 0,
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "cache_creation_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "entries": [] if include_entries else [],
    }


def _add_entry(group: dict, entry: dict, *, include_entries: bool) -> None:
    if include_entries:
        group["entries"].append(entry)
    group["request_count"] += 1
    group["error_count"] += int(_is_error(entry))
    group["duration_ms"] += _integer(entry.get("duration_ms"))
    for token_key in (
        "tokens",
        "prompt_tokens",
        "cached_tokens",
        "cache_creation_tokens",
        "completion_tokens",
        "reasoning_tokens",
    ):
        group[token_key] += _integer(entry.get(token_key))

    stage = str(entry.get("stage") or "").strip()
    model = str(entry.get("model") or "").strip()
    if stage and stage not in group["stages"]:
        group["stages"].append(stage)
    if model and model not in group["models"]:
        group["models"].append(model)

    timestamp = str(entry.get("timestamp") or entry.get("time") or "")
    if timestamp:
        group["started_at"] = min(filter(None, (group["started_at"], timestamp)))
        group["ended_at"] = max(group["ended_at"], timestamp)


def _finish_group(group: dict) -> dict:
    prompt_tokens = group["prompt_tokens"]
    group["cache_hit_rate"] = (
        round(group["cached_tokens"] / prompt_tokens, 4) if prompt_tokens else 0.0
    )
    return group


@router.get("/")
async def get_llm_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    search: str = Query("", description="Search in task_id, file_name, model, stage"),
):
    """Return task summaries without loading request payloads into the response."""
    log_path = _find_log_path()
    if not log_path:
        return {"groups": [], "total": 0, "page": page, "pages": 0}

    groups: dict[str, dict] = {}
    search_lower = search.lower().strip()
    for entry in _iter_entries(log_path):
        group_id = _group_id(entry)
        group = groups.setdefault(group_id, _new_group(group_id, entry, include_entries=False))
        _add_entry(group, entry, include_entries=False)

    finished_groups = (_finish_group(group) for group in groups.values())
    if search_lower:
        finished_groups = (
            group
            for group in finished_groups
            if any(
                search_lower in str(value).lower()
                for value in (
                    group["task_id"],
                    group["file_name"],
                    group["models"],
                    group["stages"],
                )
            )
        )
    grouped_entries = sorted(
        finished_groups,
        key=lambda group: group["ended_at"],
        reverse=True,
    )
    total = len(grouped_entries)
    pages = max(1, (total + page_size - 1) // page_size)
    start = (page - 1) * page_size
    return {
        "groups": grouped_entries[start : start + page_size],
        "total": total,
        "page": page,
        "pages": pages,
    }


@router.get("/{group_id}")
async def get_llm_log_detail(group_id: str):
    """Load request payloads only for the selected task."""
    log_path = _find_log_path()
    if not log_path:
        raise HTTPException(status_code=404, detail="LLM log task not found")

    group = None
    for entry in _iter_entries(log_path):
        if _group_id(entry) != group_id:
            continue
        if group is None:
            group = _new_group(group_id, entry, include_entries=True)
        _add_entry(group, entry, include_entries=True)
    if group is None:
        raise HTTPException(status_code=404, detail="LLM log task not found")
    return _finish_group(group)


@router.delete("/")
async def clear_llm_logs():
    """Clear the LLM log file atomically."""
    log_path = _find_log_path()
    if log_path:
        atomic_write_text(log_path, "")
    return {"status": "ok"}
