"""Helpers for normalizing OpenAI-compatible model responses."""

import re
from typing import Any

import json_repair

_REASONING_BLOCK_RE = re.compile(
    r"<(?P<tag>think|thinking|analysis|reasoning)>.*?</(?P=tag)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_UNCLOSED_REASONING_RE = re.compile(
    r"<(?P<tag>think|thinking|analysis|reasoning)>.*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def strip_reasoning_blocks(content: str) -> str:
    """Remove provider-specific visible reasoning without touching the final answer."""
    cleaned = str(content or "")
    cleaned = _REASONING_BLOCK_RE.sub("", cleaned)
    cleaned = _UNCLOSED_REASONING_RE.sub("", cleaned)
    return cleaned.strip()


def get_response_text(response: Any) -> str:
    """Return user-visible assistant text and reject reasoning-only responses."""
    content = None
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        blocks = getattr(response, "content", None)
        if isinstance(blocks, list):
            text_parts = [
                str(getattr(block, "text", ""))
                for block in blocks
                if getattr(block, "type", "") == "text" and getattr(block, "text", "")
            ]
            content = "\n".join(text_parts)
    if content is None:
        raise ValueError("Invalid LLM API response: empty content")
    cleaned = strip_reasoning_blocks(content or "")
    if not cleaned:
        raise ValueError("LLM response contained reasoning but no final answer")
    return cleaned


def _strip_markdown_fence(content: str) -> str:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _coerce_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if not isinstance(value, list):
        return None
    if len(value) == 1 and isinstance(value[0], dict):
        record = value[0]
        if not ({"key", "index", "id"} & record.keys()):
            return _coerce_object(record)

    merged: dict[str, Any] = {}
    for item in value:
        if not isinstance(item, dict):
            return None
        if len(item) == 1:
            key, translated = next(iter(item.items()))
        else:
            key = item.get("key", item.get("index", item.get("id")))
            translated = item.get(
                "translation",
                item.get("native_translation", item.get("text")),
            )
            if key is None or translated is None:
                return None
        key = str(key)
        if key in merged:
            return None
        merged[key] = translated
    return merged or None


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object from cleaned model output, including common list wrappers."""
    cleaned = _strip_markdown_fence(strip_reasoning_blocks(content))
    if not cleaned:
        raise ValueError("LLM response contained no JSON final answer")

    candidates = [cleaned]
    first_object = cleaned.find("{")
    last_object = cleaned.rfind("}")
    if first_object >= 0 and last_object > first_object:
        candidates.append(cleaned[first_object : last_object + 1])

    first_array = cleaned.find("[")
    last_array = cleaned.rfind("]")
    if first_array >= 0 and last_array > first_array:
        candidates.append(cleaned[first_array : last_array + 1])

    for candidate in candidates:
        try:
            parsed = json_repair.loads(candidate)
        except Exception:
            continue
        normalized = _coerce_object(parsed)
        if normalized is not None:
            return normalized
    raise ValueError("LLM final answer must be a JSON object keyed by subtitle index")
