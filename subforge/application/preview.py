"""Stable serialization for real-time subtitle previews."""

from __future__ import annotations

from typing import Any

from subforge.core.translate.base import BaseTranslator


def subtitle_preview_segments(data) -> list[dict[str, Any]]:
    """Serialize ASR-like data without exposing placeholder translations."""
    return [
        {
            "id": index,
            "start": segment._ms_to_srt_time(segment.start_time),
            "end": segment._ms_to_srt_time(segment.end_time),
            "text": segment.text,
            "translated": ""
            if BaseTranslator._looks_like_placeholder_translation(segment.translated_text or "")
            else segment.translated_text or "",
            "speaker": segment.speaker_id or "",
        }
        for index, segment in enumerate(data.segments, 1)
    ]
