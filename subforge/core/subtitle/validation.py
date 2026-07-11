"""Structural validation for finalized bilingual subtitles."""

from dataclasses import dataclass

from subforge.core.asr.asr_data import ASRData
from subforge.core.translate.base import BaseTranslator


@dataclass(frozen=True)
class SourceSegmentLock:
    text: str
    start_time: int
    end_time: int
    speaker_id: str


def lock_source_segments(asr_data: ASRData) -> tuple[SourceSegmentLock, ...]:
    """Capture source boundaries before translation begins."""
    return tuple(
        SourceSegmentLock(
            text=segment.text,
            start_time=segment.start_time,
            end_time=segment.end_time,
            speaker_id=segment.speaker_id,
        )
        for segment in asr_data.segments
    )


def validate_bilingual_result(
    asr_data: ASRData,
    source_lock: tuple[SourceSegmentLock, ...] | None = None,
) -> None:
    """Reject structural loss without judging translation phrasing.

    Natural translations may move clauses or compress wording, so this check
    deliberately avoids semantic-similarity thresholds. It only guarantees
    that every locked source cue still exists and owns a real translation.
    """
    errors: list[str] = []
    if source_lock is not None and len(asr_data.segments) != len(source_lock):
        errors.append(
            f"segment count changed from {len(source_lock)} to {len(asr_data.segments)}"
        )

    for index, segment in enumerate(asr_data.segments, 1):
        translated = (segment.translated_text or "").strip()
        if not translated:
            errors.append(f"empty translation at index {index}")
        elif BaseTranslator._looks_like_placeholder_translation(translated):
            errors.append(f"placeholder translation at index {index}")

        if source_lock is None or index > len(source_lock):
            continue
        locked = source_lock[index - 1]
        if segment.text != locked.text:
            errors.append(f"source text changed at index {index}")
        if (segment.start_time, segment.end_time) != (
            locked.start_time,
            locked.end_time,
        ):
            errors.append(f"source timing changed at index {index}")
        if segment.speaker_id != locked.speaker_id:
            errors.append(f"speaker changed at index {index}")

    if errors:
        detail = "; ".join(errors[:20])
        raise RuntimeError(f"Final bilingual subtitle validation failed: {detail}")
