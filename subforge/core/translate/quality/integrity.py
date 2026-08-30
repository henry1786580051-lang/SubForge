"""Source and timeline integrity checks for the translation write-back step."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from subforge.core.asr.asr_data import ASRDataSeg
from subforge.core.translate.quality.diagnostics import (
    DiagnosticCategory,
    DiagnosticSeverity,
    QualityDiagnostic,
    RepairStrategy,
)


@dataclass(frozen=True, slots=True)
class WordIntegritySnapshot:
    text: str
    start_time: int
    end_time: int
    speaker_id: str
    timing_source: str
    language_code: str


@dataclass(frozen=True, slots=True)
class SegmentIntegritySnapshot:
    text: str
    start_time: int
    end_time: int
    speaker_id: str
    timestamp_granularity: str
    timing_source: str
    language_code: str
    words: tuple[WordIntegritySnapshot, ...]


def capture_segment_integrity(
    segments: Iterable[ASRDataSeg],
) -> tuple[SegmentIntegritySnapshot, ...]:
    """Capture every source-owned field while deliberately excluding translation."""
    return tuple(
        SegmentIntegritySnapshot(
            text=segment.text,
            start_time=segment.start_time,
            end_time=segment.end_time,
            speaker_id=segment.speaker_id,
            timestamp_granularity=segment.timestamp_granularity,
            timing_source=segment.timing_source,
            language_code=segment.language_code,
            words=tuple(
                WordIntegritySnapshot(
                    text=word.text,
                    start_time=word.start_time,
                    end_time=word.end_time,
                    speaker_id=word.speaker_id,
                    timing_source=word.timing_source,
                    language_code=word.language_code,
                )
                for word in segment.words
            ),
        )
        for segment in segments
    )


def _diagnostic(
    rule_id: str,
    *,
    cue_index: int | None = None,
    evidence: tuple[tuple[str, str], ...] = (),
) -> QualityDiagnostic:
    return QualityDiagnostic(
        rule_id=rule_id,
        category=(
            DiagnosticCategory.OWNERSHIP
            if rule_id.startswith("source.")
            else DiagnosticCategory.STRUCTURE
        ),
        severity=DiagnosticSeverity.ERROR,
        confidence=1.0,
        cue_keys=(cue_index,) if cue_index is not None else (),
        evidence=evidence,
        repair_strategy=RepairStrategy.NONE,
        message=(
            "Translation write-back changed source-owned subtitle data; refusing to save "
            "the result."
        ),
    )


def inspect_segment_integrity(
    expected: Sequence[SegmentIntegritySnapshot],
    actual: Sequence[ASRDataSeg],
) -> tuple[QualityDiagnostic, ...]:
    """Return exact source/timeline mutations introduced during translation."""
    diagnostics: list[QualityDiagnostic] = []
    if len(expected) != len(actual):
        diagnostics.append(
            _diagnostic(
                "timeline.segment_count_changed",
                evidence=(
                    ("expected_count", str(len(expected))),
                    ("actual_count", str(len(actual))),
                ),
            )
        )

    for cue_index, (before, after) in enumerate(zip(expected, actual), 1):
        if before.text != after.text:
            diagnostics.append(_diagnostic("source.text_changed", cue_index=cue_index))
        if (before.start_time, before.end_time) != (after.start_time, after.end_time):
            diagnostics.append(
                _diagnostic(
                    "timeline.cue_timestamp_changed",
                    cue_index=cue_index,
                    evidence=(
                        ("expected_start", str(before.start_time)),
                        ("expected_end", str(before.end_time)),
                        ("actual_start", str(after.start_time)),
                        ("actual_end", str(after.end_time)),
                    ),
                )
            )
        metadata = (
            after.speaker_id,
            after.timestamp_granularity,
            after.timing_source,
            after.language_code,
        )
        if metadata != (
            before.speaker_id,
            before.timestamp_granularity,
            before.timing_source,
            before.language_code,
        ):
            diagnostics.append(_diagnostic("source.metadata_changed", cue_index=cue_index))
        if capture_segment_integrity((after,))[0].words != before.words:
            diagnostics.append(
                _diagnostic("timeline.word_alignment_changed", cue_index=cue_index)
            )
    return tuple(diagnostics)
