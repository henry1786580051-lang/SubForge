"""Immutable per-task context for translation quality work."""

from __future__ import annotations

from dataclasses import dataclass

from subforge.core.asr.asr_data import ASRData
from subforge.core.translate.quality.features import CueFeatures
from subforge.core.translate.types import TargetLanguage


@dataclass(frozen=True, slots=True)
class TranslationCue:
    """A lossless, immutable copy of one source subtitle cue."""

    index: int
    source: str
    start_time: int
    end_time: int
    speaker: str
    language: str
    gap_before_ms: int
    gap_after_ms: int
    timing_source: str
    timestamp_granularity: str


@dataclass(frozen=True, slots=True)
class TranslationSession:
    """Stable task input shared by future translation quality stages."""

    cues: tuple[TranslationCue, ...]
    features: tuple[CueFeatures, ...]
    target_language: TargetLanguage
    model: str
    speaker_count: int

    @property
    def is_multispeaker(self) -> bool:
        return self.speaker_count > 1

    def cue(self, index: int) -> TranslationCue | None:
        if index <= 0 or index > len(self.cues):
            return None
        candidate = self.cues[index - 1]
        return candidate if candidate.index == index else None

    def feature(self, index: int) -> CueFeatures | None:
        if index <= 0 or index > len(self.features):
            return None
        candidate = self.features[index - 1]
        return candidate if candidate.index == index else None


def build_translation_session(
    subtitle_data: ASRData,
    *,
    target_language: TargetLanguage,
    model: str,
) -> TranslationSession:
    """Copy mutable ASR input into a deterministic task-scoped snapshot."""
    segments = tuple(subtitle_data.segments)
    speaker_aliases: dict[str, str] = {}
    cues: list[TranslationCue] = []
    features: list[CueFeatures] = []

    for offset, segment in enumerate(segments):
        index = offset + 1
        raw_speaker = str(segment.speaker_id or "").strip()
        speaker = ""
        if raw_speaker:
            speaker = speaker_aliases.setdefault(
                raw_speaker,
                f"S{len(speaker_aliases) + 1}",
            )
        previous = segments[offset - 1] if offset else None
        following = segments[offset + 1] if offset + 1 < len(segments) else None
        source = str(segment.text or "")
        cues.append(
            TranslationCue(
                index=index,
                source=source,
                start_time=int(segment.start_time),
                end_time=int(segment.end_time),
                speaker=speaker,
                language=str(segment.language_code or ""),
                gap_before_ms=(
                    max(0, int(segment.start_time) - int(previous.end_time))
                    if previous is not None
                    else 0
                ),
                gap_after_ms=(
                    max(0, int(following.start_time) - int(segment.end_time))
                    if following is not None
                    else 0
                ),
                timing_source=str(segment.timing_source),
                timestamp_granularity=str(segment.timestamp_granularity),
            )
        )
        features.append(CueFeatures.from_source(index, source))

    return TranslationSession(
        cues=tuple(cues),
        features=tuple(features),
        target_language=target_language,
        model=str(model or ""),
        speaker_count=len(speaker_aliases),
    )
