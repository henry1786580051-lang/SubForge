"""Conservative DeepFilterNet calibration for multi-speaker recordings."""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from subforge.core.asr.asr_data import ASRData
from subforge.core.asr.speaker_diarization import SpeakerTurn

logger = logging.getLogger(__name__)

CALIBRATION_ATTENUATIONS_DB = (6.0, 12.0)
CALIBRATION_WINDOW_MS = 18_000
CALIBRATION_WINDOWS_PER_SPEAKER = 2
CALIBRATION_SILENCE_MS = 500


@dataclass(frozen=True)
class CalibrationWindow:
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class EnhancementCandidate:
    attenuation_db: float | None
    score: float
    word_count: int
    repeated_ratio: float
    speaker_word_rates: dict[str, float]

    @property
    def label(self) -> str:
        return "original" if self.attenuation_db is None else f"{self.attenuation_db:g} dB"


@dataclass(frozen=True)
class EnhancementCalibration:
    attenuation_db: float | None
    candidates: tuple[EnhancementCandidate, ...]


def _merge_same_speaker_turns(
    turns: Sequence[SpeakerTurn], *, max_gap_ms: int = 800
) -> list[SpeakerTurn]:
    merged: list[SpeakerTurn] = []
    for turn in sorted(turns, key=lambda item: (item.start_ms, item.end_ms)):
        if (
            merged
            and merged[-1].speaker_id == turn.speaker_id
            and turn.start_ms - merged[-1].end_ms <= max_gap_ms
        ):
            previous = merged[-1]
            merged[-1] = SpeakerTurn(
                previous.start_ms,
                max(previous.end_ms, turn.end_ms),
                previous.speaker_id,
            )
        else:
            merged.append(turn)
    return merged


def select_calibration_windows(
    turns: Sequence[SpeakerTurn],
    *,
    audio_duration_ms: int,
    window_ms: int = CALIBRATION_WINDOW_MS,
    windows_per_speaker: int = CALIBRATION_WINDOWS_PER_SPEAKER,
) -> list[CalibrationWindow]:
    """Select long, non-overlapping conversational windows for every speaker."""
    if audio_duration_ms <= 0 or window_ms <= 0:
        return []

    grouped: dict[str, list[SpeakerTurn]] = {}
    for turn in _merge_same_speaker_turns(turns):
        grouped.setdefault(turn.speaker_id, []).append(turn)

    selected: list[CalibrationWindow] = []
    for speaker_turns in grouped.values():
        ranked = sorted(
            speaker_turns,
            key=lambda turn: (turn.end_ms - turn.start_ms, -turn.start_ms),
            reverse=True,
        )
        speaker_windows = 0
        for turn in ranked:
            center = (turn.start_ms + turn.end_ms) // 2
            start = max(0, min(center - window_ms // 2, audio_duration_ms - window_ms))
            end = min(audio_duration_ms, start + window_ms)
            candidate = CalibrationWindow(start, end)
            if any(
                min(candidate.end_ms, existing.end_ms)
                - max(candidate.start_ms, existing.start_ms)
                > window_ms // 3
                for existing in selected
            ):
                continue
            selected.append(candidate)
            speaker_windows += 1
            if speaker_windows >= windows_per_speaker:
                break

    return sorted(selected, key=lambda window: window.start_ms)


def build_calibration_reel(
    audio_path: str,
    turns: Sequence[SpeakerTurn],
    output_path: str,
) -> list[SpeakerTurn]:
    """Copy selected source windows to a short WAV and remap speaker turns."""
    import numpy as np
    import soundfile as sf

    info = sf.info(audio_path)
    duration_ms = round(info.frames * 1000 / info.samplerate)
    windows = select_calibration_windows(turns, audio_duration_ms=duration_ms)
    if not windows:
        raise RuntimeError("No representative speaker windows were found")

    mapped_turns: list[SpeakerTurn] = []
    reel_cursor_ms = 0
    silence = np.zeros(
        (round(info.samplerate * CALIBRATION_SILENCE_MS / 1000), info.channels),
        dtype="float32",
    )
    with sf.SoundFile(output_path, "w", info.samplerate, info.channels, subtype="PCM_16") as output:
        for index, window in enumerate(windows):
            start_frame = round(window.start_ms * info.samplerate / 1000)
            frame_count = round((window.end_ms - window.start_ms) * info.samplerate / 1000)
            audio, _ = sf.read(
                audio_path,
                start=start_frame,
                frames=frame_count,
                dtype="float32",
                always_2d=True,
            )
            output.write(audio)
            for turn in turns:
                overlap_start = max(window.start_ms, turn.start_ms)
                overlap_end = min(window.end_ms, turn.end_ms)
                if overlap_end <= overlap_start:
                    continue
                mapped_turns.append(
                    SpeakerTurn(
                        reel_cursor_ms + overlap_start - window.start_ms,
                        reel_cursor_ms + overlap_end - window.start_ms,
                        turn.speaker_id,
                    )
                )
            reel_cursor_ms += window.end_ms - window.start_ms
            if index < len(windows) - 1:
                output.write(silence)
                reel_cursor_ms += CALIBRATION_SILENCE_MS
    return mapped_turns


def _word_tokens(data: ASRData) -> list[str]:
    return [
        token.lower()
        for segment in data.segments
        for token in re.findall(r"[A-Za-z0-9']+", segment.text)
    ]


def _repeated_ngram_ratio(tokens: Sequence[str], size: int = 3) -> float:
    if len(tokens) < size * 2:
        return 0.0
    ngrams = [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def score_candidate(
    data: ASRData,
    turns: Sequence[SpeakerTurn],
    attenuation_db: float | None,
) -> EnhancementCandidate:
    """Score recognition without treating extra (possibly hallucinated) words as quality."""
    tokens = _word_tokens(data)
    speaker_durations: dict[str, int] = {}
    speaker_words: dict[str, int] = {}
    for turn in turns:
        speaker_durations[turn.speaker_id] = speaker_durations.get(turn.speaker_id, 0) + max(
            0, turn.end_ms - turn.start_ms
        )

    invalid_timings = 0
    for segment in data.segments:
        if segment.end_time <= segment.start_time:
            invalid_timings += 1
            continue
        best_speaker = ""
        best_overlap = 0
        for turn in turns:
            overlap = min(segment.end_time, turn.end_ms) - max(segment.start_time, turn.start_ms)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker_id
        if best_speaker:
            speaker_words[best_speaker] = speaker_words.get(best_speaker, 0) + max(
                1, len(re.findall(r"[A-Za-z0-9']+", segment.text))
            )

    rates = {
        speaker: round(speaker_words.get(speaker, 0) / max(duration / 1000, 1.0), 4)
        for speaker, duration in speaker_durations.items()
    }
    min_rate = min(rates.values(), default=0.0)
    average_rate = sum(rates.values()) / max(len(rates), 1)
    repeated_ratio = _repeated_ngram_ratio(tokens)
    invalid_ratio = invalid_timings / max(len(data.segments), 1)
    score = min_rate * 2.0 + average_rate * 0.5 - repeated_ratio * 2.0 - invalid_ratio * 3.0
    return EnhancementCandidate(
        attenuation_db=attenuation_db,
        score=round(score, 4),
        word_count=len(tokens),
        repeated_ratio=round(repeated_ratio, 4),
        speaker_word_rates=rates,
    )


def choose_candidate(
    candidates: Sequence[EnhancementCandidate],
) -> EnhancementCandidate:
    """Keep original unless a denoised candidate is clearly and safely better."""
    if not candidates:
        raise ValueError("At least one enhancement candidate is required")
    baseline = next(
        (candidate for candidate in candidates if candidate.attenuation_db is None),
        candidates[0],
    )
    eligible: list[EnhancementCandidate] = []
    for candidate in candidates:
        if candidate is baseline:
            continue
        if candidate.word_count < baseline.word_count * 0.88:
            continue
        if candidate.word_count > baseline.word_count * 1.20:
            continue
        if candidate.repeated_ratio > baseline.repeated_ratio + 0.03:
            continue
        if any(
            candidate.speaker_word_rates.get(speaker, 0.0) < rate * 0.94
            for speaker, rate in baseline.speaker_word_rates.items()
        ):
            continue
        required_gain = max(0.08, abs(baseline.score) * 0.03)
        if candidate.score >= baseline.score + required_gain:
            eligible.append(candidate)
    return max(eligible, key=lambda item: item.score, default=baseline)


def calibrate_audio_enhancement(
    audio_path: str,
    turns: Sequence[SpeakerTurn],
    *,
    transcribe_sample: Callable[[str], ASRData],
    enhance: Callable[..., str],
    callback: Callable[[int, str], None] | None = None,
    attenuations_db: Iterable[float] = CALIBRATION_ATTENUATIONS_DB,
) -> EnhancementCalibration:
    """Evaluate a short calibration reel and return the safest attenuation."""
    reports: list[EnhancementCandidate] = []
    with tempfile.TemporaryDirectory(prefix="subforge_calibration_") as temp_dir:
        reel_path = str(Path(temp_dir) / "original.wav")
        mapped_turns = build_calibration_reel(audio_path, turns, reel_path)
        variants: list[tuple[float | None, str]] = [(None, reel_path)]
        for attenuation in attenuations_db:
            output_path = str(Path(temp_dir) / f"denoise_{attenuation:g}db.wav")
            try:
                variants.append(
                    (
                        float(attenuation),
                        enhance(reel_path, output_path, atten_lim_db=float(attenuation)),
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Skipping %.1f dB calibration candidate: %s",
                    attenuation,
                    exc,
                    exc_info=True,
                )

        for index, (attenuation, candidate_path) in enumerate(variants, 1):
            label = "original audio" if attenuation is None else f"{attenuation:g} dB denoise"
            if callback:
                callback(index, f"Calibrating multi-speaker audio: {label}...")
            data = transcribe_sample(candidate_path)
            report = score_candidate(data, mapped_turns, attenuation)
            reports.append(report)
            logger.info(
                "Multi-speaker calibration %s: score=%.4f words=%d rates=%s repeats=%.3f",
                report.label,
                report.score,
                report.word_count,
                report.speaker_word_rates,
                report.repeated_ratio,
            )

    selected = choose_candidate(reports)
    logger.info("Multi-speaker calibration selected %s", selected.label)
    if callback:
        callback(len(reports), f"Adaptive denoise selected: {selected.label}")
    return EnhancementCalibration(selected.attenuation_db, tuple(reports))
