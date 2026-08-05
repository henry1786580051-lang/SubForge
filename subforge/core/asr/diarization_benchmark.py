"""Dependency-light diagnostics shared by speaker-diarization benchmarks."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence

from subforge.core.asr.speaker_diarization import SpeakerTurn


@dataclass(frozen=True)
class TurnDiagnostics:
    speakers: int
    turns: int
    duration_ms: int
    active_speech_ms: int
    overlap_ms: int
    switches: int
    short_turns_500ms: int
    short_turns_1000ms: int
    short_islands_1500ms: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ReferenceWord:
    text: str
    start_ms: int
    end_ms: int
    speaker_id: str


def load_ami_words(directory: str | Path, meeting_id: str) -> list[ReferenceWord]:
    """Load AMI manual word timings for one meeting from NXT XML files."""
    root = Path(directory)
    paths = sorted(root.glob(f"{meeting_id}.*.words.xml"))
    if not paths:
        raise ValueError(f"No AMI word annotations found for {meeting_id!r}")
    words: list[ReferenceWord] = []
    for path in paths:
        parts = path.name.split(".")
        if len(parts) < 4:
            raise ValueError(f"Cannot infer AMI speaker from {path.name!r}")
        speaker_id = parts[1]
        for element in ET.parse(path).getroot():
            if element.tag.rsplit("}", 1)[-1] != "w":
                continue
            text = (element.text or "").strip()
            if element.get("punc") == "true":
                if text and words and words[-1].speaker_id == speaker_id:
                    previous = words[-1]
                    words[-1] = ReferenceWord(
                        previous.text + text,
                        previous.start_ms,
                        previous.end_ms,
                        previous.speaker_id,
                    )
                continue
            try:
                start_ms = round(float(element.attrib["starttime"]) * 1000)
                end_ms = round(float(element.attrib["endtime"]) * 1000)
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid AMI word timing in {path.name}") from exc
            if text and end_ms > start_ms:
                words.append(ReferenceWord(text, start_ms, end_ms, speaker_id))
    words.sort(key=lambda word: (word.start_ms, word.end_ms, word.speaker_id))
    if not words:
        raise ValueError(f"AMI annotations contain no timed words for {meeting_id!r}")
    return words


def word_overlaps_multiple_speakers(word: ReferenceWord, turns: Sequence[SpeakerTurn]) -> bool:
    """Return whether the word midpoint falls in simultaneous reference speech."""
    midpoint = (word.start_ms + word.end_ms) // 2
    active = {
        turn.speaker_id for turn in turns if turn.start_ms <= midpoint < turn.end_ms
    }
    return len(active) > 1


def word_speaker_error_rate(
    words: Sequence[ReferenceWord],
    predicted_labels: Sequence[str],
    *,
    include: Callable[[ReferenceWord], bool] | None = None,
) -> dict[str, object]:
    """Score speaker-attributed words after optimal anonymous-label mapping."""
    if len(words) != len(predicted_labels):
        raise ValueError("words and predicted_labels must have equal length")
    selected = [
        (word.speaker_id, label)
        for word, label in zip(words, predicted_labels)
        if include is None or include(word)
    ]
    references = sorted({reference for reference, _ in selected})
    hypotheses = sorted({hypothesis for _, hypothesis in selected if hypothesis})
    ref_index = {label: index for index, label in enumerate(references)}
    hyp_index = {label: index for index, label in enumerate(hypotheses)}
    counts = [[0 for _ in references] for _ in hypotheses]
    for reference, hypothesis in selected:
        if hypothesis:
            counts[hyp_index[hypothesis]][ref_index[reference]] += 1

    @lru_cache(maxsize=None)
    def best(hypothesis_index: int, used_references: int) -> tuple[int, tuple[int, ...]]:
        if hypothesis_index == len(hypotheses):
            return 0, ()
        score, assignment = best(hypothesis_index + 1, used_references)
        best_result = (score, (-1, *assignment))
        for reference_index in range(len(references)):
            bit = 1 << reference_index
            if used_references & bit:
                continue
            tail_score, tail_assignment = best(hypothesis_index + 1, used_references | bit)
            candidate = (
                counts[hypothesis_index][reference_index] + tail_score,
                (reference_index, *tail_assignment),
            )
            if candidate[0] > best_result[0]:
                best_result = candidate
        return best_result

    correct, assignment = best(0, 0)
    mapping = {
        hypothesis: references[reference_index]
        for hypothesis, reference_index in zip(hypotheses, assignment)
        if reference_index >= 0
    }
    total = len(selected)
    unassigned = sum(not hypothesis for _, hypothesis in selected)
    return {
        "words": total,
        "correct": correct,
        "errors": total - correct,
        "error_rate": (total - correct) / total if total else 0.0,
        "unassigned": unassigned,
        "mapping": mapping,
    }


def load_rttm(path: str | Path, *, uri: str | None = None) -> dict[str, list[SpeakerTurn]]:
    """Load SPEAKER records without requiring pyannote at import time."""
    recordings: dict[str, list[SpeakerTurn]] = defaultdict(list)
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8 or fields[0].upper() != "SPEAKER":
            raise ValueError(f"Invalid RTTM record at line {line_number}")
        recording = fields[1]
        if uri is not None and recording != uri:
            continue
        try:
            start_ms = round(float(fields[3]) * 1000)
            duration_ms = round(float(fields[4]) * 1000)
        except ValueError as exc:
            raise ValueError(f"Invalid RTTM timing at line {line_number}") from exc
        if start_ms < 0 or duration_ms <= 0 or not fields[7]:
            raise ValueError(f"Invalid RTTM interval at line {line_number}")
        recordings[recording].append(SpeakerTurn(start_ms, start_ms + duration_ms, fields[7]))
    for turns in recordings.values():
        turns.sort(key=lambda turn: (turn.start_ms, turn.end_ms, turn.speaker_id))
    if uri is not None and uri not in recordings:
        raise ValueError(f"RTTM does not contain recording {uri!r}")
    if not recordings:
        raise ValueError("RTTM contains no SPEAKER records")
    return dict(recordings)


def write_rttm(path: str | Path, recordings: dict[str, Sequence[SpeakerTurn]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for uri in sorted(recordings):
        for turn in sorted(recordings[uri], key=lambda item: (item.start_ms, item.end_ms)):
            start = turn.start_ms / 1000
            duration = (turn.end_ms - turn.start_ms) / 1000
            speaker_id = re.sub(r"\s+", "_", turn.speaker_id.strip())
            if not speaker_id:
                raise ValueError("RTTM speaker label must not be empty")
            lines.append(
                f"SPEAKER {uri} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>"
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _union_duration(intervals: Iterable[tuple[int, int]]) -> int:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _overlap_duration(turns: Sequence[SpeakerTurn]) -> int:
    events: list[tuple[int, int, str]] = []
    for turn in turns:
        events.extend(((turn.start_ms, 1, turn.speaker_id), (turn.end_ms, -1, turn.speaker_id)))
    events.sort(key=lambda event: (event[0], event[1]))
    active: dict[str, int] = {}
    previous_time: int | None = None
    overlap = 0
    for timestamp, delta, speaker in events:
        if previous_time is not None and len(active) >= 2:
            overlap += max(0, timestamp - previous_time)
        if delta < 0:
            remaining = active.get(speaker, 0) - 1
            if remaining > 0:
                active[speaker] = remaining
            else:
                active.pop(speaker, None)
        else:
            active[speaker] = active.get(speaker, 0) + 1
        previous_time = timestamp
    return overlap


def diagnose_turns(turns: Sequence[SpeakerTurn]) -> TurnDiagnostics:
    """Summarize label fragmentation without pretending it is ground-truth accuracy."""
    ordered = sorted(turns, key=lambda turn: (turn.start_ms, turn.end_ms))
    if not ordered:
        return TurnDiagnostics(0, 0, 0, 0, 0, 0, 0, 0, 0)
    switches = sum(left.speaker_id != right.speaker_id for left, right in zip(ordered, ordered[1:]))
    short_islands = 0
    for index in range(1, len(ordered) - 1):
        previous, current, following = ordered[index - 1 : index + 2]
        if (
            current.end_ms - current.start_ms <= 1500
            and previous.speaker_id == following.speaker_id != current.speaker_id
            and current.start_ms - previous.end_ms <= 300
            and following.start_ms - current.end_ms <= 300
        ):
            short_islands += 1
    return TurnDiagnostics(
        speakers=len({turn.speaker_id for turn in ordered}),
        turns=len(ordered),
        duration_ms=max(turn.end_ms for turn in ordered) - min(turn.start_ms for turn in ordered),
        active_speech_ms=_union_duration((turn.start_ms, turn.end_ms) for turn in ordered),
        overlap_ms=_overlap_duration(ordered),
        switches=switches,
        short_turns_500ms=sum(turn.end_ms - turn.start_ms <= 500 for turn in ordered),
        short_turns_1000ms=sum(turn.end_ms - turn.start_ms <= 1000 for turn in ordered),
        short_islands_1500ms=short_islands,
    )


def boundary_f1(
    reference: Sequence[SpeakerTurn],
    hypothesis: Sequence[SpeakerTurn],
    *,
    tolerance_ms: int,
) -> dict[str, float | int]:
    """Score change-point detection with one-to-one greedy temporal matching."""
    if tolerance_ms < 0:
        raise ValueError("tolerance_ms must be non-negative")

    def boundaries(turns: Sequence[SpeakerTurn]) -> list[int]:
        ordered = sorted(turns, key=lambda turn: (turn.start_ms, turn.end_ms))
        return [
            right.start_ms
            for left, right in zip(ordered, ordered[1:])
            if left.speaker_id != right.speaker_id
        ]

    expected = boundaries(reference)
    predicted = boundaries(hypothesis)
    candidates = sorted(
        (
            (abs(actual - estimate), expected_index, predicted_index)
            for expected_index, actual in enumerate(expected)
            for predicted_index, estimate in enumerate(predicted)
            if abs(actual - estimate) <= tolerance_ms
        ),
        key=lambda item: item[0],
    )
    matched_expected: set[int] = set()
    matched_predicted: set[int] = set()
    for _, expected_index, predicted_index in candidates:
        if expected_index in matched_expected or predicted_index in matched_predicted:
            continue
        matched_expected.add(expected_index)
        matched_predicted.add(predicted_index)
    true_positive = len(matched_expected)
    precision = true_positive / len(predicted) if predicted else float(not expected)
    recall = true_positive / len(expected) if expected else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tolerance_ms": tolerance_ms,
        "reference_boundaries": len(expected),
        "hypothesis_boundaries": len(predicted),
        "matched": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
