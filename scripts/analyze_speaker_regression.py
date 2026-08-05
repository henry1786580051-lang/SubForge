#!/usr/bin/env python3
"""Compare speaker labels while holding ASR text and timestamps constant."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subforge.core.asr.asr_data import ASRData, ASRDataSeg  # noqa: E402
from subforge.core.asr.speaker_diarization import smooth_speaker_assignments  # noqa: E402

LEGACY_STAGES = frozenset(
    {
        "fill_blanks",
        "suppress_islands",
        "move_prefix",
        "move_subject",
        "snap_continuations",
    }
)


def _load(path: Path) -> ASRData:
    return ASRData.from_srt(path.read_text(encoding="utf-8-sig", errors="replace"))


def _clone(data: ASRData) -> ASRData:
    return ASRData(
        [
            ASRDataSeg(
                segment.text,
                segment.start_time,
                segment.end_time,
                speaker_id=segment.speaker_id,
                timestamp_granularity=segment.timestamp_granularity,
                timing_source=segment.timing_source,
            )
            for segment in data.segments
        ],
        granularity=data.granularity,
        timing_source=data.timing_source,
    )


def _runs(data: ASRData) -> list[tuple[int, int, str]]:
    if not data.segments:
        return []
    result: list[tuple[int, int, str]] = []
    start = 0
    for index in range(1, len(data.segments) + 1):
        if index == len(data.segments) or (
            data.segments[index].speaker_id != data.segments[start].speaker_id
        ):
            result.append((start, index, data.segments[start].speaker_id))
            start = index
    return result


def _short_islands(data: ASRData) -> list[dict[str, object]]:
    runs = _runs(data)
    result = []
    for index in range(1, len(runs) - 1):
        start, end, label = runs[index]
        previous = runs[index - 1][2]
        following = runs[index + 1][2]
        duration = data.segments[end - 1].end_time - data.segments[start].start_time
        if label and previous == following != label and duration <= 1_500:
            result.append(
                {
                    "start_ms": data.segments[start].start_time,
                    "duration_ms": duration,
                    "speaker": label,
                    "text": " ".join(segment.text for segment in data.segments[start:end]),
                }
            )
    return result


def _word_counts(data: ASRData) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for segment in data.segments:
        counts[segment.speaker_id or "unassigned"] += max(
            1, len(re.findall(r"[A-Za-z0-9']+", segment.text))
        )
    return dict(counts)


def compare(baseline: ASRData, candidate: ASRData) -> dict[str, object]:
    same_content = len(baseline.segments) == len(candidate.segments) and all(
        (left.text, left.start_time, left.end_time)
        == (right.text, right.start_time, right.end_time)
        for left, right in zip(baseline.segments, candidate.segments)
    )
    changes = []
    if same_content:
        for index, (left, right) in enumerate(zip(baseline.segments, candidate.segments)):
            if left.speaker_id == right.speaker_id:
                continue
            context_start = max(0, index - 3)
            context_end = min(len(candidate.segments), index + 4)
            changes.append(
                {
                    "index": index + 1,
                    "start_ms": right.start_time,
                    "text": right.text,
                    "from": left.speaker_id or "unassigned",
                    "to": right.speaker_id or "unassigned",
                    "context": " ".join(
                        segment.text for segment in candidate.segments[context_start:context_end]
                    ),
                }
            )
    return {
        "text_and_timestamps_identical": same_content,
        "changed_labels": len(changes) if same_content else None,
        "baseline": {
            "runs": len(_runs(baseline)),
            "short_islands": _short_islands(baseline),
            "speaker_words": _word_counts(baseline),
        },
        "candidate": {
            "runs": len(_runs(candidate)),
            "short_islands": _short_islands(candidate),
            "speaker_words": _word_counts(candidate),
        },
        "changes": changes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--simulate-legacy", action="store_true")
    parser.add_argument("--legacy-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.baseline) == bool(args.simulate_legacy):
        parser.error("choose exactly one of --baseline or --simulate-legacy")

    candidate = _load(args.candidate)
    if args.baseline:
        baseline = _load(args.baseline)
        report = compare(baseline, candidate)
        report["comparison"] = "saved baseline -> conservative production"
    else:
        legacy = _clone(candidate)
        smooth_speaker_assignments(legacy, stages=LEGACY_STAGES)
        if args.legacy_output:
            args.legacy_output.parent.mkdir(parents=True, exist_ok=True)
            args.legacy_output.write_text(legacy.to_srt(), encoding="utf-8")
        report = compare(legacy, candidate)
        report["comparison"] = "simulated legacy smoothing -> conservative production"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
