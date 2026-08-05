#!/usr/bin/env python3
"""Measure production speaker assignment against AMI manual word timings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subforge.core.asr.asr_data import ASRData, ASRDataSeg  # noqa: E402
from subforge.core.asr.diarization_benchmark import (  # noqa: E402
    ReferenceWord,
    load_ami_words,
    load_rttm,
    word_overlaps_multiple_speakers,
    word_speaker_error_rate,
)
from subforge.core.asr.speaker_diarization import (  # noqa: E402
    SpeakerTurns,
    assign_speakers,
    smooth_speaker_assignments,
)


def _segments(words: list[ReferenceWord]) -> ASRData:
    return ASRData(
        [
            ASRDataSeg(
                word.text,
                word.start_ms,
                word.end_ms,
                timestamp_granularity="word",
                timing_source="imported",
            )
            for word in words
        ],
        granularity="word",
        timing_source="imported",
    )


def _score(
    words: list[ReferenceWord],
    labels: list[str],
    reference_turns,
) -> dict[str, object]:
    return {
        "all_words": word_speaker_error_rate(words, labels),
        "non_overlap_words": word_speaker_error_rate(
            words,
            labels,
            include=lambda word: not word_overlaps_multiple_speakers(word, reference_turns),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meeting", required=True, help="AMI meeting ID, for example IB4010")
    parser.add_argument("--words-dir", required=True, type=Path)
    parser.add_argument("--reference-rttm", required=True, type=Path)
    parser.add_argument("--hypothesis-rttm", required=True, type=Path)
    parser.add_argument("--uri", required=True, help="RTTM recording URI")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    words = load_ami_words(args.words_dir, args.meeting)
    reference_turns = load_rttm(args.reference_rttm, uri=args.uri)[args.uri]
    hypothesis_turns = load_rttm(args.hypothesis_rttm, uri=args.uri)[args.uri]
    overlap_regions = []
    events: list[tuple[int, int]] = []
    for turn in reference_turns:
        events.extend(((turn.start_ms, 1), (turn.end_ms, -1)))
    active = 0
    overlap_start: int | None = None
    for timestamp, delta in sorted(events, key=lambda event: (event[0], event[1])):
        was_overlapping = active >= 2
        active += delta
        is_overlapping = active >= 2
        if not was_overlapping and is_overlapping:
            overlap_start = timestamp
        elif was_overlapping and not is_overlapping and overlap_start is not None:
            overlap_regions.append((overlap_start, timestamp))
            overlap_start = None
    turns = SpeakerTurns(hypothesis_turns, overlap_regions=overlap_regions)

    raw = _segments(words)
    assign_speakers(raw, turns, smooth=False)
    raw_labels = [segment.speaker_id for segment in raw.segments]
    production = _segments(words)
    assign_speakers(production, turns)
    production_labels = [segment.speaker_id for segment in production.segments]
    profiles = {
        "fill_only": frozenset({"fill_blanks"}),
        "islands_only": frozenset({"suppress_islands"}),
        "acoustic_cleanup": frozenset({"fill_blanks", "suppress_islands"}),
        "cleanup_plus_prefix": frozenset(
            {"fill_blanks", "suppress_islands", "move_prefix"}
        ),
        "cleanup_plus_subject": frozenset(
            {"fill_blanks", "suppress_islands", "move_subject"}
        ),
        "cleanup_plus_continuations": frozenset(
            {"fill_blanks", "suppress_islands", "snap_continuations"}
        ),
        "full": frozenset(
            {
                "fill_blanks",
                "suppress_islands",
                "move_prefix",
                "move_subject",
                "snap_continuations",
            }
        ),
    }
    profile_reports = {}
    for name, stages in profiles.items():
        candidate = _segments(words)
        for segment, label in zip(candidate.segments, raw_labels):
            segment.speaker_id = label
        smooth_speaker_assignments(
            candidate,
            overlap_regions=overlap_regions,
            stages=stages,
        )
        candidate_labels = [segment.speaker_id for segment in candidate.segments]
        profile_reports[name] = {
            "changed_labels": sum(
                left != right for left, right in zip(raw_labels, candidate_labels)
            ),
            **_score(words, candidate_labels, reference_turns),
        }
    report = {
        "meeting": args.meeting,
        "uri": args.uri,
        "reference_words": len(words),
        "overlap_words": sum(
            word_overlaps_multiple_speakers(word, reference_turns) for word in words
        ),
        "raw": _score(words, raw_labels, reference_turns),
        "production": {
            "changed_labels": sum(
                left != right for left, right in zip(raw_labels, production_labels)
            ),
            **_score(words, production_labels, reference_turns),
        },
        "profiles": profile_reports,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
