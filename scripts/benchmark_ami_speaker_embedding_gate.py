#!/usr/bin/env python3
"""Score an embedding-gated legacy speaker rewrite against AMI word truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_speaker_embedding_gate import (  # noqa: E402
    LEGACY_STAGES,
    EmbeddingExtractor,
    _clip,
)

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
from subforge.core.asr.speaker_embedding_models import (  # noqa: E402
    load_speaker_verification_embedding,
)
from subforge.core.asr.speaker_verification import (  # noqa: E402
    verify_speaker_assignment_proposals,
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


def _clone(data: ASRData) -> ASRData:
    return ASRData(
        [
            ASRDataSeg(
                segment.text,
                segment.start_time,
                segment.end_time,
                speaker_id=segment.speaker_id,
                timestamp_granularity="word",
                timing_source="imported",
            )
            for segment in data.segments
        ],
        granularity="word",
        timing_source="imported",
    )


def _score(words, labels, reference_turns):
    return {
        "all_words": word_speaker_error_rate(words, labels),
        "non_overlap_words": word_speaker_error_rate(
            words,
            labels,
            include=lambda word: not word_overlaps_multiple_speakers(word, reference_turns),
        ),
    }


def _overlap_regions(turns) -> list[tuple[int, int]]:
    events = []
    for turn in turns:
        events.append((turn.start_ms, 1, turn.speaker_id))
        events.append((turn.end_ms, -1, turn.speaker_id))
    events.sort(key=lambda event: (event[0], event[1]))
    active = set()
    overlap_start = None
    regions = []
    for timestamp, delta, speaker in events:
        was_overlap = len(active) >= 2
        if delta < 0:
            active.discard(speaker)
        else:
            active.add(speaker)
        is_overlap = len(active) >= 2
        if not was_overlap and is_overlap:
            overlap_start = timestamp
        elif was_overlap and not is_overlap and overlap_start is not None:
            regions.append((overlap_start, timestamp))
            overlap_start = None
    return regions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meeting", required=True)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--words-dir", required=True, type=Path)
    parser.add_argument("--reference-rttm", required=True, type=Path)
    parser.add_argument("--hypothesis-rttm", required=True, type=Path)
    parser.add_argument("--overlap-rttm", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--confirmation-margin", type=float, default=-0.05)
    parser.add_argument("--verification-model-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    words = load_ami_words(args.words_dir, args.meeting)
    reference_turns = load_rttm(args.reference_rttm, uri=args.uri)[args.uri]
    hypothesis_turns = load_rttm(args.hypothesis_rttm, uri=args.uri)[args.uri]
    overlap_turns = (
        load_rttm(args.overlap_rttm, uri=args.uri)[args.uri]
        if args.overlap_rttm
        else hypothesis_turns
    )
    turns = SpeakerTurns(hypothesis_turns)
    production = _segments(words)
    assign_speakers(production, turns)
    legacy = _clone(production)
    smooth_speaker_assignments(legacy, stages=LEGACY_STAGES)

    waveform, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if waveform.ndim != 1:
        waveform = np.mean(waveform, axis=1, dtype=np.float32)
    extractor = EmbeddingExtractor(args.model, args.device)
    if sample_rate != extractor.sample_rate:
        raise ValueError("AMI audio must use the embedding model sample rate")
    gated = _clone(production)
    confirmation_embedding = (
        load_speaker_verification_embedding(args.verification_model_dir)
        if args.verification_model_dir
        else None
    )
    stats = verify_speaker_assignment_proposals(
        gated,
        read_audio=lambda start_ms, end_ms: _clip(waveform, sample_rate, start_ms, end_ms),
        embedding=extractor,
        confirmation_embedding=confirmation_embedding,
        overlap_regions=_overlap_regions(overlap_turns),
        similarity_margin=args.margin,
        confirmation_similarity_margin=args.confirmation_margin,
    )

    report = {
        "meeting": args.meeting,
        "uri": args.uri,
        "margin": args.margin,
        "confirmation_margin": args.confirmation_margin,
        "proposals": stats.proposals,
        "accepted_blocks": stats.accepted,
        "skipped_overlap": stats.skipped_overlap,
        "skipped_reference": stats.skipped_reference,
        "skipped_consensus": stats.skipped_consensus,
        "production": _score(
            words, [segment.speaker_id for segment in production.segments], reference_turns
        ),
        "legacy": _score(
            words, [segment.speaker_id for segment in legacy.segments], reference_turns
        ),
        "gated": _score(words, [segment.speaker_id for segment in gated.segments], reference_turns),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
