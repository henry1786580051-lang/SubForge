#!/usr/bin/env python3
"""Audit proposed speaker-label rewrites with local speaker embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subforge.core.asr.asr_data import ASRData  # noqa: E402
from subforge.core.asr.speaker_embedding_models import (  # noqa: E402
    load_speaker_verification_embedding,
)
from subforge.core.asr.speaker_verification import (  # noqa: E402
    DUAL_MODEL_RESCUE_CONFIRMATION_MARGIN,
    DUAL_MODEL_RESCUE_PRIMARY_MARGIN,
)

LEGACY_STAGES = frozenset(
    {
        "fill_blanks",
        "suppress_islands",
        "move_prefix",
        "move_subject",
        "snap_continuations",
    }
)


def _load_srt(path: Path) -> ASRData:
    return ASRData.from_srt(path.read_text(encoding="utf-8-sig", errors="replace"))


def _runs(data: ASRData) -> list[tuple[int, int, str]]:
    if not data.segments:
        return []
    result = []
    start = 0
    for index in range(1, len(data.segments) + 1):
        if index == len(data.segments) or (
            data.segments[index].speaker_id != data.segments[start].speaker_id
        ):
            result.append((start, index, data.segments[start].speaker_id))
            start = index
    return result


def _changed_blocks(baseline: ASRData, candidate: ASRData) -> list[tuple[int, int, str, str]]:
    if len(baseline.segments) != len(candidate.segments):
        raise ValueError("baseline and candidate lengths differ")
    result = []
    start = 0
    while start < len(candidate.segments):
        proposed = baseline.segments[start].speaker_id
        current = candidate.segments[start].speaker_id
        if proposed == current:
            start += 1
            continue
        end = start + 1
        while (
            end < len(candidate.segments)
            and baseline.segments[end].speaker_id == proposed
            and candidate.segments[end].speaker_id == current
        ):
            end += 1
        result.append((start, end, current, proposed))
        start = end
    return result


class EmbeddingExtractor:
    def __init__(self, model_dir: Path, device: str):
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(str(model_dir))
        if pipeline is None:
            raise RuntimeError("Unable to load Community-1")
        self.embedding = pipeline._embedding
        self.embedding.to(torch.device(device))
        self.sample_rate = int(self.embedding.sample_rate)

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        minimum = max(int(self.sample_rate * 0.5), int(self.embedding.min_num_samples))
        if len(waveform) < minimum:
            left = (minimum - len(waveform)) // 2
            right = minimum - len(waveform) - left
            waveform = np.pad(waveform, (left, right))
        batch = torch.from_numpy(waveform.astype(np.float32, copy=False)).reshape(1, 1, -1)
        vector = np.asarray(self.embedding(batch)[0], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise RuntimeError("Speaker embedding is invalid")
        return vector / norm


def _clip(waveform: np.ndarray, sample_rate: int, start_ms: int, end_ms: int) -> np.ndarray:
    start = max(0, round(start_ms * sample_rate / 1000))
    end = min(len(waveform), round(end_ms * sample_rate / 1000))
    return waveform[start:end]


def _centroids(
    data: ASRData,
    waveform: np.ndarray,
    sample_rate: int,
    extractor: EmbeddingExtractor,
) -> dict[str, np.ndarray]:
    candidates: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for start, end, label in _runs(data):
        if not label:
            continue
        start_ms = data.segments[start].start_time
        end_ms = data.segments[end - 1].end_time
        if end_ms - start_ms >= 2_000:
            candidates[label].append((start_ms, end_ms))
    centroids = {}
    for label, intervals in candidates.items():
        vectors = []
        for start_ms, end_ms in sorted(
            intervals, key=lambda interval: interval[1] - interval[0], reverse=True
        )[:12]:
            if end_ms - start_ms > 4_000:
                midpoint = (start_ms + end_ms) // 2
                start_ms, end_ms = midpoint - 2_000, midpoint + 2_000
            vectors.append(extractor(_clip(waveform, sample_rate, start_ms, end_ms)))
        centroid = np.mean(vectors, axis=0)
        centroids[label] = centroid / np.linalg.norm(centroid)
    return centroids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("candidate", type=Path, help="Conservative production SRT")
    parser.add_argument("baseline", type=Path, help="Legacy or proposed-label SRT")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--confirmation-margin", type=float, default=-0.05)
    parser.add_argument("--verification-model-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate = _load_srt(args.candidate)
    baseline = _load_srt(args.baseline)
    waveform, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if waveform.ndim != 1:
        waveform = np.mean(waveform, axis=1, dtype=np.float32)
    extractor = EmbeddingExtractor(args.model, args.device)
    if sample_rate != extractor.sample_rate:
        raise ValueError(
            f"audio sample rate {sample_rate} does not match embedding model {extractor.sample_rate}"
        )
    centroids = _centroids(candidate, waveform, sample_rate, extractor)
    confirmation_extractor = (
        load_speaker_verification_embedding(args.verification_model_dir)
        if args.verification_model_dir
        else None
    )
    confirmation_centroids = (
        _centroids(candidate, waveform, sample_rate, confirmation_extractor)
        if confirmation_extractor is not None
        else {}
    )
    decisions = []
    for start, end, current, proposed in _changed_blocks(baseline, candidate):
        first = candidate.segments[start]
        last = candidate.segments[end - 1]
        if not current or not proposed or current not in centroids or proposed not in centroids:
            decisions.append(
                {
                    "start_ms": first.start_time,
                    "text": " ".join(segment.text for segment in candidate.segments[start:end]),
                    "current": current or "unassigned",
                    "proposed": proposed or "unassigned",
                    "decision": "insufficient_reference",
                }
            )
            continue
        vector = extractor(_clip(waveform, sample_rate, first.start_time, last.end_time))
        current_score = float(vector @ centroids[current])
        proposed_score = float(vector @ centroids[proposed])
        confirmation_margin = None
        confirmation_supported = True
        if confirmation_extractor is not None:
            if current not in confirmation_centroids or proposed not in confirmation_centroids:
                confirmation_supported = False
            else:
                confirmation_vector = confirmation_extractor(
                    _clip(waveform, sample_rate, first.start_time, last.end_time)
                )
                confirmation_margin = float(
                    confirmation_vector @ confirmation_centroids[proposed]
                    - confirmation_vector @ confirmation_centroids[current]
                )
                confirmation_supported = confirmation_margin >= args.confirmation_margin
        primary_margin = proposed_score - current_score
        rescued = (
            confirmation_margin is not None
            and len(centroids) == 2
            and primary_margin >= DUAL_MODEL_RESCUE_PRIMARY_MARGIN
            and confirmation_margin >= DUAL_MODEL_RESCUE_CONFIRMATION_MARGIN
        )
        accepted = (primary_margin >= args.margin and confirmation_supported) or rescued
        decisions.append(
            {
                "start_ms": first.start_time,
                "duration_ms": last.end_time - first.start_time,
                "text": " ".join(segment.text for segment in candidate.segments[start:end]),
                "current": current,
                "proposed": proposed,
                "current_similarity": current_score,
                "proposed_similarity": proposed_score,
                "margin": primary_margin,
                "confirmation_margin": confirmation_margin,
                "decision": "accept_proposed" if accepted else "keep_current",
            }
        )
    report = {
        "device": args.device,
        "threshold_margin": args.margin,
        "confirmation_threshold_margin": args.confirmation_margin,
        "centroid_speakers": sorted(centroids),
        "proposals": len(decisions),
        "accepted": sum(item["decision"] == "accept_proposed" for item in decisions),
        "decisions": decisions,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
