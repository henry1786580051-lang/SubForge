"""Acoustic verification for conservative speaker-label corrections."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from subforge.core.asr.asr_data import ASRData, ASRDataSeg

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_MARGIN = 0.10
DEFAULT_CONFIRMATION_SIMILARITY_MARGIN = -0.05
DUAL_MODEL_RESCUE_PRIMARY_MARGIN = 0.075
DUAL_MODEL_RESCUE_CONFIRMATION_MARGIN = 0.15
REFERENCE_MIN_DURATION_MS = 2_000
REFERENCE_MAX_DURATION_MS = 4_000
REFERENCE_LIMIT_PER_SPEAKER = 12
PROPOSAL_STAGES = frozenset(
    {
        "fill_blanks",
        "suppress_islands",
        "move_prefix",
        "move_subject",
        "snap_continuations",
    }
)


@dataclass(frozen=True)
class SpeakerVerificationStats:
    """Summary of an optional acoustic speaker-label verification pass."""

    proposals: int = 0
    accepted: int = 0
    skipped_overlap: int = 0
    skipped_reference: int = 0
    skipped_consensus: int = 0


def _runs(segments: list[ASRDataSeg]) -> list[tuple[int, int, str]]:
    if not segments:
        return []
    result: list[tuple[int, int, str]] = []
    start = 0
    for index in range(1, len(segments) + 1):
        if index == len(segments) or segments[index].speaker_id != segments[start].speaker_id:
            result.append((start, index, segments[start].speaker_id))
            start = index
    return result


def _intersects(
    start_ms: int,
    end_ms: int,
    regions: Iterable[tuple[int, int]],
) -> bool:
    return any(
        region_start < end_ms and region_end > start_ms for region_start, region_end in regions
    )


def _proposed_labels(asr_data: ASRData) -> list[str]:
    """Run legacy semantic stages as proposals without mutating production data."""
    from subforge.core.asr.speaker_diarization import smooth_speaker_assignments

    original = [segment.speaker_id for segment in asr_data.segments]
    try:
        smooth_speaker_assignments(asr_data, stages=PROPOSAL_STAGES)
        return [segment.speaker_id for segment in asr_data.segments]
    finally:
        for segment, label in zip(asr_data.segments, original):
            segment.speaker_id = label


def _changed_blocks(current: list[str], proposed: list[str]) -> list[tuple[int, int, str, str]]:
    result: list[tuple[int, int, str, str]] = []
    start = 0
    while start < len(current):
        if current[start] == proposed[start]:
            start += 1
            continue
        end = start + 1
        while (
            end < len(current)
            and current[end] == current[start]
            and proposed[end] == proposed[start]
        ):
            end += 1
        result.append((start, end, current[start], proposed[start]))
        start = end
    return result


def _speaker_centroids(
    segments: list[ASRDataSeg],
    read_audio: Callable[[int, int], Any],
    embedding: Callable[[Any], Any],
    overlap_regions: tuple[tuple[int, int], ...],
    *,
    robust: bool = False,
) -> dict[str, Any]:
    import numpy as np

    candidates: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for start, end, label in _runs(segments):
        if not label:
            continue
        start_ms = segments[start].start_time
        end_ms = segments[end - 1].end_time
        if end_ms - start_ms < REFERENCE_MIN_DURATION_MS:
            continue
        if _intersects(start_ms, end_ms, overlap_regions):
            continue
        candidates[label].append((start_ms, end_ms))

    centroids: dict[str, Any] = {}
    for label, intervals in candidates.items():
        vectors = []
        for start_ms, end_ms in sorted(
            intervals,
            key=lambda interval: interval[1] - interval[0],
            reverse=True,
        )[:REFERENCE_LIMIT_PER_SPEAKER]:
            if end_ms - start_ms > REFERENCE_MAX_DURATION_MS:
                midpoint = (start_ms + end_ms) // 2
                start_ms = midpoint - REFERENCE_MAX_DURATION_MS // 2
                end_ms = midpoint + REFERENCE_MAX_DURATION_MS // 2
            vectors.append(embedding(read_audio(start_ms, end_ms)))
        if not vectors:
            continue
        if robust and len(vectors) >= 4:
            initial = np.mean(vectors, axis=0)
            initial_norm = float(np.linalg.norm(initial))
            if np.isfinite(initial_norm) and initial_norm > 0:
                initial /= initial_norm
                keep = max(3, (len(vectors) * 3 + 3) // 4)
                vectors = sorted(
                    vectors,
                    key=lambda vector: float(np.dot(vector, initial)),
                    reverse=True,
                )[:keep]
        centroid = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(centroid))
        if np.isfinite(norm) and norm > 0:
            centroids[label] = centroid / norm
    return centroids


def verify_speaker_assignment_proposals(
    asr_data: ASRData,
    *,
    read_audio: Callable[[int, int], Any],
    embedding: Callable[[Any], Any],
    confirmation_embedding: Callable[[Any], Any] | None = None,
    overlap_regions: Iterable[tuple[int, int]] = (),
    similarity_margin: float = DEFAULT_SIMILARITY_MARGIN,
    confirmation_similarity_margin: float = DEFAULT_CONFIRMATION_SIMILARITY_MARGIN,
) -> SpeakerVerificationStats:
    """Apply only semantic label proposals supported by stronger acoustic evidence."""
    import numpy as np

    segments = asr_data.segments
    if len(segments) < 2:
        return SpeakerVerificationStats()
    current = [segment.speaker_id for segment in segments]
    speaker_count = len({label for label in current if label})
    if speaker_count < 2:
        return SpeakerVerificationStats()

    regions = tuple(overlap_regions)
    proposed = _proposed_labels(asr_data)
    blocks = _changed_blocks(current, proposed)
    if not blocks:
        return SpeakerVerificationStats()

    centroids = _speaker_centroids(segments, read_audio, embedding, regions)
    confirmation_centroids = (
        _speaker_centroids(
            segments,
            read_audio,
            confirmation_embedding,
            regions,
            robust=True,
        )
        if confirmation_embedding is not None
        else {}
    )
    accepted = 0
    skipped_overlap = 0
    skipped_reference = 0
    skipped_consensus = 0
    for start, end, current_label, proposed_label in blocks:
        start_ms = segments[start].start_time
        end_ms = segments[end - 1].end_time
        if _intersects(start_ms, end_ms, regions):
            skipped_overlap += 1
            continue
        if (
            not current_label
            or not proposed_label
            or current_label not in centroids
            or proposed_label not in centroids
        ):
            skipped_reference += 1
            continue
        samples = read_audio(start_ms, end_ms)
        vector = embedding(samples)
        current_score = float(np.dot(vector, centroids[current_label]))
        proposed_score = float(np.dot(vector, centroids[proposed_label]))
        primary_delta = proposed_score - current_score
        if confirmation_embedding is None and primary_delta < similarity_margin:
            continue
        if confirmation_embedding is not None:
            if (
                current_label not in confirmation_centroids
                or proposed_label not in confirmation_centroids
            ):
                skipped_reference += 1
                continue
            confirmation_vector = confirmation_embedding(samples)
            confirmation_current = float(
                np.dot(confirmation_vector, confirmation_centroids[current_label])
            )
            confirmation_proposed = float(
                np.dot(confirmation_vector, confirmation_centroids[proposed_label])
            )
            confirmation_delta = confirmation_proposed - confirmation_current
            standard_supported = (
                primary_delta >= similarity_margin
                and confirmation_delta >= confirmation_similarity_margin
            )
            consensus_rescue = (
                speaker_count == 2
                and primary_delta >= DUAL_MODEL_RESCUE_PRIMARY_MARGIN
                and confirmation_delta >= DUAL_MODEL_RESCUE_CONFIRMATION_MARGIN
            )
            if not standard_supported and not consensus_rescue:
                skipped_consensus += 1
                continue
        for segment in segments[start:end]:
            segment.speaker_id = proposed_label
        accepted += 1

    return SpeakerVerificationStats(
        proposals=len(blocks),
        accepted=accepted,
        skipped_overlap=skipped_overlap,
        skipped_reference=skipped_reference,
        skipped_consensus=skipped_consensus,
    )


def verify_speakers_with_pipeline(
    asr_data: ASRData,
    audio_path: str,
    *,
    pipeline: Any,
    device: str,
    model_dir: str | None = None,
    overlap_regions: Iterable[tuple[int, int]] = (),
) -> SpeakerVerificationStats:
    """Run acoustic verification with an already acquired Community-1 pipeline."""
    import numpy as np
    import soundfile as sf
    import torch

    embedding_model = getattr(pipeline, "_embedding", None)
    if embedding_model is None:
        raise RuntimeError("Community-1 embedding verifier is unavailable")
    embedding_model.to(torch.device(device))
    expected_rate = int(embedding_model.sample_rate)

    with sf.SoundFile(audio_path) as audio_file:
        sample_rate = int(audio_file.samplerate)
        if len(audio_file) <= 0:
            return SpeakerVerificationStats()
        if sample_rate != expected_rate:
            raise RuntimeError(
                f"Audio is {sample_rate} Hz, but the speaker model expects {expected_rate} Hz"
            )

        def _read_audio(start_ms: int, end_ms: int):
            start = max(0, round(start_ms * sample_rate / 1000))
            end = min(len(audio_file), round(end_ms * sample_rate / 1000))
            audio_file.seek(start)
            samples = audio_file.read(
                max(0, end - start),
                dtype="float32",
                always_2d=True,
            )
            return np.mean(samples, axis=1, dtype=np.float32)

        def _embedding(samples):
            minimum = max(
                int(expected_rate * 0.5),
                int(embedding_model.min_num_samples),
            )
            if len(samples) < minimum:
                left = (minimum - len(samples)) // 2
                samples = np.pad(samples, (left, minimum - len(samples) - left))
            batch = torch.from_numpy(samples.astype(np.float32, copy=False)).reshape(1, 1, -1)
            vector = np.asarray(embedding_model(batch)[0], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm <= 0:
                raise RuntimeError("Speaker embedding is invalid")
            return vector / norm

        confirmation_embedding = None
        if model_dir:
            from subforge.core.asr.speaker_embedding_models import (
                is_speaker_verification_model_ready,
                load_speaker_verification_embedding,
            )

            if is_speaker_verification_model_ready(model_dir):
                confirmation_embedding = load_speaker_verification_embedding(model_dir)
            else:
                logger.info(
                    "WeSpeaker ECAPA-TDNN512-LM is not installed; using Community-1 verification only"
                )

        return verify_speaker_assignment_proposals(
            asr_data,
            read_audio=_read_audio,
            embedding=_embedding,
            confirmation_embedding=confirmation_embedding,
            overlap_regions=overlap_regions,
        )
