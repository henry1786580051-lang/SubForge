"""Speaker diarization and conservative speaker assignment for ASR results."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from subforge.core.asr.asr_data import ASRData

logger = logging.getLogger(__name__)

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
LOCAL_DIARIZATION_DIR = "pyannote-speaker-diarization-community-1"
DIARIZATION_CACHE_VERSION = 1


@dataclass(frozen=True)
class SpeakerTurn:
    """A single speaker-active interval in milliseconds."""

    start_ms: int
    end_ms: int
    speaker_id: str


def is_diarization_model_dir(path: str | Path) -> bool:
    """Return whether a local pyannote pipeline snapshot is usable."""
    model_dir = Path(path).expanduser()
    return model_dir.is_dir() and (model_dir / "config.yaml").is_file()


def resolve_diarization_model(model: str, model_dir: str | Path | None = None) -> str:
    """Prefer the managed offline model snapshot over the Hub repository."""
    configured = str(model or DEFAULT_DIARIZATION_MODEL).strip()
    configured_path = Path(configured).expanduser()
    if is_diarization_model_dir(configured_path):
        return str(configured_path)
    if model_dir:
        local = Path(model_dir).expanduser() / LOCAL_DIARIZATION_DIR
        if is_diarization_model_dir(local):
            return str(local)
    return configured or DEFAULT_DIARIZATION_MODEL


def require_local_diarization_model(model: str, model_dir: str | Path | None = None) -> str:
    """Resolve a local model or fail before the expensive ASR stage starts."""
    resolved = resolve_diarization_model(model, model_dir)
    if not is_diarization_model_dir(resolved):
        raise RuntimeError(
            "Speaker diarization is enabled but Community-1 is not downloaded. "
            "Download the speaker model in WhisperX settings before transcription."
        )
    return resolved


def _load_waveform(audio_path: str):
    """Load audio in memory to avoid torchcodec/FFmpeg ABI dependencies."""
    import numpy as np
    import soundfile as sf
    import torch

    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    if waveform.size == 0:
        raise RuntimeError("Speaker diarization received an empty audio file")
    mono = np.mean(waveform, axis=1, dtype=np.float32)
    return {
        "waveform": torch.from_numpy(mono).unsqueeze(0),
        "sample_rate": int(sample_rate),
    }


def _iter_labeled_turns(annotation) -> Iterable[tuple[float, float, str]]:
    if annotation is None:
        return
    if hasattr(annotation, "itertracks"):
        for segment, _, label in annotation.itertracks(yield_label=True):
            yield float(segment.start), float(segment.end), str(label)
        return
    for item in annotation:
        if len(item) == 2:
            segment, label = item
            yield float(segment.start), float(segment.end), str(label)


def _audio_content_hash(audio_path: str) -> str:
    digest = hashlib.sha256()
    with open(audio_path, "rb") as audio_file:
        for chunk in iter(lambda: audio_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _diarization_cache_key(
    audio_path: str,
    resolved_model: str,
    num_speakers: int | None,
) -> str:
    from subforge.core.utils.cache import generate_cache_key

    model_config = Path(resolved_model) / "config.yaml"
    model_signature = ""
    if model_config.is_file():
        stat = model_config.stat()
        model_signature = f"{stat.st_size}:{stat.st_mtime_ns}"
    return generate_cache_key(
        {
            "version": DIARIZATION_CACHE_VERSION,
            "audio_sha256": _audio_content_hash(audio_path),
            "model": str(Path(resolved_model).resolve()),
            "model_signature": model_signature,
            "num_speakers": num_speakers,
        }
    )


def _deserialize_cached_turns(value: Any) -> list[SpeakerTurn]:
    if not isinstance(value, list):
        return []
    turns: list[SpeakerTurn] = []
    for item in value:
        if not isinstance(item, dict):
            return []
        try:
            turn = SpeakerTurn(
                int(item["start_ms"]),
                int(item["end_ms"]),
                str(item["speaker_id"]),
            )
        except (KeyError, TypeError, ValueError):
            return []
        if turn.end_ms <= turn.start_ms or not turn.speaker_id:
            return []
        turns.append(turn)
    return turns


def diarize_audio(
    audio_path: str,
    *,
    model: str = DEFAULT_DIARIZATION_MODEL,
    token: str = "",
    model_dir: str | Path | None = None,
    num_speakers: int | None = 2,
    callback: Callable[[int, str], None] | None = None,
) -> list[SpeakerTurn]:
    """Run pyannote on the original audio and return first-appearance labels."""
    resolved_model = require_local_diarization_model(model, model_dir)
    cache_key = _diarization_cache_key(audio_path, resolved_model, num_speakers)
    try:
        from subforge.core.utils.cache import get_diarization_cache, is_cache_enabled

        if is_cache_enabled():
            cached_turns = _deserialize_cached_turns(
                get_diarization_cache().get(cache_key)
            )
            if cached_turns:
                logger.info("Using cached speaker diarization (%d turns)", len(cached_turns))
                if callback:
                    callback(94, "Using cached speaker analysis...")
                return cached_turns
    except Exception as exc:
        logger.debug("Speaker diarization cache lookup failed: %s", exc)

    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Speaker diarization runtime is unavailable. Install the WhisperX "
            "dependencies that include pyannote.audio."
        ) from exc

    if callback:
        callback(92, "Loading speaker diarization model...")
    try:
        pipeline = Pipeline.from_pretrained(
            resolved_model,
            token=token or None,
            cache_dir=str(model_dir) if model_dir else None,
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to load the speaker diarization model. Verify the local model or "
            "Hugging Face token and Community-1 access."
        ) from exc
    if pipeline is None:
        raise RuntimeError("Speaker diarization model could not be loaded")

    # pyannote is reliable on CPU in packaged macOS builds. MPS support varies
    # by model/operator and must not make the existing MLX transcription fail.
    pipeline.to(torch.device("cpu"))
    audio = _load_waveform(audio_path)
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    if callback:
        callback(94, "Identifying speakers from the original audio...")
    pipeline_call: Any = pipeline
    output = pipeline_call(audio, **kwargs)
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", output)

    label_map: dict[str, str] = {}
    turns: list[SpeakerTurn] = []
    for start, end, raw_label in _iter_labeled_turns(annotation):
        if end <= start:
            continue
        label_map.setdefault(raw_label, f"Speaker {len(label_map) + 1}")
        turns.append(
            SpeakerTurn(
                start_ms=max(0, round(start * 1000)),
                end_ms=max(1, round(end * 1000)),
                speaker_id=label_map[raw_label],
            )
        )
    turns.sort(key=lambda turn: (turn.start_ms, turn.end_ms))
    if not turns:
        raise RuntimeError("Speaker diarization produced no speaker turns")
    try:
        from subforge.core.utils.cache import get_diarization_cache, is_cache_enabled

        if is_cache_enabled():
            get_diarization_cache().set(
                cache_key,
                [
                    {
                        "start_ms": turn.start_ms,
                        "end_ms": turn.end_ms,
                        "speaker_id": turn.speaker_id,
                    }
                    for turn in turns
                ],
                expire=90 * 24 * 60 * 60,
            )
    except Exception as exc:
        logger.debug("Speaker diarization cache write failed: %s", exc)
    logger.info("Speaker diarization found %d speakers in %d turns", len(label_map), len(turns))
    return turns


def assign_speakers(
    asr_data: ASRData,
    turns: list[SpeakerTurn],
    *,
    nearest_gap_ms: int = 120,
    suppress_flip_ms: int = 300,
) -> ASRData:
    """Attach speaker labels without changing ASR text or timestamps."""
    if not asr_data.segments or not turns:
        return asr_data

    turn_cursor = 0
    labels: list[str] = []
    for segment in asr_data.segments:
        while turn_cursor < len(turns) and turns[turn_cursor].end_ms <= segment.start_time:
            turn_cursor += 1

        best_label = ""
        best_overlap = 0
        index = max(0, turn_cursor - 1)
        while index < len(turns) and turns[index].start_ms < segment.end_time:
            turn = turns[index]
            overlap = min(segment.end_time, turn.end_ms) - max(segment.start_time, turn.start_ms)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = turn.speaker_id
            index += 1

        if not best_label:
            midpoint = (segment.start_time + segment.end_time) // 2
            candidates = turns[max(0, turn_cursor - 1) : min(len(turns), turn_cursor + 2)]
            nearest = min(
                candidates,
                key=lambda turn: min(abs(midpoint - turn.start_ms), abs(midpoint - turn.end_ms)),
                default=None,
            )
            if nearest is not None:
                gap = max(nearest.start_ms - midpoint, midpoint - nearest.end_ms, 0)
                if gap <= nearest_gap_ms:
                    best_label = nearest.speaker_id
        labels.append(best_label)

    for segment, label in zip(asr_data.segments, labels):
        segment.speaker_id = label
    return smooth_speaker_assignments(
        asr_data,
        nearest_gap_ms=nearest_gap_ms,
        suppress_flip_ms=suppress_flip_ms,
    )


_BOUNDARY_PREFIX_WORDS = {
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "he'd",
    "he's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "is",
    "it'd",
    "it'll",
    "it's",
    "she'd",
    "she's",
    "should",
    "that's",
    "there's",
    "they're",
    "was",
    "we'd",
    "we'll",
    "we're",
    "were",
    "will",
    "would",
    "you'd",
    "you'll",
    "you're",
    "you've",
}


def _normalized_word(text: str) -> str:
    words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text.lower())
    return words[-1].replace("’", "'") if words else ""


def smooth_speaker_assignments(
    asr_data: ASRData,
    *,
    nearest_gap_ms: int = 120,
    suppress_flip_ms: int = 300,
) -> ASRData:
    """Conservatively remove word-level diarization boundary jitter.

    This changes speaker metadata only. Long turns and uncertain gaps remain
    untouched; short blanks, isolated label islands, and auxiliaries stranded
    before a continuous lowercase phrase are repaired.
    """
    segments = asr_data.segments
    if len(segments) < 2 or not any(segment.speaker_id for segment in segments):
        return asr_data

    labels = [segment.speaker_id for segment in segments]
    max_fill_duration = max(700, suppress_flip_ms * 2)
    max_edge_gap = max(250, nearest_gap_ms * 2)

    index = 0
    while index < len(labels):
        if labels[index]:
            index += 1
            continue
        end = index + 1
        while end < len(labels) and not labels[end]:
            end += 1
        previous = labels[index - 1] if index > 0 else ""
        following = labels[end] if end < len(labels) else ""
        duration = segments[end - 1].end_time - segments[index].start_time
        candidate = previous if previous and previous == following else ""
        if not candidate and duration <= max_fill_duration:
            if previous and not following:
                gap = segments[index].start_time - segments[index - 1].end_time
                if gap <= max_edge_gap:
                    candidate = previous
            elif following and not previous:
                gap = segments[end].start_time - segments[end - 1].end_time
                if gap <= max_edge_gap:
                    candidate = following
        if candidate and duration <= max_fill_duration:
            labels[index:end] = [candidate] * (end - index)
        index = end

    changed = True
    while changed:
        changed = False
        runs: list[tuple[int, int, str]] = []
        start = 0
        for index in range(1, len(labels) + 1):
            if index == len(labels) or labels[index] != labels[start]:
                runs.append((start, index, labels[start]))
                start = index
        for run_index in range(1, len(runs) - 1):
            start, end, label = runs[run_index]
            previous = runs[run_index - 1][2]
            following = runs[run_index + 1][2]
            duration = segments[end - 1].end_time - segments[start].start_time
            if (
                label
                and previous
                and previous == following
                and label != previous
                and end - start <= 3
                and duration <= max(700, suppress_flip_ms)
                and segments[start].start_time - segments[start - 1].end_time <= max_edge_gap
                and segments[end].start_time - segments[end - 1].end_time <= max_edge_gap
            ):
                labels[start:end] = [previous] * (end - start)
                changed = True
                break

    for index in range(len(labels) - 1):
        current = labels[index]
        following = labels[index + 1]
        if not current or not following or current == following:
            continue
        word = _normalized_word(segments[index].text)
        next_text = segments[index + 1].text.strip()
        if (
            word in _BOUNDARY_PREFIX_WORDS
            and next_text
            and next_text[0].islower()
            and segments[index].end_time - segments[index].start_time <= 450
            and segments[index + 1].start_time - segments[index].end_time <= max_edge_gap
        ):
            labels[index] = following

    for segment, label in zip(segments, labels):
        segment.speaker_id = label
    return asr_data
