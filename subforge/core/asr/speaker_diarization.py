"""Speaker diarization and conservative speaker assignment for ASR results."""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.asr.model_cache import SingleEntryModelCache

logger = logging.getLogger(__name__)

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
LOCAL_DIARIZATION_DIR = "pyannote-speaker-diarization-community-1"
DIARIZATION_CACHE_VERSION = 3
_DIARIZATION_MODEL_CACHE = SingleEntryModelCache()
_DARWIN_DATALESS_FLAG = 0x40000000

if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


@dataclass(frozen=True)
class SpeakerTurn:
    """A single speaker-active interval in milliseconds."""

    start_ms: int
    end_ms: int
    speaker_id: str


class SpeakerTurns(list[SpeakerTurn]):
    """Speaker turns plus overlap regions from regular diarization output."""

    def __init__(
        self,
        turns: Iterable[SpeakerTurn] = (),
        *,
        overlap_regions: Iterable[tuple[int, int]] = (),
        regular_turns: Iterable[SpeakerTurn] = (),
        execution_device: str = "unknown",
    ) -> None:
        super().__init__(turns)
        self.overlap_regions = list(overlap_regions)
        self.regular_turns = list(regular_turns)
        self.execution_device = execution_device


def _is_local_file_available(path: Path, minimum_size: int) -> bool:
    """Reject cloud placeholders that have metadata but no readable local data."""
    try:
        metadata = path.stat()
    except OSError:
        return False
    flags = int(getattr(metadata, "st_flags", 0) or 0)
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_size >= minimum_size
        and not flags & _DARWIN_DATALESS_FLAG
    )


def _dataless_model_files(path: Path) -> list[Path]:
    if platform.system() != "Darwin":
        return []
    required = (
        "config.yaml",
        "segmentation/pytorch_model.bin",
        "embedding/pytorch_model.bin",
        "plda/plda.npz",
        "plda/xvec_transform.npz",
    )
    result = []
    for name in required:
        candidate = path / name
        try:
            flags = int(getattr(candidate.stat(), "st_flags", 0) or 0)
        except OSError:
            continue
        if flags & _DARWIN_DATALESS_FLAG:
            result.append(candidate)
    return result


def is_diarization_model_dir(path: str | Path) -> bool:
    """Return whether a local pyannote pipeline snapshot is usable."""
    model_dir = Path(path).expanduser()
    required = {
        "config.yaml": 100,
        "segmentation/pytorch_model.bin": 1024 * 1024,
        "embedding/pytorch_model.bin": 1024 * 1024,
        "plda/plda.npz": 1024,
        "plda/xvec_transform.npz": 1024,
    }
    return model_dir.is_dir() and all(
        _is_local_file_available(model_dir / name, minimum) for name, minimum in required.items()
    )


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
    candidates = [Path(str(model)).expanduser()]
    if model_dir:
        candidates.append(Path(model_dir).expanduser() / LOCAL_DIARIZATION_DIR)
    for candidate in candidates:
        dataless = _dataless_model_files(candidate)
        if dataless:
            raise RuntimeError(
                "Community-1 model files are cloud placeholders and are not available "
                "offline. In Finder, download the model folder or download the speaker "
                "model again in WhisperX settings."
            )
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
    min_speakers: int | None,
    max_speakers: int | None,
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
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }
    )


def _deserialize_cached_turns(value: Any) -> SpeakerTurns:
    overlap_regions: list[tuple[int, int]] = []
    regular_turns: list[SpeakerTurn] = []
    execution_device = "cache"
    if isinstance(value, dict):
        raw_turns = value.get("turns")
        raw_regions = value.get("overlap_regions", [])
        raw_regular_turns = value.get("regular_turns", [])
        execution_device = str(value.get("execution_device") or "cache")
        if not isinstance(raw_turns, list) or not isinstance(raw_regions, list):
            return SpeakerTurns()
        for region in raw_regions:
            if not isinstance(region, (list, tuple)) or len(region) != 2:
                return SpeakerTurns()
            try:
                start_ms, end_ms = int(region[0]), int(region[1])
            except (TypeError, ValueError):
                return SpeakerTurns()
            if end_ms <= start_ms:
                return SpeakerTurns()
            overlap_regions.append((start_ms, end_ms))
        if not isinstance(raw_regular_turns, list):
            return SpeakerTurns()
        for item in raw_regular_turns:
            if not isinstance(item, dict):
                return SpeakerTurns()
            try:
                turn = SpeakerTurn(
                    int(item["start_ms"]),
                    int(item["end_ms"]),
                    str(item["speaker_id"]),
                )
            except (KeyError, TypeError, ValueError):
                return SpeakerTurns()
            if turn.end_ms <= turn.start_ms or not turn.speaker_id:
                return SpeakerTurns()
            regular_turns.append(turn)
        value = raw_turns
    if not isinstance(value, list):
        return SpeakerTurns()
    turns: list[SpeakerTurn] = []
    for item in value:
        if not isinstance(item, dict):
            return SpeakerTurns()
        try:
            turn = SpeakerTurn(
                int(item["start_ms"]),
                int(item["end_ms"]),
                str(item["speaker_id"]),
            )
        except (KeyError, TypeError, ValueError):
            return SpeakerTurns()
        if turn.end_ms <= turn.start_ms or not turn.speaker_id:
            return SpeakerTurns()
        turns.append(turn)
    return SpeakerTurns(
        turns,
        overlap_regions=overlap_regions,
        regular_turns=regular_turns,
        execution_device=execution_device,
    )


def _find_overlap_regions(annotation: Any) -> list[tuple[int, int]]:
    """Return intervals where regular diarization has two active speakers."""
    events: list[tuple[float, int, str]] = []
    for start, end, label in _iter_labeled_turns(annotation):
        if end <= start:
            continue
        events.append((start, 1, label))
        events.append((end, -1, label))
    events.sort(key=lambda event: (event[0], event[1]))
    active: dict[str, int] = {}
    overlap_start: float | None = None
    regions: list[tuple[int, int]] = []
    for timestamp, delta, label in events:
        was_overlapping = len(active) >= 2
        if delta < 0:
            count = active.get(label, 0) - 1
            if count > 0:
                active[label] = count
            else:
                active.pop(label, None)
        else:
            active[label] = active.get(label, 0) + 1
        is_overlapping = len(active) >= 2
        if not was_overlapping and is_overlapping:
            overlap_start = timestamp
        elif was_overlapping and not is_overlapping and overlap_start is not None:
            start_ms = max(0, round(overlap_start * 1000))
            end_ms = max(1, round(timestamp * 1000))
            if end_ms > start_ms:
                regions.append((start_ms, end_ms))
            overlap_start = None
    return regions


def _select_diarization_device(torch_module: Any) -> str:
    """Select a safe pyannote device, with an override for diagnostics."""
    configured = os.environ.get("SUBFORGE_DIARIZATION_DEVICE", "auto").strip().lower()
    if configured not in {"auto", "cpu", "mps", "cuda"}:
        logger.warning("Ignoring unsupported diarization device: %s", configured)
        configured = "auto"
    if configured == "cpu":
        return "cpu"

    cuda = getattr(torch_module, "cuda", None)
    cuda_available = bool(cuda and cuda.is_available())
    if configured == "cuda":
        if not cuda_available:
            raise RuntimeError("NVIDIA CUDA was requested but is not available")
        return "cuda"

    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    if configured == "mps":
        if not mps_available:
            raise RuntimeError("Apple MPS was requested but is not available")
        return "mps"
    if platform.system() == "Darwin" and platform.machine() == "arm64" and mps_available:
        return "mps"
    if platform.system() == "Windows" and cuda_available:
        return "cuda"
    return "cpu"


def _load_diarization_pipeline(
    resolved_model: str,
    *,
    token: str,
    model_dir: str | Path | None,
    device: str,
):
    """Acquire the shared Community-1 pipeline on the requested device."""
    from pyannote.audio import Pipeline

    def _loader():
        loaded = Pipeline.from_pretrained(
            resolved_model,
            token=token or None,
            cache_dir=str(model_dir) if model_dir else None,
        )
        if loaded is None:
            raise RuntimeError("Speaker diarization model could not be loaded")
        return loaded

    cache_key = (str(Path(resolved_model).resolve()), device, id(Pipeline))
    return _DIARIZATION_MODEL_CACHE.acquire(cache_key, _loader)


def diarize_audio(
    audio_path: str,
    *,
    model: str = DEFAULT_DIARIZATION_MODEL,
    token: str = "",
    model_dir: str | Path | None = None,
    num_speakers: int | None = 2,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    callback: Callable[[int, str], None] | None = None,
) -> SpeakerTurns:
    """Run pyannote on the original audio and return first-appearance labels."""
    if num_speakers is not None:
        min_speakers = None
        max_speakers = None
    elif min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise ValueError("min_speakers must not exceed max_speakers")
    resolved_model = require_local_diarization_model(model, model_dir)
    result_cache_key = _diarization_cache_key(
        audio_path,
        resolved_model,
        num_speakers,
        min_speakers,
        max_speakers,
    )
    try:
        from subforge.core.utils.cache import get_diarization_cache, is_cache_enabled

        if is_cache_enabled():
            cached_turns = _deserialize_cached_turns(get_diarization_cache().get(result_cache_key))
            if cached_turns:
                logger.info("Using cached speaker diarization (%d turns)", len(cached_turns))
                if callback:
                    callback(94, "Using cached speaker analysis...")
                return cached_turns
    except Exception as exc:
        logger.debug("Speaker diarization cache lookup failed: %s", exc)

    try:
        import pyannote.audio as pyannote_audio
        import torch

        if not hasattr(pyannote_audio, "Pipeline"):
            raise ImportError("pyannote.audio.Pipeline is unavailable")
    except ImportError as exc:
        raise RuntimeError(
            "Speaker diarization runtime is unavailable. Install the WhisperX "
            "dependencies that include pyannote.audio."
        ) from exc

    if callback:
        callback(92, "Loading speaker diarization model...")

    audio = _load_waveform(audio_path)
    kwargs: dict[str, int] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers
    selected_device = _select_diarization_device(torch)

    def _run_pipeline(active_pipeline: Any, device_name: str):
        active_pipeline.to(torch.device(device_name))
        if callback:
            count = f" {num_speakers}" if num_speakers else ""
            callback(
                94,
                f"Identifying{count} speakers on {device_name.upper()}...",
            )
        pipeline_call: Any = active_pipeline
        return pipeline_call(audio, **kwargs)

    model_loaded = False
    try:
        pipeline_context = _load_diarization_pipeline(
            resolved_model,
            token=token,
            model_dir=model_dir,
            device=selected_device,
        )
        with pipeline_context as pipeline:
            model_loaded = True
            output = _run_pipeline(pipeline, selected_device)
    except Exception as exc:
        if not model_loaded:
            raise RuntimeError(
                "Unable to load the local Community-1 model. Verify that every model "
                "file is stored locally and readable."
            ) from exc
        if selected_device not in {"mps", "cuda"}:
            raise RuntimeError(
                "Unable to load or run the speaker diarization model. Verify the local "
                "model and Community-1 runtime."
            ) from exc
        logger.warning(
            "Community-1 %s inference failed; retrying on CPU: %s",
            selected_device.upper(),
            exc,
            exc_info=True,
        )
        if callback:
            callback(
                94,
                f"{selected_device.upper()} unavailable; retrying speaker analysis on CPU...",
            )
        try:
            cpu_model_loaded = False
            pipeline_context = _load_diarization_pipeline(
                resolved_model,
                token=token,
                model_dir=model_dir,
                device="cpu",
            )
            with pipeline_context as pipeline:
                cpu_model_loaded = True
                output = _run_pipeline(pipeline, "cpu")
            selected_device = "cpu"
        except Exception as cpu_exc:
            if not cpu_model_loaded:
                raise RuntimeError(
                    "Unable to reload the local Community-1 model for CPU fallback."
                ) from cpu_exc
            raise RuntimeError("Speaker diarization failed on both Apple MPS and CPU") from cpu_exc
    regular_annotation = getattr(output, "speaker_diarization", None)
    overlap_regions = _find_overlap_regions(regular_annotation)
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", output)

    label_map: dict[str, str] = {}
    turns = SpeakerTurns(overlap_regions=overlap_regions, execution_device=selected_device)
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
    regular_turns: list[SpeakerTurn] = []
    for start, end, raw_label in _iter_labeled_turns(regular_annotation):
        if end <= start:
            continue
        label_map.setdefault(raw_label, f"Speaker {len(label_map) + 1}")
        regular_turns.append(
            SpeakerTurn(
                start_ms=max(0, round(start * 1000)),
                end_ms=max(1, round(end * 1000)),
                speaker_id=label_map[raw_label],
            )
        )
    regular_turns.sort(key=lambda turn: (turn.start_ms, turn.end_ms, turn.speaker_id))
    turns.regular_turns = regular_turns
    try:
        from subforge.core.utils.cache import get_diarization_cache, is_cache_enabled

        if is_cache_enabled():
            get_diarization_cache().set(
                result_cache_key,
                {
                    "turns": [
                        {
                            "start_ms": turn.start_ms,
                            "end_ms": turn.end_ms,
                            "speaker_id": turn.speaker_id,
                        }
                        for turn in turns
                    ],
                    "overlap_regions": [list(region) for region in overlap_regions],
                    "regular_turns": [
                        {
                            "start_ms": turn.start_ms,
                            "end_ms": turn.end_ms,
                            "speaker_id": turn.speaker_id,
                        }
                        for turn in regular_turns
                    ],
                    "execution_device": selected_device,
                },
                expire=90 * 24 * 60 * 60,
            )
    except Exception as exc:
        logger.debug("Speaker diarization cache write failed: %s", exc)
    logger.info(
        "Speaker diarization found %d speakers in %d turns on %s",
        len(label_map),
        len(turns),
        selected_device.upper(),
    )
    return turns


def acoustically_verify_speakers(
    asr_data: ASRData,
    audio_path: str,
    turns: SpeakerTurns,
    *,
    model: str = DEFAULT_DIARIZATION_MODEL,
    token: str = "",
    model_dir: str | Path | None = None,
) -> ASRData:
    """Verify risky semantic corrections with independent speaker embeddings.

    This pass is deliberately best-effort. It never changes text or timestamps,
    and any verifier failure leaves the conservative diarization labels intact.
    """
    try:
        import torch

        from subforge.core.asr.speaker_verification import verify_speakers_with_pipeline

        resolved_model = require_local_diarization_model(model, model_dir)
        device = getattr(turns, "execution_device", "unknown")
        if device not in {"cpu", "mps", "cuda"}:
            device = _select_diarization_device(torch)
        pipeline_context = _load_diarization_pipeline(
            resolved_model,
            token=token,
            model_dir=model_dir,
            device=device,
        )
        with pipeline_context as pipeline:
            stats = verify_speakers_with_pipeline(
                asr_data,
                audio_path,
                pipeline=pipeline,
                device=device,
                model_dir=str(model_dir) if model_dir else None,
                overlap_regions=getattr(turns, "overlap_regions", ()),
            )
        logger.info(
            "Acoustic speaker verification accepted %d/%d proposals "
            "(%d overlap, %d reference, %d consensus skips)",
            stats.accepted,
            stats.proposals,
            stats.skipped_overlap,
            stats.skipped_reference,
            stats.skipped_consensus,
        )
    except Exception as exc:
        logger.warning(
            "Acoustic speaker verification failed; keeping conservative labels: %s",
            exc,
            exc_info=True,
        )
    return asr_data


def assign_speakers(
    asr_data: ASRData,
    turns: list[SpeakerTurn],
    *,
    nearest_gap_ms: int = 120,
    suppress_flip_ms: int = 300,
    smooth: bool = True,
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
    if not smooth:
        return asr_data
    return smooth_speaker_assignments(
        asr_data,
        nearest_gap_ms=nearest_gap_ms,
        suppress_flip_ms=suppress_flip_ms,
        overlap_regions=getattr(turns, "overlap_regions", ()),
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

_CONTINUATION_START_WORDS = {
    "because",
    "but",
    "does",
    "for",
    "how",
    "if",
    "that",
    "then",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
}

_BOUNDARY_CONTINUATION_WORDS = _CONTINUATION_START_WORDS | {
    "and",
    "of",
    "to",
    "with",
}

_QUESTION_START_WORDS = {
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "how",
    "is",
    "should",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "would",
}

_SHORT_INTERJECTIONS = {
    "ah",
    "hmm",
    "mm-hmm",
    "no",
    "okay",
    "right",
    "uh-huh",
    "wow",
    "yeah",
    "yes",
}

_BOUNDARY_SUBJECT_WORDS = {
    "he",
    "i",
    "it",
    "she",
    "they",
    "we",
    "who",
    "you",
}


def _normalized_word(text: str) -> str:
    words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text.lower())
    return words[-1].replace("’", "'") if words else ""


def _first_normalized_word(text: str) -> str:
    words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text.lower())
    return words[0].replace("’", "'") if words else ""


def _ends_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?][\"')\]]*\s*$", text.strip()))


def _intersects_overlap(
    segments: list[ASRDataSeg],
    start: int,
    end: int,
    overlap_regions: Iterable[tuple[int, int]],
) -> bool:
    if start >= end:
        return False
    start_ms = segments[start].start_time
    end_ms = segments[end - 1].end_time
    return any(
        region_start < end_ms and region_end > start_ms
        for region_start, region_end in overlap_regions
    )


def _snap_boundary_continuations(
    segments: list[ASRDataSeg],
    labels: list[str],
    overlap_regions: Iterable[tuple[int, int]],
    max_edge_gap: int,
) -> None:
    """Repair grammatical fragments stranded beside a diarization boundary."""
    overlap_regions = tuple(overlap_regions)
    original_labels = list(labels)
    boundaries = [
        index
        for index in range(1, len(labels))
        if original_labels[index]
        and original_labels[index - 1]
        and original_labels[index] != original_labels[index - 1]
    ]

    # Move a tiny unfinished suffix with its following continuation, as in
    # "I'm glad / to be here.". Stop at the prior sentence boundary.
    for boundary in boundaries:
        following_word = _first_normalized_word(segments[boundary].text)
        if following_word not in _BOUNDARY_CONTINUATION_WORDS:
            continue
        start = boundary
        while (
            start > 0
            and boundary - start < 4
            and labels[start - 1] == labels[boundary - 1]
            and not _ends_sentence(segments[start - 1].text)
        ):
            start -= 1
        if start == boundary:
            continue
        phrase = segments[start:boundary]
        duration = phrase[-1].end_time - phrase[0].start_time
        gap = segments[boundary].start_time - phrase[-1].end_time
        first_word = _first_normalized_word(phrase[0].text)
        if (
            duration <= 1_300
            and gap <= max_edge_gap
            and first_word in _BOUNDARY_PREFIX_WORDS
            and not _intersects_overlap(segments, start, boundary, overlap_regions)
        ):
            labels[start:boundary] = [labels[boundary]] * (boundary - start)

    # Exclusive diarization can land before a lowercase sentence tail. Move
    # only the first completed sentence and, at most, one immediate question.
    for boundary in boundaries:
        previous_label = labels[boundary - 1]
        current_label = labels[boundary]
        first_text = segments[boundary].text.strip()
        first_word = _first_normalized_word(first_text)
        boundary_gap = segments[boundary].start_time - segments[boundary - 1].end_time
        extended_gap_allowed = (
            first_word in _BOUNDARY_CONTINUATION_WORDS
            or first_word in _BOUNDARY_PREFIX_WORDS
            or first_word.endswith("ing")
        )
        if (
            not previous_label
            or not current_label
            or previous_label == current_label
            or not first_text
            or not first_text[0].islower()
            or _ends_sentence(segments[boundary - 1].text)
            or (boundary_gap > max_edge_gap and (boundary_gap > 1_000 or not extended_gap_allowed))
        ):
            continue
        run_end = boundary + 1
        while run_end < len(labels) and labels[run_end] == current_label:
            run_end += 1
        terminal = next(
            (
                index
                for index in range(boundary, min(run_end, boundary + 8))
                if _ends_sentence(segments[index].text)
            ),
            None,
        )
        if terminal is None:
            continue
        move_end = terminal + 1
        word_count = sum(
            len(re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", segment.text))
            for segment in segments[boundary:move_end]
        )
        duration = segments[move_end - 1].end_time - segments[boundary].start_time
        if word_count > 8 or duration > 3_500:
            continue

        if (
            move_end < run_end
            and _first_normalized_word(segments[move_end].text) in _QUESTION_START_WORDS
        ):
            question_terminal = next(
                (
                    index
                    for index in range(move_end, min(run_end, move_end + 12))
                    if _ends_sentence(segments[index].text)
                ),
                None,
            )
            if question_terminal is not None:
                question_end = question_terminal + 1
                total_duration = segments[question_end - 1].end_time - segments[boundary].start_time
                total_words = sum(
                    len(re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", segment.text))
                    for segment in segments[boundary:question_end]
                )
                if total_words <= 20 and total_duration <= 8_000:
                    move_end = question_end

        labels[boundary:move_end] = [previous_label] * (move_end - boundary)


def _is_short_interjection(segments: list[ASRDataSeg]) -> bool:
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text.lower())
    normalized = " ".join(word.replace("’", "'") for word in words)
    return normalized in _SHORT_INTERJECTIONS


def _is_incomplete_speaker_island(segments: list[ASRDataSeg]) -> bool:
    """Identify a tiny grammatical continuation, not a complete short reply."""
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
    words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text.lower())
    if not words or len(words) > 3 or re.search(r"[.!?]\s*$", text):
        return False
    if _is_short_interjection(segments):
        return False
    return text[0].islower() or words[0] in _CONTINUATION_START_WORDS


def smooth_speaker_assignments(
    asr_data: ASRData,
    *,
    nearest_gap_ms: int = 120,
    suppress_flip_ms: int = 300,
    overlap_regions: Iterable[tuple[int, int]] = (),
    stages: frozenset[str] | None = None,
) -> ASRData:
    """Fill only short unlabeled gaps in word-level diarization output.

    Semantic boundary rewrites are available to the offline benchmark through
    ``stages`` but are intentionally disabled in production: AMI word-level
    scoring showed that they move more correctly attributed words than errors.
    """
    segments = asr_data.segments
    if len(segments) < 2 or not any(segment.speaker_id for segment in segments):
        return asr_data

    labels = [segment.speaker_id for segment in segments]
    max_fill_duration = max(700, suppress_flip_ms * 2)
    max_edge_gap = max(300, nearest_gap_ms * 2)
    enabled = stages if stages is not None else frozenset({"fill_blanks"})

    if "fill_blanks" in enabled:
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
                next_text = segments[end].text.strip() if end < len(segments) else ""
                blank_words = [
                    _first_normalized_word(segment.text) for segment in segments[index:end]
                ]
                if (
                    following
                    and next_text
                    and next_text[0].islower()
                    and all(word in _BOUNDARY_PREFIX_WORDS for word in blank_words)
                    and segments[end].start_time - segments[end - 1].end_time <= 600
                    and not _intersects_overlap(segments, index, end, overlap_regions)
                ):
                    candidate = following
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

    if "assign_interjections" in enabled:
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
            previous_gap = (
                segments[index].start_time - segments[index - 1].end_time
                if index > 0
                else max_edge_gap + 1
            )
            following_gap = (
                segments[end].start_time - segments[end - 1].end_time
                if end < len(segments)
                else max_edge_gap + 1
            )
            if (
                previous
                and following
                and previous != following
                and duration <= 350
                and previous_gap <= max_edge_gap
                and following_gap <= max_edge_gap
                and _is_short_interjection(segments[index:end])
                and not _intersects_overlap(segments, index, end, overlap_regions)
            ):
                # Acknowledgements commonly introduce the following speaker's
                # turn. This remains only a proposal until acoustics confirm it.
                labels[index:end] = [following] * (end - index)
            index = end

    if "suppress_islands" in enabled:
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
                previous_gap = segments[start].start_time - segments[start - 1].end_time
                following_gap = segments[end].start_time - segments[end - 1].end_time
                standard_short_flip = (
                    duration <= max(700, suppress_flip_ms)
                    and previous_gap <= max_edge_gap
                    and following_gap <= max_edge_gap
                    and not _is_short_interjection(segments[start:end])
                )
                incomplete_continuation = (
                    duration <= 1200
                    and previous_gap <= 1200
                    and following_gap <= 450
                    and _is_incomplete_speaker_island(segments[start:end])
                )
                if (
                    label
                    and previous
                    and previous == following
                    and label != previous
                    and end - start <= 3
                    and (standard_short_flip or incomplete_continuation)
                ):
                    labels[start:end] = [previous] * (end - start)
                    changed = True
                    break

    if "move_prefix" in enabled:
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

    # A diarization boundary can land after an adverb and strand a tiny
    # subject phrase ("He certainly / delivered...") on the prior speaker.
    # Keep this deliberately narrow so complete replies and interruptions are
    # not absorbed into the next turn.
    for boundary in range(2, len(labels)) if "move_subject" in enabled else ():
        previous = labels[boundary - 1]
        following = labels[boundary]
        next_text = segments[boundary].text.strip()
        if not previous or not following or previous == following:
            continue
        if not next_text or not next_text[0].islower():
            continue
        for phrase_length in (3, 2):
            start = boundary - phrase_length
            if start < 0 or any(labels[index] != previous for index in range(start, boundary)):
                continue
            first_word = _normalized_word(segments[start].text)
            phrase = " ".join(segment.text.strip() for segment in segments[start:boundary])
            duration = segments[boundary - 1].end_time - segments[start].start_time
            gap = segments[boundary].start_time - segments[boundary - 1].end_time
            if (
                first_word in _BOUNDARY_SUBJECT_WORDS
                and duration <= 450
                and gap <= 200
                and not re.search(r"[.!?—]\s*$", phrase)
            ):
                labels[start:boundary] = [following] * phrase_length
                break

    if "snap_continuations" in enabled:
        _snap_boundary_continuations(
            segments,
            labels,
            overlap_regions,
            max_edge_gap,
        )

    for segment, label in zip(segments, labels):
        segment.speaker_id = label
        for word in segment.words:
            word.speaker_id = label
    return asr_data
