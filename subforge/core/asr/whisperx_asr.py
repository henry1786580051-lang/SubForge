import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Union, cast

from ...config import MODEL_PATH
from ..utils.logger import setup_logger
from .alignment_models import alignment_model_for_language, is_alignment_model_ready
from .asr_data import ASRData, ASRDataSeg, ASRWord, TimestampSource
from .base import BaseASR
from .compound_numbers import parse_tire_size
from .faster_whisper import find_faster_whisper_model_dir, resolve_faster_whisper_runtime
from .model_cache import SingleEntryModelCache
from .status import ASRStatus
from .worker_runtime import atomic_json_write as _atomic_json_write
from .worker_runtime import log_tail as _worker_log_tail
from .worker_runtime import stop_process as _stop_worker_process

logger = setup_logger("whisperx_asr")
_ALIGNMENT_MODEL_CACHE = SingleEntryModelCache()

_MLX_WORKER_FLAG = "SUBFORGE_MLX_WHISPER_WORKER"
_MLX_WORKER_REQUEST = "SUBFORGE_MLX_WHISPER_REQUEST"
_MLX_WORKER_OUTPUT = "SUBFORGE_MLX_WHISPER_OUTPUT"
_MLX_PREVIEW_LINE = re.compile(
    r"^\[(?P<start>\d{2}(?::\d{2})?:\d{2}\.\d{3}) --> "
    r"(?P<end>\d{2}(?::\d{2})?:\d{2}\.\d{3})\]\s*(?P<text>.+?)\s*$"
)

MIXED_LANGUAGE_MIN_CONFIDENCE = 0.80
MIXED_LANGUAGE_MAX_PRIMARY_CONFIDENCE = 0.20
MIXED_LANGUAGE_MIN_MARGIN = 0.15
MIXED_LANGUAGE_MAX_GAP_SECONDS = 2.50
MIXED_LANGUAGE_CONTEXT_SECONDS = 0.75
MIXED_LANGUAGE_WINDOW_SECONDS = 8.0
MIXED_LANGUAGE_WINDOW_STRIDE_SECONDS = 4.0
MIXED_LANGUAGE_MIN_LANGUAGE_SUPPORT_SECONDS = 6.0
MIXED_LANGUAGE_MIN_PROBE_COUNT = 2
MIXED_LANGUAGE_MIN_DOMINANCE = 0.60
MIXED_LANGUAGE_MIN_VOICED_SECONDS = 2.5
MIXED_LANGUAGE_MIN_VOICED_RATIO = 0.30
MIXED_LANGUAGE_MIN_DECODED_UNITS = 2
MIXED_LANGUAGE_MINORITY_CJK_SUPPORT_SECONDS = 8.0
MIXED_LANGUAGE_MINORITY_CJK_AGREED_EVENTS = 2
MIXED_LANGUAGE_MINORITY_CJK_SUPPORT_RATIO = 0.50
MIXED_LANGUAGE_MAX_CONFIRMED_BRIDGE_SECONDS = 20.0
MIXED_LANGUAGE_MIN_BRIDGE_SPEECH_SECONDS = 2.0
MIXED_LANGUAGE_MIN_BRIDGE_SPEECH_RATIO = 0.30
MIXED_LANGUAGE_REPETITION_NEARBY_SECONDS = 5.0
MIXED_LANGUAGE_REPETITION_MAX_RANGE_SECONDS = 90.0
MLX_GAP_RECOVERY_MIN_GAP_SECONDS = 6.0
MLX_GAP_RECOVERY_MIN_SPEECH_SECONDS = 2.5
MLX_GAP_RECOVERY_MIN_SPEECH_RATIO = 0.45
MLX_GAP_RECOVERY_CONTEXT_SECONDS = 2.0
MLX_GAP_RECOVERY_FALLBACK_CONTEXT_SECONDS = 15.0
MLX_SHORT_GAP_MIN_SECONDS = 0.8
MLX_SHORT_GAP_MAX_SECONDS = MLX_GAP_RECOVERY_MIN_GAP_SECONDS
MLX_SHORT_GAP_MIN_SPEECH_SECONDS = 0.55
MLX_SHORT_GAP_MIN_SPEECH_RATIO = 0.40
MLX_SHORT_GAP_CONTEXT_SECONDS = (2.0, 5.0)
MLX_SHORT_GAP_MAX_CANDIDATES = 10
MLX_SHORT_GAP_MAX_DECODE_SECONDS = 180.0
MLX_SPARSE_RECOVERY_MIN_DURATION_SECONDS = 10.0
MLX_SPARSE_RECOVERY_MIN_SPEECH_SECONDS = 5.0
MLX_SPARSE_RECOVERY_MIN_SPEECH_RATIO = 0.60
MLX_SPARSE_RECOVERY_MAX_UNITS_PER_SPEECH_SECOND = 0.75
MLX_SPARSE_RECOVERY_CONTEXT_SECONDS = 2.0
MLX_SPARSE_RECOVERY_FALLBACK_CONTEXT_SECONDS = (15.0, 45.0)
MLX_SHORT_GAP_REVIEW_MIN_SPEECH_SECONDS = 2.5
MLX_SHORT_GAP_REVIEW_MIN_SPEECH_RATIO = 0.75
MLX_FINAL_AUDIT_MIN_SPEECH_SECONDS = 5.0
MLX_FINAL_AUDIT_MIN_SPEECH_RATIO = 0.70
MLX_AUDIO_SAMPLE_RATE = 16_000


DEFAULT_EN_ALIGN_MODEL = "WAV2VEC2_ASR_LARGE_LV60K_960H"
DEFAULT_EN_ALIGN_FILENAME = "wav2vec2_fairseq_large_lv60k_asr_ls960.pth"
LOCAL_MLX_MODEL_NAMES = (
    "whisper-large-v3-fp16",
    "mlx-whisper-large-v3-fp16",
    "large-v3-fp16",
)
MLX_WHISPER_MODELS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-mlx",
    "large-v1": "mlx-community/whisper-large-v1-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def _candidate_local_model_dirs() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("SUBFORGE_MLX_WHISPER_MODEL", "").strip()
    if env_path:
        candidates.insert(0, Path(env_path).expanduser())

    roots = [
        Path.home() / "Desktop" / "YouTube" / "model",
        Path.home() / "SubForge" / "models",
        Path.home() / "Library" / "Application Support" / "SubForge" / "models",
        MODEL_PATH,
    ]
    for root in roots:
        for name in LOCAL_MLX_MODEL_NAMES:
            candidates.append(root / name)

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _is_valid_mlx_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "config.json").is_file()
    has_weights = any(
        (path / name).is_file()
        for name in ("weights.safetensors", "weights.npz", "model.safetensors")
    )
    return has_config and has_weights


def _find_local_mlx_model() -> Path | None:
    for candidate in _candidate_local_model_dirs():
        if _is_valid_mlx_model_dir(candidate):
            return candidate
    return None


def default_mlx_model() -> str:
    path = _find_local_mlx_model()
    if path:
        return str(path)
    return "large-v3"


def resolve_mlx_model(model: str = "") -> str:
    """Resolve a configured MLX model name to its local path or HF repo."""
    return _mlx_model_repo(model)


def is_valid_mlx_model_dir(path: str | Path) -> bool:
    """Return whether a directory contains a usable MLX Whisper model."""
    return _is_valid_mlx_model_dir(Path(path).expanduser())


def _normalize_language(language: str | None) -> str | None:
    value = (language or "").strip().lower()
    if not value or value == "auto":
        return None
    return value


def _normalize_align_device(device: str | None) -> str:
    value = (device or "cpu").strip().lower()
    if value in {"auto", "cuda"}:
        try:
            import torch

            torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
            if torch_cuda_version is not None and torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    if value == "mps":
        # MLX handles Apple Silicon acceleration; WhisperX alignment is more stable on CPU.
        return "cpu"
    if value == "cpu":
        return value
    return "cpu"


def _normalize_align_model(model: str | None) -> str:
    """Map the downloaded torchaudio weight file to its pipeline identifier."""
    value = (model or "").strip()
    if value and Path(value).name.lower() == DEFAULT_EN_ALIGN_FILENAME:
        return DEFAULT_EN_ALIGN_MODEL
    return value


def _normalize_compute_type(device: str, compute_type: str | None) -> str:
    value = (compute_type or "default").strip().lower()
    if value and value != "default":
        return value
    return "float16" if device == "cuda" else "int8"


def _mlx_model_repo(model: str) -> str:
    value = (model or default_mlx_model()).strip()
    if value in {"", "auto", "default"}:
        return default_mlx_model()
    if "/" in value or Path(value).exists():
        return value
    # Prefer a local MLX Large V3 FP16 directory for common short aliases. This
    # keeps the packaged app from unexpectedly falling back to an online
    # HuggingFace repo when the user has already downloaded the local model.
    if value in {"large-v3", "large-v3-fp16", "whisper-large-v3-fp16"}:
        local = _find_local_mlx_model()
        if local:
            return str(local)
    return MLX_WHISPER_MODELS.get(value, value)


def _prepare_mlx_model_path(model: str, tmp_dir: Path) -> str:
    model_path = Path(model)
    if not model_path.exists() or not model_path.is_dir():
        return model
    if not _is_valid_mlx_model_dir(model_path):
        raise RuntimeError(
            "Invalid MLX Whisper model directory. Expected config.json and "
            f"weights/model safetensors in: {model_path}"
        )

    weights = model_path / "weights.safetensors"
    weights_npz = model_path / "weights.npz"
    model_weights = model_path / "model.safetensors"
    if weights.exists() or weights_npz.exists() or not model_weights.exists():
        return str(model_path)

    alias_dir = tmp_dir / "mlx_model"
    alias_dir.mkdir(parents=True, exist_ok=True)
    for item in model_path.iterdir():
        if not item.is_file() or item.name == ".DS_Store":
            continue
        target_name = "weights.safetensors" if item.name == "model.safetensors" else item.name
        target = alias_dir / target_name
        if target.exists():
            continue
        try:
            os.symlink(item, target)
        except OSError:
            os.link(item, target)
    return str(alias_dir)


def _mlx_preview_timestamp_ms(value: str) -> int:
    parts = value.split(":")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Invalid MLX preview timestamp: {value}")
    return round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000)


def _parse_mlx_preview_lines(lines: list[str]) -> list[ASRDataSeg]:
    """Parse MLX's stable verbose segment format for UI-only live previews."""
    segments: list[ASRDataSeg] = []
    for line in lines:
        match = _MLX_PREVIEW_LINE.match(line.strip())
        if not match:
            continue
        text = match.group("text").strip()
        start_time = _mlx_preview_timestamp_ms(match.group("start"))
        end_time = _mlx_preview_timestamp_ms(match.group("end"))
        if text and end_time > start_time:
            segments.append(
                ASRDataSeg(
                    text,
                    start_time,
                    end_time,
                    timestamp_granularity="sentence",
                    timing_source="native",
                )
            )
    return segments


@dataclass(frozen=True)
class _SpeechBackedGap:
    start: float
    end: float
    speech_seconds: float
    speech_ratio: float
    is_internal: bool


@dataclass(frozen=True)
class _SparseMLXSegment:
    index: int
    start: float
    end: float
    text: str
    lexical_units: int


def _lexical_unit_count(text: str) -> int:
    """Estimate spoken units across whitespace-delimited and CJK languages."""
    latin_words = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", text)
    cjk_units = re.findall(
        r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]",
        text,
    )
    return len(latin_words) + len(cjk_units)


def _find_sparse_mlx_segments(segments: list[dict[str, Any]]) -> list[_SparseMLXSegment]:
    """Find long decoder spans whose text is implausibly sparse for their duration."""
    candidates: list[_SparseMLXSegment] = []
    for index, segment in enumerate(segments):
        text = str(segment.get("text") or "").strip()
        start = segment.get("start")
        end = segment.get("end")
        if not text or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        duration = float(end) - float(start)
        if duration < MLX_SPARSE_RECOVERY_MIN_DURATION_SECONDS:
            continue
        lexical_units = _lexical_unit_count(text)
        if lexical_units <= 0:
            continue
        if lexical_units / duration >= MLX_SPARSE_RECOVERY_MAX_UNITS_PER_SPEECH_SECOND:
            continue
        candidates.append(
            _SparseMLXSegment(
                index=index,
                start=float(start),
                end=float(end),
                text=text,
                lexical_units=lexical_units,
            )
        )
    return candidates


def _speech_overlap_metrics(
    start: float,
    end: float,
    speech_segments_ms: list[tuple[int, int]],
) -> tuple[float, float]:
    overlaps: list[tuple[int, int]] = []
    start_ms = round(start * 1000)
    end_ms = round(end * 1000)
    for speech_start, speech_end in speech_segments_ms:
        overlap_start = max(start_ms, int(speech_start))
        overlap_end = min(end_ms, int(speech_end))
        if overlap_end > overlap_start:
            overlaps.append((overlap_start, overlap_end))
    if not overlaps:
        return 0.0, 0.0
    overlaps.sort()
    merged: list[tuple[int, int]] = []
    for overlap_start, overlap_end in overlaps:
        if merged and overlap_start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], overlap_end))
        else:
            merged.append((overlap_start, overlap_end))
    speech_seconds = sum(item_end - item_start for item_start, item_end in merged) / 1000
    duration = max(0.001, end - start)
    return speech_seconds, speech_seconds / duration


def _find_uncovered_mlx_gaps(
    segments: list[dict[str, Any]],
    audio_duration: float,
    *,
    min_gap_seconds: float = MLX_GAP_RECOVERY_MIN_GAP_SECONDS,
    max_gap_seconds: float | None = None,
) -> list[tuple[float, float]]:
    """Return ASR timeline holes large enough to justify a coverage audit."""
    timed: list[tuple[float, float]] = []
    for segment in segments:
        if not str(segment.get("text") or "").strip():
            continue
        start = segment.get("start")
        end = segment.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        bounded_start = max(0.0, float(start))
        bounded_end = min(max(0.0, audio_duration), float(end))
        if bounded_end > bounded_start:
            timed.append((bounded_start, bounded_end))
    timed.sort()

    uncovered: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in timed:
        gap_seconds = start - cursor
        if gap_seconds >= min_gap_seconds and (
            max_gap_seconds is None or gap_seconds <= max_gap_seconds
        ):
            uncovered.append((cursor, start))
        cursor = max(cursor, end)
    trailing_gap = audio_duration - cursor
    if trailing_gap >= min_gap_seconds and (
        max_gap_seconds is None or trailing_gap <= max_gap_seconds
    ):
        uncovered.append((cursor, audio_duration))
    return uncovered


def _find_speech_backed_mlx_gaps(
    segments: list[dict[str, Any]],
    speech_segments_ms: list[tuple[int, int]],
    audio_duration: float,
    *,
    min_gap_seconds: float = MLX_GAP_RECOVERY_MIN_GAP_SECONDS,
) -> list[_SpeechBackedGap]:
    """Find long ASR holes that independent VAD confirms contain speech."""
    uncovered = _find_uncovered_mlx_gaps(segments, audio_duration, min_gap_seconds=min_gap_seconds)

    candidates: list[_SpeechBackedGap] = []
    for gap_start, gap_end in uncovered:
        gap_duration = gap_end - gap_start
        gap_start_ms = round(gap_start * 1000)
        gap_end_ms = round(gap_end * 1000)
        overlaps: list[tuple[int, int]] = []
        for speech_start, speech_end in speech_segments_ms:
            overlap_start = max(gap_start_ms, int(speech_start))
            overlap_end = min(gap_end_ms, int(speech_end))
            if overlap_end > overlap_start:
                overlaps.append((overlap_start, overlap_end))
        speech_seconds = sum(end - start for start, end in overlaps) / 1000
        if speech_seconds < MLX_GAP_RECOVERY_MIN_SPEECH_SECONDS:
            continue
        speech_ratio = speech_seconds / gap_duration
        if speech_ratio < MLX_GAP_RECOVERY_MIN_SPEECH_RATIO:
            continue
        speech_start = min(start for start, _end in overlaps) / 1000
        speech_end = max(end for _start, end in overlaps) / 1000
        candidates.append(
            _SpeechBackedGap(
                start=max(
                    gap_start,
                    speech_start - MLX_GAP_RECOVERY_CONTEXT_SECONDS,
                ),
                end=min(
                    gap_end,
                    speech_end + MLX_GAP_RECOVERY_CONTEXT_SECONDS,
                ),
                speech_seconds=speech_seconds,
                speech_ratio=speech_ratio,
                is_internal=gap_start > 0.05 and gap_end < audio_duration - 0.05,
            )
        )
    return candidates


def _detect_speech_in_mlx_gaps(
    audio: Any,
    sample_rate: int,
    uncovered_gaps: list[tuple[float, float]],
    *,
    threshold: float = 0.5,
    min_speech_ms: int = 160,
    min_silence_ms: int = 180,
) -> list[tuple[int, int]]:
    """Run VAD only inside long ASR holes instead of rescanning the full recording."""
    if not uncovered_gaps:
        return []

    from subforge.core.asr import silero_vad, ten_vad

    def _run(vad_backend: Any) -> list[tuple[int, int]]:
        speech_segments: list[tuple[int, int]] = []
        for gap_start, gap_end in uncovered_gaps:
            start_sample = max(0, round(gap_start * sample_rate))
            end_sample = min(len(audio), round(gap_end * sample_rate))
            if end_sample <= start_sample:
                continue
            local_speech = vad_backend.run_vad_inference(
                audio[start_sample:end_sample],
                sample_rate=sample_rate,
                threshold=threshold,
                min_speech_ms=min_speech_ms,
                min_silence_ms=min_silence_ms,
                speech_pad_ms=0,
            )
            gap_start_ms = round(gap_start * 1000)
            speech_segments.extend(
                (gap_start_ms + int(start), gap_start_ms + int(end)) for start, end in local_speech
            )
        return speech_segments

    if ten_vad.is_available():
        try:
            return _run(ten_vad)
        except Exception as exc:
            logger.warning("TEN-VAD gap audit failed; using Silero VAD: %s", exc)
    return _run(silero_vad)


def _usable_mlx_recovery_segment(
    segment: dict[str, Any],
    *,
    speech_ratio: float,
) -> bool:
    text = str(segment.get("text") or "").strip()
    start = segment.get("start")
    end = segment.get("end")
    if not text or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return False
    if float(end) <= float(start):
        return False
    no_speech_prob = segment.get("no_speech_prob")
    average_log_probability = segment.get("avg_logprob")
    if isinstance(average_log_probability, (int, float)) and float(average_log_probability) < -1.0:
        return False
    compression_ratio = segment.get("compression_ratio")
    if isinstance(compression_ratio, (int, float)) and float(compression_ratio) > 2.8:
        return False
    if isinstance(no_speech_prob, (int, float)) and float(no_speech_prob) > 0.50:
        # Speech mixed under music can raise Whisper's no-speech estimate even
        # when the decoded text is strong. Only let independent, dense VAD
        # support override it, and keep a stricter text-confidence gate.
        if speech_ratio < 0.75 or float(no_speech_prob) > 0.85:
            return False
        if not isinstance(average_log_probability, (int, float)):
            return False
        if float(average_log_probability) < -0.60 or _lexical_unit_count(text) < 3:
            return False
    return True


def _recover_mlx_sparse_segments(
    result: dict[str, Any],
    audio: Any,
    sample_rate: int,
    speech_segments_ms: list[tuple[int, int]],
    transcribe_clip: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    """Re-decode long MLX spans that contain far more speech than transcript text."""
    original_segments = [
        dict(segment) for segment in result.get("segments") or [] if isinstance(segment, dict)
    ]
    candidates = _find_sparse_mlx_segments(original_segments)
    if not candidates:
        return result

    replacements: dict[int, list[dict[str, Any]]] = {}
    recovered_ranges: list[dict[str, Any]] = []
    unresolved_ranges: list[dict[str, Any]] = []
    audio_duration = len(audio) / sample_rate if sample_rate > 0 else 0.0
    for candidate in candidates:
        speech_seconds, speech_ratio = _speech_overlap_metrics(
            candidate.start,
            candidate.end,
            speech_segments_ms,
        )
        if (
            speech_seconds < MLX_SPARSE_RECOVERY_MIN_SPEECH_SECONDS
            or speech_ratio < MLX_SPARSE_RECOVERY_MIN_SPEECH_RATIO
            or candidate.lexical_units / max(0.001, speech_seconds)
            >= MLX_SPARSE_RECOVERY_MAX_UNITS_PER_SPEECH_SECOND
        ):
            continue

        minimum_gain = max(6, int(speech_seconds * 0.4))
        accepted: list[dict[str, Any]] = []
        recovered_units = 0
        for context_seconds in (
            MLX_SPARSE_RECOVERY_CONTEXT_SECONDS,
            *MLX_SPARSE_RECOVERY_FALLBACK_CONTEXT_SECONDS,
        ):
            decode_start = max(0.0, candidate.start - context_seconds)
            decode_end = min(audio_duration, candidate.end + context_seconds)
            start_sample = max(0, round(decode_start * sample_rate))
            end_sample = min(len(audio), round(decode_end * sample_rate))
            local_result = transcribe_clip(audio[start_sample:end_sample])
            attempt: list[dict[str, Any]] = []
            for segment in local_result.get("segments") or []:
                if not isinstance(segment, dict) or not _usable_mlx_recovery_segment(
                    segment,
                    speech_ratio=speech_ratio,
                ):
                    continue
                shifted = dict(segment)
                words = segment.get("words")
                if isinstance(words, list) and words:
                    selected_words: list[dict[str, Any]] = []
                    for word in words:
                        if not isinstance(word, dict):
                            continue
                        word_start = word.get("start")
                        word_end = word.get("end")
                        if not isinstance(word_start, (int, float)) or not isinstance(
                            word_end,
                            (int, float),
                        ):
                            continue
                        absolute_start = float(word_start) + decode_start
                        absolute_end = float(word_end) + decode_start
                        if (
                            absolute_start < candidate.start
                            or absolute_end > candidate.end
                            or absolute_end <= absolute_start
                        ):
                            continue
                        selected = dict(word)
                        selected["start"] = absolute_start
                        selected["end"] = absolute_end
                        selected_words.append(selected)
                    if not selected_words:
                        continue
                    shifted["text"] = "".join(
                        str(word.get("word") or "") for word in selected_words
                    ).strip()
                    shifted["start"] = float(selected_words[0]["start"])
                    shifted["end"] = float(selected_words[-1]["end"])
                    shifted["words"] = selected_words
                else:
                    shifted["start"] = max(
                        candidate.start,
                        float(segment.get("start", 0.0)) + decode_start,
                    )
                    shifted["end"] = min(
                        candidate.end,
                        float(segment.get("end", 0.0)) + decode_start,
                        audio_duration,
                    )
                if shifted["end"] <= shifted["start"]:
                    continue
                if not str(shifted.get("text") or "").strip():
                    continue
                shifted["recovered_sparse_segment"] = True
                attempt.append(shifted)

            attempt_units = sum(
                _lexical_unit_count(str(segment.get("text") or "")) for segment in attempt
            )
            if attempt_units > recovered_units:
                accepted = attempt
                recovered_units = attempt_units
            if (
                attempt
                and attempt_units >= candidate.lexical_units + minimum_gain
                and attempt_units >= round(candidate.lexical_units * 1.5)
            ):
                accepted = attempt
                recovered_units = attempt_units
                break

        if (
            not accepted
            or recovered_units < candidate.lexical_units + minimum_gain
            or recovered_units < round(candidate.lexical_units * 1.5)
        ):
            unresolved_ranges.append(
                {
                    "start": candidate.start,
                    "end": candidate.end,
                    "speech_seconds": speech_seconds,
                    "speech_ratio": speech_ratio,
                    "original_units": candidate.lexical_units,
                    "recovered_units": recovered_units,
                    "is_internal": candidate.start > 0.05 and candidate.end < audio_duration - 0.05,
                }
            )
            continue

        replacements[candidate.index] = accepted
        recovered_ranges.append(
            {
                "start": candidate.start,
                "end": candidate.end,
                "speech_seconds": speech_seconds,
                "speech_ratio": speech_ratio,
                "original_units": candidate.lexical_units,
                "recovered_units": recovered_units,
                "segments": len(accepted),
            }
        )

    if not replacements and not unresolved_ranges:
        return result

    merged_segments: list[dict[str, Any]] = []
    for index, segment in enumerate(original_segments):
        merged_segments.extend(replacements.get(index, [segment]))
    merged_segments.sort(
        key=lambda segment: (
            float(segment.get("start", 0.0)),
            float(segment.get("end", 0.0)),
        )
    )
    merged = dict(result)
    merged["segments"] = merged_segments
    if recovered_ranges:
        merged["sparse_segment_recovery"] = recovered_ranges
    if unresolved_ranges:
        merged["unresolved_sparse_segments"] = unresolved_ranges
    return merged


def _stitch_mlx_recovery_boundary(
    accepted: list[dict[str, Any]],
    original_segments: list[dict[str, Any]],
    gap: _SpeechBackedGap,
) -> list[dict[str, Any]]:
    """Remove decode-boundary repetition without rewriting existing ASR text."""
    if not accepted:
        return accepted

    next_segment = next(
        (
            segment
            for segment in original_segments
            if isinstance(segment.get("start"), (int, float))
            and float(segment["start"]) >= gap.end - 0.05
        ),
        None,
    )
    if next_segment is None:
        return accepted

    recovered_text = str(accepted[-1].get("text") or "")
    next_text = str(next_segment.get("text") or "")
    recovered_spans = list(re.finditer(r"[\w']+", recovered_text.lower()))
    next_tokens = re.findall(r"[\w']+", next_text.lower())
    max_overlap = min(len(recovered_spans), len(next_tokens), 12)
    overlap = 0
    for length in range(max_overlap, 0, -1):
        recovered_tokens = [item.group() for item in recovered_spans[-length:]]
        if recovered_tokens != next_tokens[:length]:
            continue
        if length >= 2 or len(recovered_tokens[0]) >= 4:
            overlap = length
            break

    if overlap:
        cut_at = recovered_spans[-overlap].start()
        accepted[-1]["text"] = recovered_text[:cut_at].rstrip(" \t\r\n,.;:!?，。！？；：-–—")
        if not str(accepted[-1]["text"]).strip():
            accepted.pop()
        return accepted

    # A clipped decode may insert sentence-final punctuation immediately
    # before a lowercase continuation already covered by the next segment.
    continuation_words = {
        "about",
        "after",
        "although",
        "and",
        "as",
        "at",
        "before",
        "because",
        "but",
        "by",
        "for",
        "from",
        "if",
        "in",
        "into",
        "of",
        "on",
        "or",
        "over",
        "so",
        "than",
        "that",
        "though",
        "through",
        "to",
        "under",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whose",
        "with",
        "within",
        "without",
    }
    if (
        next_tokens
        and next_tokens[0] in continuation_words
        and next_text.lstrip()[:1].islower()
        and recovered_text.rstrip().endswith((".", "!", "?"))
    ):
        accepted[-1]["text"] = recovered_text.rstrip().rstrip(".!?")
    return accepted


def _recover_mlx_speech_gaps(
    result: dict[str, Any],
    audio: Any,
    sample_rate: int,
    speech_segments_ms: list[tuple[int, int]],
    transcribe_clip: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    """Decode only VAD-confirmed holes while preserving every original segment."""
    original_segments = [
        dict(segment) for segment in result.get("segments") or [] if isinstance(segment, dict)
    ]
    audio_duration = len(audio) / sample_rate if sample_rate > 0 else 0.0
    gaps = _find_speech_backed_mlx_gaps(
        original_segments,
        speech_segments_ms,
        audio_duration,
    )
    if not gaps:
        return result

    recovered: list[dict[str, Any]] = []
    recovered_gaps: list[dict[str, Any]] = []
    unresolved_gaps: list[dict[str, Any]] = []

    def decode_gap(
        gap: _SpeechBackedGap,
        context_seconds: float,
    ) -> list[dict[str, Any]]:
        decode_start = max(0.0, gap.start - context_seconds)
        decode_end = min(
            audio_duration,
            gap.end + context_seconds,
        )
        start_sample = max(0, round(decode_start * sample_rate))
        end_sample = min(len(audio), round(decode_end * sample_rate))
        if end_sample <= start_sample:
            return []
        local_result = transcribe_clip(audio[start_sample:end_sample])
        accepted: list[dict[str, Any]] = []
        for segment in local_result.get("segments") or []:
            if not isinstance(segment, dict) or not _usable_mlx_recovery_segment(
                segment,
                speech_ratio=gap.speech_ratio,
            ):
                continue
            shifted = dict(segment)
            words = segment.get("words")
            if isinstance(words, list) and words:
                selected_words: list[dict[str, Any]] = []
                for word in words:
                    if not isinstance(word, dict):
                        continue
                    word_start = word.get("start")
                    word_end = word.get("end")
                    if not isinstance(word_start, (int, float)) or not isinstance(
                        word_end,
                        (int, float),
                    ):
                        continue
                    absolute_start = float(word_start) + decode_start
                    absolute_end = float(word_end) + decode_start
                    midpoint = (absolute_start + absolute_end) / 2
                    if midpoint < gap.start - 0.05 or midpoint > gap.end + 0.05:
                        continue
                    if absolute_end <= absolute_start:
                        continue
                    selected = dict(word)
                    selected["start"] = max(gap.start, absolute_start)
                    selected["end"] = min(gap.end, absolute_end)
                    if selected["end"] > selected["start"]:
                        selected_words.append(selected)
                if not selected_words:
                    continue
                shifted["text"] = "".join(
                    str(word.get("word") or "") for word in selected_words
                ).strip()
                shifted["start"] = float(selected_words[0]["start"])
                shifted["end"] = float(selected_words[-1]["end"])
                shifted["words"] = selected_words
            else:
                shifted["start"] = max(
                    gap.start,
                    float(segment["start"]) + decode_start,
                )
                shifted["end"] = min(
                    gap.end,
                    float(segment["end"]) + decode_start,
                )
            if shifted["end"] <= shifted["start"]:
                continue
            if not str(shifted.get("text") or "").strip():
                continue
            shifted["recovered_speech_gap"] = True
            accepted.append(shifted)
        return accepted

    for gap in gaps:
        accepted = decode_gap(gap, MLX_GAP_RECOVERY_CONTEXT_SECONDS)
        if not accepted:
            accepted = decode_gap(
                gap,
                MLX_GAP_RECOVERY_FALLBACK_CONTEXT_SECONDS,
            )
        accepted = _stitch_mlx_recovery_boundary(accepted, original_segments, gap)
        if not accepted:
            unresolved_gaps.append(
                {
                    "start": gap.start,
                    "end": gap.end,
                    "speech_seconds": gap.speech_seconds,
                    "speech_ratio": gap.speech_ratio,
                    "is_internal": gap.is_internal,
                }
            )
            continue
        recovered.extend(accepted)
        recovered_gaps.append(
            {
                "start": gap.start,
                "end": gap.end,
                "speech_seconds": gap.speech_seconds,
                "speech_ratio": gap.speech_ratio,
                "segments": len(accepted),
            }
        )

    merged = dict(result)
    if recovered:
        merged["segments"] = sorted(
            [*original_segments, *recovered],
            key=lambda segment: (
                float(segment.get("start", 0.0)),
                float(segment.get("end", 0.0)),
            ),
        )
        merged["speech_gap_recovery"] = recovered_gaps
    if unresolved_gaps:
        merged["unresolved_speech_gaps"] = unresolved_gaps
    return merged


def _alignment_word_coverage(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return individual aligned word spans so segment envelopes cannot hide omissions."""
    coverage: list[dict[str, Any]] = []
    for segment in result.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        words = segment.get("words")
        if not isinstance(words, list):
            continue
        for word in words:
            if not isinstance(word, dict):
                continue
            start = word.get("start")
            end = word.get("end")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                coverage.append(
                    {
                        "text": str(word.get("word") or "speech"),
                        "start": float(start),
                        "end": float(end),
                    }
                )
    if coverage:
        return coverage
    for word in result.get("word_segments") or []:
        if not isinstance(word, dict):
            continue
        start = word.get("start")
        end = word.get("end")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            coverage.append(
                {
                    "text": str(word.get("word") or "speech"),
                    "start": float(start),
                    "end": float(end),
                }
            )
    return coverage


def _recover_short_mlx_speech_gaps(
    result: dict[str, Any],
    audio: Any,
    sample_rate: int,
    speech_segments_ms: list[tuple[int, int]],
    transcribe_clip: Callable[[Any], dict[str, Any]],
    max_candidates: int = MLX_SHORT_GAP_MAX_CANDIDATES,
    max_decode_seconds: float = MLX_SHORT_GAP_MAX_DECODE_SECONDS,
) -> dict[str, Any]:
    """Recover short omissions only when two isolated decodes agree.

    Long-form Whisper can skip a brief phrase even though the surrounding words
    are correct.  VAD alone cannot distinguish that from music or an ordinary
    pause, so this pass requires independent high-confidence speech plus exact
    lexical agreement across two context windows.
    """
    from .speech_gap_repair import (
        anchored_candidate,
        confirmation_window,
        corroborates,
        insert_anchored_gap,
        timed_words,
    )

    coverage = sorted(
        _alignment_word_coverage(result),
        key=lambda item: (float(item["start"]), float(item["end"])),
    )
    if len(coverage) < 2 or not speech_segments_ms:
        return result

    audio_duration = len(audio) / sample_rate if sample_rate > 0 else 0.0
    uncovered = _find_uncovered_mlx_gaps(
        coverage,
        audio_duration,
        min_gap_seconds=MLX_SHORT_GAP_MIN_SECONDS,
        max_gap_seconds=MLX_SHORT_GAP_MAX_SECONDS,
    )
    if not uncovered:
        return result

    def _tokens(text: str) -> list[str]:
        return [
            token.casefold().replace("’", "'")
            for token in re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", text)
        ]

    def _boundary_word(gap_start: float, gap_end: float) -> tuple[str, str]:
        previous = ""
        following = ""
        for item in coverage:
            if float(item["end"]) <= gap_start + 0.05:
                previous = str(item.get("text") or "")
                continue
            if float(item["start"]) >= gap_end - 0.05:
                following = str(item.get("text") or "")
                break
        return previous, following

    def _trim_boundary_echoes(
        words: list[dict[str, Any]],
        previous_text: str,
        following_text: str,
    ) -> list[dict[str, Any]]:
        trimmed = list(words)
        previous_tokens = _tokens(previous_text)
        following_tokens = _tokens(following_text)
        while trimmed and previous_tokens:
            current = _tokens(str(trimmed[0].get("word") or ""))
            if len(current) != 1 or current[0] != previous_tokens[-1]:
                break
            trimmed.pop(0)
        while trimmed and following_tokens:
            current = _tokens(str(trimmed[-1].get("word") or ""))
            if len(current) != 1 or current[0] != following_tokens[0]:
                break
            trimmed.pop()
        return trimmed

    def _decode_words(
        decode_start: float, decode_end: float, speech_ratio: float
    ) -> list[dict[str, Any]]:
        local = transcribe_clip(
            audio[round(decode_start * sample_rate) : round(decode_end * sample_rate)]
        )
        return timed_words(
            local,
            decode_start,
            lambda s: _usable_mlx_recovery_segment(s, speech_ratio=speech_ratio),
        )

    full_attempts: list[list[dict[str, Any]]] = []

    def _decode_gap(
        gap_start: float,
        gap_end: float,
        speech_ratio: float,
        context_seconds: float,
    ) -> list[dict[str, Any]]:
        decode_start = max(0.0, gap_start - context_seconds)
        decode_end = min(audio_duration, gap_end + context_seconds)
        start_sample = max(0, round(decode_start * sample_rate))
        end_sample = min(len(audio), round(decode_end * sample_rate))
        if end_sample <= start_sample:
            return []

        all_words = _decode_words(decode_start, decode_end, speech_ratio)
        full_attempts.append(all_words)
        selected: list[dict[str, Any]] = []
        for word in all_words:
            absolute_start, absolute_end = word["start"], word["end"]
            midpoint = (absolute_start + absolute_end) / 2
            if not (gap_start + 0.03 <= midpoint <= gap_end - 0.03):
                continue
            selected_word = dict(word)
            selected_word["start"] = max(gap_start, absolute_start)
            selected_word["end"] = min(gap_end, absolute_end)
            if selected_word["end"] > selected_word["start"]:
                selected.append(selected_word)

        previous_text, following_text = _boundary_word(gap_start, gap_end)
        return _trim_boundary_echoes(selected, previous_text, following_text)

    candidates: list[tuple[float, float, float, float]] = []
    for gap_start, gap_end in uncovered:
        speech_seconds, speech_ratio = _speech_overlap_metrics(
            gap_start,
            gap_end,
            speech_segments_ms,
        )
        if (
            speech_seconds < MLX_SHORT_GAP_MIN_SPEECH_SECONDS
            or speech_ratio < MLX_SHORT_GAP_MIN_SPEECH_RATIO
        ):
            continue

        candidates.append((gap_start, gap_end, speech_seconds, speech_ratio))

    candidates.sort(key=lambda item: (item[3], item[2]), reverse=True)
    deferred = candidates[max(0, int(max_candidates)) :]
    candidates = candidates[: max(0, int(max_candidates))]
    candidates.sort(key=lambda item: item[0])

    recovered_segments: list[dict[str, Any]] = []
    recovered_ranges: list[dict[str, Any]] = []
    decoded_seconds = 0.0
    skipped_for_budget = 0
    updated = result
    issues = list(result.get("coverage_issues") or [])

    def _mark_unresolved(candidate, reason):
        start, end, speech_seconds, speech_ratio = candidate
        if (
            start > 0.05
            and end < audio_duration - 0.05
            and speech_seconds >= MLX_SHORT_GAP_REVIEW_MIN_SPEECH_SECONDS
            and speech_ratio >= MLX_SHORT_GAP_REVIEW_MIN_SPEECH_RATIO
        ):
            issues.append(
                {
                    "start": start,
                    "end": end,
                    "speech_seconds": speech_seconds,
                    "speech_ratio": speech_ratio,
                    "reason": reason,
                }
            )

    for candidate in deferred:
        _mark_unresolved(candidate, "candidate_budget")
    for gap_start, gap_end, speech_seconds, speech_ratio in candidates:
        candidate = (gap_start, gap_end, speech_seconds, speech_ratio)
        decode_cost = sum(
            min(audio_duration, gap_end + context_seconds) - max(0.0, gap_start - context_seconds)
            for context_seconds in MLX_SHORT_GAP_CONTEXT_SECONDS
        )
        if decoded_seconds + decode_cost > max(0.0, float(max_decode_seconds)):
            skipped_for_budget += 1
            _mark_unresolved(candidate, "decode_budget")
            continue
        decoded_seconds += decode_cost

        full_attempts.clear()
        attempts = [
            _decode_gap(gap_start, gap_end, speech_ratio, context_seconds)
            for context_seconds in MLX_SHORT_GAP_CONTEXT_SECONDS
        ]
        token_attempts = [
            _tokens("".join(str(word.get("word") or "") for word in attempt))
            for attempt in attempts
        ]
        agreed = not (
            len(attempts) != 2
            or not attempts[0]
            or token_attempts[0] != token_attempts[1]
            or len(token_attempts[0]) < 2
        )
        # VAD nominates the gap but never supplies words. If the right neighbor
        # starts too early, lexical anchors retain the omitted utterance's tail.
        anchored = [anchored_candidate(w, coverage, gap_start, gap_end) for w in full_attempts]
        needs_boundary_repair = any(w and w[-1]["end"] > gap_end + 0.05 for w in anchored)
        if not agreed or needs_boundary_repair:
            repaired = None
            strong_speech = (
                speech_seconds >= MLX_SHORT_GAP_REVIEW_MIN_SPEECH_SECONDS
                and speech_ratio >= MLX_SHORT_GAP_REVIEW_MIN_SPEECH_RATIO
            )
            if strong_speech and any(anchored):
                left, right = confirmation_window(result, gap_start, gap_end, audio_duration)
                if decoded_seconds + right - left <= max_decode_seconds:
                    decoded_seconds += right - left
                    confirmation = anchored_candidate(
                        _decode_words(left, right, speech_ratio),
                        coverage,
                        gap_start,
                        gap_end,
                    )
                    for proposed in anchored:
                        if corroborates(proposed, confirmation):
                            repaired = insert_anchored_gap(updated, proposed, gap_start, gap_end)
                            if repaired is not None:
                                updated = repaired
                                recovered_ranges.append(
                                    {
                                        "start": gap_start,
                                        "end": gap_end,
                                        "speech_seconds": speech_seconds,
                                        "speech_ratio": speech_ratio,
                                        "text": "".join(w["word"] for w in proposed).strip(),
                                        "method": "anchored_context_consensus",
                                    }
                                )
                                break
                else:
                    skipped_for_budget += 1
            if repaired is None:
                _mark_unresolved(candidate, "context_disagreement")
            continue

        words = attempts[0]
        text = "".join(str(word.get("word") or "") for word in words).strip()
        if not text:
            continue
        recovered_segments.append(
            {
                "text": text,
                "start": float(words[0]["start"]),
                "end": float(words[-1]["end"]),
                "words": words,
                "recovered_short_speech_gap": True,
            }
        )
        recovered_ranges.append(
            {
                "start": gap_start,
                "end": gap_end,
                "speech_seconds": speech_seconds,
                "speech_ratio": speech_ratio,
                "text": text,
            }
        )

    if not recovered_segments and not recovered_ranges and not issues:
        return result

    updated = dict(updated)
    updated["segments"] = sorted(
        [
            *[
                dict(segment)
                for segment in updated.get("segments") or []
                if isinstance(segment, dict)
            ],
            *recovered_segments,
        ],
        key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
    )
    updated["short_speech_gap_recovery"] = recovered_ranges
    updated["coverage_issues"] = issues
    updated["short_speech_gap_recovery_budget"] = {
        "candidates": len(candidates),
        "decoded_seconds": round(decoded_seconds, 3),
        "skipped": skipped_for_budget,
    }
    return updated


def _critical_aligned_speech_gaps(
    aligned: dict[str, Any],
    speech_segments_ms: list[tuple[int, int]],
    audio_duration: float,
) -> list[_SpeechBackedGap]:
    """Find internal VAD speech holes after alignment using word-level coverage."""
    coverage = _alignment_word_coverage(aligned)
    if not coverage:
        return []
    return [
        gap
        for gap in _find_speech_backed_mlx_gaps(
            coverage,
            speech_segments_ms,
            audio_duration,
            min_gap_seconds=MLX_SHORT_GAP_MIN_SECONDS,
        )
        if gap.is_internal
        and (
            (
                gap.speech_seconds >= MLX_FINAL_AUDIT_MIN_SPEECH_SECONDS
                and gap.speech_ratio >= MLX_FINAL_AUDIT_MIN_SPEECH_RATIO
            )
            or (
                gap.speech_seconds >= MLX_SHORT_GAP_REVIEW_MIN_SPEECH_SECONDS
                and gap.speech_ratio >= MLX_SHORT_GAP_REVIEW_MIN_SPEECH_RATIO
            )
        )
    ]


def _audit_aligned_speech_coverage(
    aligned: dict, audio: Any, sample_rate: int
) -> list[_SpeechBackedGap]:
    duration = len(audio) / sample_rate
    uncovered = _find_uncovered_mlx_gaps(
        _alignment_word_coverage(aligned),
        duration,
        min_gap_seconds=MLX_SHORT_GAP_MIN_SECONDS,
    )
    # Preserve the existing long-gap sensitivity; require stricter VAD only for
    # newly audited short holes, where music transients are more ambiguous.
    long_ranges = [(a, b) for a, b in uncovered if b - a >= MLX_GAP_RECOVERY_MIN_GAP_SECONDS]
    short_ranges = [(a, b) for a, b in uncovered if b - a < MLX_GAP_RECOVERY_MIN_GAP_SECONDS]
    speech = _detect_speech_in_mlx_gaps(audio, sample_rate, long_ranges)
    speech.extend(_detect_speech_in_mlx_gaps(audio, sample_rate, short_ranges, threshold=0.75))
    return _critical_aligned_speech_gaps(aligned, speech, duration)


def _classify_foreign_unresolved_ranges(
    ranges: list[dict[str, Any]],
    audio: Any,
    sample_rate: int,
    primary_language: str,
    transcribe_clip_auto: Callable[[Any], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate confirmed foreign speech from unresolved primary-language speech."""
    primary = _normalize_language(primary_language)
    unresolved: list[dict[str, Any]] = []
    foreign: list[dict[str, Any]] = []
    audio_duration = len(audio) / sample_rate if sample_rate > 0 else 0.0
    for item in ranges:
        # Unlike recovery decoding, language identification must not pull in
        # adjacent primary-language speech at a language-switch boundary.
        start = max(0.0, float(item.get("start", 0.0)))
        end = min(audio_duration, float(item.get("end", 0.0)))
        start_sample = max(0, round(start * sample_rate))
        end_sample = min(len(audio), round(end * sample_rate))
        if end_sample <= start_sample:
            unresolved.append(item)
            continue
        try:
            probe = transcribe_clip_auto(audio[start_sample:end_sample])
        except Exception:
            unresolved.append(item)
            continue
        detected = _normalize_language(str(probe.get("language") or ""))
        detected_text = str(probe.get("text") or "").strip()
        script_language = _script_language_evidence(detected_text)
        resolved_language = script_language or detected
        recovered_units = _lexical_unit_count(detected_text)
        if resolved_language and resolved_language != primary and recovered_units >= 2:
            foreign.append(
                {
                    **item,
                    "detected_language": resolved_language,
                    "model_language": detected,
                    "script_language": script_language,
                    "detected_text": detected_text,
                    "detected_units": recovered_units,
                }
            )
        else:
            unresolved.append(item)
    return unresolved, foreign


def _gap_is_covered_by_foreign_range(
    gap: _SpeechBackedGap,
    foreign_ranges: list[dict[str, Any]],
) -> bool:
    return _time_range_is_covered_by_ranges(gap.start, gap.end, foreign_ranges)


def _time_range_is_covered_by_ranges(
    start: float,
    end: float,
    ranges: list[dict[str, Any]],
) -> bool:
    duration = max(0.001, end - start)
    overlap = sum(
        max(
            0.0,
            min(end, float(item.get("end", 0.0))) - max(start, float(item.get("start", 0.0))),
        )
        for item in ranges
    )
    return overlap / duration >= 0.60


def _recover_aligned_gaps_from_native_words(
    aligned: dict[str, Any],
    native_result: dict[str, Any],
    gaps: list[_SpeechBackedGap],
) -> dict[str, Any]:
    """Fill forced-alignment holes with MLX's native word timestamps only."""
    if not gaps:
        return aligned

    recovered_segments: list[dict[str, Any]] = []
    recovered_words: list[dict[str, Any]] = []
    for segment in native_result.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        words = segment.get("words")
        if not isinstance(words, list):
            continue
        selected: list[dict[str, Any]] = []
        for word in words:
            if not isinstance(word, dict):
                continue
            start = _float_seconds(word.get("start"))
            end = _float_seconds(word.get("end"))
            if start is None or end is None or end <= start:
                continue
            midpoint = (start + end) / 2
            if not any(gap.start <= midpoint <= gap.end for gap in gaps):
                continue
            selected_word = dict(word)
            selected_word["timing_source"] = "native"
            selected.append(selected_word)
        if not selected:
            continue
        text = "".join(str(word.get("word") or "") for word in selected).strip()
        if not text:
            continue
        recovered_segments.append(
            {
                "text": text,
                "start": float(selected[0]["start"]),
                "end": float(selected[-1]["end"]),
                "words": selected,
                "native_word_fallback": True,
            }
        )
        recovered_words.extend(selected)

    if not recovered_segments:
        return aligned
    updated = dict(aligned)
    updated["segments"] = sorted(
        [
            *[
                dict(segment)
                for segment in aligned.get("segments") or []
                if isinstance(segment, dict)
            ],
            *recovered_segments,
        ],
        key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
    )
    updated["word_segments"] = sorted(
        [
            *[dict(word) for word in aligned.get("word_segments") or [] if isinstance(word, dict)],
            *recovered_words,
        ],
        key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
    )
    updated["native_word_gap_recovery"] = len(recovered_segments)
    return updated


def _exit_mlx_worker(exit_code: int) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream:
                stream.flush()
        except Exception:
            pass
    os._exit(exit_code)


def run_packaged_mlx_whisper_worker() -> None:
    """Run MLX decoding outside the desktop process so the UI remains responsive."""
    request_path = Path(os.environ[_MLX_WORKER_REQUEST])
    output_path = Path(os.environ[_MLX_WORKER_OUTPUT])
    try:
        import mlx_whisper

        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(line_buffering=True, write_through=True)

        request = json.loads(request_path.read_text(encoding="utf-8"))
        if request.get("operation") == "classify_ranges":
            from mlx_whisper.audio import SAMPLE_RATE
            from mlx_whisper.audio import load_audio as load_mlx_audio

            audio = load_mlx_audio(str(request["audio"]))

            def transcribe_clip_auto(clip: Any) -> dict[str, Any]:
                local_kwargs: dict[str, Any] = {
                    "path_or_hf_repo": str(request["model"]),
                    "word_timestamps": False,
                    "condition_on_previous_text": False,
                    "verbose": None,
                }
                try:
                    return dict(mlx_whisper.transcribe(clip, **local_kwargs))
                except TypeError:
                    local_kwargs.pop("condition_on_previous_text", None)
                    return dict(mlx_whisper.transcribe(clip, **local_kwargs))

            review_ranges = [
                dict(item) for item in request.get("ranges") or [] if isinstance(item, dict)
            ]
            unresolved, foreign_ranges = _classify_foreign_unresolved_ranges(
                review_ranges,
                audio,
                SAMPLE_RATE,
                str(request.get("primary_language") or ""),
                transcribe_clip_auto,
            )
            _atomic_json_write(
                output_path,
                {
                    "ok": True,
                    "data": {
                        "unresolved_ranges": unresolved,
                        "foreign_language_speech_ranges": foreign_ranges,
                    },
                },
            )
            _exit_mlx_worker(0)

        kwargs: dict[str, Any] = {
            "path_or_hf_repo": str(request["model"]),
            "word_timestamps": bool(request.get("word_timestamps")),
            "condition_on_previous_text": False,
            # MLX emits each completed segment in a stable timestamped format.
            # The parent process tails these lines for UI previews; the final
            # result still comes from the structured JSON payload below.
            "verbose": True,
        }
        language = str(request.get("language") or "").strip()
        if language:
            kwargs["language"] = language
        try:
            result = mlx_whisper.transcribe(str(request["audio"]), **kwargs)
        except TypeError:
            kwargs.pop("condition_on_previous_text", None)
            result = mlx_whisper.transcribe(str(request["audio"]), **kwargs)
        try:
            from mlx_whisper.audio import SAMPLE_RATE
            from mlx_whisper.audio import load_audio as load_mlx_audio

            audio = load_mlx_audio(str(request["audio"]))
            source_segments = [
                dict(segment)
                for segment in result.get("segments") or []
                if isinstance(segment, dict)
            ]

            def transcribe_clip(
                clip: Any,
                *,
                word_timestamps: bool = False,
            ) -> dict[str, Any]:
                local_kwargs: dict[str, Any] = {
                    "path_or_hf_repo": str(request["model"]),
                    "word_timestamps": word_timestamps,
                    "condition_on_previous_text": False,
                    "verbose": None,
                }
                if language:
                    local_kwargs["language"] = language
                try:
                    return dict(mlx_whisper.transcribe(clip, **local_kwargs))
                except TypeError:
                    local_kwargs.pop("condition_on_previous_text", None)
                    return dict(mlx_whisper.transcribe(clip, **local_kwargs))

            def transcribe_clip_auto(clip: Any) -> dict[str, Any]:
                local_kwargs: dict[str, Any] = {
                    "path_or_hf_repo": str(request["model"]),
                    "word_timestamps": False,
                    "condition_on_previous_text": False,
                    "verbose": None,
                }
                try:
                    return dict(mlx_whisper.transcribe(clip, **local_kwargs))
                except TypeError:
                    local_kwargs.pop("condition_on_previous_text", None)
                    return dict(mlx_whisper.transcribe(clip, **local_kwargs))

            sparse_candidates = _find_sparse_mlx_segments(source_segments)
            sparse_ranges = [(item.start, item.end) for item in sparse_candidates]
            sparse_speech = _detect_speech_in_mlx_gaps(
                audio,
                SAMPLE_RATE,
                sparse_ranges,
            )
            result = _recover_mlx_sparse_segments(
                dict(result),
                audio,
                SAMPLE_RATE,
                sparse_speech,
                lambda clip: transcribe_clip(clip, word_timestamps=True),
            )

            source_segments = [
                dict(segment)
                for segment in result.get("segments") or []
                if isinstance(segment, dict)
            ]
            uncovered_gaps = _find_uncovered_mlx_gaps(
                source_segments,
                len(audio) / SAMPLE_RATE,
            )
            speech_segments = _detect_speech_in_mlx_gaps(
                audio,
                SAMPLE_RATE,
                uncovered_gaps,
            )
            result = _recover_mlx_speech_gaps(
                dict(result),
                audio,
                SAMPLE_RATE,
                speech_segments,
                lambda clip: transcribe_clip(clip, word_timestamps=True),
            )

            # Audit brief internal holes separately. Natural pauses and music
            # are common, so a short phrase is restored only when high-confidence
            # VAD finds speech and two context windows decode the same new words.
            short_coverage = _alignment_word_coverage(result)
            short_uncovered = _find_uncovered_mlx_gaps(
                short_coverage,
                len(audio) / SAMPLE_RATE,
                min_gap_seconds=MLX_SHORT_GAP_MIN_SECONDS,
                max_gap_seconds=MLX_SHORT_GAP_MAX_SECONDS,
            )
            short_speech = _detect_speech_in_mlx_gaps(
                audio,
                SAMPLE_RATE,
                short_uncovered,
                threshold=0.75,
                min_speech_ms=120,
                min_silence_ms=180,
            )
            result = _recover_short_mlx_speech_gaps(
                dict(result),
                audio,
                SAMPLE_RATE,
                short_speech,
                lambda clip: transcribe_clip(clip, word_timestamps=True),
            )
        except Exception as exc:
            raise RuntimeError("MLX speech-coverage recovery failed") from exc

        critical_unresolved = [
            gap
            for gap in result.get("unresolved_speech_gaps") or []
            if gap.get("is_internal")
            and float(gap.get("speech_seconds", 0.0)) >= 5.0
            and float(gap.get("speech_ratio", 0.0)) >= 0.75
        ]
        critical_unresolved.extend(
            gap
            for gap in result.get("unresolved_sparse_segments") or []
            if gap.get("is_internal")
            and float(gap.get("speech_seconds", 0.0)) >= MLX_SPARSE_RECOVERY_MIN_SPEECH_SECONDS
            and float(gap.get("speech_ratio", 0.0)) >= MLX_SPARSE_RECOVERY_MIN_SPEECH_RATIO
        )
        if language:
            recovered_ranges = [
                dict(item)
                for key in ("speech_gap_recovery", "sparse_segment_recovery")
                for item in result.get(key) or []
                if isinstance(item, dict)
            ]
            review_ranges = [*critical_unresolved, *recovered_ranges]
            _, foreign_ranges = _classify_foreign_unresolved_ranges(
                review_ranges,
                audio,
                SAMPLE_RATE,
                language,
                transcribe_clip_auto,
            )
            if foreign_ranges:
                result["foreign_language_speech_ranges"] = foreign_ranges
                critical_unresolved = [
                    item
                    for item in critical_unresolved
                    if not _time_range_is_covered_by_ranges(
                        float(item.get("start", 0.0)),
                        float(item.get("end", 0.0)),
                        foreign_ranges,
                    )
                ]
        elif critical_unresolved:
            # Auto-language jobs resolve these ranges in the parent process,
            # where per-language alignment models and user prompts are available.
            result["deferred_unresolved_coverage"] = critical_unresolved
        if critical_unresolved:
            windows = ", ".join(
                f"{float(gap['start']):.1f}-{float(gap['end']):.1f}s" for gap in critical_unresolved
            )
            raise RuntimeError(
                "MLX Whisper left confirmed speech untranslated after retry: " + windows
            )
        _atomic_json_write(output_path, {"ok": True, "data": result})
        exit_code = 0
    except BaseException:
        try:
            _atomic_json_write(output_path, {"ok": False, "error": traceback.format_exc()})
        except OSError:
            pass
        exit_code = 1
    _exit_mlx_worker(exit_code)


def _segments_for_alignment(result: dict) -> list[dict]:
    segments: list[dict] = []
    for item in result.get("segments") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        start = item.get("start")
        end = item.get("end")
        if not text or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        segments.append({"text": text, "start": float(start), "end": float(end)})
    return segments


_SMALL_NUMBERS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_UNIT_NAMES = {
    "hp": "horsepower",
    "mph": "miles per hour",
    "kph": "kilometers per hour",
    "kmh": "kilometers per hour",
    "kg": "kilograms",
    "lb": "pounds",
    "lbs": "pounds",
    "ft": "feet",
    "nm": "newton meters",
    "rpm": "R P M",
}


@dataclass(frozen=True)
class _AlignmentToken:
    display_text: str
    spoken_text: str
    spoken_word_count: int


@dataclass(frozen=True)
class _AlignmentSegmentPlan:
    text: str
    start: float
    end: float
    tokens: tuple[_AlignmentToken, ...]


@dataclass(frozen=True)
class _LanguageProbe:
    start: float
    end: float
    language: str
    confidence: float
    primary_confidence: float
    runner_up_language: str = ""
    runner_up_confidence: float = 0.0
    speech_seconds: float = 0.0
    speech_ratio: float = 0.0
    source: str = "window"


@dataclass(frozen=True)
class _LanguageRange:
    start: float
    end: float
    language: str
    confidence: float
    support_seconds: float = 0.0
    probe_count: int = 0
    dominance: float = 1.0
    needs_confirmation: bool = False
    decoded_language: str = ""
    script_language: str = ""
    confirmation_agreement: bool = False


_JAPANESE_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_KOREAN_HANGUL_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")
_CJK_CONFUSION_LANGUAGES = {"ja", "ko", "zh", "yue"}


def _union_duration(ranges: list[tuple[float, float]]) -> float:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def _script_language_evidence(text: str) -> str:
    """Return strong script evidence without guessing from shared Han characters."""
    kana = len(_JAPANESE_KANA_RE.findall(text))
    hangul = len(_KOREAN_HANGUL_RE.findall(text))
    if kana >= 2 and kana >= hangul * 2:
        return "ja"
    if hangul >= 2 and hangul >= kana * 2:
        return "ko"
    return ""


def _language_ranges_from_foreign_audit(
    ranges: list[dict[str, Any]],
    primary_language: str,
) -> list[_LanguageRange]:
    primary = str(_normalize_language(primary_language) or primary_language).lower()
    converted: list[_LanguageRange] = []
    for item in ranges:
        start = item.get("start")
        end = item.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        start = float(start)
        end = float(end)
        if end <= start:
            continue
        text = str(item.get("detected_text") or "")
        script_language = str(item.get("script_language") or "").lower()
        detected_language = str(item.get("detected_language") or "").lower()
        language = script_language or _script_language_evidence(text) or detected_language
        if not language or language == primary:
            continue
        converted.append(
            _LanguageRange(
                start=start,
                end=end,
                language=language,
                confidence=1.0,
                support_seconds=float(item.get("speech_seconds") or end - start),
                probe_count=1,
                dominance=1.0,
                decoded_language=str(item.get("model_language") or detected_language).lower(),
                script_language=script_language or _script_language_evidence(text),
                confirmation_agreement=detected_language == language,
            )
        )
    return _merge_confirmed_language_ranges(converted)


def _candidate_foreign_language_ranges(
    probes: list[_LanguageProbe],
    primary_language: str,
) -> list[_LanguageRange]:
    """Build local foreign-language events without counting overlap globally."""
    candidates = []
    for probe in probes:
        competing_confidence = max(
            probe.primary_confidence,
            probe.runner_up_confidence,
        )
        if (
            probe.language != primary_language
            and probe.confidence >= MIXED_LANGUAGE_MIN_CONFIDENCE
            and probe.primary_confidence <= MIXED_LANGUAGE_MAX_PRIMARY_CONFIDENCE
            and probe.confidence - competing_confidence >= MIXED_LANGUAGE_MIN_MARGIN
        ):
            candidates.append(probe)
    if not candidates:
        return []

    episodes: list[list[_LanguageProbe]] = []
    for probe in sorted(candidates, key=lambda item: (item.start, item.end)):
        if episodes and probe.start - max(item.end for item in episodes[-1]) <= (
            MIXED_LANGUAGE_MAX_GAP_SECONDS
        ):
            episodes[-1].append(probe)
        else:
            episodes.append([probe])

    ranges: list[_LanguageRange] = []
    for episode in episodes:
        # Window cores are non-overlapping evidence. Segment probes refine the
        # boundaries but must not double-count the same speech.
        support_probes = [item for item in episode if item.source == "window"] or episode
        by_language: dict[str, list[_LanguageProbe]] = {}
        for probe in support_probes:
            by_language.setdefault(probe.language, []).append(probe)

        support_by_language = {
            language: sum(
                min(max(0.0, item.end - item.start), max(0.0, item.speech_seconds))
                if item.speech_seconds > 0
                else max(0.0, item.end - item.start)
                for item in items
            )
            for language, items in by_language.items()
        }
        winner = max(
            support_by_language,
            key=lambda language: (
                support_by_language[language],
                max(item.confidence for item in by_language[language]),
            ),
        )
        winner_probes = by_language[winner]
        total_support = sum(
            min(max(0.0, item.end - item.start), max(0.0, item.speech_seconds))
            if item.speech_seconds > 0
            else max(0.0, item.end - item.start)
            for item in support_probes
        )
        winner_support = support_by_language[winner]
        dominance = winner_support / max(0.001, total_support)
        stable = (
            winner_support >= MIXED_LANGUAGE_MIN_LANGUAGE_SUPPORT_SECONDS
            and len(winner_probes) >= MIXED_LANGUAGE_MIN_PROBE_COUNT
            and dominance >= MIXED_LANGUAGE_MIN_DOMINANCE
        )
        ranges.append(
            _LanguageRange(
                start=min(item.start for item in episode),
                end=max(item.end for item in episode),
                language=winner,
                confidence=max(item.confidence for item in winner_probes),
                support_seconds=winner_support,
                probe_count=len(winner_probes),
                dominance=dominance,
                needs_confirmation=not stable
                or len({item.language for item in episode} & _CJK_CONFUSION_LANGUAGES) > 1,
            )
        )
    return ranges


def _select_foreign_language_ranges(
    probes: list[_LanguageProbe],
    primary_language: str,
) -> list[_LanguageRange]:
    """Return stable events; tentative short events require transcript confirmation."""
    return [
        item
        for item in _candidate_foreign_language_ranges(probes, primary_language)
        if not item.needs_confirmation
    ]


def _confirm_foreign_language_range(
    candidate: _LanguageRange,
    primary_language: str,
    decoded_language: str,
    decoded_text: str,
) -> _LanguageRange | None:
    """Confirm an event with an independent full-event decode and script evidence."""
    primary = str(_normalize_language(primary_language) or primary_language).lower()
    detected = str(_normalize_language(decoded_language) or "").lower()
    units = _lexical_unit_count(decoded_text)
    if units < MIXED_LANGUAGE_MIN_DECODED_UNITS:
        return None

    script_language = _script_language_evidence(decoded_text)
    if script_language:
        if script_language == primary:
            return None
        agreement = candidate.language == script_language and detected == script_language
        if candidate.needs_confirmation and (
            units < MIXED_LANGUAGE_MIN_DECODED_UNITS
            or (
                detected
                and detected != script_language
                and not {
                    detected,
                    candidate.language,
                    script_language,
                }.issubset(_CJK_CONFUSION_LANGUAGES)
            )
        ):
            return None
        return _LanguageRange(
            start=candidate.start,
            end=candidate.end,
            language=script_language,
            confidence=candidate.confidence,
            support_seconds=candidate.support_seconds,
            probe_count=candidate.probe_count,
            dominance=candidate.dominance,
            needs_confirmation=False,
            decoded_language=detected,
            script_language=script_language,
            confirmation_agreement=agreement,
        )

    candidate_is_cjk = candidate.language in _CJK_CONFUSION_LANGUAGES
    detected_is_cjk = detected in _CJK_CONFUSION_LANGUAGES
    if candidate_is_cjk or detected_is_cjk:
        # Han-only text cannot distinguish Chinese from Japanese, and a short
        # Japanese/Korean decode without kana or Hangul is not strong evidence.
        if (
            candidate.needs_confirmation
            or detected != candidate.language
            or candidate.dominance < 0.75
        ):
            return None
    elif candidate.needs_confirmation:
        if detected != candidate.language or units < 3 or candidate.confidence < 0.90:
            return None
    elif detected and detected != candidate.language:
        return None

    if detected == primary:
        return None
    if not detected and candidate.dominance < 0.80:
        return None
    return _LanguageRange(
        start=candidate.start,
        end=candidate.end,
        language=candidate.language,
        confidence=candidate.confidence,
        support_seconds=candidate.support_seconds,
        probe_count=candidate.probe_count,
        dominance=candidate.dominance,
        needs_confirmation=False,
        decoded_language=detected,
        confirmation_agreement=detected == candidate.language,
    )


def _stabilize_confirmed_language_ranges(
    ranges: list[_LanguageRange],
) -> tuple[list[_LanguageRange], list[_LanguageRange]]:
    """Suppress weak minority CJK labels within one recording or chunk."""
    cjk_ranges = [item for item in ranges if item.language in _CJK_CONFUSION_LANGUAGES]
    cjk_languages = {item.language for item in cjk_ranges}
    if len(cjk_languages) <= 1:
        return ranges, []

    support_by_language = {
        language: sum(item.support_seconds for item in cjk_ranges if item.language == language)
        for language in cjk_languages
    }
    dominant_language = max(
        support_by_language,
        key=lambda language: support_by_language[language],
    )
    dominant_support = support_by_language[dominant_language]
    retained_languages = {dominant_language}
    for language in cjk_languages - {dominant_language}:
        language_ranges = [item for item in cjk_ranges if item.language == language]
        agreed_events = sum(item.confirmation_agreement for item in language_ranges)
        support = support_by_language[language]
        if (
            support >= MIXED_LANGUAGE_MINORITY_CJK_SUPPORT_SECONDS
            and agreed_events >= MIXED_LANGUAGE_MINORITY_CJK_AGREED_EVENTS
            and support >= dominant_support * MIXED_LANGUAGE_MINORITY_CJK_SUPPORT_RATIO
        ):
            retained_languages.add(language)

    retained = [
        item
        for item in ranges
        if item.language not in _CJK_CONFUSION_LANGUAGES or item.language in retained_languages
    ]
    rejected = [item for item in ranges if item not in retained]
    return retained, rejected


def _candidate_confirmed_language_bridges(
    ranges: list[_LanguageRange],
    speech_segments_ms: list[tuple[int, int]],
) -> list[tuple[_LanguageRange, _LanguageRange, float, float]]:
    """Find short voiced gaps that may continue the same confirmed CJK language."""
    candidates: list[tuple[_LanguageRange, _LanguageRange, float, float]] = []
    ordered = sorted(ranges, key=lambda item: (item.start, item.end))
    for left, right in zip(ordered, ordered[1:]):
        gap = right.start - left.end
        if (
            left.language != right.language
            or left.language not in _CJK_CONFUSION_LANGUAGES
            or gap <= 0
            or gap > MIXED_LANGUAGE_MAX_CONFIRMED_BRIDGE_SECONDS
        ):
            continue
        speech_seconds, speech_ratio = _speech_overlap_metrics(
            left.end,
            right.start,
            speech_segments_ms,
        )
        if (
            speech_seconds >= MIXED_LANGUAGE_MIN_BRIDGE_SPEECH_SECONDS
            and speech_ratio >= MIXED_LANGUAGE_MIN_BRIDGE_SPEECH_RATIO
        ):
            candidates.append((left, right, speech_seconds, speech_ratio))
    return candidates


def _merge_confirmed_language_ranges(
    ranges: list[_LanguageRange],
) -> list[_LanguageRange]:
    merged: list[_LanguageRange] = []
    for item in sorted(ranges, key=lambda value: (value.start, value.end)):
        if (
            merged
            and merged[-1].language == item.language
            and item.start - merged[-1].end <= MIXED_LANGUAGE_MAX_GAP_SECONDS
        ):
            previous = merged[-1]
            total_support = previous.support_seconds + item.support_seconds
            dominance = (
                previous.dominance * previous.support_seconds
                + item.dominance * item.support_seconds
            ) / max(0.001, total_support)
            merged[-1] = _LanguageRange(
                start=previous.start,
                end=max(previous.end, item.end),
                language=previous.language,
                confidence=max(previous.confidence, item.confidence),
                support_seconds=total_support,
                probe_count=previous.probe_count + item.probe_count,
                dominance=dominance,
                needs_confirmation=False,
            )
        else:
            merged.append(item)
    return merged


def _is_severe_repetition_hallucination(text: str) -> bool:
    tokens = re.findall(r"[\w']+", text.lower())
    if len(tokens) < 12:
        return False
    counts: dict[str, int] = {}
    longest_run = 1
    current_run = 1
    for index, token in enumerate(tokens):
        counts[token] = counts.get(token, 0) + 1
        if index > 0 and token == tokens[index - 1]:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1
    peak = max(counts.values(), default=0)
    return longest_run >= 8 or (peak >= 12 and peak / len(tokens) >= 0.60)


def _expand_language_ranges_over_repetition(
    ranges: list[_LanguageRange],
    source_segments: list[dict[str, Any]],
) -> tuple[list[_LanguageRange], list[tuple[float, float]]]:
    """Replace decoder loops next to confirmed foreign speech with that language."""
    if not ranges:
        return ranges, []

    additions: list[_LanguageRange] = []
    expanded_segments: list[tuple[float, float]] = []
    for segment in source_segments:
        text = str(segment.get("text") or "").strip()
        start = segment.get("start")
        end = segment.get("end")
        if (
            not _is_severe_repetition_hallucination(text)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
        ):
            continue
        start_seconds = float(start)
        end_seconds = float(end)
        if (
            end_seconds <= start_seconds
            or end_seconds - start_seconds > MIXED_LANGUAGE_REPETITION_MAX_RANGE_SECONDS
        ):
            continue
        nearby = [
            item
            for item in ranges
            if max(item.start - end_seconds, start_seconds - item.end, 0.0)
            <= MIXED_LANGUAGE_REPETITION_NEARBY_SECONDS
        ]
        if not nearby:
            continue
        closest = min(
            nearby,
            key=lambda item: (
                max(item.start - end_seconds, start_seconds - item.end, 0.0),
                -item.confidence,
            ),
        )
        additions.append(
            _LanguageRange(
                start=start_seconds,
                end=end_seconds,
                language=closest.language,
                confidence=closest.confidence,
                support_seconds=closest.support_seconds,
                probe_count=closest.probe_count,
                dominance=closest.dominance,
                decoded_language=closest.decoded_language,
                script_language=closest.script_language,
                confirmation_agreement=closest.confirmation_agreement,
            )
        )
        expanded_segments.append((start_seconds, end_seconds))

    return _merge_confirmed_language_ranges(ranges + additions), expanded_segments


def _subtract_language_ranges(
    start: float,
    end: float,
    ranges: list[_LanguageRange],
) -> list[tuple[float, float]]:
    """Return portions of an ASR segment not covered by replacement ranges."""
    if end <= start:
        return []
    remaining: list[tuple[float, float]] = []
    cursor = start
    for item in sorted(ranges, key=lambda value: (value.start, value.end)):
        if item.end <= cursor:
            continue
        if item.start >= end:
            break
        if item.start > cursor:
            remaining.append((cursor, min(end, item.start)))
        cursor = max(cursor, min(end, item.end))
        if cursor >= end:
            break
    if cursor < end:
        remaining.append((cursor, end))
    return [(part_start, part_end) for part_start, part_end in remaining if part_end > part_start]


def _integer_to_english(value: int) -> str:
    if value < 0:
        return f"minus {_integer_to_english(-value)}"
    if value < 20:
        return _SMALL_NUMBERS[value]
    if value < 100:
        tens, remainder = divmod(value, 10)
        return _TENS[tens] if not remainder else f"{_TENS[tens]} {_SMALL_NUMBERS[remainder]}"
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        result = f"{_SMALL_NUMBERS[hundreds]} hundred"
        return result if not remainder else f"{result} {_integer_to_english(remainder)}"
    for scale, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")):
        if value >= scale:
            leading, remainder = divmod(value, scale)
            result = f"{_integer_to_english(leading)} {name}"
            return result if not remainder else f"{result} {_integer_to_english(remainder)}"
    return " ".join(_SMALL_NUMBERS[int(digit)] for digit in str(value))


def _number_to_english(value: str, *, allow_year: bool = True) -> str | None:
    compact = value.replace(",", "")
    if not re.fullmatch(r"\d+(?:\.\d+)?", compact):
        return None
    if "." in compact:
        whole, fraction = compact.split(".", 1)
        return f"{_integer_to_english(int(whole))} point {' '.join(_SMALL_NUMBERS[int(d)] for d in fraction)}"
    number = int(compact)
    if allow_year and len(compact) == 4 and 1900 <= number <= 2099:
        first, second = divmod(number, 100)
        if second == 0:
            return f"{_integer_to_english(first)} hundred"
        if number < 2000:
            return f"{_integer_to_english(first)} {_integer_to_english(second)}"
        if number < 2010:
            return f"two thousand {_integer_to_english(second)}"
        return f"twenty {_integer_to_english(second)}"
    return _integer_to_english(number)


def _spoken_token(token: str) -> str:
    """Return deterministic English speech text for one display token."""
    leading = re.match(r"^[\(\[\{\"']*", token).group(0)  # type: ignore[union-attr]
    trailing_match = re.search(r"[\)\]\}\"',.!?;:]*$", token)
    trailing = trailing_match.group(0) if trailing_match else ""
    core_end = len(token) - len(trailing) if trailing else len(token)
    core = token[len(leading) : core_end]
    if not core:
        return token

    punctuation = next((ch for ch in reversed(trailing) if ch in ".!?"), "")

    def finish(text: str) -> str:
        return f"{text}{punctuation}" if punctuation else text

    tire = parse_tire_size(core)
    if tire:
        hundreds, remainder = divmod(tire.width, 100)
        if remainder == 0:
            width = f"{_integer_to_english(hundreds)} hundred"
        elif remainder < 10:
            width = f"{_integer_to_english(hundreds)} oh {_integer_to_english(remainder)}"
        else:
            width = f"{_integer_to_english(hundreds)} {_integer_to_english(remainder)}"
        return finish(
            f"{width} {_integer_to_english(tire.aspect)} "
            f"{' '.join(tire.construction)} {_integer_to_english(tire.rim)}"
        )

    if core == "&":
        return finish("and")
    if core == "+":
        return finish("plus")

    currency_names = {"$": "dollars", "£": "pounds", "€": "euros"}
    magnitude_names = {
        "k": "thousand",
        "m": "million",
        "mn": "million",
        "b": "billion",
        "bn": "billion",
    }

    currency_range = re.fullmatch(
        r"([\$£€])(\d[\d,]*(?:\.\d+)?)[-–](\d[\d,]*(?:\.\d+)?)"
        r"(K|M|MN|B|BN)?",
        core,
        re.IGNORECASE,
    )
    if currency_range:
        left = _number_to_english(currency_range.group(2), allow_year=False)
        right = _number_to_english(currency_range.group(3), allow_year=False)
        magnitude = magnitude_names.get((currency_range.group(4) or "").lower(), "")
        parts = [left, "to", right, magnitude, currency_names[currency_range.group(1)]]
        return finish(" ".join(part for part in parts if part))

    currency_magnitude = re.fullmatch(
        r"([\$£€])(\d[\d,]*(?:\.\d+)?)(K|M|MN|B|BN)",
        core,
        re.IGNORECASE,
    )
    if currency_magnitude:
        amount = _number_to_english(currency_magnitude.group(2), allow_year=False)
        magnitude = magnitude_names[currency_magnitude.group(3).lower()]
        return finish(f"{amount} {magnitude} {currency_names[currency_magnitude.group(1)]}")

    currency = re.fullmatch(r"([\$£€])(\d[\d,]*(?:\.\d+)?)", core)
    if currency:
        amount = _number_to_english(currency.group(2), allow_year=False)
        return finish(f"{amount} {currency_names[currency.group(1)]}")

    percent = re.fullmatch(r"(\d[\d,]*(?:\.\d+)?)%", core)
    if percent:
        return finish(f"{_number_to_english(percent.group(1), allow_year=False)} percent")

    number_range = re.fullmatch(r"(\d+(?:\.\d+)?)[-–](\d+(?:\.\d+)?)", core)
    if number_range:
        left = _number_to_english(number_range.group(1), allow_year=False)
        right = _number_to_english(number_range.group(2), allow_year=False)
        return finish(f"{left} to {right}")

    number_unit = re.fullmatch(r"(\d[\d,]*(?:\.\d+)?)([A-Za-z]+)", core)
    if number_unit:
        number = _number_to_english(number_unit.group(1), allow_year=False)
        unit = number_unit.group(2)
        spoken_unit = _UNIT_NAMES.get(unit.lower())
        if spoken_unit:
            return finish(f"{number} {spoken_unit}")
        if len(unit) <= 3:
            return finish(f"{number} {' '.join(unit.upper())}")

    model_name = re.fullmatch(r"([A-Za-z]{1,3})(\d+)", core)
    if model_name:
        letters = " ".join(model_name.group(1).upper())
        number = _number_to_english(model_name.group(2), allow_year=False)
        return finish(f"{letters} {number}")

    number = _number_to_english(core)
    if number:
        return finish(number)

    if core.isupper() and core.isalpha() and 3 <= len(core) <= 6:
        return finish(" ".join(core))
    return token


def _compound_timing_candidates(data: ASRData) -> list[tuple[int, int]]:
    """Find size endings followed by a suspicious gap, without filling pauses."""
    if not data.is_word_timestamp():
        return []
    candidates = []
    for index, segment in enumerate(data.segments[:-1]):
        if segment.language_code not in {"", "en"}:
            continue
        last = index
        tire = parse_tire_size(segment.text)
        if tire is None:
            tire = parse_tire_size(f"{segment.text} {data.segments[index + 1].text}")
            last += 1
        if tire is None or last + 1 >= len(data.segments):
            continue
        gap = data.segments[last + 1].start_time - data.segments[last].end_time
        if 220 <= gap <= 3200:
            candidates.append((index, last))
    return candidates


def _prepare_spoken_alignment(
    segments: list[dict], language_code: str
) -> tuple[list[dict], list[_AlignmentSegmentPlan] | None]:
    if language_code != "en":
        return segments, None

    normalized: list[dict] = []
    plans: list[_AlignmentSegmentPlan] = []
    changed = False
    for segment in segments:
        display_tokens = re.findall(r"\S+", segment["text"])
        spoken_tokens = [_spoken_token(token) for token in display_tokens]
        # Keep the two display tokens separate when the transcription writes
        # "245/35 ZR19", but align the size as speech rather than as a range.
        for index in range(len(display_tokens) - 1):
            left, right = display_tokens[index : index + 2]
            tire = parse_tire_size(f"{left} {right}")
            if tire is not None:
                spoken = _spoken_token(f"{left}{right}").split()
                suffix_count = len(_spoken_token(right).split())
                spoken_tokens[index] = " ".join(spoken[:-suffix_count])
                spoken_tokens[index + 1] = " ".join(spoken[-suffix_count:])
        changed |= spoken_tokens != display_tokens
        plan_tokens = tuple(
            _AlignmentToken(display, spoken, max(1, len(spoken.split())))
            for display, spoken in zip(display_tokens, spoken_tokens)
        )
        normalized.append({**segment, "text": " ".join(spoken_tokens)})
        plans.append(
            _AlignmentSegmentPlan(
                text=segment["text"],
                start=float(segment["start"]),
                end=float(segment["end"]),
                tokens=plan_tokens,
            )
        )
    return (normalized, plans) if changed else (segments, None)


def _alignment_char_key(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _restore_segment_display_words(
    aligned_words: list[dict], plan: _AlignmentSegmentPlan
) -> list[dict] | None:
    """Map display tokens onto aligned speech without assuming word-token parity.

    WhisperX can split contractions or omit one unalignable word. The previous
    global word-count check discarded number expansion for the entire video in
    that case, shortening every compact numeric token. Character ranges keep
    the mapping local and preserve all unaffected numeric/model tokens.
    """
    expected_ranges: list[tuple[int, int]] = []
    expected_text = ""
    for token in plan.tokens:
        key = _alignment_char_key(token.spoken_text)
        start = len(expected_text)
        expected_text += key
        expected_ranges.append((start, len(expected_text)))

    actual_ranges: list[tuple[int, int, dict]] = []
    actual_text = ""
    for word in aligned_words:
        key = _alignment_char_key(_word_text(word))
        if not key:
            continue
        start = len(actual_text)
        actual_text += key
        actual_ranges.append((start, len(actual_text), word))

    if not expected_text or not actual_text:
        return None

    from difflib import SequenceMatcher

    matcher = SequenceMatcher(None, expected_text, actual_text, autojunk=False)
    expected_to_actual: dict[int, int] = {}
    for expected_start, actual_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            expected_to_actual[expected_start + offset] = actual_start + offset

    restored_words: list[dict] = []
    mapped_tokens = 0
    for token, (expected_start, expected_end) in zip(plan.tokens, expected_ranges):
        mapped_positions = [
            expected_to_actual[position]
            for position in range(expected_start, expected_end)
            if position in expected_to_actual
        ]
        restored: dict[str, Any] = {"word": token.display_text}
        if mapped_positions:
            coverage = len(mapped_positions) / max(1, expected_end - expected_start)
            overlapping = [
                word
                for actual_start, actual_end, word in actual_ranges
                if actual_end > min(mapped_positions) and actual_start <= max(mapped_positions)
            ]
            starts = [_float_seconds(word.get("start")) for word in overlapping]
            ends = [_float_seconds(word.get("end")) for word in overlapping]
            valid_starts = [value for value in starts if value is not None]
            valid_ends = [value for value in ends if value is not None]
            # A compact number may represent several spoken words. Require
            # broad character coverage so a partial match cannot shorten it.
            minimum_coverage = 0.75 if token.spoken_word_count > 1 else 0.5
            complete_edges = token.spoken_word_count == 1 or (
                expected_start in expected_to_actual and expected_end - 1 in expected_to_actual
            )
            if complete_edges and token.spoken_word_count > 1:
                # A matching suffix with no acoustic timing is still partial:
                # do not silently use the preceding spoken word as its end.
                for position, time_key in ((expected_start, "start"), (expected_end - 1, "end")):
                    actual_position = expected_to_actual[position]
                    edge_word = next(
                        word
                        for left, right, word in actual_ranges
                        if left <= actual_position < right
                    )
                    if _float_seconds(edge_word.get(time_key)) is None:
                        complete_edges = False
                        break
            if coverage >= minimum_coverage and complete_edges and valid_starts and valid_ends:
                restored["start"] = min(valid_starts)
                restored["end"] = max(valid_ends)
                scores = [
                    float(score)
                    for word in overlapping
                    if isinstance((score := word.get("score")), (int, float))
                ]
                if scores:
                    restored["score"] = round(sum(scores) / len(scores), 3)
                mapped_tokens += 1
        restored_words.append(restored)

    return restored_words if mapped_tokens else None


def _restore_display_alignment(aligned: dict, plans: list[_AlignmentSegmentPlan]) -> dict | None:
    aligned_segments = [
        segment for segment in aligned.get("segments") or [] if isinstance(segment, dict)
    ]
    if not aligned_segments or not plans:
        return None

    word_buckets: list[list[dict]] = [[] for _ in plans]
    for aligned_segment in aligned_segments:
        segment_words = [
            word for word in aligned_segment.get("words") or [] if isinstance(word, dict)
        ]
        starts = [_float_seconds(word.get("start")) for word in segment_words]
        ends = [_float_seconds(word.get("end")) for word in segment_words]
        valid_starts = [value for value in starts if value is not None]
        valid_ends = [value for value in ends if value is not None]
        segment_start = _float_seconds(aligned_segment.get("start"))
        segment_end = _float_seconds(aligned_segment.get("end"))
        if valid_starts:
            segment_start = min(valid_starts)
        if valid_ends:
            segment_end = max(valid_ends)

        def plan_score(index: int) -> tuple[float, float]:
            plan = plans[index]
            if segment_start is None or segment_end is None:
                return (0.0, -float(index))
            overlap = max(0.0, min(segment_end, plan.end) - max(segment_start, plan.start))
            midpoint_distance = abs((segment_start + segment_end) / 2 - (plan.start + plan.end) / 2)
            return (overlap, -midpoint_distance)

        owner = max(range(len(plans)), key=plan_score)
        word_buckets[owner].extend(segment_words)

    restored_segments: list[dict] = []
    restored_words: list[dict] = []
    mapped_any = False
    for source_words, plan in zip(word_buckets, plans):
        words = _restore_segment_display_words(source_words, plan)
        if words is None:
            words = [{"word": token.display_text} for token in plan.tokens]
        else:
            mapped_any = True
        restored_words.extend(words)

        timed_starts = [_float_seconds(word.get("start")) for word in words]
        timed_ends = [_float_seconds(word.get("end")) for word in words]
        starts = [value for value in timed_starts if value is not None]
        ends = [value for value in timed_ends if value is not None]
        restored_segments.append(
            {
                "text": plan.text,
                "start": min(starts) if starts else plan.start,
                "end": max(ends) if ends else plan.end,
                "words": words,
            }
        )

    result = dict(aligned)
    if not mapped_any:
        return None
    result["segments"] = restored_segments
    result["word_segments"] = restored_words
    return result


def install_whisperx_runtime_stubs() -> None:
    """Avoid importing WhisperX features that SubForge does not execute.

    Older WhisperX releases eagerly import transcribe/diarize, while frozen
    builds can still probe those modules during collection. SubForge uses MLX
    Whisper for transcription and only needs WhisperX audio/alignment.
    """

    def _unsupported(*_args, **_kwargs):
        raise RuntimeError("This WhisperX feature is not bundled in SubForge")

    class _DiarizationSegment:
        """Compatibility value object imported by WhisperX's Pyannote VAD."""

        def __init__(self, start: int, end: int, speaker: str | None = None):
            self.start = start
            self.end = end
            self.speaker = speaker

    if "whisperx.transcribe" not in sys.modules:
        transcribe = types.ModuleType("whisperx.transcribe")
        setattr(transcribe, "load_model", _unsupported)
        sys.modules["whisperx.transcribe"] = transcribe

    if "whisperx.diarize" not in sys.modules:
        diarize = types.ModuleType("whisperx.diarize")
        setattr(diarize, "assign_word_speakers", _unsupported)
        setattr(diarize, "DiarizationPipeline", _unsupported)
        setattr(diarize, "Segment", _DiarizationSegment)
        sys.modules["whisperx.diarize"] = diarize


class _OfflineSentenceTokenizer:
    """Small Punkt fallback used when the packaged app has no NLTK data."""

    def span_tokenize(self, text: str):
        start = 0
        for match in re.finditer(r"[.!?。！？]+(?=\s+|$)", text):
            end = match.end()
            if end > start:
                yield start, end
            start = end
            while start < len(text) and text[start].isspace():
                start += 1
        if start < len(text):
            yield start, len(text)


def _install_offline_sentence_tokenizer(alignment_module: Any) -> None:
    """Prevent WhisperX 3.8+ from downloading Punkt data at runtime."""
    if getattr(alignment_module, "_subforge_offline_tokenizer_installed", False):
        return
    nltk_load = getattr(alignment_module, "nltk_load", None)
    if not callable(nltk_load):
        return

    fallback_tokenizer: _OfflineSentenceTokenizer | None = None
    fallback_logged = False
    unavailable_resources: set[str] = set()

    def _load_or_fallback(resource: str):
        nonlocal fallback_tokenizer, fallback_logged
        if resource in unavailable_resources:
            if fallback_tokenizer is None:
                fallback_tokenizer = _OfflineSentenceTokenizer()
            return fallback_tokenizer
        try:
            return nltk_load(resource)
        except LookupError:
            unavailable_resources.add(resource)
            if not fallback_logged:
                logger.warning("NLTK Punkt data is unavailable; using offline sentence boundaries")
                fallback_logged = True
            if fallback_tokenizer is None:
                fallback_tokenizer = _OfflineSentenceTokenizer()
            return fallback_tokenizer

    alignment_module.nltk_load = _load_or_fallback
    alignment_module._subforge_offline_tokenizer_installed = True


def _float_seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    return None


def _word_text(word: dict) -> str:
    return str(word.get("word") or word.get("text") or "").strip()


def _word_has_timing(word: dict) -> bool:
    return (
        _float_seconds(word.get("start")) is not None
        and _float_seconds(word.get("end")) is not None
    )


def _word_duration_weight(text: str) -> int:
    alnum = sum(1 for ch in text if ch.isalnum())
    return max(1, alnum)


def _refine_words_with_char_alignments(words: list[dict], chars: list[dict] | None) -> list[dict]:
    """Use WhisperX character timings to tighten aligned word boundaries.

    WhisperX already derives words from characters, but older releases can
    leave a word partially timed when punctuation or an unsupported character
    is involved. Character data is treated as supporting evidence only: text
    order and existing timings are preserved when an exact sequential match is
    unavailable.
    """
    if not chars:
        return words

    timed_chars = [char for char in chars if isinstance(char, dict) and str(char.get("char") or "")]
    uses_space_delimiters = any(str(char.get("char") or "").isspace() for char in timed_chars)
    char_index = 0
    refined: list[dict] = []

    for original in words:
        word = dict(original)
        target = "".join(ch.lower() for ch in _word_text(word) if not ch.isspace())
        if not target:
            refined.append(word)
            continue

        matched: list[dict] = []
        target_index = 0
        scan_index = char_index
        while scan_index < len(timed_chars):
            value = str(timed_chars[scan_index].get("char") or "")
            if not value.isspace():
                break
            scan_index += 1

        if uses_space_delimiters:
            while scan_index < len(timed_chars):
                item = timed_chars[scan_index]
                value = str(item.get("char") or "")
                if value.isspace():
                    break
                matched.append(item)
                scan_index += 1
            candidate = "".join(str(item.get("char") or "").lower() for item in matched)
            target_index = len(target) if candidate == target else 0
            char_index = scan_index
        else:
            while scan_index < len(timed_chars) and target_index < len(target):
                item = timed_chars[scan_index]
                value = str(item.get("char") or "")
                scan_index += 1
                if value.lower() != target[target_index]:
                    matched = []
                    break
                matched.append(item)
                target_index += 1

        if target_index == len(target) and matched:
            starts = [_float_seconds(item.get("start")) for item in matched]
            ends = [_float_seconds(item.get("end")) for item in matched]
            valid_starts = [value for value in starts if value is not None]
            valid_ends = [value for value in ends if value is not None]
            if valid_starts:
                word["start"] = min(valid_starts)
            if valid_ends:
                word["end"] = max(valid_ends)
            char_index = scan_index

        refined.append(word)

    return refined


def _append_word_run(
    output: list[ASRDataSeg],
    words: list[dict],
    left_bound: float | None,
    right_bound: float | None,
    language_code: str = "",
) -> None:
    """Add unaligned WhisperX words by distributing the adjacent aligned gap.

    WhisperX keeps unalignable words such as numerals in the segment word list,
    but without start/end. Dropping them makes the exported transcript lose real
    content. The safest recovery is local interpolation between the surrounding
    aligned words inside the same ASR segment.
    """
    texts = [_word_text(word) for word in words]
    valid = [(word, text) for word, text in zip(words, texts) if text]
    if not valid:
        return

    if left_bound is None and right_bound is None:
        return
    if left_bound is None:
        left_bound = max(0.0, right_bound - 0.25 * len(valid))  # type: ignore[operator]
    if right_bound is None:
        right_bound = left_bound + 0.25 * len(valid)
    if right_bound <= left_bound:
        right_bound = left_bound + 0.02 * len(valid)

    weights = [_word_duration_weight(text) for _, text in valid]
    total_weight = max(1, sum(weights))
    current = left_bound
    for index, ((word, text), weight) in enumerate(zip(valid, weights)):
        if index == len(valid) - 1:
            end = right_bound
        else:
            end = current + (right_bound - left_bound) * weight / total_weight
        start_ms = max(0, int(round(current * 1000)))
        end_ms = max(start_ms, int(round(end * 1000)))
        output.append(
            _make_word_segment(
                text,
                start_ms,
                end_ms,
                timing_source="estimated",
                language_code=str(word.get("language") or language_code),
            )
        )
        current = end


def _make_word_segment(
    text: str,
    start_ms: int,
    end_ms: int,
    *,
    timing_source: TimestampSource,
    confidence: float | None = None,
    language_code: str = "",
) -> ASRDataSeg:
    word = ASRWord(
        text=text,
        start_time=start_ms,
        end_time=end_ms,
        confidence=confidence,
        alignment_score=confidence if timing_source == "forced_alignment" else None,
        timing_source=timing_source,
        language_code=language_code,
    )
    return ASRDataSeg(
        text,
        start_ms,
        end_ms,
        words=[word],
        timestamp_granularity="word",
        timing_source=timing_source,
        language_code=language_code,
    )


def _words_to_segments(
    words: list[dict],
    segment_start: float | None = None,
    segment_end: float | None = None,
    language_code: str = "",
) -> list[ASRDataSeg]:
    output: list[ASRDataSeg] = []
    pending: list[dict] = []
    last_known_end = segment_start

    for word in words:
        text = _word_text(word)
        if not text:
            continue

        if _word_has_timing(word):
            start = _float_seconds(word.get("start"))
            end = _float_seconds(word.get("end"))
            if start is None or end is None:
                continue
            if pending:
                if last_known_end is not None or segment_start is not None:
                    _append_word_run(
                        output,
                        pending,
                        last_known_end,
                        start,
                        language_code,
                    )
                pending = []
            start_ms = max(0, int(round(start * 1000)))
            end_ms = max(start_ms, int(round(end * 1000)))
            score = word.get("score")
            confidence = float(score) if isinstance(score, (int, float)) else None
            output.append(
                _make_word_segment(
                    text,
                    start_ms,
                    end_ms,
                    timing_source="forced_alignment",
                    confidence=confidence,
                    language_code=str(word.get("language") or language_code),
                )
            )
            last_known_end = end
        else:
            pending.append(word)

    if pending:
        if segment_end is not None:
            _append_word_run(
                output,
                pending,
                last_known_end,
                segment_end,
                language_code,
            )

    return output


class WhisperXASR(BaseASR):
    """Platform-native Whisper transcription followed by WhisperX alignment.

    Apple Silicon uses MLX Whisper. Other platforms use WhisperX's
    Faster-Whisper backend so Windows can use the same forced-alignment model.
    """

    def __init__(
        self,
        audio_input: Union[str, bytes],
        whisper_model: str = "",
        model_dir: str = "",
        language: str = "en",
        device: str = "auto",
        compute_type: str = "default",
        align_model: str = "",
        batch_size: int = 4,
        segment_callback: Optional[Callable[[ASRData], None]] = None,
        missing_alignment_model_callback: Optional[Callable[[list[dict[str, Any]]], str]] = None,
        detect_additional_languages: bool = False,
        cancel_event: Any = None,
        use_cache: bool = False,
        need_word_time_stamp: bool = True,
    ):
        super().__init__(audio_input, use_cache, need_word_time_stamp)
        self.uses_mlx = platform.system() == "Darwin" and platform.machine() == "arm64"
        self.model_dir = model_dir or str(MODEL_PATH)
        requested_model = whisper_model or (default_mlx_model() if self.uses_mlx else "large-v3")
        if not self.uses_mlx:
            local_model = find_faster_whisper_model_dir(requested_model, self.model_dir)
            if local_model is not None:
                requested_model = str(local_model)
        self.whisper_model = requested_model
        self.mlx_model = _mlx_model_repo(self.whisper_model) if self.uses_mlx else ""
        self.language = _normalize_language(language)
        if self.uses_mlx:
            self.transcribe_device = "mlx"
            self.compute_type = _normalize_compute_type("cpu", compute_type)
        else:
            self.transcribe_device, self.compute_type = resolve_faster_whisper_runtime(
                device, compute_type
            )
        self.align_device = _normalize_align_device(device)
        self.align_model = _normalize_align_model(align_model)
        self.batch_size = max(1, int(batch_size or 4))
        self.segment_callback = segment_callback
        self.missing_alignment_model_callback = missing_alignment_model_callback
        self.detect_additional_languages = bool(detect_additional_languages)
        self.cancel_event = cancel_event
        self.need_word_time_stamp = need_word_time_stamp

    def _raise_if_cancelled(self, stage: str) -> None:
        cancel_event = getattr(self, "cancel_event", None)
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError(f"WhisperX transcription was cancelled during {stage}")

    def realign_compound_word_gaps(self, data: ASRData, original_audio_path: str) -> int:
        """Confirm compact size tails against original audio, even after denoising.

        Only a bounded local context is aligned. Text and following word onsets
        are immutable; timing can grow only with scored alignment and matching
        neighboring anchors. A true pause therefore remains a pause.
        """
        candidates = _compound_timing_candidates(data)
        if not candidates:
            return 0
        import whisperx.alignment as alignment
        from whisperx.audio import load_audio

        _install_offline_sentence_tokenizer(alignment)
        self._raise_if_cancelled("compound number alignment")
        audio = load_audio(original_audio_path)
        repaired = 0
        for first, last in candidates:
            self._raise_if_cancelled("compound number alignment")
            left, right = max(0, first - 2), min(len(data.segments), last + 4)
            context = data.segments[left:right]
            start_ms = max(0, context[0].start_time - 350)
            end_ms = min(round(len(audio) / 16), context[-1].end_time + 350)
            if end_ms <= start_ms or end_ms - start_ms > 12000:
                continue
            text = " ".join(segment.text for segment in context)
            aligned = self._align_result(
                {"segments": [{"text": text, "start": 0, "end": (end_ms - start_ms) / 1000}]},
                audio[start_ms * 16 : end_ms * 16],
                "en",
                lambda *_: None,
                alignment,
            )
            words = aligned.get("word_segments") or []
            if [_word_text(word) for word in words] != [segment.text for segment in context]:
                continue
            target = words[last - left]
            onset = _float_seconds(words[first - left].get("start"))
            ending = _float_seconds(target.get("end"))
            following = _float_seconds(words[last + 1 - left].get("start"))
            score = target.get("score")
            if (
                onset is None
                or ending is None
                or following is None
                or ending > following
                or not isinstance(score, (int, float))
                or score < 0.70
            ):
                continue
            next_start = data.segments[last + 1].start_time
            if (
                abs(round(onset * 1000) + start_ms - data.segments[first].start_time) > 180
                or abs(round(following * 1000) + start_ms - next_start) > 180
            ):
                continue
            new_end = min(round(ending * 1000) + start_ms, next_start - 30)
            target_segment = data.segments[last]
            if not 120 <= new_end - target_segment.end_time <= 3200:
                continue
            logger.info(
                "Re-aligned compound number end %s -> %s: %s",
                target_segment.end_time,
                new_end,
                target_segment.text,
            )
            target_segment.end_time = new_end
            if target_segment.words:
                target_segment.words[-1].end_time = new_end
            repaired += 1
        return repaired

    def _transcribe_mlx_in_worker(
        self,
        audio_path: str,
        mlx_model_path: str,
        callback: Callable[[int, str], None],
    ) -> dict:
        """Decode MLX audio in a child process without starving the desktop backend."""
        segment_callback = getattr(self, "segment_callback", None)
        with tempfile.TemporaryDirectory(prefix="subforge-mlx-worker-") as temp_path:
            temp_dir = Path(temp_path)
            request_path = temp_dir / "request.json"
            output_path = temp_dir / "output.json"
            log_path = temp_dir / "worker.log"
            _atomic_json_write(
                request_path,
                {
                    "audio": audio_path,
                    "model": mlx_model_path,
                    "language": self.language or "",
                    "word_timestamps": getattr(self, "need_word_time_stamp", True),
                },
            )
            env = os.environ.copy()
            env.update(
                {
                    _MLX_WORKER_FLAG: "1",
                    _MLX_WORKER_REQUEST: str(request_path),
                    _MLX_WORKER_OUTPUT: str(output_path),
                }
            )
            command = (
                [sys.executable]
                if getattr(sys, "frozen", False)
                else [sys.executable, "-m", "subforge.core.asr.mlx_worker"]
            )
            started_at = time.monotonic()
            last_status_second = -1
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                preview_reader = log_path.open("r", encoding="utf-8", errors="replace")
                preview_buffer = ""
                preview_segments: list[ASRDataSeg] = []

                def publish_new_preview_lines(*, include_tail: bool = False) -> None:
                    nonlocal preview_buffer
                    preview_buffer += preview_reader.read()
                    lines = preview_buffer.split("\n")
                    preview_buffer = "" if include_tail else lines.pop()
                    parsed = _parse_mlx_preview_lines(lines)
                    if not parsed:
                        return
                    preview_segments.extend(parsed)
                    if segment_callback:
                        segment_callback(ASRData(list(preview_segments)))

                try:
                    while process.poll() is None:
                        if self.cancel_event is not None and self.cancel_event.is_set():
                            raise RuntimeError("MLX Whisper transcription was cancelled")
                        elapsed = int(time.monotonic() - started_at)
                        if elapsed // 5 != last_status_second // 5:
                            callback(
                                20,
                                f"MLX Whisper is transcribing ({elapsed // 60}:{elapsed % 60:02d})...",
                            )
                            last_status_second = elapsed
                        if segment_callback:
                            publish_new_preview_lines()
                        time.sleep(0.2)
                    if segment_callback:
                        publish_new_preview_lines(include_tail=True)
                finally:
                    preview_reader.close()
                    _stop_worker_process(process)

            payload: dict[str, Any] | None = None
            if output_path.is_file():
                try:
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
            if process.returncode != 0 or not payload or not payload.get("ok"):
                detail = str((payload or {}).get("error") or "").strip()
                if not detail:
                    detail = _worker_log_tail(log_path)
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(
                    f"MLX Whisper worker exited with code {process.returncode}{suffix}"
                )
            result = payload.get("data")
            if not isinstance(result, dict):
                raise RuntimeError("MLX Whisper worker returned invalid transcription data")
            logger.info(
                "MLX Whisper worker completed in %.1fs",
                time.monotonic() - started_at,
            )
            recovered_gaps = result.get("speech_gap_recovery") or []
            if recovered_gaps:
                logger.warning(
                    "Recovered %d VAD-confirmed speech gaps after long-form MLX decoding: %s",
                    len(recovered_gaps),
                    [
                        f"{float(item['start']):.1f}-{float(item['end']):.1f}s"
                        for item in recovered_gaps
                    ],
                )
                callback(32, f"Recovered {len(recovered_gaps)} missed speech section(s)...")
            short_recoveries = result.get("short_speech_gap_recovery") or []
            if short_recoveries:
                logger.warning(
                    "Recovered %d short speech omission(s) by dual-context consensus: %s",
                    len(short_recoveries),
                    [
                        {
                            "window": f"{float(item['start']):.1f}-{float(item['end']):.1f}s",
                            "text": str(item.get("text") or ""),
                        }
                        for item in short_recoveries
                    ],
                )
                callback(33, f"Recovered {len(short_recoveries)} short speech omission(s)...")
            sparse_recoveries = result.get("sparse_segment_recovery") or []
            if sparse_recoveries:
                logger.warning(
                    "Re-decoded %d under-transcribed MLX spans: %s",
                    len(sparse_recoveries),
                    [
                        {
                            "window": f"{float(item['start']):.1f}-{float(item['end']):.1f}s",
                            "before": int(item["original_units"]),
                            "after": int(item["recovered_units"]),
                        }
                        for item in sparse_recoveries
                    ],
                )
                callback(
                    33,
                    f"Recovered {len(sparse_recoveries)} under-transcribed section(s)...",
                )
            return result

    def _classify_mlx_ranges_in_worker(
        self,
        audio_path: str,
        mlx_model_path: str,
        ranges: list[_SpeechBackedGap],
        callback: Callable[[int, str], None],
    ) -> list[dict[str, Any]]:
        """Classify final audit gaps without loading MLX into the desktop process."""
        if not ranges or not self.language:
            return []

        with tempfile.TemporaryDirectory(prefix="subforge-mlx-language-audit-") as temp_path:
            temp_dir = Path(temp_path)
            request_path = temp_dir / "request.json"
            output_path = temp_dir / "output.json"
            log_path = temp_dir / "worker.log"
            _atomic_json_write(
                request_path,
                {
                    "operation": "classify_ranges",
                    "audio": audio_path,
                    "model": mlx_model_path,
                    "primary_language": self.language,
                    "ranges": [
                        {
                            "start": gap.start,
                            "end": gap.end,
                            "speech_seconds": gap.speech_seconds,
                            "speech_ratio": gap.speech_ratio,
                            "is_internal": gap.is_internal,
                        }
                        for gap in ranges
                    ],
                },
            )
            env = os.environ.copy()
            env.update(
                {
                    _MLX_WORKER_FLAG: "1",
                    _MLX_WORKER_REQUEST: str(request_path),
                    _MLX_WORKER_OUTPUT: str(output_path),
                }
            )
            command = (
                [sys.executable]
                if getattr(sys, "frozen", False)
                else [sys.executable, "-m", "subforge.core.asr.mlx_worker"]
            )
            callback(93, "Checking the language of uncovered speech...")
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                try:
                    while process.poll() is None:
                        if self.cancel_event is not None and self.cancel_event.is_set():
                            raise RuntimeError("MLX language audit was cancelled")
                        time.sleep(0.2)
                finally:
                    _stop_worker_process(process)

            payload: dict[str, Any] | None = None
            if output_path.is_file():
                try:
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
            if process.returncode != 0 or not payload or not payload.get("ok"):
                detail = str((payload or {}).get("error") or "").strip()
                if not detail:
                    detail = _worker_log_tail(log_path)
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(
                    f"MLX language audit worker exited with code {process.returncode}{suffix}"
                )
            data = payload.get("data")
            if not isinstance(data, dict):
                raise RuntimeError("MLX language audit worker returned invalid data")
            return [
                dict(item)
                for item in data.get("foreign_language_speech_ranges") or []
                if isinstance(item, dict)
            ]

    def _write_audio_to_temp(self, tmp_dir: Path) -> str:
        if isinstance(self.audio_input, str):
            return self.audio_input
        if not self.file_binary:
            raise ValueError("No audio data available")
        audio_path = tmp_dir / "whisperx_audio.wav"
        audio_path.write_bytes(self.file_binary)
        return str(audio_path)

    def _resolve_align_model_name(self, language_code: str) -> str | None:
        normalized_language = _normalize_language(language_code)
        if self.align_model and self.align_model != DEFAULT_EN_ALIGN_MODEL:
            return self.align_model
        if self.align_model == DEFAULT_EN_ALIGN_MODEL:
            if normalized_language == "en":
                return DEFAULT_EN_ALIGN_MODEL
            logger.info(
                "Ignoring English-only forced alignment model for language=%s",
                normalized_language,
            )
        default_model = alignment_model_for_language(normalized_language)
        return default_model.model_name if default_model else None

    def _run(
        self,
        callback: Optional[Callable[[int, str], None]] = None,
        **kwargs: Any,
    ) -> dict:
        if self.uses_mlx:
            return self._run_mlx(callback=callback, **kwargs)
        return self._run_standard(callback=callback, **kwargs)

    def _run_standard(
        self,
        callback: Optional[Callable[[int, str], None]] = None,
        **_kwargs: Any,
    ) -> dict:
        def _default_callback(_progress: int, _message: str) -> None:
            pass

        callback = callback or _default_callback
        self._raise_if_cancelled("startup")
        try:
            import whisperx.alignment as whisperx_alignment
            from whisperx.asr import load_model
            from whisperx.audio import load_audio
        except ImportError as exc:
            raise RuntimeError(
                "WhisperX is not installed in this desktop build. Reinstall SubForge with "
                "the WhisperX desktop runtime included."
            ) from exc

        with tempfile.TemporaryDirectory() as tmp:
            audio_path = self._write_audio_to_temp(Path(tmp))
            self._raise_if_cancelled("model loading")
            callback(15, "Loading WhisperX transcription model...")
            logger.info(
                "Transcribing with standard WhisperX model=%s transcribe_device=%s "
                "align_device=%s compute=%s",
                self.whisper_model,
                self.transcribe_device,
                self.align_device,
                self.compute_type,
            )
            model = load_model(
                self.whisper_model,
                self.transcribe_device,
                compute_type=self.compute_type,
                language=self.language,
                vad_method="silero",
                download_root=self.model_dir,
            )
            self._raise_if_cancelled("model loading")
            callback(30, "Loading audio...")
            audio = load_audio(audio_path)
            self._raise_if_cancelled("audio loading")
            callback(40, "Transcribing with WhisperX...")
            result = dict(
                model.transcribe(
                    audio,
                    batch_size=self.batch_size,
                    language=self.language,
                )
            )
            self._raise_if_cancelled("ASR decoding")

            if self.segment_callback:
                raw_segments = self._make_segments(result)
                if raw_segments:
                    self.segment_callback(ASRData(raw_segments))

            language_code = str(result.get("language") or self.language or "en").lower()
            skip_alignment_languages: set[str] = set()
            if self.language is None:
                _, skip_alignment_languages = self._resolve_missing_alignment_models(
                    language_code,
                    [],
                )
            aligned = self._align_multilingual_result(
                result,
                audio,
                language_code,
                callback,
                whisperx_alignment,
                skip_alignment_languages,
            )
            self._raise_if_cancelled("forced alignment")
            aligned["asr_backend"] = "faster-whisper"
            aligned["whisper_model"] = self.whisper_model

            if self.segment_callback:
                aligned_segments = self._make_segments(aligned)
                if aligned_segments:
                    self.segment_callback(ASRData(aligned_segments))

            try:
                del model
                if self.align_device == "cuda":
                    import torch

                    torch.cuda.empty_cache()
            except Exception:
                pass

            callback(*ASRStatus.COMPLETED.callback_tuple())
            return aligned

    def _align_result(
        self,
        result: dict,
        audio: Any,
        language_code: str,
        callback: Callable[[int, str], None],
        whisperx_alignment: Any,
    ) -> dict:
        align_segments = _segments_for_alignment(result)
        self._raise_if_cancelled("forced alignment preparation")
        if not align_segments:
            raise RuntimeError("WhisperX did not return alignable transcript segments")

        spoken_align_segments, alignment_plans = _prepare_spoken_alignment(
            align_segments, language_code
        )
        callback(65, "Loading forced alignment model...")
        align_model_name = self._resolve_align_model_name(language_code)
        align_spec = alignment_model_for_language(language_code)
        if align_spec is not None and not is_alignment_model_ready(align_spec, self.model_dir):
            raise RuntimeError(
                f"The {align_spec.language_name} forced alignment model is not "
                "available locally. Download it again in WhisperX settings; cloud "
                "placeholder files cannot be used for offline alignment."
            )
        align_kwargs: dict[str, Any] = {
            "language_code": language_code,
            "device": self.align_device,
        }
        if align_model_name:
            align_kwargs["model_name"] = align_model_name
        if self.model_dir:
            align_kwargs["model_dir"] = self.model_dir

        def _load_alignment_model():
            try:
                return whisperx_alignment.load_align_model(**align_kwargs)
            except TypeError:
                fallback_kwargs = dict(align_kwargs)
                fallback_kwargs.pop("model_dir", None)
                return whisperx_alignment.load_align_model(**fallback_kwargs)

        cache_key = (
            language_code,
            align_model_name or "",
            self.align_device,
            str(self.model_dir or ""),
            id(whisperx_alignment.load_align_model),
        )
        with _ALIGNMENT_MODEL_CACHE.acquire(cache_key, _load_alignment_model) as loaded:
            self._raise_if_cancelled("alignment model loading")
            model_a, metadata = loaded
            callback(78, "Running forced alignment...")

            def _align(segments_to_align: list[dict]) -> dict:
                try:
                    return dict(
                        whisperx_alignment.align(
                            segments_to_align,
                            model_a,
                            metadata,
                            audio,
                            self.align_device,
                            return_char_alignments=True,
                        )
                    )
                except TypeError:
                    return dict(
                        whisperx_alignment.align(
                            segments_to_align,
                            model_a,
                            metadata,
                            audio,
                            self.align_device,
                        )
                    )

            aligned = _align(spoken_align_segments)
            self._raise_if_cancelled("forced alignment")
            if alignment_plans:
                restored = _restore_display_alignment(aligned, alignment_plans)
                if restored is None:
                    logger.warning(
                        "Spoken alignment mapping was incomplete; retrying original text"
                    )
                    aligned = _align(align_segments)
                    self._raise_if_cancelled("forced alignment retry")
                else:
                    aligned = restored
        aligned["language"] = language_code
        aligned["align_model"] = align_model_name or ""

        return aligned

    def _detect_mlx_language_ranges(
        self,
        audio_path: str,
        mlx_model_path: str,
        result: dict,
        primary_language: str,
    ) -> list[_LanguageRange]:
        """Detect confident language switches without decoding correct primary speech again."""
        try:
            import mlx.core as mx
            import mlx_whisper
            from mlx_whisper.audio import (
                N_FRAMES,
                SAMPLE_RATE,
                log_mel_spectrogram,
                pad_or_trim,
            )
            from mlx_whisper.audio import (
                load_audio as load_mlx_audio,
            )
            from mlx_whisper.transcribe import ModelHolder
        except (ImportError, RuntimeError) as exc:
            logger.info("Mixed-language probing is unavailable: %s", exc)
            return []

        source_segments = _segments_for_alignment(result)
        if not source_segments:
            return []

        audio = load_mlx_audio(audio_path)
        model = ModelHolder.get_model(mlx_model_path, mx.float16)
        audio_duration = float(audio.shape[0]) / SAMPLE_RATE
        full_mel = log_mel_spectrogram(cast(Any, audio), n_mels=model.dims.n_mels)
        frames_per_second = 100
        probes: list[_LanguageProbe] = []
        try:
            speech_segments_ms = _detect_speech_in_mlx_gaps(
                audio,
                SAMPLE_RATE,
                [(0.0, audio_duration)],
            )
        except Exception as exc:
            logger.warning("Mixed-language VAD failed; skipping language switches: %s", exc)
            return []
        if not speech_segments_ms:
            return []

        def _probe(
            start: float,
            end: float,
            *,
            range_start: Optional[float] = None,
            range_end: Optional[float] = None,
            source: str = "window",
        ) -> None:
            if end - start < 0.75:
                return
            speech_seconds, speech_ratio = _speech_overlap_metrics(
                start,
                end,
                speech_segments_ms,
            )
            minimum_speech = min(
                MIXED_LANGUAGE_MIN_VOICED_SECONDS,
                max(0.60, (end - start) * MIXED_LANGUAGE_MIN_VOICED_RATIO),
            )
            if speech_seconds < minimum_speech or speech_ratio < MIXED_LANGUAGE_MIN_VOICED_RATIO:
                return
            frame_start = max(0, int(start * frames_per_second))
            frame_end = min(int(full_mel.shape[-2]), int(end * frames_per_second))
            mel_segment = pad_or_trim(full_mel[frame_start:frame_end], N_FRAMES, axis=-2).astype(
                mx.float16
            )
            _, raw_probabilities = model.detect_language(mel_segment)
            probabilities = cast(dict[str, float], raw_probabilities)
            if not probabilities:
                return
            ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            language, confidence = ranked[0]
            runner_up_language, runner_up_confidence = ranked[1] if len(ranked) > 1 else ("", 0.0)
            evidence_start = start if range_start is None else range_start
            evidence_end = end if range_end is None else range_end
            evidence_speech_seconds, evidence_speech_ratio = _speech_overlap_metrics(
                evidence_start,
                evidence_end,
                speech_segments_ms,
            )
            probes.append(
                _LanguageProbe(
                    start=evidence_start,
                    end=evidence_end,
                    language=str(language).lower(),
                    confidence=float(confidence),
                    primary_confidence=float(probabilities.get(primary_language, 0.0)),
                    runner_up_language=str(runner_up_language).lower(),
                    runner_up_confidence=float(runner_up_confidence),
                    speech_seconds=evidence_speech_seconds,
                    speech_ratio=evidence_speech_ratio,
                    source=source,
                )
            )

        window_start = 0.0
        while window_start < audio_duration:
            window_end = min(audio_duration, window_start + MIXED_LANGUAGE_WINDOW_SECONDS)
            # Probe the complete recording. Forced-primary ASR can omit foreign speech,
            # so its segment boundaries are not a reliable VAD gate here. Use only the
            # center of each overlapping window as the candidate replacement range to
            # avoid pulling adjacent primary-language speech into local re-decoding.
            core_margin = min(
                MIXED_LANGUAGE_WINDOW_STRIDE_SECONDS / 2,
                max(0.0, (window_end - window_start) / 2 - 0.05),
            )
            _probe(
                window_start,
                window_end,
                range_start=window_start + core_margin,
                range_end=window_end - core_margin,
            )
            window_start += MIXED_LANGUAGE_WINDOW_STRIDE_SECONDS

        # Fixed windows find switches even when forced-primary ASR omitted the
        # foreign speech. Refine only near those candidates with the ASR's more
        # precise speech boundaries; probing every primary segment roughly
        # doubles this stage on long recordings without improving recall.
        preliminary_ranges = _candidate_foreign_language_ranges(probes, primary_language)
        for segment in source_segments:
            start = max(0.0, segment["start"])
            end = min(audio_duration, segment["end"])
            if any(
                end >= item.start - MIXED_LANGUAGE_WINDOW_STRIDE_SECONDS
                and start <= item.end + MIXED_LANGUAGE_WINDOW_STRIDE_SECONDS
                for item in preliminary_ranges
            ):
                _probe(start, end, source="segment")

        candidate_ranges = _candidate_foreign_language_ranges(probes, primary_language)
        confirmed_ranges: list[_LanguageRange] = []
        rejected_ranges: list[dict[str, Any]] = []
        for candidate in candidate_ranges:
            context_start = max(0.0, candidate.start - MIXED_LANGUAGE_CONTEXT_SECONDS)
            context_end = min(
                audio_duration,
                candidate.end + MIXED_LANGUAGE_CONTEXT_SECONDS,
            )
            _event_speech_seconds, event_speech_ratio = _speech_overlap_metrics(
                candidate.start,
                candidate.end,
                speech_segments_ms,
            )
            clip = audio[int(context_start * SAMPLE_RATE) : int(context_end * SAMPLE_RATE)]
            try:
                decoded = mlx_whisper.transcribe(
                    clip,
                    path_or_hf_repo=mlx_model_path,
                    task="transcribe",
                    word_timestamps=False,
                    condition_on_previous_text=False,
                    verbose=None,
                )
            except Exception as exc:
                logger.warning(
                    "Mixed-language confirmation failed for %.2f-%.2f: %s",
                    candidate.start,
                    candidate.end,
                    exc,
                )
                continue
            decoded_segments = [
                segment
                for segment in decoded.get("segments") or []
                if isinstance(segment, dict)
                and _usable_mlx_recovery_segment(
                    segment,
                    # The event has already passed independent VAD gating. Do not
                    # reject a valid language decode a second time merely because
                    # natural pauses lower the event-wide speech ratio.
                    speech_ratio=max(0.75, event_speech_ratio),
                )
            ]
            decoded_text = " ".join(
                str(segment.get("text") or "").strip()
                for segment in decoded_segments
                if str(segment.get("text") or "").strip()
            ).strip()
            confirmed = _confirm_foreign_language_range(
                candidate,
                primary_language,
                str(decoded.get("language") or ""),
                decoded_text,
            )
            if confirmed is None:
                rejected_ranges.append(
                    {
                        "start": round(candidate.start, 2),
                        "end": round(candidate.end, 2),
                        "candidate": candidate.language,
                        "decoded": str(decoded.get("language") or ""),
                        "script": _script_language_evidence(decoded_text),
                    }
                )
                continue
            confirmed_ranges.append(confirmed)

        confirmed_ranges, rejected_cjk_ranges = _stabilize_confirmed_language_ranges(
            confirmed_ranges
        )
        rejected_ranges.extend(
            {
                "start": round(item.start, 2),
                "end": round(item.end, 2),
                "candidate": item.language,
                "decoded": item.decoded_language,
                "script": item.script_language,
                "reason": "weak minority CJK evidence",
            }
            for item in rejected_cjk_ranges
        )
        bridge_ranges: list[_LanguageRange] = []
        for left, right, speech_seconds, speech_ratio in _candidate_confirmed_language_bridges(
            confirmed_ranges,
            speech_segments_ms,
        ):
            clip = audio[int(left.end * SAMPLE_RATE) : int(right.start * SAMPLE_RATE)]
            try:
                decoded = mlx_whisper.transcribe(
                    clip,
                    path_or_hf_repo=mlx_model_path,
                    task="transcribe",
                    word_timestamps=False,
                    condition_on_previous_text=False,
                    verbose=None,
                )
            except Exception as exc:
                logger.warning(
                    "Mixed-language bridge confirmation failed for %.2f-%.2f: %s",
                    left.end,
                    right.start,
                    exc,
                )
                continue
            decoded_text = " ".join(
                str(segment.get("text") or "").strip()
                for segment in decoded.get("segments") or []
                if isinstance(segment, dict)
                and _usable_mlx_recovery_segment(
                    segment,
                    speech_ratio=max(0.75, speech_ratio),
                )
                and str(segment.get("text") or "").strip()
            ).strip()
            if _script_language_evidence(decoded_text) != left.language:
                continue
            bridge_ranges.append(
                _LanguageRange(
                    start=left.end,
                    end=right.start,
                    language=left.language,
                    confidence=min(left.confidence, right.confidence),
                    support_seconds=speech_seconds,
                    probe_count=1,
                    dominance=1.0,
                    decoded_language=str(decoded.get("language") or "").lower(),
                    script_language=left.language,
                    confirmation_agreement=True,
                )
            )
        if bridge_ranges:
            logger.info(
                "Confirmed %d same-language bridge(s): %s",
                len(bridge_ranges),
                [f"{item.start:.1f}-{item.end:.1f}s {item.language}" for item in bridge_ranges],
            )
            confirmed_ranges.extend(bridge_ranges)
        ranges = _merge_confirmed_language_ranges(confirmed_ranges)
        ranges, expanded_repetitions = _expand_language_ranges_over_repetition(
            ranges,
            source_segments,
        )
        if expanded_repetitions:
            logger.warning(
                "Expanded confirmed foreign-language ranges over %d repeated decoder loop(s): %s",
                len(expanded_repetitions),
                [f"{start:.1f}-{end:.1f}s" for start, end in expanded_repetitions],
            )
        if rejected_ranges:
            logger.info("Rejected ambiguous mixed-language ranges: %s", rejected_ranges)
        if ranges:
            logger.info(
                "Detected mixed-language ranges: %s",
                [
                    {
                        "start": round(item.start, 2),
                        "end": round(item.end, 2),
                        "language": item.language,
                        "confidence": round(item.confidence, 3),
                        "support": round(item.support_seconds, 2),
                        "probes": item.probe_count,
                        "dominance": round(item.dominance, 3),
                    }
                    for item in ranges
                ],
            )
        return ranges

    def _retranscribe_mlx_language_ranges(
        self,
        audio_path: str,
        mlx_model_path: str,
        result: dict,
        ranges: list[_LanguageRange],
        primary_language: str,
    ) -> dict:
        """Replace only confirmed foreign ranges with original-language decoding."""
        if not ranges:
            tagged = dict(result)
            tagged["segments"] = [
                {**segment, "language": primary_language}
                for segment in result.get("segments") or []
                if isinstance(segment, dict)
            ]
            return tagged

        import mlx_whisper
        from mlx_whisper.audio import SAMPLE_RATE
        from mlx_whisper.audio import load_audio as load_mlx_audio

        audio = load_mlx_audio(audio_path)
        audio_duration = float(audio.shape[0]) / SAMPLE_RATE
        replacements: list[tuple[_LanguageRange, list[dict]]] = []
        for language_range in ranges:
            context_start = max(0.0, language_range.start - MIXED_LANGUAGE_CONTEXT_SECONDS)
            context_end = min(
                audio_duration,
                language_range.end + MIXED_LANGUAGE_CONTEXT_SECONDS,
            )
            clip = audio[int(context_start * SAMPLE_RATE) : int(context_end * SAMPLE_RATE)]
            local_result = mlx_whisper.transcribe(
                clip,
                path_or_hf_repo=mlx_model_path,
                language=language_range.language,
                task="transcribe",
                word_timestamps=False,
                condition_on_previous_text=False,
                verbose=None,
            )
            localized: list[dict] = []
            for segment in local_result.get("segments") or []:
                if not isinstance(segment, dict):
                    continue
                shifted = dict(segment)
                shifted["start"] = float(segment.get("start", 0.0)) + context_start
                shifted["end"] = float(segment.get("end", 0.0)) + context_start
                midpoint = (shifted["start"] + shifted["end"]) / 2
                if not language_range.start <= midpoint <= language_range.end:
                    continue
                shifted["start"] = max(language_range.start, shifted["start"])
                shifted["end"] = min(language_range.end, shifted["end"])
                shifted["language"] = language_range.language
                if str(shifted.get("text") or "").strip() and shifted["end"] > shifted["start"]:
                    localized.append(shifted)
            if localized:
                replacements.append((language_range, localized))
            else:
                logger.warning(
                    "Foreign-language re-transcription returned no speech for %.2f-%.2f (%s)",
                    language_range.start,
                    language_range.end,
                    language_range.language,
                )

        original_segments = [
            {**segment, "language": primary_language}
            for segment in result.get("segments") or []
            if isinstance(segment, dict)
        ]
        replaced_ranges = [item[0] for item in replacements]
        kept: list[dict] = []
        primary_replacements: list[dict] = []
        for segment in original_segments:
            start = segment.get("start")
            end = segment.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            segment_start = float(start)
            segment_end = float(end)
            overlapping = [
                item
                for item in replaced_ranges
                if item.end > segment_start and item.start < segment_end
            ]
            if not overlapping:
                kept.append(segment)
                continue

            # Initial MLX segments can span both languages. Removing an entire
            # segment by midpoint drops valid primary speech on either side of a
            # short language switch. Re-decode only the uncovered portions.
            for part_start, part_end in _subtract_language_ranges(
                segment_start,
                segment_end,
                overlapping,
            ):
                if part_end - part_start < 0.75:
                    continue
                context_start = max(0.0, part_start - MIXED_LANGUAGE_CONTEXT_SECONDS)
                context_end = min(audio_duration, part_end + MIXED_LANGUAGE_CONTEXT_SECONDS)
                clip = audio[int(context_start * SAMPLE_RATE) : int(context_end * SAMPLE_RATE)]
                local_result = mlx_whisper.transcribe(
                    clip,
                    path_or_hf_repo=mlx_model_path,
                    language=primary_language,
                    task="transcribe",
                    word_timestamps=False,
                    condition_on_previous_text=False,
                    verbose=None,
                )
                for local_segment in local_result.get("segments") or []:
                    if not isinstance(local_segment, dict):
                        continue
                    shifted = dict(local_segment)
                    shifted["start"] = float(local_segment.get("start", 0.0)) + context_start
                    shifted["end"] = float(local_segment.get("end", 0.0)) + context_start
                    midpoint = (shifted["start"] + shifted["end"]) / 2
                    if not part_start <= midpoint <= part_end:
                        continue
                    shifted["start"] = max(part_start, shifted["start"])
                    shifted["end"] = min(part_end, shifted["end"])
                    shifted["language"] = primary_language
                    if str(shifted.get("text") or "").strip() and shifted["end"] > shifted["start"]:
                        primary_replacements.append(shifted)

        combined = (
            kept
            + primary_replacements
            + [segment for _, segments in replacements for segment in segments]
        )
        combined.sort(key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))
        updated = dict(result)
        updated["segments"] = combined
        updated["languages"] = sorted(
            {str(item.get("language") or primary_language) for item in combined}
        )
        return updated

    def _resolve_missing_alignment_models(
        self,
        primary_language: str,
        ranges: list[_LanguageRange],
    ) -> tuple[list[_LanguageRange], set[str]]:
        """Pause auto-language jobs for missing models and apply the user's decision."""
        hybrid_mode = self.language is not None and bool(
            getattr(self, "detect_additional_languages", False)
        )
        detected_languages = {item.language for item in ranges}
        if not hybrid_mode:
            detected_languages.add(primary_language)
        unsupported_languages = {
            language
            for language in detected_languages
            if alignment_model_for_language(language) is None
        }
        if (
            self.language is not None and not hybrid_mode
        ) or self.missing_alignment_model_callback is None:
            return ranges, unsupported_languages

        range_by_language: dict[str, list[_LanguageRange]] = {}
        for item in ranges:
            range_by_language.setdefault(item.language, []).append(item)

        while True:
            missing: list[dict[str, Any]] = []
            for language in sorted(detected_languages):
                spec = alignment_model_for_language(language)
                if spec is None or is_alignment_model_ready(spec, self.model_dir):
                    continue
                language_ranges = range_by_language.get(language, [])
                missing.append(
                    {
                        "language": language,
                        "language_name": spec.language_name,
                        "model_id": spec.id,
                        "model_name": spec.model_name,
                        "size": spec.size,
                        "source": spec.source,
                        "confidence": max(
                            (item.confidence for item in language_ranges),
                            default=1.0 if language == primary_language else 0.0,
                        ),
                        "ranges": [
                            {"start": item.start, "end": item.end} for item in language_ranges
                        ],
                    }
                )
            if not missing:
                return ranges, unsupported_languages

            decision = self.missing_alignment_model_callback(missing)
            if decision == "retry":
                continue
            missing_languages = {str(item["language"]) for item in missing}
            if decision == "ignore":
                return (
                    [item for item in ranges if item.language not in missing_languages],
                    unsupported_languages | ({primary_language} & missing_languages),
                )
            if decision == "continue":
                return ranges, unsupported_languages | missing_languages
            raise RuntimeError(f"Unsupported alignment model decision: {decision}")

    def _align_multilingual_result(
        self,
        result: dict,
        audio: Any,
        primary_language: str,
        callback: Callable[[int, str], None],
        whisperx_alignment: Any,
        skip_alignment_languages: Optional[set[str]] = None,
    ) -> dict:
        """Align each detected language with its own acoustic model, then restore order."""
        grouped: dict[str, list[dict]] = {}
        for segment in result.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            language = str(segment.get("language") or primary_language).lower()
            grouped.setdefault(language, []).append(segment)

        skip_alignment_languages = skip_alignment_languages or set()
        if len(grouped) <= 1 and not (set(grouped) & skip_alignment_languages):
            return self._align_result(
                result,
                audio,
                primary_language,
                callback,
                whisperx_alignment,
            )

        aligned_segments: list[dict] = []
        aligned_words: list[dict] = []
        align_models: dict[str, str] = {}
        for language, segments in grouped.items():
            self._raise_if_cancelled(f"{language} alignment")
            if language in skip_alignment_languages:
                aligned_segments.extend(
                    {**segment, "language": language, "alignment_skipped": True}
                    for segment in segments
                )
                align_models[language] = ""
                continue
            callback(65, f"Loading {language} forced alignment model...")
            aligned = self._align_result(
                {"segments": segments},
                audio,
                language,
                callback,
                whisperx_alignment,
            )
            align_models[language] = str(aligned.get("align_model") or "")
            for segment in aligned.get("segments") or []:
                if isinstance(segment, dict):
                    aligned_segments.append({**segment, "language": language})
            for word in aligned.get("word_segments") or []:
                if isinstance(word, dict):
                    aligned_words.append({**word, "language": language})

        aligned_segments.sort(
            key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0)))
        )
        aligned_words.sort(
            key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0)))
        )
        return {
            "segments": aligned_segments,
            "word_segments": aligned_words,
            "language": primary_language,
            "languages": sorted(grouped),
            "align_models": align_models,
        }

    def _run_mlx(
        self,
        callback: Optional[Callable[[int, str], None]] = None,
        **kwargs: Any,
    ) -> dict:
        def _default_callback(_progress: int, _message: str) -> None:
            pass

        if callback is None:
            callback = _default_callback

        try:
            install_whisperx_runtime_stubs()
            import whisperx.alignment as whisperx_alignment
            from whisperx.audio import load_audio
        except ImportError as exc:
            raise RuntimeError(
                "WhisperX/MLX Whisper is not installed. Install the optional WhisperX stack "
                "or rebuild SubForge with the whisperx extra."
            ) from exc
        except RuntimeError as exc:
            message = str(exc)
            if "No Metal device available" in message:
                raise RuntimeError(
                    "MLX Whisper could not access an Apple Metal device. Launch SubForge from "
                    "the macOS app bundle/Finder on Apple Silicon, and avoid running it in a "
                    "headless or sandboxed terminal environment."
                ) from exc
            if "metallib" in message.lower():
                raise RuntimeError(
                    "MLX Whisper could not load mlx.metallib. Rebuild or reinstall SubForge "
                    "with the bundled MLX Metal resources."
                ) from exc
            raise

        _install_offline_sentence_tokenizer(whisperx_alignment)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            audio_path = self._write_audio_to_temp(tmp_dir)
            mlx_model_path = _prepare_mlx_model_path(self.mlx_model, tmp_dir)
            callback(20, "Transcribing with MLX Whisper...")
            logger.info(
                "Transcribing with MLX Whisper model=%s align_device=%s",
                mlx_model_path,
                self.align_device,
            )
            if not Path(mlx_model_path).exists() and "/" not in mlx_model_path:
                logger.warning(
                    "MLX Whisper model %s is not a local path. mlx-whisper may try to download it.",
                    mlx_model_path,
                )
            result = self._transcribe_mlx_in_worker(
                audio_path,
                mlx_model_path,
                callback,
            )
            foreign_language_speech_ranges = list(
                result.get("foreign_language_speech_ranges") or []
            )
            native_result = dict(result)
            language_ranges: list[_LanguageRange] = []

            if self.segment_callback:
                raw_segments = self._make_segments(result)
                if raw_segments:
                    self.segment_callback(ASRData(raw_segments))

            callback(35, "Loading audio...")
            audio = load_audio(audio_path)

            language_code = str(result.get("language") or self.language or "en").lower()
            hybrid_language_mode = self.language is not None and bool(
                getattr(self, "detect_additional_languages", False)
            )
            if self.language is None or hybrid_language_mode:
                callback(48, "Checking for language switches...")
                detected_language_ranges = self._detect_mlx_language_ranges(
                    audio_path,
                    mlx_model_path,
                    result,
                    language_code,
                )
                audited_language_ranges = _language_ranges_from_foreign_audit(
                    foreign_language_speech_ranges,
                    language_code,
                )
                language_ranges = _merge_confirmed_language_ranges(
                    [*detected_language_ranges, *audited_language_ranges]
                )
                language_ranges, skip_alignment_languages = self._resolve_missing_alignment_models(
                    language_code,
                    language_ranges,
                )
                result = self._retranscribe_mlx_language_ranges(
                    audio_path,
                    mlx_model_path,
                    result,
                    language_ranges,
                    language_code,
                )
            else:
                skip_alignment_languages = set()
            aligned = self._align_multilingual_result(
                result,
                audio,
                language_code,
                callback,
                whisperx_alignment,
                skip_alignment_languages,
            )
            callback(92, "Verifying speech coverage...")
            try:
                critical_gaps = _audit_aligned_speech_coverage(
                    aligned, audio, MLX_AUDIO_SAMPLE_RATE
                )
            except Exception as exc:
                raise RuntimeError("Final MLX speech-coverage audit failed") from exc
            if (
                self.language is not None
                and not hybrid_language_mode
                and foreign_language_speech_ranges
            ):
                ignored_foreign_gaps = [
                    gap
                    for gap in critical_gaps
                    if _gap_is_covered_by_foreign_range(
                        gap,
                        foreign_language_speech_ranges,
                    )
                ]
                if ignored_foreign_gaps:
                    logger.info(
                        "Ignored %d speech coverage gap(s) confirmed as outside the fixed "
                        "source language %s",
                        len(ignored_foreign_gaps),
                        self.language,
                    )
                critical_gaps = [gap for gap in critical_gaps if gap not in ignored_foreign_gaps]
            if self.language is not None and critical_gaps:
                newly_confirmed_foreign = self._classify_mlx_ranges_in_worker(
                    audio_path,
                    mlx_model_path,
                    critical_gaps,
                    callback,
                )
                if newly_confirmed_foreign:
                    foreign_language_speech_ranges.extend(newly_confirmed_foreign)
                    if hybrid_language_mode:
                        audit_language_ranges = _language_ranges_from_foreign_audit(
                            newly_confirmed_foreign,
                            language_code,
                        )
                        if audit_language_ranges:
                            logger.warning(
                                "Final audit recovered %d foreign-language range(s); "
                                "rebuilding multilingual alignment",
                                len(audit_language_ranges),
                            )
                            language_ranges = _merge_confirmed_language_ranges(
                                [*language_ranges, *audit_language_ranges]
                            )
                            (
                                language_ranges,
                                skip_alignment_languages,
                            ) = self._resolve_missing_alignment_models(
                                language_code,
                                language_ranges,
                            )
                            result = self._retranscribe_mlx_language_ranges(
                                audio_path,
                                mlx_model_path,
                                native_result,
                                language_ranges,
                                language_code,
                            )
                            aligned = self._align_multilingual_result(
                                result,
                                audio,
                                language_code,
                                callback,
                                whisperx_alignment,
                                skip_alignment_languages,
                            )
                            critical_gaps = _audit_aligned_speech_coverage(
                                aligned, audio, MLX_AUDIO_SAMPLE_RATE
                            )
                    else:
                        ignored_foreign_gaps = [
                            gap
                            for gap in critical_gaps
                            if _gap_is_covered_by_foreign_range(
                                gap,
                                newly_confirmed_foreign,
                            )
                        ]
                        logger.info(
                            "Final audit confirmed %d uncovered speech gap(s) as outside "
                            "the fixed source language %s",
                            len(ignored_foreign_gaps),
                            self.language,
                        )
                        critical_gaps = [
                            gap for gap in critical_gaps if gap not in ignored_foreign_gaps
                        ]
            if critical_gaps and self.need_word_time_stamp:
                aligned = _recover_aligned_gaps_from_native_words(
                    aligned,
                    native_result,
                    critical_gaps,
                )
                if aligned.get("native_word_gap_recovery"):
                    critical_gaps = _audit_aligned_speech_coverage(
                        aligned, audio, MLX_AUDIO_SAMPLE_RATE
                    )
                    if (
                        self.language is not None
                        and not hybrid_language_mode
                        and foreign_language_speech_ranges
                    ):
                        critical_gaps = [
                            gap
                            for gap in critical_gaps
                            if not _gap_is_covered_by_foreign_range(
                                gap,
                                foreign_language_speech_ranges,
                            )
                        ]
                    logger.warning(
                        "Recovered %d forced-alignment hole(s) with native MLX word timestamps",
                        int(aligned["native_word_gap_recovery"]),
                    )
            # Retain adaptive chunk retries for the pre-existing long-gap case.
            # New short-gap warnings must preserve partial output instead of
            # silently passing or discarding an otherwise complete transcript.
            long_gaps = [g for g in critical_gaps if g.end - g.start >= 6 and g.speech_seconds >= 5]
            if long_gaps:
                windows = ", ".join(f"{gap.start:.1f}-{gap.end:.1f}s" for gap in long_gaps)
                raise RuntimeError(
                    "MLX Whisper left VAD-confirmed speech without aligned words: " + windows
                )
            issues = list(native_result.get("coverage_issues") or [])
            for gap in critical_gaps:
                if not any(i["start"] < gap.end and i["end"] > gap.start for i in issues):
                    issues.append(
                        {
                            "start": gap.start,
                            "end": gap.end,
                            "speech_seconds": gap.speech_seconds,
                            "speech_ratio": gap.speech_ratio,
                            "reason": "final_alignment_gap",
                        }
                    )
            if self.language is not None and not hybrid_language_mode:
                issues = [
                    i
                    for i in issues
                    if not _gap_is_covered_by_foreign_range(
                        _SpeechBackedGap(
                            start=i["start"],
                            end=i["end"],
                            speech_seconds=i["speech_seconds"],
                            speech_ratio=i["speech_ratio"],
                            is_internal=True,
                        ),
                        foreign_language_speech_ranges,
                    )
                ]
            aligned["coverage_issues"] = issues
            aligned["asr_backend"] = "mlx-whisper"
            aligned["mlx_model"] = self.mlx_model

            if self.segment_callback:
                aligned_segments = self._make_segments(aligned)
                if aligned_segments:
                    self.segment_callback(ASRData(aligned_segments))

            try:
                if self.align_device == "cuda":
                    import torch

                    torch.cuda.empty_cache()
            except Exception:
                pass

            if issues:
                from .speech_gap_repair import coverage_issue_message

                callback(98, coverage_issue_message(issues))
            else:
                callback(*ASRStatus.COMPLETED.callback_tuple())
            return aligned

    def _make_segments(self, resp_data: dict) -> List[ASRDataSeg]:
        segments: list[ASRDataSeg] = []

        if self.need_word_time_stamp:
            for item in resp_data.get("segments") or []:
                if not isinstance(item, dict):
                    continue
                words = item.get("words")
                if not isinstance(words, list):
                    if item.get("alignment_skipped"):
                        text = str(item.get("text") or "").strip()
                        start = _float_seconds(item.get("start"))
                        end = _float_seconds(item.get("end"))
                        if text and start is not None and end is not None and end > start:
                            segments.append(
                                ASRDataSeg(
                                    text,
                                    max(0, int(round(start * 1000))),
                                    max(0, int(round(end * 1000))),
                                    timestamp_granularity="sentence",
                                    timing_source="native",
                                    language_code=str(
                                        item.get("language") or resp_data.get("language") or ""
                                    ),
                                )
                            )
                    continue
                word_dicts = [word for word in words if isinstance(word, dict)]
                chars = item.get("chars")
                word_dicts = _refine_words_with_char_alignments(
                    word_dicts,
                    chars if isinstance(chars, list) else None,
                )
                segment_start = _float_seconds(item.get("start"))
                segment_end = _float_seconds(item.get("end"))
                segments.extend(
                    _words_to_segments(
                        word_dicts,
                        segment_start,
                        segment_end,
                        str(item.get("language") or resp_data.get("language") or ""),
                    )
                )

            if segments:
                segments.sort(key=lambda item: (item.start_time, item.end_time))
                return segments

            words = resp_data.get("word_segments") or []
            if isinstance(words, list):
                word_dicts = [word for word in words if isinstance(word, dict)]
                segments.extend(
                    _words_to_segments(
                        word_dicts,
                        language_code=str(resp_data.get("language") or ""),
                    )
                )

            if segments:
                return segments

        for item in resp_data.get("segments") or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            start = item.get("start")
            end = item.get("end")
            if not text or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            start_ms = max(0, int(round(float(start) * 1000)))
            end_ms = max(start_ms, int(round(float(end) * 1000)))
            segments.append(
                ASRDataSeg(
                    text,
                    start_ms,
                    end_ms,
                    timestamp_granularity="sentence",
                    timing_source="native",
                    language_code=str(item.get("language") or resp_data.get("language") or ""),
                )
            )

        return segments

    def _get_key(self) -> str:
        key = {
            "crc32": self.crc32_hex,
            "model": self.mlx_model,
            "language": self.language or "auto",
            "mixed_language_revision": 5 if self.uses_mlx and self.language is None else 0,
            "speech_coverage_revision": 5 if self.uses_mlx else 0,
            "align_device": self.align_device,
            "compute_type": self.compute_type,
            "align_model": self.align_model or "auto",
            "batch_size": self.batch_size,
            "word": self.need_word_time_stamp,
            "spoken_alignment_revision": 2,
        }
        return json.dumps(key, sort_keys=True)
