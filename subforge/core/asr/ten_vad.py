"""TEN-VAD speech detection for conservative timestamp validation."""

from __future__ import annotations

import ctypes
import logging
import os
import platform
from pathlib import Path
from typing import List, Tuple

import numpy as np

from subforge.config import RESOURCE_PATH

logger = logging.getLogger(__name__)

HOP_SIZE = 256
SAMPLE_RATE = 16000
_library: ctypes.CDLL | None = None


def _library_candidates() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("SUBFORGE_TEN_VAD_LIBRARY", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    if platform.system() == "Darwin":
        candidates.append(RESOURCE_PATH / "ten_vad" / "macos" / "ten_vad")
    return candidates


def _library_path() -> Path | None:
    return next((path for path in _library_candidates() if path.is_file()), None)


def is_available() -> bool:
    """Return whether the bundled native library supports this platform."""
    return _library_path() is not None


def _load_library() -> ctypes.CDLL:
    global _library
    if _library is not None:
        return _library

    path = _library_path()
    if path is None:
        raise RuntimeError("TEN-VAD native library is unavailable on this platform")

    library = ctypes.CDLL(str(path))
    library.ten_vad_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_float,
    ]
    library.ten_vad_create.restype = ctypes.c_int
    library.ten_vad_destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    library.ten_vad_destroy.restype = ctypes.c_int
    library.ten_vad_process.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.ten_vad_process.restype = ctypes.c_int
    _library = library
    return library


class _TenVadDetector:
    def __init__(self, threshold: float) -> None:
        self._library = _load_library()
        self._handler = ctypes.c_void_p()
        result = self._library.ten_vad_create(
            ctypes.byref(self._handler),
            ctypes.c_size_t(HOP_SIZE),
            ctypes.c_float(threshold),
        )
        if result != 0 or not self._handler:
            raise RuntimeError(f"TEN-VAD initialization failed with code {result}")

    def process(self, frame: np.ndarray) -> tuple[float, int]:
        contiguous = np.ascontiguousarray(frame, dtype=np.int16)
        probability = ctypes.c_float()
        flag = ctypes.c_int32()
        result = self._library.ten_vad_process(
            self._handler,
            contiguous.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(HOP_SIZE),
            ctypes.byref(probability),
            ctypes.byref(flag),
        )
        if result != 0:
            raise RuntimeError(f"TEN-VAD inference failed with code {result}")
        return probability.value, flag.value

    def close(self) -> None:
        if not self._handler:
            return
        result = self._library.ten_vad_destroy(ctypes.byref(self._handler))
        self._handler = ctypes.c_void_p()
        if result != 0:
            logger.warning("TEN-VAD cleanup failed with code %s", result)

    def __enter__(self) -> "_TenVadDetector":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _create_detector(threshold: float) -> _TenVadDetector:
    return _TenVadDetector(threshold)


def group_speech_frames(
    flags: list[int],
    *,
    audio_len_ms: int,
    min_speech_ms: int,
    min_silence_ms: int,
    speech_pad_ms: int,
) -> List[Tuple[int, int]]:
    frame_ms = HOP_SIZE / SAMPLE_RATE * 1000
    raw: list[tuple[float, float]] = []
    start: float | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index * frame_ms
        elif not flag and start is not None:
            end = index * frame_ms
            if end - start >= min_speech_ms:
                raw.append((start, end))
            start = None
    if start is not None:
        end = min(audio_len_ms, len(flags) * frame_ms)
        if end - start >= min_speech_ms:
            raw.append((start, end))
    if not raw:
        return []

    merged = [raw[0]]
    for start, end in raw[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end < min_silence_ms:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))

    padded: list[tuple[int, int]] = []
    for index, (start, end) in enumerate(merged):
        left_room = float(speech_pad_ms)
        if index:
            left_room = min(left_room, (start - merged[index - 1][1]) / 2)
        right_room = float(speech_pad_ms)
        if index + 1 < len(merged):
            right_room = min(right_room, (merged[index + 1][0] - end) / 2)
        padded.append(
            (
                max(0, int(round(start - left_room))),
                min(audio_len_ms, int(round(end + right_room))),
            )
        )
    return padded


def infer_vad_flags(samples: np.ndarray, threshold: float = 0.5) -> list[int]:
    """Run TEN-VAD once and retain frame decisions for multiple groupings."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return []
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = np.rint(pcm * 32767.0).astype(np.int16)
    flags: list[int] = []
    with _create_detector(threshold) as detector:
        for offset in range(0, pcm.size, HOP_SIZE):
            frame = pcm[offset : offset + HOP_SIZE]
            if frame.size < HOP_SIZE:
                frame = np.pad(frame, (0, HOP_SIZE - frame.size))
            _, flag = detector.process(frame)
            flags.append(int(bool(flag)))
    return flags


def run_vad_inference(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 300,
    speech_pad_ms: int = 300,
    audio_len_ms: int = 0,
) -> List[Tuple[int, int]]:
    if sample_rate != SAMPLE_RATE:
        raise ValueError("TEN-VAD requires 16 kHz audio")
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return []
    if audio_len_ms <= 0:
        audio_len_ms = int(round(samples.size / sample_rate * 1000))

    flags = infer_vad_flags(samples, threshold)

    return group_speech_frames(
        flags,
        audio_len_ms=audio_len_ms,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )


def detect_speech_segments(
    audio_path: str,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 300,
    speech_pad_ms: int = 300,
) -> List[Tuple[int, int]]:
    from pydub import AudioSegment

    path = Path(audio_path) if audio_path else None
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    samples = np.asarray(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    segments = run_vad_inference(
        samples,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        audio_len_ms=len(audio),
    )
    logger.info("TEN-VAD: %s speech segments over %.1fs", len(segments), len(audio) / 1000)
    return segments
