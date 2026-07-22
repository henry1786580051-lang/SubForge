"""Per-transcription audio analysis shared by timing post-processors."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any


class AudioAnalysisContext:
    """Lazily decode one 16 kHz mono source and cache compatible analyses."""

    def __init__(self, audio_path: str) -> None:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        self.audio_path = str(path)
        self._audio = None
        self._samples = None
        self._energy_cache: dict[int, list[dict[str, int]]] = {}
        self._speech_cache: dict[tuple[Any, ...], list[tuple[int, int]]] = {}
        self._ten_flags: dict[float, list[int]] = {}
        self._lock = threading.RLock()

    def audio_segment(self):
        with self._lock:
            if self._audio is None:
                from pydub import AudioSegment

                self._audio = (
                    AudioSegment.from_file(self.audio_path)
                    .set_frame_rate(16000)
                    .set_channels(1)
                    .set_sample_width(2)
                )
            return self._audio

    def samples(self):
        with self._lock:
            if self._samples is None:
                import numpy as np

                audio = self.audio_segment()
                self._samples = (
                    np.asarray(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
                )
            return self._samples

    def energy_windows(self, window_ms: int = 50) -> list[dict[str, int]]:
        with self._lock:
            cached = self._energy_cache.get(window_ms)
            if cached is not None:
                return cached
            audio = self.audio_segment()
            energies = [
                {"time_ms": offset, "rms": audio[offset : offset + window_ms].rms}
                for offset in range(0, len(audio), window_ms)
            ]
            self._energy_cache[window_ms] = energies
            return energies

    def speech_segments(
        self,
        *,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 300,
        speech_pad_ms: int = 300,
    ) -> list[tuple[int, int]]:
        key = (threshold, min_speech_ms, min_silence_ms, speech_pad_ms)
        with self._lock:
            cached = self._speech_cache.get(key)
            if cached is not None:
                return cached

            from subforge.core.asr import silero_vad, ten_vad

            audio = self.audio_segment()
            samples = self.samples()
            if ten_vad.is_available():
                try:
                    flags = self._ten_flags.get(threshold)
                    if flags is None:
                        flags = ten_vad.infer_vad_flags(samples, threshold)
                        self._ten_flags[threshold] = flags
                    segments = ten_vad.group_speech_frames(
                        flags,
                        audio_len_ms=len(audio),
                        min_speech_ms=min_speech_ms,
                        min_silence_ms=min_silence_ms,
                        speech_pad_ms=speech_pad_ms,
                    )
                except Exception:
                    segments = silero_vad.run_vad_inference(
                        samples,
                        sample_rate=16000,
                        threshold=threshold,
                        min_speech_ms=min_speech_ms,
                        min_silence_ms=min_silence_ms,
                        speech_pad_ms=speech_pad_ms,
                        audio_len_ms=len(audio),
                    )
            else:
                segments = silero_vad.run_vad_inference(
                    samples,
                    sample_rate=16000,
                    threshold=threshold,
                    min_speech_ms=min_speech_ms,
                    min_silence_ms=min_silence_ms,
                    speech_pad_ms=speech_pad_ms,
                    audio_len_ms=len(audio),
                )
            self._speech_cache[key] = segments
            return segments
