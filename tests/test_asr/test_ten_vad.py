"""Tests for TEN-VAD timestamp validation and Silero fallback."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from subforge.core.asr import speech_vad, ten_vad


class _FakeDetector:
    def __init__(self, flags: list[int]) -> None:
        self.flags = iter(flags)

    def process(self, _frame):
        flag = next(self.flags, 0)
        return float(flag), flag

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def test_ten_vad_groups_frames_without_padding_overlap(monkeypatch):
    flags = [0] * 10 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 10
    monkeypatch.setattr(ten_vad, "_create_detector", lambda _threshold: _FakeDetector(flags))
    samples = np.zeros(len(flags) * ten_vad.HOP_SIZE, dtype=np.float32)

    segments = ten_vad.run_vad_inference(
        samples,
        min_speech_ms=160,
        min_silence_ms=180,
        speech_pad_ms=300,
    )

    assert len(segments) == 2
    assert segments[0][1] <= segments[1][0]
    assert segments[0][0] >= 0
    assert segments[-1][1] <= round(len(samples) / ten_vad.SAMPLE_RATE * 1000)


def test_speech_vad_prefers_ten_without_falling_back_on_silence(monkeypatch):
    calls = SimpleNamespace(ten=0, silero=0)

    monkeypatch.setattr(ten_vad, "is_available", lambda: True)
    monkeypatch.setattr(
        ten_vad,
        "detect_speech_segments",
        lambda *_args, **_kwargs: setattr(calls, "ten", calls.ten + 1) or [],
    )
    from subforge.core.asr import silero_vad

    monkeypatch.setattr(
        silero_vad,
        "detect_speech_segments",
        lambda *_args, **_kwargs: setattr(calls, "silero", calls.silero + 1) or [(0, 1)],
    )

    assert speech_vad.detect_speech_segments("silent.wav") == []
    assert calls.ten == 1
    assert calls.silero == 0


def test_speech_vad_falls_back_to_silero_on_ten_failure(monkeypatch):
    from subforge.core.asr import silero_vad

    monkeypatch.setattr(ten_vad, "is_available", lambda: True)
    monkeypatch.setattr(
        ten_vad,
        "detect_speech_segments",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("native failure")),
    )
    monkeypatch.setattr(
        silero_vad,
        "detect_speech_segments",
        lambda *_args, **_kwargs: [(100, 500)],
    )

    assert speech_vad.detect_speech_segments("audio.wav") == [(100, 500)]
