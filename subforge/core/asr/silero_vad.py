"""Silero VAD — standalone speech segment detection as preprocessing."""

import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

_vad_model = None


def _load_model():
    """Lazy-load Silero VAD model via torch.hub."""
    global _vad_model
    if _vad_model is not None:
        return

    import torch
    logger.info("Loading Silero VAD model...")
    _vad_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    if _vad_model is None:
        raise RuntimeError("Silero VAD model failed to load")
    logger.info("Silero VAD model loaded")


def run_vad_inference(
    samples,
    sample_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 300,
    speech_pad_ms: int = 300,
    audio_len_ms: int = 0,
) -> List[Tuple[int, int]]:
    """Core VAD inference on audio samples.

    Args:
        samples: numpy array of float32 audio samples (16kHz mono)
        sample_rate: sample rate (default 16000)
        threshold: speech probability threshold (0.0-1.0)
        min_speech_ms: minimum speech segment duration in ms
        min_silence_ms: minimum silence gap to split segments in ms
        speech_pad_ms: padding added to both sides of each segment in ms
        audio_len_ms: total audio length in ms (for padding bounds)

    Returns:
        List of (start_ms, end_ms) tuples for detected speech segments
    """
    import numpy as np
    import torch

    _load_model()
    assert _vad_model is not None  # guaranteed by _load_model

    window_size = 512  # 32ms at 16kHz
    speech_probs = []
    _vad_model.reset_states()

    for i in range(0, len(samples), window_size):
        chunk = samples[i:i + window_size]
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))
        tensor = torch.from_numpy(chunk)
        prob = _vad_model(tensor, sample_rate).item()
        speech_probs.append(prob)

    # Group consecutive speech frames into segments
    frame_duration_ms = window_size / sample_rate * 1000
    segments = []
    in_speech = False
    seg_start = 0.0

    for i, prob in enumerate(speech_probs):
        if prob >= threshold and not in_speech:
            in_speech = True
            seg_start = i * frame_duration_ms
        elif prob < threshold and in_speech:
            in_speech = False
            seg_end = i * frame_duration_ms
            if seg_end - seg_start >= min_speech_ms:
                segments.append((seg_start, seg_end))

    # Handle speech continuing to end of audio
    if in_speech:
        seg_end = len(speech_probs) * frame_duration_ms
        if seg_end - seg_start >= min_speech_ms:
            segments.append((seg_start, seg_end))

    if not segments:
        return []

    # Merge segments separated by short silence
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < min_silence_ms:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # Apply padding and clamp to audio bounds
    padded = []
    for start, end in merged:
        p_start = max(0, start - speech_pad_ms)
        p_end = min(audio_len_ms, end + speech_pad_ms)
        padded.append((int(p_start), int(p_end)))

    return padded


def detect_speech_segments(
    audio_path: str,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 300,
    speech_pad_ms: int = 300,
) -> List[Tuple[int, int]]:
    """Detect speech segments in audio using Silero VAD.

    Args:
        audio_path: Path to audio file (any format ffmpeg/pydub supports)
        threshold: Speech probability threshold (0.0-1.0)
        min_speech_ms: Minimum speech segment duration in ms
        min_silence_ms: Minimum silence gap to split segments in ms
        speech_pad_ms: Padding added to both sides of each speech segment in ms

    Returns:
        List of (start_ms, end_ms) tuples for detected speech segments
    """
    import numpy as np
    from pydub import AudioSegment

    if not audio_path or not Path(audio_path).is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio as 16kHz mono
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    audio_len_ms = len(audio)

    padded = run_vad_inference(
        samples,
        sample_rate=16000,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        audio_len_ms=audio_len_ms,
    )

    total_speech_ms = sum(end - start for start, end in padded)
    coverage = total_speech_ms / audio_len_ms * 100 if audio_len_ms > 0 else 0
    logger.info(
        f"Silero VAD: {len(padded)} speech segments, "
        f"{total_speech_ms/1000:.1f}s / {audio_len_ms/1000:.1f}s ({coverage:.1f}%)"
    )

    return padded


def is_available() -> bool:
    """Check if Silero VAD is available (requires torch)."""
    import importlib.util
    return importlib.util.find_spec("torch") is not None
