"""Select the best available VAD for post-ASR timestamp validation."""

from __future__ import annotations

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def is_available() -> bool:
    from subforge.core.asr import silero_vad, ten_vad

    return ten_vad.is_available() or silero_vad.is_available()


def detect_speech_segments(audio_path: str, **kwargs) -> List[Tuple[int, int]]:
    """Prefer TEN-VAD and fall back to Silero only on backend failure.

    An empty TEN-VAD result is valid for silent audio and must not trigger a
    second detector with different semantics.
    """
    from subforge.core.asr import silero_vad, ten_vad

    if ten_vad.is_available():
        try:
            return ten_vad.detect_speech_segments(audio_path, **kwargs)
        except Exception as exc:
            logger.warning("TEN-VAD failed; falling back to Silero VAD: %s", exc)
    return silero_vad.detect_speech_segments(audio_path, **kwargs)
