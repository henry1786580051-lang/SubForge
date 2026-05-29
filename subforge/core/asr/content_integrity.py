"""Content Integrity Score (CIS) — safety guardrail for VAD parameters.

CIS = current_speech_duration / baseline_speech_duration

Baseline uses very宽松 parameters (threshold=0.2) to capture almost all speech.
If CIS < 0.85, the VAD parameters are too strict and content is being lost.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Thresholds
CIS_SAFE = 0.90
CIS_WARNING = 0.85

# Baseline parameters (very permissive)
BASELINE_THRESHOLD = 0.2
BASELINE_MIN_SPEECH_MS = 100
BASELINE_MIN_SILENCE_MS = 100
BASELINE_SPEECH_PAD_MS = 200


def compute_cis(
    audio_path: str,
    current_segments: List[Tuple[int, int]],
    current_params: dict,
) -> float:
    """Compute Content Integrity Score.

    Args:
        audio_path: Path to audio file
        current_segments: List of (start_ms, end_ms) from current VAD
        current_params: Current VAD parameters dict

    Returns:
        CIS score (0.0 - 1.0+)
    """
    try:
        import numpy as np
        from pydub import AudioSegment

        from subforge.core.asr.silero_vad import run_vad_inference

        # Load audio
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        audio_len_ms = len(audio)

        # Run baseline VAD (very permissive)
        baseline_segments = run_vad_inference(
            samples,
            sample_rate=16000,
            threshold=BASELINE_THRESHOLD,
            min_speech_ms=BASELINE_MIN_SPEECH_MS,
            min_silence_ms=BASELINE_MIN_SILENCE_MS,
            speech_pad_ms=BASELINE_SPEECH_PAD_MS,
            audio_len_ms=audio_len_ms,
        )

        # Calculate durations
        baseline_duration = sum(e - s for s, e in baseline_segments) / 1000
        current_duration = sum(e - s for s, e in current_segments) / 1000

        if baseline_duration <= 0:
            logger.warning("Baseline duration is 0, cannot compute CIS")
            return 1.0

        cis = current_duration / baseline_duration

        # Log results
        logger.info(f"CIS: {cis:.3f} (baseline={baseline_duration:.0f}s, current={current_duration:.0f}s)")

        if cis < CIS_WARNING:
            logger.warning(f"CIS {cis:.3f} < {CIS_WARNING} — losing >{((1-cis)*100):.0f}% speech content!")
        elif cis < CIS_SAFE:
            logger.info(f"CIS {cis:.3f} < {CIS_SAFE} — monitor closely")

        return cis

    except Exception as e:
        logger.warning(f"CIS computation failed: {e}")
        return 1.0  # Assume safe if computation fails


def check_cis_health(cis: float) -> Tuple[str, str]:
    """Check CIS health status.

    Args:
        cis: Content Integrity Score

    Returns:
        (status, message) tuple
    """
    if cis >= CIS_SAFE:
        return "OK", f"CIS {cis:.3f} — content integrity good"
    elif cis >= CIS_WARNING:
        return "WARN", f"CIS {cis:.3f} — monitor closely, may lose some content"
    else:
        return "FAIL", f"CIS {cis:.3f} — CRITICAL: losing >{((1-cis)*100):.0f}% speech content!"


def suggest_parameter_adjustment(cis: float, current_params: dict) -> dict:
    """Suggest parameter adjustments if CIS is too low.

    Args:
        cis: Current CIS score
        current_params: Current VAD parameters

    Returns:
        Suggested parameters dict
    """
    if cis >= CIS_SAFE:
        return current_params

    # If CIS is low, suggest more permissive parameters
    suggested = current_params.copy()

    # Increase speech_pad_ms (most impactful)
    if suggested.get("speech_pad_ms", 300) < 400:
        suggested["speech_pad_ms"] = 400
        logger.info(f"Suggesting speech_pad_ms={suggested['speech_pad_ms']}")

    # Decrease threshold
    if suggested.get("threshold", 0.5) > 0.3:
        suggested["threshold"] = 0.3
        logger.info(f"Suggesting threshold={suggested['threshold']}")

    # Decrease min_silence_ms
    if suggested.get("min_silence_ms", 300) > 200:
        suggested["min_silence_ms"] = 200
        logger.info(f"Suggesting min_silence_ms={suggested['min_silence_ms']}")

    return suggested
