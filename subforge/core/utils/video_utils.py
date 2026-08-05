"""Media conversion helpers used by speech transcription."""

import os
import subprocess
from pathlib import Path

from subforge.core.utils.logger import setup_logger

logger = setup_logger("video_utils")


def video2audio(
    input_file: str,
    output: str = "",
    audio_track_index: int = 0,
    cancel_event=None,
) -> bool:
    """Extract one media audio track as mono 16 kHz PCM WAV."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i",
        input_file,
        "-map",
        f"0:a:{audio_track_index}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-y",
        str(output_path),
    ]
    logger.debug("Extracting audio track %s", audio_track_index)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            ),
        )
        while True:
            try:
                _stdout, stderr = process.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if cancel_event is None or not cancel_event.is_set():
                    continue
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=5)
                logger.info("Audio extraction cancelled")
                return False

        if process.returncode == 0 and output_path.is_file():
            return True
        logger.error("Audio conversion failed: %s", stderr.strip())
        return False
    except Exception:
        logger.exception("Audio conversion failed")
        return False
