"""Resource-safe pydub file helpers."""

from __future__ import annotations

from os import PathLike
from typing import Any

from pydub import AudioSegment


def load_audio_file(path: str | PathLike[str]) -> Any:
    """Decode an audio file while deterministically closing its input handle."""
    with open(path, "rb") as source:
        return AudioSegment.from_file(source)


def export_audio_file(
    audio: Any,
    path: str | PathLike[str],
    *,
    format: str,
    **kwargs: Any,
) -> None:
    """Export to a path and close the file object returned by pydub."""
    output: Any = audio.export(str(path), format=format, **kwargs)
    close = getattr(output, "close", None)
    if callable(close):
        close()
