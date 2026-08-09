"""Small atomic file-write helpers for durable user-facing outputs."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Replace *path* atomically after fully flushing a sibling temp file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = stat.S_IMODE(destination.stat().st_mode) if destination.exists() else None
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if existing_mode is not None:
            os.chmod(temp_path, existing_mode)
        temp_path.replace(destination)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    """Encode and atomically replace a text file."""
    atomic_write_bytes(path, text.encode(encoding))


def encode_srt_text(text: str) -> bytes:
    """Encode SRT for broad desktop-editor compatibility.

    Word does not reliably detect BOM-less UTF-8 subtitle files, especially on
    Windows. A UTF-8 BOM and CRLF line endings keep the file unambiguous while
    remaining valid SRT for players and subtitle editors.
    """
    normalized = text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\n", "\r\n").encode("utf-8-sig")


def atomic_write_srt(path: str | Path, text: str) -> None:
    """Atomically write a Word-compatible UTF-8 SRT file."""
    atomic_write_bytes(path, encode_srt_text(text))
