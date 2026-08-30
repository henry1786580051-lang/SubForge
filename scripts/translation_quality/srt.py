"""Strict, read-only SRT parsing for translation-quality evaluation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SrtLayout = Literal["source_only", "target_above", "source_above", "auto"]

_TIME_PATTERN = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})(?P<settings>.*)$"
)


class SrtParseError(ValueError):
    """Raised when an SRT cannot be parsed without guessing its structure."""


@dataclass(frozen=True)
class SrtCue:
    index: int
    timeline: str
    start_ms: int
    end_ms: int
    source: str
    target: str
    text_lines: tuple[str, ...]


@dataclass(frozen=True)
class SrtDocument:
    path: Path
    cues: tuple[SrtCue, ...]
    encoding: str
    newline: str
    has_bom: bool


def _timestamp_ms(value: str) -> int:
    normalized = value.replace(".", ",")
    hours, minutes, remainder = normalized.split(":")
    seconds, milliseconds = remainder.split(",")
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(milliseconds)
    )


def _read_text(path: Path) -> tuple[str, str, bool]:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16"), "utf-16", True
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding), encoding, has_bom
        except UnicodeDecodeError:
            continue
    raise SrtParseError(f"Unable to decode subtitle file: {path}")


def _line_family(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text.strip())
    if not normalized:
        return "empty"
    han = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", normalized))
    kana = len(re.findall(r"[\u3040-\u30ff\u31f0-\u31ff]", normalized))
    hangul = len(
        re.findall(
            r"[\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff]",
            normalized,
        )
    )
    latin = len(re.findall(r"[A-Za-z]", normalized))
    if hangul >= 2 and hangul >= han:
        return "hangul"
    if kana:
        return "japanese"
    if han >= 2:
        return "han"
    if latin >= 2 and not (han or kana or hangul):
        return "latin"
    if hangul:
        return "hangul"
    if han:
        return "han"
    return "neutral"


def _group_family(lines: list[str]) -> str:
    families = [_line_family(line) for line in lines if line.strip()]
    if not families:
        return "empty"
    counts = {name: families.count(name) for name in ("han", "japanese", "hangul", "latin")}
    winner, count = max(counts.items(), key=lambda item: item[1])
    if count:
        return winner
    return "neutral"


def _split_language_groups(lines: list[str]) -> tuple[list[str], list[str]] | None:
    if len(lines) < 2:
        return None
    for boundary in range(1, len(lines)):
        left = lines[:boundary]
        right = lines[boundary:]
        left_family = _group_family(left)
        right_family = _group_family(right)
        if left_family == right_family:
            continue
        if "han" in {left_family, right_family} and {left_family, right_family} & {
            "latin",
            "japanese",
            "hangul",
        }:
            return left, right
    return None


def _assign_text(lines: list[str], layout: SrtLayout) -> tuple[str, str]:
    if layout == "source_only":
        return "\n".join(lines).strip(), ""

    groups = _split_language_groups(lines)
    if groups is not None:
        left, right = groups
        left_family = _group_family(left)
        right_family = _group_family(right)
        if left_family == "han" and right_family != "han":
            return "\n".join(right).strip(), "\n".join(left).strip()
        if right_family == "han" and left_family != "han":
            return "\n".join(left).strip(), "\n".join(right).strip()

    if layout == "target_above":
        if len(lines) < 2:
            return "\n".join(lines).strip(), ""
        return "\n".join(lines[1:]).strip(), lines[0].strip()
    if layout == "source_above":
        if len(lines) < 2:
            return "\n".join(lines).strip(), ""
        return lines[0].strip(), "\n".join(lines[1:]).strip()

    if len(lines) == 1:
        return lines[0].strip(), ""
    first_family = _line_family(lines[0])
    last_family = _line_family(lines[-1])
    if first_family == "han" and last_family != "han":
        return "\n".join(lines[1:]).strip(), lines[0].strip()
    if last_family == "han" and first_family != "han":
        return lines[0].strip(), "\n".join(lines[1:]).strip()
    return "\n".join(lines).strip(), ""


def parse_srt(path: Path, *, layout: SrtLayout = "auto") -> SrtDocument:
    path = path.expanduser().resolve()
    text, encoding, has_bom = _read_text(path)
    newline = "crlf" if "\r\n" in text else "lf"
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    blocks = [block for block in re.split(r"\n[ \t]*\n", normalized.strip()) if block.strip()]
    cues: list[SrtCue] = []
    seen_indices: set[int] = set()

    for position, block in enumerate(blocks, 1):
        raw_lines = block.split("\n")
        lines = [line.rstrip("\ufeff") for line in raw_lines]
        if len(lines) < 2:
            raise SrtParseError(f"Block {position} has fewer than two lines in {path}")
        try:
            index = int(lines[0].strip())
        except ValueError as exc:
            raise SrtParseError(f"Invalid cue index at block {position} in {path}") from exc
        if index in seen_indices:
            raise SrtParseError(f"Duplicate cue index {index} in {path}")
        seen_indices.add(index)

        timeline = lines[1].strip()
        match = _TIME_PATTERN.match(timeline)
        if not match:
            raise SrtParseError(f"Invalid timeline for cue {index} in {path}: {timeline}")
        start_ms = _timestamp_ms(match.group("start"))
        end_ms = _timestamp_ms(match.group("end"))
        if end_ms < start_ms:
            raise SrtParseError(f"Cue {index} ends before it starts in {path}")

        text_lines = [line.strip() for line in lines[2:] if line.strip()]
        source, target = _assign_text(text_lines, layout)
        cues.append(
            SrtCue(
                index=index,
                timeline=timeline,
                start_ms=start_ms,
                end_ms=end_ms,
                source=source,
                target=target,
                text_lines=tuple(text_lines),
            )
        )

    return SrtDocument(
        path=path,
        cues=tuple(cues),
        encoding=encoding,
        newline=newline,
        has_bom=has_bom,
    )
