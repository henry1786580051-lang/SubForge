"""Immutable features shared by Chinese boundary detectors."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TRIM_CHARS = " \t\r\n，。！？；：、,.!?;:…（）()【】[]‘’“”\"'"
_DISPLAY_PUNCTUATION_RE = re.compile(r"[\s，。！？；：、,.!?;:]+")
_TRAILING_FILLER_RE = re.compile(r"(?:[ \t]*你知道(?:的|吗)?)+$")
_LEADING_FILLER_RE = re.compile(r"^(?:你知道(?:的|吗)?[ \t]*)+")
_TERMINAL_PUNCTUATION_RE = re.compile(r"[。！？!?…][）)】\]\"'’”]*$")


@dataclass(frozen=True, slots=True)
class ChineseBoundaryFeatures:
    """Stable text forms derived once for one adjacent Chinese cue pair."""

    raw_left: str
    raw_right: str
    left: str
    right: str
    compact_left: str
    compact_right: str
    canonical_left: str
    canonical_right: str
    display_left_compact: str
    display_right_compact: str
    left_has_terminal_punctuation: bool
    gap_ms: int

    @classmethod
    def from_text(cls, left: str, right: str, *, gap_ms: int = 0) -> "ChineseBoundaryFeatures":
        raw_left = str(left or "").strip()
        raw_right = str(right or "").strip()
        normalized_left = raw_left.strip(_TRIM_CHARS)
        normalized_right = raw_right.strip(_TRIM_CHARS)
        normalized_left = _TRAILING_FILLER_RE.sub("", normalized_left).strip(_TRIM_CHARS)
        normalized_right = _LEADING_FILLER_RE.sub("", normalized_right).strip(_TRIM_CHARS)
        compact_left = re.sub(r"\s+", "", normalized_left)
        compact_right = re.sub(r"\s+", "", normalized_right)
        return cls(
            raw_left=raw_left,
            raw_right=raw_right,
            left=normalized_left,
            right=normalized_right,
            compact_left=compact_left,
            compact_right=compact_right,
            canonical_left=_DISPLAY_PUNCTUATION_RE.sub("", normalized_left).replace(
                "那种", "这种"
            ),
            canonical_right=_DISPLAY_PUNCTUATION_RE.sub("", normalized_right).replace(
                "那种", "这种"
            ),
            display_left_compact=_DISPLAY_PUNCTUATION_RE.sub("", raw_left),
            display_right_compact=_DISPLAY_PUNCTUATION_RE.sub("", raw_right),
            left_has_terminal_punctuation=bool(_TERMINAL_PUNCTUATION_RE.search(raw_left)),
            gap_ms=int(gap_ms),
        )
