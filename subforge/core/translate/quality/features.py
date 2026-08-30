"""Immutable source-cue features for translation quality pipelines."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f]")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_HANGUL_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_LEXICAL_TOKEN_RE = re.compile(
    r"[A-Za-z\u00c0-\u024f]+(?:['’][A-Za-z\u00c0-\u024f]+)?"
    r"|\d+(?:[.,:/-]\d+)*"
    r"|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
    r"|[\u3040-\u30ff\u31f0-\u31ff]+"
    r"|[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]+"
)
_NUMBER_RE = re.compile(r"(?<!\w)[+-]?\d+(?:[.,:/-]\d+)*(?:%|[A-Za-z]{1,5})?(?!\w)")
_IDENTIFIER_RE = re.compile(
    r"(?<!\w)(?:[A-Z]{2,}(?:-?[A-Z0-9]+)*|[A-Za-z]+-\d+[A-Za-z0-9-]*|"
    r"[A-Za-z]+\d+[A-Za-z0-9-]*|"
    r"\d+[A-Za-z][A-Za-z0-9-]*)(?!\w)"
)
_TERMINAL_PUNCTUATION = frozenset(".!?。！？…")


def normalize_source_text(text: str) -> str:
    """Return a stable comparison form without changing source semantics."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text or ""))).strip()


def detect_scripts(text: str) -> frozenset[str]:
    """Detect only script families that are useful to subtitle validation."""
    scripts: set[str] = set()
    checks = (
        ("latin", _LATIN_RE),
        ("cjk", _CJK_RE),
        ("kana", _KANA_RE),
        ("hangul", _HANGUL_RE),
        ("cyrillic", _CYRILLIC_RE),
        ("arabic", _ARABIC_RE),
    )
    for name, pattern in checks:
        if pattern.search(text):
            scripts.add(name)
    return frozenset(scripts)


@dataclass(frozen=True, slots=True)
class CueFeatures:
    """Deterministic facts derived from one source cue exactly once per task."""

    index: int
    normalized_source: str
    scripts: frozenset[str]
    lexical_tokens: tuple[str, ...]
    numeric_tokens: tuple[str, ...]
    identifier_tokens: tuple[str, ...]
    character_count: int
    starts_lowercase: bool
    ends_terminal_punctuation: bool

    @classmethod
    def from_source(cls, index: int, source: str) -> "CueFeatures":
        normalized = normalize_source_text(source)
        first_alpha = next((char for char in normalized if char.isalpha()), "")
        return cls(
            index=index,
            normalized_source=normalized,
            scripts=detect_scripts(normalized),
            lexical_tokens=tuple(_LEXICAL_TOKEN_RE.findall(normalized)),
            numeric_tokens=tuple(_NUMBER_RE.findall(normalized)),
            identifier_tokens=tuple(_IDENTIFIER_RE.findall(normalized)),
            character_count=len(normalized),
            starts_lowercase=bool(first_alpha and first_alpha.islower()),
            ends_terminal_punctuation=bool(
                normalized and normalized.rstrip()[-1] in _TERMINAL_PUNCTUATION
            ),
        )
