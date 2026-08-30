"""Immutable input features shared by English boundary rules."""

from __future__ import annotations

import re
from dataclasses import dataclass

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
TERMINAL_RE = re.compile(r"[.!?][\"')\]]*$")
CLAUSE_RE = re.compile(r"[,;:][\"')\]]*$")


def tokenize_english(text: str) -> list[str]:
    return [token.replace("’", "'").lower() for token in TOKEN_RE.findall(text)]


def has_terminal_punctuation(text: str) -> bool:
    """Return whether *text* closes a sentence rather than trailing off.

    ASR commonly renders a hesitation as three periods. Treating the final dot
    as a sentence terminator hides dependencies such as ``I think it's... /``.
    """
    stripped = re.sub(r"[\"')\]]+$", "", str(text or "").strip())
    if stripped.endswith(("...", "…")):
        return False
    return bool(TERMINAL_RE.search(str(text or "").strip()))


@dataclass(frozen=True, slots=True)
class EnglishBoundaryFeatures:
    left: str
    right: str
    left_tokens: tuple[str, ...]
    right_tokens: tuple[str, ...]
    tail: str
    previous: str
    head: str
    semantic_left: str
    semantic_right: str
    semantic_tokens: tuple[str, ...]
    semantic_tail: str
    capitalized_dependent_of: bool
    complete_does_clause: bool
    complete_degree_adverb: bool
    complete_superlative: bool
    complete_fixed_phrase: bool
    complete_ease_of_use_phrase: bool
    complete_demonstrative_object: bool
    complete_own_idiom: bool
    complete_visibility_preposition: bool
    complete_predicative_adjective: bool
    complete_scalar_complement: bool

    @property
    def eligible(self) -> bool:
        return bool(
            self.left
            and self.right
            and self.left_tokens
            and self.right_tokens
            and (
                not has_terminal_punctuation(self.left)
                or self.capitalized_dependent_of
            )
        )


def extract_english_boundary_features(left: str, right: str) -> EnglishBoundaryFeatures:
    normalized_left = str(left or "").strip()
    normalized_right = str(right or "").strip()
    left_tokens = tuple(tokenize_english(normalized_left))
    right_tokens = tuple(tokenize_english(normalized_right))
    tail = left_tokens[-1] if left_tokens else ""
    previous = left_tokens[-2] if len(left_tokens) > 1 else ""
    head = right_tokens[0] if right_tokens else ""

    semantic_left = re.sub(
        r"(?:[,;:]?\s*(?:you\s+know|i\s+mean))[,;:]?\s*$",
        "",
        normalized_left,
        flags=re.IGNORECASE,
    ).strip()
    semantic_tokens = tuple(tokenize_english(semantic_left))
    semantic_tail = semantic_tokens[-1] if semantic_tokens else tail
    semantic_right = re.sub(
        r"^(?:(?:you\s+know|i\s+mean)[,;:]?\s*)+",
        "",
        normalized_right,
        flags=re.IGNORECASE,
    ).strip()

    capitalized_dependent_of = bool(
        normalized_left
        and normalized_right
        and re.match(r"^Of\s+(?!course\b)", normalized_right, re.IGNORECASE)
        and not CLAUSE_RE.search(normalized_right)
    )
    complete_does_clause = bool(
        tail == "does"
        and re.search(
            r"\b(?:see|show|check|find\s+out)\s+how\s+"
            r"(?:he|she|it|this|that)\s+does$",
            normalized_left,
            flags=re.IGNORECASE,
        )
    )
    complete_degree_adverb = bool(
        tail == "much"
        and head in {"and", "but", "like"}
        and re.search(
            r"\b(?:elevate|help|improve|like|love|matter)\b[^.!?]*\bso\s+much$",
            normalized_left,
            flags=re.IGNORECASE,
        )
    )
    complete_superlative = bool(
        tail == "most"
        and head in {"and", "but", "or"}
        and re.search(r"\b[a-z][a-z'’-]*\s+the\s+most$", semantic_left, re.IGNORECASE)
    )
    complete_fixed_phrase = bool(
        head in {"a", "an", "the"}
        and re.search(
            r"\b(?:don['’]t|do\s+not)\s+get\s+me\s+wrong$",
            normalized_left,
            flags=re.IGNORECASE,
        )
    )
    complete_ease_of_use_phrase = bool(
        head == "in"
        and re.search(
            r"\b(?:the\s+)?ease\s+of\s+use$",
            normalized_left,
            flags=re.IGNORECASE,
        )
    )
    complete_demonstrative_object = bool(
        tail in {"this", "that"}
        and re.search(
            r"\b(?:hear|see|watch|feel|like|love|hate|want|need|recommend|"
            r"prefer|use)\s+(?:all\s+of\s+)?(?:this|that)$",
            semantic_left,
            flags=re.IGNORECASE,
        )
    )
    complete_own_idiom = bool(
        tail == "own"
        and re.search(
            r"\bof\s+(?:his|her|its|my|our|their|your)\s+own[,;:]?$",
            semantic_left,
            flags=re.IGNORECASE,
        )
    )
    complete_visibility_preposition = bool(
        tail == "of"
        and head in {"and", "but", "because", "so", "while"}
        and re.search(
            r"\b(?:easy|hard|difficult)\s+to\s+(?:see|look)\s+out\s+of$",
            semantic_left,
            flags=re.IGNORECASE,
        )
    )
    complete_predicative_adjective = bool(
        head in {"and", "but", "or"}
        and bool(CLAUSE_RE.search(normalized_left))
        and re.search(
            r"\b(?:am|is|are|was|were|be|been|being|become(?:s|d)?|"
            r"feel(?:s|t)?|look(?:s|ed)?|remain(?:s|ed)?|seem(?:s|ed)?)\s+"
            r"(?:quite\s+|rather\s+|really\s+|so\s+|too\s+|very\s+)?"
            rf"{re.escape(tail)}[,;:]?$",
            semantic_left,
            re.IGNORECASE,
        )
    )
    complete_scalar_complement = bool(
        tail in {"less", "more"}
        and head in {"and", "but", "or"}
        and bool(CLAUSE_RE.search(normalized_left))
        and re.search(
            r"\b(?:cost|last|measure|take|weigh)(?:s|ed)?\s+(?:less|more)[,;:]?$",
            semantic_left,
            re.IGNORECASE,
        )
    )

    return EnglishBoundaryFeatures(
        left=normalized_left,
        right=normalized_right,
        left_tokens=left_tokens,
        right_tokens=right_tokens,
        tail=tail,
        previous=previous,
        head=head,
        semantic_left=semantic_left,
        semantic_right=semantic_right,
        semantic_tokens=semantic_tokens,
        semantic_tail=semantic_tail,
        capitalized_dependent_of=capitalized_dependent_of,
        complete_does_clause=complete_does_clause,
        complete_degree_adverb=complete_degree_adverb,
        complete_superlative=complete_superlative,
        complete_fixed_phrase=complete_fixed_phrase,
        complete_ease_of_use_phrase=complete_ease_of_use_phrase,
        complete_demonstrative_object=complete_demonstrative_object,
        complete_own_idiom=complete_own_idiom,
        complete_visibility_preposition=complete_visibility_preposition,
        complete_predicative_adjective=complete_predicative_adjective,
        complete_scalar_complement=complete_scalar_complement,
    )
