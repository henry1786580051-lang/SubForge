"""Recognize a stranded deadline after a second coordinated statistical clause."""

from __future__ import annotations

import re

_TIME = r"(?:by|at)\s+the\s+end\s+of\s+(?:\d{4}|(?:last|this|next)\s+year)(?:\s*,?\s*\d{4})?"
_TIME_ONLY = re.compile(rf"^{_TIME}[,.]?$", re.IGNORECASE)
_CLAUSE = re.compile(
    r"^(?:and|but)\s+(?:that|this|the)\s+(?:number|figure|share|proportion|percentage|rate)\s+"
    r"(?:(?:could|may|might|would|will)\s+)?(?:have\s+)?"
    r"(?:increased?|risen|rise|grown|grow|fallen|fall|decreased?|reached?|doubled?|halved?)\b",
    re.IGNORECASE,
)


def temporal_clause_split(left: str, right: str) -> int | None:
    """Return token boundary only for a time tail + new clause + stranded time tail.

    Ordinary standalone dates, corrections, and single-clause temporal tails are
    intentionally left alone. This is a local source repair, not a fact checker.
    """
    if not _TIME_ONLY.fullmatch(right.strip()):
        return None
    tokens = left.split()
    for position in range(1, len(tokens)):
        if _TIME_ONLY.fullmatch(" ".join(tokens[:position])) and _CLAUSE.match(
            " ".join(tokens[position:])
        ):
            return position
    return None


def is_temporal_clause_boundary(left: str, right: str) -> bool:
    """A completed date tail followed by a new statistic is a safe clause break."""
    return bool(_TIME_ONLY.fullmatch(left.strip()) and _CLAUSE.match(right.strip()))
