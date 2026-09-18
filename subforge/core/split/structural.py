"""Evidence for open syntax across cues; absence of evidence is not completeness."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class StructuralDependency:
    kind: str
    left_evidence: str
    right_evidence: str


def structural_dependency(left: str, right: str) -> StructuralDependency | None:
    left, right = left.strip(), right.strip()
    if not left or not right or re.search(r"[.!?][\"\')\]]*$", left):
        return None
    patterns = (
        (
            "fraction_quantity",
            r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)$",
            r"^and a (?:half|quarter)\b",
        ),
        ("open_what_subject", r"\bwhat$", r"^(?:i|you|we|they|he|she)\b"),
        ("road_compound", r"\b(?:dirt|gravel|paved|unpaved)$", r"^roads?\b"),
        (
            "directional_predicate",
            r"\b(?:hauling|pulling|racing|speeding)$",
            r"^(?:up|down|past)\b",
        ),
        (
            "negative_parenthetical",
            r"\b(?:cannot|can't|couldn't)(?: for the life of me)?[,]?$",
            r"^(?:after|figure|work|find|get|tell|understand|see)\b",
        ),
        (
            "description_completion",
            r"\b(?:this|that|a|an)\s+(?:like\s+)?[^,.;!?]{1,60}$",
            r"^(?:sort|kind) of (?:thing|design|pattern|finish)\b",
        ),
        (
            "attributive_head",
            r"\b(?:a|an|this|that|these|those)\s+(?:sort of\s+)?(?:very|rather|quite|particularly)\s+[a-z]+(?:ing|ive|ous|ful|less|able|al)$",
            r"^(?!(?:and|but|or|so|i|we|you|they|it|is|are|was|were|can|will|have|has)\b)[a-z]+\b",
        ),
        (
            "interrupted_negative_predicate",
            r"\b(?:cannot|can't|couldn't)(?: for the life of me)?[,]?\s+after [^.!?;]{1,70},$",
            r"^(?:figure|work|find|get|tell|understand|see)\b",
        ),
        ("negative_progressive", r"\b(?:no one|nobody|someone|anyone)['’]s$", r"^[a-z]+ing\b"),
        (
            "restarted_predicate",
            r"\b(?:because|since|that)\s+(it|they|we|you)[,]$",
            r"^(?:it|they|we|you)\s+(?:[a-z]+s|can|will|could|would|has|have)\b",
        ),
        (
            "relative_nominal_complement",
            r"\bthat\s+(?:has|have|had)(?:,?\s+like)?[,]?$",
            r"^(?:a|an|the|some|more|less)\b",
        ),
        (
            "nominal_what_complement",
            r"\bwhat (?:i|we|you|they) (?:really |particularly )?(?:enjoyed|liked|loved)$",
            r"^about\b",
        ),
    )
    for kind, lp, rp in patterns:
        a, b = re.search(lp, left, re.I), re.match(rp, right, re.I)
        if a and b:
            if kind == "restarted_predicate" and a.group(1).lower() != right.split()[0].lower():
                continue
            return StructuralDependency(kind, a.group(), b.group())
    return None
