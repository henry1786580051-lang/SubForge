"""Compact, source-aware guidance for subtitle translation and repair."""

from __future__ import annotations

import re
from collections.abc import Iterable

_CHINESE_TARGETS = {"简体中文", "繁体中文", "粤语"}


def _source_text(source_texts: Iterable[str]) -> str:
    return " ".join(str(text or "").strip() for text in source_texts).lower()


def _contains(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def target_language_style_rules(
    target_language: str,
    source_texts: Iterable[str],
) -> str:
    """Return concise Chinese guidance selected by the current source batch.

    Structural validators remain authoritative. These hints help the first pass
    avoid predictable errors without attaching every historical case to every
    request.
    """
    if target_language not in _CHINESE_TARGETS:
        return ""

    source = _source_text(source_texts)
    rules = [
        "Reconstruct idiomatic Chinese syntax instead of mirroring English word order.",
        "Keep every material subject, predicate, object, modifier, negation, comparison, "
        "number, name, and conclusion exactly once under its owning key.",
        "Make adjacent cues read naturally in sequence without completing one cue with "
        "meaning borrowed from another.",
        "Map the complete source clause before translating fragments: a conjunction, subject, "
        "linking predicate, complement, number, and unit must each appear under exactly one key; "
        "never restart the same clause with a second Chinese subject or connective.",
        "Resolve explicit spoken self-corrections to the final intended value when nearby source "
        "makes that choice certain; do not turn alternatives such as 'X or Y' into a range.",
        "Resolve contrastive references from the full local construction, especially 'A rather "
        "than B, which ...'; do not attach the following belief or comment to the wrong option.",
        "Avoid repeating the same Chinese head noun twice inside one cue when one coherent noun "
        "phrase can express the source exactly.",
        "Omit semantically empty oral fillers, but preserve discourse markers that carry "
        "contrast, correction, uncertainty, or turn-taking intent.",
    ]

    conditional_rules = (
        (
            r"\bhow\s+(?:quiet|loud|fast|slow|good|bad|hard|soft)\b|\bhow\s+\w+\s+.*\bgets?\b",
            "Render English degree constructions as natural Chinese results or states; do "
            "not mechanically translate the surface 'how ...' structure.",
        ),
        (
            r"\b\d{1,3}%\b|\bpercent\b|\buse cases?\b",
            "For percentages and use cases, identify the actual evaluated feature and make "
            "it the Chinese subject; do not mistake an example for the use case itself.",
        ),
        (
            r"\b(?:rpm|horsepower|torque|mpg|gear|clutch|steering|suspension|trim|"
            r"reverse|cargo|wheel|tire|tyre|engine|vehicle|truck|sedan|suv)\b",
            "Use established automotive Chinese, preserve trims and model identifiers, and "
            "recover an elliptical unit only when the local vehicle context makes it unique.",
        ),
        (
            r"\b(?:reading|writing|literate|literacy|post[- ]?literacy)\b",
            "In reading and literacy discussions, distinguish literacy from general culture "
            "or education and keep recurring academic terms consistent.",
        ),
        (
            r"\b(?:email|e-mail|drop us a line|at\s+\w+\s+dot|dot\s+(?:com|org|net))\b",
            "When speech explicitly introduces an email address, normalize an unambiguous "
            "spoken at/dot form as an email rather than a website.",
        ),
        (
            r"\b(?:not|never|isn't|aren't|wasn't|weren't|don't|doesn't|didn't|"
            r"can't|couldn't|won't|wouldn't)\b",
            "Preserve the complete logical scope of negation and comparison across adjacent "
            "fragments; avoid meaning-reversing Chinese double negatives.",
        ),
        (
            r"\b(?:sarcasm|sarcastic|ironically|not actually|yeah,? right)\b",
            "Express clearly supported irony through natural wording without adding editorial "
            "labels or turning it into a sincere statement.",
        ),
        (
            r"\bi\s+don['’]t\s+know\s+if\s+i\s+(?:think|believe)\b",
            "Render nested uncertainty by its actual stance and polarity in idiomatic Chinese; "
            "avoid literal frames such as '我不确定我认为'.",
        ),
        (
            r"\btalk\s+about\s+(?:a|an|the)\s+(?:case|example|illustration|lesson)\b",
            "Treat 'talk about' as an emphatic example marker when the context supports it, "
            "not automatically as an instruction to discuss something.",
        ),
        (
            r"\bnot\b.{0,80}\bcalling\s+it\b.{0,120}\b(?:named|called)\s+it\b",
            "When a common word is also an official all-caps name and the speaker explicitly "
            "jokes that it is not merely their description but the real name, preserve both "
            "layers: translate the ordinary meaning at the joke and introduce the canonical "
            "name at the naming clause. Terminology must not erase the wordplay.",
        ),
        (
            r"\bsynonymous\s+with\b",
            "Render metalinguistic emotional associations as natural Chinese such as "
            "'让人联想到' or '几乎意味着'; avoid tautologies built from '等同于' and '同义词'.",
        ),
        (
            r"\bbeam(?:ed|ing|s)?\b.*\b(?:content|information|stuff)\b|"
            r"\b(?:content|information|stuff)\b.*\bbeam(?:ed|ing|s)?\b",
            "When digital content is figuratively beamed toward a face or eyes, express the "
            "effect as content being pushed or flooding into view, not as a literal ray.",
        ),
    )
    rules.extend(rule for pattern, rule in conditional_rules if _contains(source, pattern))

    return "\n\n<target_language_style>\n" + "\n".join(
        f"{index}. {rule}" for index, rule in enumerate(rules, 1)
    ) + "\n</target_language_style>"


def repair_mode_guidance(multispeaker: bool) -> str:
    """Return distinct repair constraints for monologue and dialogue content."""
    shared = (
        "Treat cue-level readability as a hard requirement. Reconstruct the complete local "
        "idea first, then render each key as a concise Chinese reading unit. Do not strand a "
        "subject, predicate, object, modifier, measure word, complement, or connective across "
        "the boundary. Preserve the combined meaning exactly once and prefer rephrasing within "
        "each key over moving content. When a coordinated subject list spans several keys before "
        "one shared predicate, keep each noun under its source key, make the pre-predicate keys "
        "readable nominal units, and begin the predicate key with only the minimum collective "
        "reference needed in Chinese, such as 'these groups'; do not duplicate the predicate or "
        "any fact. Map the complete source clause before repairing individual fragments. A "
        "conjunction, subject, linking predicate, complement, number, and unit must each appear "
        "under exactly one key; never restart the same clause with a second Chinese subject or "
        "connective. Resolve an explicit spoken self-correction to its final value only when the "
        "nearby source proves that value, and resolve contrastive references from the complete "
        "local construction rather than the nearest noun. Avoid repeating the same Chinese head "
        "noun twice inside one cue when one coherent noun phrase conveys the source exactly."
    )
    if not multispeaker:
        return (
            shared
            + " This is a continuous single-speaker passage: preserve the speaker's argument "
            "and register across cues, but do not repeat the same subject or conclusion merely "
            "to make an isolated fragment grammatical."
        )
    return (
        shared
        + " Speaker values are read-only metadata. Preserve every turn, question, answer, "
        "interruption, qualification, tone, and speaker-specific viewpoint. A speaker change "
        "is normally a hard semantic boundary. At a tightly edited handoff that cuts one "
        "grammatical phrase, restate only the minimum shared grammatical frame needed for both "
        "cues to read naturally; never move or duplicate a name, number, fact, opinion, answer, "
        "or conclusion between speakers. Never emit speaker labels."
    )
