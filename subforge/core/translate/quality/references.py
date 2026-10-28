"""Read-only, local evidence for physical motion pronouns in vehicle controls."""

from __future__ import annotations

import re


def physical_motion_reference(
    index: int, source: str, sources: dict[int, str], gaps: dict[int, int]
) -> dict:
    motion = re.search(
        r"\b(?:this|that|it|them|windows?|roof|front)\s+(?:to\s+)?go(?:es)?\s+(down|up)\b",
        source,
        re.I,
    )
    if not motion or re.search(
        r"\bgo(?:es)?\s+down\s+(?:in\s+history|well|as\s+one)\b", source, re.I
    ):
        return {}
    if not re.search(
        r"\b(?:prematurely|automatically|manually|button)\b", source, re.I
    ) and not re.search(
        r"\b(?:windows?|roof|tailgate|front)\s+(?:to\s+)?go(?:es)?\s+(?:down|up)\b", source, re.I
    ):
        return {}
    anchors = []
    for i in range(max(1, index - 8), index + 9):
        if any(gaps.get(p, 0) > 20_000 for p in range(min(i, index), max(i, index))):
            continue
        text = sources.get(i, source if i == index else "")
        match = re.search(
            r"\b(?:front[- ](?:end\s+)?lift|front\s+end\s+lift|smart\s+lift|windows?|roof|tailgate)\b",
            text,
            re.I,
        )
        if match:
            anchors.append((abs(i - index), i > index, match.group()))
    if not anchors:
        return {}
    anchor = min(anchors)[2]
    direction = motion.group(1).lower()
    return {
        "source_anchor": anchor,
        "direction": direction,
        "instruction": (
            f"Nearby source explicitly discusses '{anchor}'. Here go {direction} is physical "
            f"{'lowering (下降/降下)' if direction == 'down' else 'raising (升起/抬升)'}, not an event happening. "
            "Resolve the pronoun from that local control context and retain the condition and timing qualifier."
        ),
    }
