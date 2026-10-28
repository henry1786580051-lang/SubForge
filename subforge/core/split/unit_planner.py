"""Bounded variable-count repair around an observed source phrase break."""

from __future__ import annotations

import re
from typing import Callable, Sequence

from subforge.core.asr.asr_data import ASRDataSeg, ASRWord
from subforge.core.split.syntax_units import is_atomic_seam
from subforge.core.split.vehicle_names import splits_vehicle_name
from subforge.core.utils.text_utils import count_words


def _broken(left: str, right: str) -> bool:
    return is_atomic_seam(left, right) or splits_vehicle_name(left, right)


def plan_syntax_units(
    segments: Sequence[ASRDataSeg],
    *,
    soft_max: int,
    hard_max: int,
    join_words: Callable[[Sequence[ASRWord]], str],
    make_cue: Callable[[Sequence[ASRWord], str], ASRDataSeg],
    hard_boundary: Callable[[ASRDataSeg, ASRDataSeg], bool],
    assess: Callable,
) -> list[ASRDataSeg]:
    result = list(segments)
    if any(c.translated_text for c in result):
        return result
    regions: list[tuple[int, int]] = []
    for i, (left, right) in enumerate(zip(result, result[1:])):
        if _broken(left.text, right.text) and not hard_boundary(left, right):
            if regions and regions[-1][1] == i:
                regions[-1] = (regions[-1][0], i + 1)
            else:
                regions.append((i, i + 1))
    for start, stop in reversed(regions):
        cues = result[start : stop + 1]
        words = [w for c in cues for w in c.words]
        if not words or len(cues) > 6 or len(words) > 100 or any(not c.words for c in cues):
            continue
        if (
            len(
                {w.speaker_id or c.speaker_id for c in cues for w in c.words}
                | {c.speaker_id for c in cues if c.speaker_id}
            )
            > 1
        ):
            continue
        if (
            len(
                {w.language_code or c.language_code for c in cues for w in c.words}
                | {c.language_code for c in cues if c.language_code}
            )
            > 1
        ):
            continue
        if any(re.search(r"[\u3400-\u9fff\u3040-\u30ff]", c.text) for c in cues):
            continue
        if any(w.end_time <= w.start_time or w.timing_source == "estimated" for w in words):
            continue
        if any(b.start_time < a.end_time for a, b in zip(words, words[1:])):
            continue
        pauses = {
            i
            for i, (a, b) in enumerate(zip(words, words[1:]), 1)
            if b.start_time - a.end_time > 1800
        }
        if any(
            re.findall(r"\w+", join_words(c.words).casefold())
            != re.findall(r"\w+", c.text.casefold())
            for c in cues
        ):
            continue
        original, total = [], 0
        for c in cues[:-1]:
            total += len(c.words)
            original.append(total)
        seam_cost: dict[int, float] = {}
        for p in range(1, len(words)):
            left, right = join_words(words[:p]), join_words(words[p:])
            if _broken(left, right):
                continue
            # An unknown low rule score is not a license to cut inside a new
            # noun phrase. New seams need punctuation or a bounded clause frame.
            if (
                p not in original
                and p not in pauses
                and not (
                    re.search(r"[,;:.!?]$", words[p - 1].text)
                    or re.match(r"^(?:and|but|because|if|when|where|to)\b", right, re.I)
                    or re.match(r"^as\s+(?:far|well|soon)\s+as\b", right, re.I)
                    or re.match(r"^(?:is|are|was|were)\s+that\b", right, re.I)
                )
            ):
                continue
            risk = assess(left, right).risk
            cost = risk * 1.5 + min(abs(p - old) for old in original) * 0.15
            if re.search(r"[.!?]$", words[p - 1].text):
                cost -= 22
            elif re.search(r"[,;:]$", words[p - 1].text):
                cost -= 8
            if re.match(r"^(?:and|but|because|if|when|where|to)\b", right, re.I):
                cost -= 6
            seam_cost[p] = cost
        states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {(0, 0): (0, ())}
        for end in range(1, len(words) + 1):
            if end < len(words) and end not in seam_cost:
                continue
            for begin in range(max(0, end - hard_max), end):
                part = words[begin:end]
                count = count_words(join_words(part))
                duration = part[-1].end_time - part[0].start_time
                if (
                    count > hard_max
                    or not 600 <= duration <= 8000
                    or any(begin < pause < end for pause in pauses)
                ):
                    continue
                for n in range(1, min(8, len(cues) + 2) + 1):
                    previous = states.get((begin, n - 1))
                    if previous is None:
                        continue
                    cost = previous[0] + 14 + max(0, count - soft_max) * 5 + max(0, 3 - count) * 6
                    if end < len(words):
                        cost += seam_cost[end]
                    candidate = (cost, (*previous[1], end))
                    if (end, n) not in states or candidate[0] < states[end, n][0]:
                        states[end, n] = candidate
        choices = [value for (end, n), value in states.items() if end == len(words) and n]
        if not choices:
            continue
        cuts = min(choices, key=lambda value: value[0])[1]
        parts, previous = [], 0
        for cut in cuts:
            parts.append(make_cue(words[previous:cut], cues[0].speaker_id))
            previous = cut
        if any(_broken(a.text, b.text) for a, b in zip(parts, parts[1:])):
            continue
        result[start : stop + 1] = parts
    return result
