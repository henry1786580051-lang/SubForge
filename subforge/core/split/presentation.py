"""Bounded resegmentation of a spoken introduction with word provenance intact."""

from __future__ import annotations

import re
from typing import Callable, Sequence

from subforge.core.asr.asr_data import ASRDataSeg, ASRWord
from subforge.core.split.vehicle_names import introduction_seam, vehicle_apposition, vehicle_names
from subforge.core.utils.text_utils import count_words


def opening_presentation_indices(segments: Sequence[ASRDataSeg]) -> tuple[int, ...]:
    """Locate one short, already separated introduction for translation polish."""
    for i, (left, right) in enumerate(zip(segments, segments[1:])):
        if left.start_time > 60_000 or not introduction_seam(left.text, right.text):
            continue
        end = i + 1
        while (
            end + 1 < len(segments) and end - i < 6 and not re.search(r"[.!?]$", segments[end].text)
        ):
            end += 1
        window = segments[i : end + 1]
        if (
            not re.search(r"[.!?]$", window[-1].text)
            or len({c.speaker_id for c in window}) > 1
            or len({c.language_code for c in window}) > 1
            or any(not 0 <= b.start_time - a.end_time <= 1800 for a, b in zip(window, window[1:]))
        ):
            continue
        return tuple(range(i + 1, end + 2))
    return ()


def plan_presentations(
    segments: Sequence[ASRDataSeg],
    *,
    hard_max_words: int,
    join_words: Callable[[Sequence[ASRWord]], str],
    make_cue: Callable[[Sequence[ASRWord], str], ASRDataSeg],
    hard_boundary: Callable[[ASRDataSeg, ASRDataSeg], bool],
) -> list[ASRDataSeg]:
    """Permit a changed cue count only for an evidenced opening presentation.

    The planner cannot consume translated text, estimated timing, speaker turns,
    language switches, or pauses. Other sentence repair remains unchanged.
    """
    result = list(segments)
    for start in range(len(result)):
        first = result[start]
        if first.start_time > 60_000 or not re.match(
            r"^(?:hey|hi|hello|welcome|good\s+(?:morning|afternoon|evening))\b", first.text, re.I
        ):
            continue
        stop = start
        while (
            stop + 1 < len(result)
            and stop - start < 6
            and not re.search(r"[.!?]$", result[stop].text)
        ):
            if hard_boundary(result[stop], result[stop + 1]):
                break
            stop += 1
        cues = result[start : stop + 1]
        words = [w for c in cues for w in c.words]
        if not words or len(words) > 70 or any(not c.words or c.translated_text for c in cues):
            continue
        speakers = {w.speaker_id or c.speaker_id for c in cues for w in c.words} | {
            c.speaker_id for c in cues if c.speaker_id
        }
        languages = {w.language_code for w in words if w.language_code} | {
            c.language_code for c in cues if c.language_code
        }
        if (
            len(speakers) > 1
            or len(languages) > 1
            or any(w.end_time <= w.start_time or w.timing_source == "estimated" for w in words)
        ):
            continue
        if any(
            a.end_time > b.start_time or b.start_time - a.end_time > 1800
            for a, b in zip(words, words[1:])
        ):
            continue
        text = join_words(words)
        # Only a one-to-one text/timing projection is eligible. Optimized text
        # with deletions or replacements is handled by mapped_boundary instead.
        if re.findall(r"\w+", text.casefold()) != re.findall(
            r"\w+", " ".join(c.text for c in cues).casefold()
        ):
            continue
        seams = [
            p
            for p in range(1, len(words))
            if introduction_seam(join_words(words[:p]), join_words(words[p:]))
        ]
        if not seams:
            continue
        intro = seams[0]
        names = [n for n in vehicle_names(text) if n.start >= len(join_words(words[:intro]))]
        if not names:
            continue
        name = names[0]
        entity_end = next(
            (p for p in range(intro + 1, len(words) + 1) if len(join_words(words[:p])) == name.end),
            None,
        )
        if entity_end is None:
            continue
        cuts = [intro]
        if entity_end < len(words) and vehicle_apposition(
            join_words(words[:entity_end]), join_words(words[entity_end:])
        ):
            cuts.append(entity_end)
            # Coordinated descriptions can be separate cues; their shared
            # subject is the complete vehicle named in the preceding block.
            for p in range(entity_end + 2, len(words)):
                if (
                    words[p - 1].text.rstrip(",").casefold() == "generation"
                    and words[p].text.casefold() == "and"
                ):
                    cuts.append(p)
                    break
        parts = []
        previous = 0
        for cut in [*cuts, len(words)]:
            part = words[previous:cut]
            parts.append(part)
            previous = cut
        if any(
            not p
            or count_words(join_words(p)) > hard_max_words
            or not 600 <= p[-1].end_time - p[0].start_time <= 8000
            for p in parts
        ):
            continue
        original = [len(c.words) for c in cues]
        if original != [len(p) for p in parts]:
            result[start : stop + 1] = [make_cue(p, first.speaker_id) for p in parts]
            return result
    return result
