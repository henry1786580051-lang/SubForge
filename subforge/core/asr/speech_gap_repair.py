"""Conservative lexical anchors for repairing speech omitted by long-form ASR.

This module makes no model calls. Decoder timestamps nominate a small region;
independent local decodes must agree on the added words before it can change.
"""

import math
import re
from typing import Any

Word = dict[str, Any]
MAX_BOUNDARY_SHIFT_SECONDS = 3.0


def word_key(word: Word) -> str:
    text = str(word.get("word", word.get("text", "")))
    return "".join(re.findall(r"\w+", text.casefold().replace("’", "'")))


def keys(words: list[Word]) -> list[str]:
    return [word_key(word) for word in words]


def timed_words(result: dict, offset: float, usable) -> list[Word]:
    words: list[Word] = []
    for segment in result.get("segments") or []:
        if not isinstance(segment, dict) or not usable(segment):
            continue
        for word in segment.get("words") or []:
            if not isinstance(word, dict) or not word_key(word):
                continue
            start, end = word.get("start"), word.get("end")
            if (
                not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not math.isfinite(start)
                or not math.isfinite(end)
                or end <= start
            ):
                continue
            words.append({**word, "start": start + offset, "end": end + offset})
    return words


def neighbor_words(coverage: list[Word], start: float, end: float) -> tuple[list[Word], list[Word]]:
    before = [w for w in coverage if float(w["end"]) <= start + 0.05][-4:]
    after = [w for w in coverage if float(w["start"]) >= end - 0.05][:4]
    return before, after


def anchored_candidate(
    words: list[Word],
    coverage: list[Word],
    start: float,
    end: float,
) -> list[Word]:
    """Extract an insertion after a known left anchor, including a displaced tail."""
    before, after = neighbor_words(coverage, start, end)
    source_keys, left_keys, right_keys = keys(words), keys(before), keys(after)
    left = None
    for length in range(len(left_keys), 1, -1):
        matches = [
            i + length
            for i in range(len(source_keys) - length + 1)
            if source_keys[i : i + length] == left_keys[-length:]
            and abs(float(words[i + length - 1]["end"]) - start) <= 0.8
        ]
        if len(matches) == 1:
            left = matches[0]
            break
    if left is None:
        return []
    right = len(words)
    for length in range(len(right_keys), 1, -1):
        matches = [
            i
            for i in range(left, len(source_keys) - length + 1)
            if source_keys[i : i + length] == right_keys[:length]
        ]
        if len(matches) == 1:
            right = matches[0]
            break
    candidate = words[left:right]
    if (
        len(candidate) < 3
        or len(candidate) > 90
        or float(candidate[0]["start"]) < start - 0.2
        or float(candidate[0]["start"]) > start + 0.8
        or float(candidate[-1]["end"]) > end + MAX_BOUNDARY_SHIFT_SECONDS
    ):
        return []
    return candidate


def corroborates(candidate: list[Word], confirmation: list[Word]) -> bool:
    """Require exact anchored text and matching positions in the same audio."""
    if not candidate or keys(candidate) != keys(confirmation):
        return False
    drift = [
        abs((float(a["start"]) + float(a["end"]) - float(b["start"]) - float(b["end"])) / 2)
        for a, b in zip(candidate, confirmation)
    ]
    return max(drift) <= 0.8 and sum(drift) / len(drift) <= 0.35


def confirmation_window(
    result: dict, start: float, end: float, duration: float
) -> tuple[float, float]:
    """Keep up to two surrounding utterances, within Whisper's 30s window."""
    segments = result.get("segments") or []
    before = [s for s in segments if float(s["end"]) <= start + 0.05][-2:]
    after = [s for s in segments if float(s["start"]) >= end - 0.05][:2]
    left = float(before[0]["start"]) - 0.2 if before else start - 8.0
    right = float(after[-1]["end"]) + 0.2 if after else end + 8.0
    padding = max(0.0, (30.0 - (end - start)) / 2)
    return max(0.0, left, start - padding), min(duration, right, end + padding)


def insert_anchored_gap(result: dict, words: list[Word], start: float, end: float) -> dict | None:
    """Preserve source text and constrain only an overlapping right alignment edge."""
    segments = result.get("segments") or []
    following = next((i for i, s in enumerate(segments) if float(s["start"]) >= end - 0.05), None)
    if following is None:
        return None
    # A within-segment hole needs a separate split operation; never overlap a
    # new utterance with an unchanged source segment envelope.
    if any(float(s["start"]) < end - 0.05 and float(s["end"]) > start + 0.05 for s in segments):
        return None
    new_start, new_end = float(words[0]["start"]), float(words[-1]["end"])
    right = segments[following]
    adjusted_start = max(float(right["start"]), new_end + 0.01)
    if (
        new_start < start - 0.05
        or adjusted_start - float(right["start"]) > MAX_BOUNDARY_SHIFT_SECONDS
        or adjusted_start >= float(right["end"]) - 0.1
    ):
        return None
    updated_segments = list(segments)
    if adjusted_start > float(right["start"]):
        updated_segments[following] = {**right, "start": adjusted_start}
    inserted = {
        "text": "".join(str(w["word"]) for w in words).strip(),
        "start": new_start,
        "end": new_end,
        "words": words,
        "recovered_short_speech_gap": True,
    }
    updated_segments.append(inserted)
    updated = dict(result)
    updated["segments"] = sorted(
        updated_segments, key=lambda s: (float(s["start"]), float(s["end"]))
    )
    updated["text"] = " ".join(str(s["text"]).strip() for s in updated["segments"])
    return updated


def coverage_issue_message(issues: list[dict]) -> str:
    def timestamp(seconds: float) -> str:
        value = max(0, round(seconds * 1000))
        return f"{value // 3600000:02}:{value // 60000 % 60:02}:{value // 1000 % 60:02}.{value % 1000:03}"

    ranges = ", ".join(f"{timestamp(i['start'])} - {timestamp(i['end'])}" for i in issues[:10])
    return f"Speech coverage needs review ({len(issues)} region(s)): {ranges}. Partial subtitles have been preserved."
