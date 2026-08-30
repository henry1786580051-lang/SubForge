"""Shared bilingual cue parsing for subtitle import and editor round trips."""

import re
import unicodedata

from langdetect import LangDetectException, detect


def _line_family(text: str) -> str:
    """Classify a subtitle text line without being fooled by model names.

    A Chinese translation often contains Latin tokens such as W126,
    AMG, Mercedes-Benz, or email addresses. Presence of meaningful CJK
    text therefore wins over embedded Latin identifiers. Han, Japanese,
    and Korean remain distinct so bilingual CJK cues can be separated.
    """
    stripped = text.strip()
    if not stripped:
        return "empty"

    normalized = unicodedata.normalize("NFC", stripped)
    counts = {
        "han": len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", normalized)),
        "kana": len(re.findall(r"[\u3040-\u30ff\u31f0-\u31ff]", normalized)),
        "hangul": len(
            re.findall(
                r"[\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff]",
                normalized,
            )
        ),
        "latin": len(re.findall(r"[A-Za-z]", normalized)),
    }
    non_latin = counts["han"] + counts["kana"] + counts["hangul"]
    if counts["hangul"] >= 2 and counts["hangul"] >= counts["han"]:
        return "hangul"
    if counts["kana"]:
        return "japanese"
    if counts["han"] >= 2:
        return "han"
    if counts["latin"] >= 2 and non_latin == 0:
        return "latin"
    if counts["hangul"]:
        return "hangul"
    if counts["han"]:
        return "han"
    return "other"


def _group_family(lines: list[str]) -> str:
    joined = " ".join(line.strip() for line in lines if line.strip())
    family = _line_family(joined)
    if family in {"han", "japanese", "hangul", "latin"}:
        return family
    families = [_line_family(line) for line in lines if line.strip()]
    counts = {
        family_name: families.count(family_name)
        for family_name in ("han", "japanese", "hangul", "latin")
    }
    winner, winner_count = max(counts.items(), key=lambda item: item[1])
    if winner_count > 0 and list(counts.values()).count(winner_count) == 1:
        return winner
    return "other"


def _fallback_different_language(left: str, right: str) -> bool:
    try:
        return detect(left) != detect(right)
    except LangDetectException:
        return False


def split_bilingual_lines(text_lines: list[str]) -> tuple[str, str] | None:
    non_empty = [line for line in text_lines if line.strip()]
    if len(non_empty) < 2:
        return None

    dialogue_marked = all(re.match(r"^-\s+", line.strip()) for line in non_empty)
    if dialogue_marked:
        non_empty = [re.sub(r"^-\s+", "", line.strip()) for line in non_empty]

    # Generated translated-on-top subtitles may contain language-neutral
    # amounts or identical numeric lines. Keep these round-trippable even
    # when speaker display markers are intentionally hidden.
    if len(non_empty) == 2:
        left_family = _line_family(non_empty[0])
        right_family = _line_family(non_empty[1])
        short_latin_line = re.compile(r"^[A-Za-z][.!?]?$", re.IGNORECASE)
        left_title = re.fullmatch(r"[《“\"'](.+?)[》”\"']?[。.]?", non_empty[0].strip())
        right_title = re.fullmatch(r"[《“\"'](.+?)[》”\"']?[。.]?", non_empty[1].strip())
        if left_family == right_family == "latin":
            canonical_locator = re.compile(
                r"^(?:https?://)?[A-Za-z0-9._%+-]+"
                r"(?:@[A-Za-z0-9.-]+|(?:\.[A-Za-z0-9-]+)+)"
                r"(?:/[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]*)?[.!]?$",
                re.IGNORECASE,
            )
            spoken_locator = re.compile(
                r"\b(?:dot|slash|at)\b",
                re.IGNORECASE,
            )
            left_is_locator = bool(canonical_locator.fullmatch(non_empty[0].strip()))
            right_is_locator = bool(canonical_locator.fullmatch(non_empty[1].strip()))
            left_is_spoken = bool(spoken_locator.search(non_empty[0]))
            right_is_spoken = bool(spoken_locator.search(non_empty[1]))
            # A generated bilingual URL may remain Latin in both languages:
            # canonical target above, spoken ASR source below (or vice versa).
            # Preserve that pair instead of reopening it as one source-only cue.
            if left_is_locator and right_is_spoken:
                return non_empty[1], non_empty[0]
            if right_is_locator and left_is_spoken:
                return non_empty[0], non_empty[1]
            if (
                left_is_locator
                and right_is_locator
                and non_empty[0].rstrip(".!").casefold() == non_empty[1].rstrip(".!").casefold()
            ):
                return non_empty[1], non_empty[0]
            if (
                left_title
                and re.sub(r"\W+", "", left_title.group(1)).lower()
                == re.sub(r"\W+", "", non_empty[1]).lower()
            ):
                return non_empty[1], non_empty[0]
            if (
                right_title
                and re.sub(r"\W+", "", right_title.group(1)).lower()
                == re.sub(r"\W+", "", non_empty[0]).lower()
            ):
                return non_empty[0], non_empty[1]
        if (
            left_family == "han"
            and right_family == "other"
            and (re.search(r"\d", non_empty[1]) or short_latin_line.fullmatch(non_empty[1].strip()))
        ):
            return non_empty[1], non_empty[0]
        if (
            right_family == "han"
            and left_family == "other"
            and (re.search(r"\d", non_empty[0]) or short_latin_line.fullmatch(non_empty[0].strip()))
        ):
            return non_empty[0], non_empty[1]
        normalized_left = re.sub(r"\s+", "", non_empty[0])
        normalized_right = re.sub(r"\s+", "", non_empty[1])
        neutral_left = re.sub(r"[^\w]+", "", normalized_left)
        neutral_right = re.sub(r"[^\w]+", "", normalized_right)
        if (
            left_family == right_family == "other"
            and re.search(r"\d", normalized_left + normalized_right)
            and (
                dialogue_marked
                or normalized_left == normalized_right
                or neutral_left == neutral_right
            )
        ):
            return non_empty[1], non_empty[0]

    best: tuple[int, int, str, str] | None = None
    for split_index in range(1, len(non_empty)):
        left = non_empty[:split_index]
        right = non_empty[split_index:]
        left_family = _group_family(left)
        right_family = _group_family(right)

        score = 0
        language_pair = {left_family, right_family}
        if language_pair in (
            {"han", "latin"},
            {"han", "hangul"},
            {"han", "japanese"},
        ):
            score = 100
        elif left_family != right_family and left_family != "other" and right_family != "other":
            score = 60
        elif (
            left_family != right_family
            and len(left) == 1
            and len(right) == 1
            and _fallback_different_language(left[0], right[0])
        ):
            score = 40

        if score <= 0:
            continue
        # Prefer balanced split points when multiple options look valid.
        balance_penalty = abs(len(left) - len(right))
        candidate = (score - balance_penalty, split_index, left_family, right_family)
        if best is None or candidate > best:
            best = candidate

    if best is None:
        return None

    _, split_index, left_family, right_family = best
    left_text = "\n".join(non_empty[:split_index]).strip()
    right_text = "\n".join(non_empty[split_index:]).strip()

    language_pair = {left_family, right_family}
    if language_pair in (
        {"han", "latin"},
        {"han", "hangul"},
        {"han", "japanese"},
    ):
        # SubForge's bilingual SRT workflow translates source speech to
        # Chinese. Resolve either display order back to source/target.
        if left_family != "han":
            return left_text, right_text
        return right_text, left_text

    return left_text, right_text


_SPEAKER_PATTERN = re.compile(r"^\[(说话人\d+|Speaker \d+)\]\s*")


def parse_subtitle_text(text: str) -> tuple[str, str, str]:
    """Return source, translation and optional speaker using existing SRT rules."""
    lines = text.splitlines()
    speaker = ""
    if lines:
        match = _SPEAKER_PATTERN.match(lines[0])
        if match:
            speaker = match.group(1)
            lines[0] = lines[0][match.end() :]
    bilingual = split_bilingual_lines(lines)
    if bilingual:
        return bilingual[0], bilingual[1], speaker
    return "\n".join(lines), "", speaker
