"""Conservative subtitle boundary scoring and word-timestamp repair."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence, cast

from subforge.core.asr.asr_data import ASRDataSeg, ASRWord, TimestampSource
from subforge.core.utils.logger import setup_logger

logger = setup_logger("subtitle_boundary")

SOFT_MAX_WORDS = 18
HARD_MAX_WORDS = 22
MAX_BOUNDARY_SHIFT_WORDS = 8
MAX_RELOCATABLE_GAP_MS = 1800
MAX_DIARIZATION_GLITCH_GAP_MS = 250
MAX_DUPLICATE_CORRECTION_GAP_MS = 600
MIN_REPAIR_IMPROVEMENT = 8.0

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_TERMINAL_RE = re.compile(r"[.!?][\"')\]]*$")
_CLAUSE_RE = re.compile(r"[,;:][\"')\]]*$")

_HARD_DANGLING_TAILS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "because",
    "although",
    "though",
    "unless",
    "while",
    "whereas",
    "if",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "over",
    "through",
    "that",
    "to",
    "under",
    "with",
}

_SUBJECT_TAILS = {
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "this",
    "these",
    "those",
    "who",
    "which",
}

_INCOMPLETE_PREDICATE_TAILS = {
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "will",
    "would",
    "not",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "can't",
    "couldn't",
    "won't",
    "wouldn't",
}

_SUBJECT_AUX_TAILS = {
    "i'm",
    "you're",
    "he's",
    "she's",
    "it's",
    "we're",
    "they're",
    "i've",
    "you've",
    "we've",
    "they've",
    "i'll",
    "you'll",
    "he'll",
    "she'll",
    "it'll",
    "we'll",
    "they'll",
}

_MODIFIER_TAILS = {
    "another",
    "any",
    "big",
    "each",
    "every",
    "few",
    "good",
    "great",
    "high",
    "large",
    "less",
    "little",
    "low",
    "many",
    "more",
    "most",
    "much",
    "new",
    "other",
    "our",
    "public",
    "red",
    "drastically",
    "her",
    "his",
    "its",
    "massive",
    "my",
    "really",
    "serrated",
    "several",
    "small",
    "some",
    "specialized",
    "traditional",
    "their",
    "very",
    "your",
}

_OPEN_COMPLEMENT_TAILS = {"give", "gives", "gave", "get", "gets", "got", "just"}
_PHRASAL_PARTICLES = {"away", "back", "down", "in", "off", "on", "out", "over", "up"}

_DANGLING_PHRASES = {
    ("a", "lot", "of"),
    ("as", "much", "as"),
    ("because", "of"),
    ("going", "to"),
    ("has", "been"),
    ("have", "been"),
    ("high", "end"),
    ("higher", "end"),
    ("in", "order"),
    ("kind", "of"),
    ("need", "to"),
    ("not", "only"),
    ("one", "of"),
    ("start", "generating"),
    ("tends", "to"),
    ("used", "to"),
    ("want", "to"),
}

_DEPENDENCY_PAIRS = {
    ("ago", "or"),
    ("better", "sound"),
    ("big", "picture"),
    ("condition", "or"),
    ("fall", "out"),
    ("flip", "switch"),
    ("generating", "power"),
    ("good", "jobs"),
    ("high", "end"),
    ("higher", "echelon"),
    ("nuclear", "plants"),
    ("nuclear", "power"),
    ("power", "plants"),
    ("pretty", "standard"),
    ("public", "college"),
    ("same", "sort"),
    ("serrated", "edge"),
    ("specialized", "employees"),
    ("traditional", "hybrid"),
}

_COPULA_COMPLEMENT_TAILS = {
    "car",
    "here",
    "interior",
    "point",
    "problem",
    "reason",
    "thing",
}

_DEPENDENT_RIGHT_HEADS = {
    "a",
    "an",
    "the",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "over",
    "through",
    "to",
    "under",
    "with",
}

_PREFERRED_CLAUSE_HEADS = {
    "although",
    "and",
    "because",
    "but",
    "however",
    "if",
    "now",
    "or",
    "so",
    "that",
    "then",
    "unless",
    "when",
    "while",
}

_RELATIVE_CLAUSE_HEADS = {"that", "which", "who", "whom", "whose"}
_THAT_COMPLEMENT_TAILS = {
    ("find", "out"),
    ("found", "out"),
    ("know",),
    ("knew",),
    ("say",),
    ("said",),
    ("think",),
    ("thought",),
}


def _tokens(text: str) -> list[str]:
    return [token.replace("’", "'").lower() for token in _TOKEN_RE.findall(text)]


def _ends_with_phrase(tokens: Sequence[str]) -> bool:
    return any(len(tokens) >= len(phrase) and tuple(tokens[-len(phrase) :]) == phrase for phrase in _DANGLING_PHRASES)


@dataclass(frozen=True)
class BoundaryAssessment:
    risk: int
    reasons: tuple[str, ...]

    @property
    def unstable(self) -> bool:
        return self.risk >= 20


def assess_english_boundary(left: str, right: str) -> BoundaryAssessment:
    """Assess whether an English cue boundary splits a dependent phrase."""
    left = str(left or "").strip()
    right = str(right or "").strip()
    if not left or not right or _TERMINAL_RE.search(left):
        return BoundaryAssessment(0, ())

    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return BoundaryAssessment(0, ())

    tail = left_tokens[-1]
    head = right_tokens[0]
    risk = 0
    reasons: list[str] = []

    if _ends_with_phrase(left_tokens):
        risk += 36
        reasons.append("incomplete multi-word phrase")
    if tail in _HARD_DANGLING_TAILS:
        risk += 32
        reasons.append(f"dangling function word '{tail}'")
    if (
        tail in _SUBJECT_TAILS
        and right[:1].islower()
        and not _CLAUSE_RE.search(left)
    ):
        risk += 26
        reasons.append(f"dangling subject '{tail}'")
    if tail in _INCOMPLETE_PREDICATE_TAILS:
        risk += 26
        reasons.append(f"incomplete predicate '{tail}'")
    if tail in _SUBJECT_AUX_TAILS and right[:1].islower():
        risk += 30
        reasons.append(f"subject and auxiliary stranded at '{tail}'")
    if tail in _MODIFIER_TAILS:
        risk += 24
        reasons.append(f"dangling modifier '{tail}'")
    if tail in _OPEN_COMPLEMENT_TAILS and right[:1].islower():
        risk += 26
        reasons.append(f"open complement after '{tail}'")
    if head in _PHRASAL_PARTICLES and tail in {"fall", "get", "gets", "go", "look", "take"}:
        risk += 32
        reasons.append(f"split phrasal verb '{tail} {head}'")
    if head in {"is", "are", "was", "were"} and tail in _COPULA_COMPLEMENT_TAILS:
        risk += 28
        reasons.append(f"subject complement split before '{head}'")
    if tail == "one" and right[:1].islower():
        risk += 24
        reasons.append("omitted relative clause separated from 'one'")
    if re.search(r"\b(?:this|that|it)\s+(?:is|was),?\s+i\s+(?:think|guess),?$", left, re.I):
        risk += 30
        reasons.append("copula separated from its complement by a parenthetical")
    if re.search(r"(?:^|[.!?]\s+)(?:and|but|now|so),?$", left, re.I):
        risk += 34
        reasons.append("new-clause connective stranded at previous cue end")
    if tail == "now" and head in (_SUBJECT_TAILS | {"our", "the", "a", "an"}):
        risk += 30
        reasons.append("sentence-opening time adverb belongs to the next cue")
    if re.search(r"\bbut\s+for\s+(?:them|him|her|us|you),?$", left, re.IGNORECASE):
        risk += 34
        reasons.append("contrastive beneficiary phrase belongs to the next cue")
    if (tail, head) in _DEPENDENCY_PAIRS:
        risk += 34
        reasons.append(f"split lexical unit '{tail} {head}'")
    if (
        head in _DEPENDENT_RIGHT_HEADS
        and right[:1].islower()
        and not _CLAUSE_RE.search(left)
    ):
        risk += 30
        reasons.append(f"dependent phrase beginning with '{head}'")
    that_starts_complement = head == "that" and any(
        len(left_tokens) >= len(phrase)
        and tuple(left_tokens[-len(phrase) :]) == phrase
        for phrase in _THAT_COMPLEMENT_TAILS
    )
    if (
        head in _RELATIVE_CLAUSE_HEADS
        and not that_starts_complement
        and not _CLAUSE_RE.search(left)
    ):
        risk += 30
        reasons.append(f"relative clause '{head}' separated from its antecedent")

    # A lowercase continuation makes a dangling tail more likely, but is not
    # sufficient by itself: natural subtitle clauses often continue lowercase.
    if risk and right[:1].islower():
        risk += 4

    return BoundaryAssessment(risk, tuple(dict.fromkeys(reasons)))


def has_unstable_english_boundary(left: str, right: str) -> bool:
    return assess_english_boundary(left, right).unstable


def _join_words(words: Iterable[ASRWord]) -> str:
    result = ""
    no_space_before = set(",.;:!?)]}，。！？；：、）】》")
    no_space_after = set("([{（【《")
    for word in words:
        text = word.text.strip()
        if not text:
            continue
        if not result:
            result = text
        elif text[0] in no_space_before or result[-1] in no_space_after:
            result += text
        else:
            result += f" {text}"
    return re.sub(r"\s+([,.;:!?，。！？；：、])", r"\1", result).strip()


def _word_count(words: Sequence[ASRWord]) -> int:
    return sum(max(1, len(_tokens(word.text))) for word in words if word.text.strip())


def _dominant_speaker(words: Sequence[ASRWord], fallback: str = "") -> str:
    durations: dict[str, int] = {}
    for word in words:
        if word.speaker_id:
            durations[word.speaker_id] = durations.get(word.speaker_id, 0) + max(
                1, word.end_time - word.start_time
            )
    return max(durations, key=durations.__getitem__) if durations else fallback


def _make_cue(words: Sequence[ASRWord], fallback_speaker: str) -> ASRDataSeg:
    sources = {word.timing_source for word in words if word.timing_source != "unknown"}
    timing_source = cast(
        TimestampSource,
        next(iter(sources)) if len(sources) == 1 else ("mixed" if sources else "unknown"),
    )
    return ASRDataSeg(
        text=_join_words(words),
        start_time=words[0].start_time,
        end_time=words[-1].end_time,
        speaker_id=_dominant_speaker(words, fallback_speaker),
        words=list(words),
        timestamp_granularity="sentence",
        timing_source=timing_source,
    )


def _is_singular_correction(left: ASRWord, right: ASRWord) -> bool:
    left_text = left.text.strip()
    left_tokens = _tokens(left_text)
    right_tokens = _tokens(right.text)
    return bool(
        left_text.endswith(",")
        and len(left_tokens) == 1
        and len(right_tokens) == 1
        and len(left_tokens[0]) >= 3
        and right_tokens[0] == f"{left_tokens[0]}s"
        and right.start_time - left.end_time <= MAX_DUPLICATE_CORRECTION_GAP_MS
        and (
            not left.speaker_id
            or not right.speaker_id
            or left.speaker_id == right.speaker_id
        )
    )


def _remove_singular_corrections(
    segments: Sequence[ASRDataSeg],
) -> list[ASRDataSeg]:
    """Drop a spoken singular false start immediately corrected to its plural."""
    result = list(segments)
    for index, segment in enumerate(result):
        words = list(segment.words)
        keep = [True] * len(words)
        for word_index in range(len(words) - 1):
            if _is_singular_correction(words[word_index], words[word_index + 1]):
                keep[word_index] = False
                logger.info(
                    "Removed in-subtitle singular correction at subtitle %s: %s -> %s",
                    index + 1,
                    _tokens(words[word_index].text)[0],
                    _tokens(words[word_index + 1].text)[0],
                )
        kept_words = [word for word, should_keep in zip(words, keep) if should_keep]
        if len(kept_words) != len(words) and kept_words:
            result[index] = _make_cue(kept_words, segment.speaker_id)

    for index in range(len(result) - 1):
        left = result[index]
        right = result[index + 1]
        if not left.words or not right.words:
            continue
        if left.speaker_id and right.speaker_id and left.speaker_id != right.speaker_id:
            continue
        if right.start_time - left.end_time > MAX_DUPLICATE_CORRECTION_GAP_MS:
            continue
        if not _is_singular_correction(left.words[-1], right.words[0]) or len(left.words) == 1:
            continue
        left_tokens = _tokens(left.words[-1].text)
        right_tokens = _tokens(right.words[0].text)
        result[index] = _make_cue(left.words[:-1], left.speaker_id)
        logger.info(
            "Removed cross-boundary singular correction at subtitles %s-%s: %s -> %s",
            index + 1,
            index + 2,
            left_tokens[0],
            right_tokens[0],
        )
    return result


def _is_hard_boundary(left: ASRDataSeg, right: ASRDataSeg) -> bool:
    if left.speaker_id and right.speaker_id and left.speaker_id != right.speaker_id:
        gap = max(0, right.start_time - left.end_time)
        assessment = assess_english_boundary(left.text, right.text)
        # Diarization can briefly flip speaker IDs inside one sentence. Only a
        # very short gap plus a strong syntactic dependency may override it.
        if gap > MAX_DIARIZATION_GLITCH_GAP_MS or assessment.risk < 30:
            return True
    return right.start_time - left.end_time > MAX_RELOCATABLE_GAP_MS


def _merge_compact_unstable_pairs(
    segments: Sequence[ASRDataSeg],
    *,
    hard_max_words: int,
) -> list[ASRDataSeg]:
    """Merge only compact pairs that still have no stable internal boundary."""
    result = list(segments)
    index = 0
    while index < len(result) - 1:
        left = result[index]
        right = result[index + 1]
        assessment = assess_english_boundary(left.text, right.text)
        words = [*left.words, *right.words]
        combined_text = _join_words(words)
        if (
            assessment.unstable
            and not _is_hard_boundary(left, right)
            and left.words
            and right.words
            and _word_count(words) <= hard_max_words
            and words[-1].end_time - words[0].start_time <= 8000
            and not re.search(r"[.!?][\"')\]]*\s+\S", combined_text)
        ):
            result[index : index + 2] = [
                _make_cue(words, left.speaker_id or right.speaker_id)
            ]
            logger.info(
                "Merged compact unstable subtitles %s-%s: %s",
                index + 1,
                index + 2,
                assessment.reasons,
            )
            if index:
                index -= 1
            continue
        index += 1
    return result


def _segment_cost(words: Sequence[ASRWord], soft_max: int, hard_max: int) -> float:
    count = _word_count(words)
    if count <= 0 or count > hard_max:
        return float("inf")
    cost = 0.0
    if count < 3:
        cost += (3 - count) * 7.0
    if count > soft_max:
        cost += (count - soft_max) * 5.0
    duration = words[-1].end_time - words[0].start_time
    if duration > 8000:
        cost += (duration - 8000) / 500.0
    return cost


def _boundary_cost(
    words: Sequence[ASRWord], position: int, original_position: int
) -> tuple[float, int]:
    left = _join_words(words[:position])
    right = _join_words(words[position:])
    assessment = assess_english_boundary(left, right)
    tail = words[position - 1].text.strip()
    right_tokens = _tokens(words[position].text)
    head = right_tokens[0] if right_tokens else ""
    gap = max(0, words[position].start_time - words[position - 1].end_time)

    cost = float(assessment.risk)
    if _TERMINAL_RE.search(tail):
        cost -= 24.0
    elif _CLAUSE_RE.search(tail):
        cost -= 5.0
    if head in _PREFERRED_CLAUSE_HEADS:
        cost -= 10.0
    cost -= min(5.0, gap / 300.0)
    cost += abs(position - original_position) * 0.8
    if position == original_position:
        cost -= 1.5
    return cost, assessment.risk


def _best_region_breaks(
    words: Sequence[ASRWord],
    original_positions: Sequence[int],
    *,
    soft_max: int,
    hard_max: int,
) -> tuple[list[int], float, int] | None:
    """Return same-count local breaks using dynamic programming."""
    candidates: list[list[int]] = []
    total = len(words)
    for original in original_positions:
        start = max(1, original - MAX_BOUNDARY_SHIFT_WORDS)
        end = min(total - 1, original + MAX_BOUNDARY_SHIFT_WORDS)
        candidates.append(list(range(start, end + 1)))

    states: dict[int, tuple[float, list[int], int]] = {0: (0.0, [], 0)}
    for boundary_index, positions in enumerate(candidates):
        next_states: dict[int, tuple[float, list[int], int]] = {}
        original = original_positions[boundary_index]
        for position in positions:
            for previous, (cost, path, risk) in states.items():
                if position <= previous:
                    continue
                segment_cost = _segment_cost(words[previous:position], soft_max, hard_max)
                if segment_cost == float("inf"):
                    continue
                boundary_cost, boundary_risk = _boundary_cost(words, position, original)
                candidate = (cost + segment_cost + boundary_cost, path + [position], risk + boundary_risk)
                existing = next_states.get(position)
                if existing is None or candidate[0] < existing[0]:
                    next_states[position] = candidate
        states = next_states
        if not states:
            return None

    best: tuple[float, list[int], int] | None = None
    for previous, (cost, path, risk) in states.items():
        final_cost = _segment_cost(words[previous:], soft_max, hard_max)
        if final_cost == float("inf"):
            continue
        candidate = (cost + final_cost, path, risk)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        return None
    return best[1], best[0], best[2]


def _region_metrics(
    words: Sequence[ASRWord], positions: Sequence[int], soft_max: int, hard_max: int
) -> tuple[float, int]:
    cost = 0.0
    risk = 0
    previous = 0
    for index, position in enumerate(positions):
        cost += _segment_cost(words[previous:position], soft_max, hard_max)
        boundary_cost, boundary_risk = _boundary_cost(words, position, positions[index])
        cost += boundary_cost
        risk += boundary_risk
        previous = position
    cost += _segment_cost(words[previous:], soft_max, hard_max)
    return cost, risk


def normalize_boundaries(
    segments: Sequence[ASRDataSeg],
    *,
    soft_max_words: int = SOFT_MAX_WORDS,
    hard_max_words: int = HARD_MAX_WORDS,
) -> list[ASRDataSeg]:
    """Move only high-risk boundaries while retaining every atomic word timing."""
    result = _remove_singular_corrections(segments)
    result = _merge_compact_unstable_pairs(
        result,
        hard_max_words=hard_max_words,
    )
    if len(result) < 2 or any(segment.translated_text for segment in result):
        return result

    for _ in range(3):
        risky = [
            index
            for index in range(len(result) - 1)
            if not _is_hard_boundary(result[index], result[index + 1])
            and has_unstable_english_boundary(result[index].text, result[index + 1].text)
        ]
        if not risky:
            break

        regions: list[tuple[int, int]] = []
        start = previous = risky[0]
        for index in risky[1:]:
            if index == previous + 1:
                previous = index
                continue
            regions.append((start, previous + 1))
            start = previous = index
        regions.append((start, previous + 1))

        changed = False
        for cue_start, cue_end in reversed(regions):
            cues = result[cue_start : cue_end + 1]
            if any(not cue.words for cue in cues) or any(
                _is_hard_boundary(cues[index], cues[index + 1])
                for index in range(len(cues) - 1)
            ):
                continue

            words = [word for cue in cues for word in cue.words]
            original_positions: list[int] = []
            cursor = 0
            for cue in cues[:-1]:
                cursor += len(cue.words)
                original_positions.append(cursor)

            baseline_cost, baseline_risk = _region_metrics(
                words, original_positions, soft_max_words, hard_max_words
            )
            best = _best_region_breaks(
                words,
                original_positions,
                soft_max=soft_max_words,
                hard_max=hard_max_words,
            )
            if best is None:
                continue
            positions, candidate_cost, candidate_risk = best
            if (
                positions == original_positions
                or candidate_risk >= baseline_risk
                or baseline_cost - candidate_cost < MIN_REPAIR_IMPROVEMENT
            ):
                continue

            rebuilt: list[ASRDataSeg] = []
            previous_position = 0
            fallback_speaker = cues[0].speaker_id
            for position in [*positions, len(words)]:
                rebuilt.append(
                    _make_cue(words[previous_position:position], fallback_speaker)
                )
                previous_position = position
            result[cue_start : cue_end + 1] = rebuilt
            changed = True
            logger.info(
                "Repaired subtitle boundaries %s-%s: %s -> %s (risk %s -> %s)",
                cue_start + 1,
                cue_end + 1,
                original_positions,
                positions,
                baseline_risk,
                candidate_risk,
            )

        if not changed:
            break

    return _merge_compact_unstable_pairs(
        result,
        hard_max_words=hard_max_words,
    )
