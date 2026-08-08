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
    "how",
    "if",
    "as",
    "at",
    "about",
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
    "whether",
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
    "closest",
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
    "puny",
    "red",
    "drastically",
    "her",
    "his",
    "its",
    "massive",
    "my",
    "near",
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

_ATTRIBUTIVE_TAILS = {"first", "old", "same", "second", "similar"}

_OPEN_COMPLEMENT_TAILS = {
    "give",
    "gives",
    "gave",
    "get",
    "gets",
    "got",
    "just",
    "spend",
    "spending",
    "spent",
}
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
    ("american", "sedans"),
    ("better", "sound"),
    ("big", "picture"),
    ("body", "american"),
    ("condition", "or"),
    ("damn", "near"),
    ("european", "influence"),
    ("exhaust", "tips"),
    ("fall", "out"),
    ("flip", "switch"),
    ("generating", "power"),
    ("good", "jobs"),
    ("high", "end"),
    ("higher", "echelon"),
    ("nuclear", "plants"),
    ("nuclear", "power"),
    ("power", "plants"),
    ("performance", "pack"),
    ("pretty", "standard"),
    ("public", "college"),
    ("rpm", "gauge"),
    ("same", "sort"),
    ("serrated", "edge"),
    ("so", "much"),
    ("specialized", "employees"),
    ("traditional", "hybrid"),
    ("turn", "signals"),
    ("which", "i"),
    ("that's", "what"),
}

_COPULA_COMPLEMENT_TAILS = {
    "car",
    "here",
    "interior",
    "point",
    "problem",
    "reason",
    "story",
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
    "than",
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
_TRANSLATION_SENSITIVE_HEADS = {"after", "before", "when", "where"}
_US_STATE_NAMES = {
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "wisconsin",
    "wyoming",
}
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
    return any(
        len(tokens) >= len(phrase) and tuple(tokens[-len(phrase) :]) == phrase
        for phrase in _DANGLING_PHRASES
    )


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

    complete_does_clause = bool(
        tail == "does"
        and re.search(
            r"\b(?:see|show|check|find\s+out)\s+how\s+"
            r"(?:he|she|it|this|that)\s+does$",
            left,
            flags=re.IGNORECASE,
        )
    )
    complete_degree_adverb = bool(
        tail == "much"
        and head in {"and", "but", "like"}
        and re.search(
            r"\b(?:elevate|help|improve|like|love|matter)\b[^.!?]*\bso\s+much$",
            left,
            flags=re.IGNORECASE,
        )
    )
    complete_fixed_phrase = bool(
        head in {"a", "an", "the"}
        and re.search(
            r"\b(?:don['’]t|do\s+not)\s+get\s+me\s+wrong$",
            left,
            flags=re.IGNORECASE,
        )
    )

    if _ends_with_phrase(left_tokens):
        risk += 36
        reasons.append("incomplete multi-word phrase")
    if tail in _HARD_DANGLING_TAILS:
        risk += 32
        reasons.append(f"dangling function word '{tail}'")
    if tail in _SUBJECT_TAILS and right[:1].islower() and not _CLAUSE_RE.search(left):
        risk += 26
        reasons.append(f"dangling subject '{tail}'")
    if tail in _INCOMPLETE_PREDICATE_TAILS and not complete_does_clause:
        risk += 26
        reasons.append(f"incomplete predicate '{tail}'")
    if tail in _SUBJECT_AUX_TAILS and right[:1].islower():
        risk += 30
        reasons.append(f"subject and auxiliary stranded at '{tail}'")
    if tail in _MODIFIER_TAILS and not complete_degree_adverb:
        risk += 24
        reasons.append(f"dangling modifier '{tail}'")
    if tail in _ATTRIBUTIVE_TAILS and right[:1].islower():
        risk += 24
        reasons.append(f"dangling attributive '{tail}'")
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
    if re.search(r"[.!?,;]\s+i\s+mean,?$", left, re.IGNORECASE):
        risk += 36
        reasons.append("sentence-opening filler belongs to the next cue")
    if re.search(
        r"[.!?]\s+(?:i|we|you)\s+(?:think|guess|believe),?$",
        left,
        re.IGNORECASE,
    ):
        risk += 36
        reasons.append("sentence-opening opinion marker belongs to the next cue")
    if (
        head == "wrong"
        and re.search(r"\b(?:don['’]t|do\s+not)\s+get\s+me$", left, re.IGNORECASE)
    ):
        risk += 38
        reasons.append("fixed phrase split inside 'do not get me wrong'")
    if re.search(r"\bbecause\s+at\s+the\s+time,?$", left, re.IGNORECASE):
        risk += 36
        reasons.append("reason clause opener separated from its subject")
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
        tail in {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
        and re.match(r"^and\s+a\s+half\s+(?:foot|feet|inch|inches)\b", right, re.IGNORECASE)
    ):
        risk += 38
        reasons.append("mixed-number measurement split before 'and a half'")
    if tail == "intake" and tuple(right_tokens[:2]) == ("and", "exhaust"):
        risk += 34
        reasons.append("coordinate automotive term split between intake and exhaust")
    if head == "or" and re.fullmatch(r"[a-z]+\d+", tail):
        risk += 34
        reasons.append("alphanumeric model alternative split before 'or'")
    if tail == "same" and head == "as":
        risk += 36
        reasons.append("comparison split between 'same' and 'as'")
    if re.search(r"\bthis\s+is\s+what['’]s$", left, re.IGNORECASE) and head == "so":
        risk += 36
        reasons.append("'what is so' complement split")
    if re.search(r"\bwhat['’]s\s+so$", left, re.IGNORECASE) and head in {
        "good",
        "great",
        "special",
    }:
        risk += 36
        reasons.append("'what is so' complement split")
    if tail == "like" and head in {"this", "that", "it", "these", "those"}:
        risk += 32
        reasons.append(f"comparison complement split before '{head}'")
    if (
        re.search(
            r"\bwhat\s+(?:i|you|we|they|he|she|it)(?:['’][a-z]+)?"
            r"(?:\s+[a-z]+){0,6}\s+use$",
            left,
            flags=re.IGNORECASE,
        )
        and re.match(
            r"^(?:a|an|the|this|that|these|those|it|them|him|her|us)\b"
            r"(?:\s+\S+){0,4}\s+for\b",
            right,
            flags=re.IGNORECASE,
        )
    ):
        risk += 36
        reasons.append("object split inside 'what ... use ... for' construction")
    if (
        head in _DEPENDENT_RIGHT_HEADS
        and right[:1].islower()
        and not _CLAUSE_RE.search(left)
        and not complete_fixed_phrase
    ):
        risk += 30
        reasons.append(f"dependent phrase beginning with '{head}'")
    that_starts_complement = head == "that" and any(
        len(left_tokens) >= len(phrase) and tuple(left_tokens[-len(phrase) :]) == phrase
        for phrase in _THAT_COMPLEMENT_TAILS
    )
    if (
        head in _RELATIVE_CLAUSE_HEADS
        and not that_starts_complement
        and not _CLAUSE_RE.search(left)
    ):
        risk += 30
        reasons.append(f"relative clause '{head}' separated from its antecedent")
    if head in _TRANSLATION_SENSITIVE_HEADS and right[:1].islower() and not _CLAUSE_RE.search(left):
        risk += 26
        reasons.append(f"dependent adverbial clause beginning with '{head}'")
    if (
        head == "here"
        and len(right_tokens) >= 2
        and right_tokens[1] in {"in", "on", "with"}
        and right[:1].islower()
    ):
        risk += 30
        reasons.append("dependent locative phrase separated from its clause")
    if (
        head in {"is", "was"}
        and re.match(
            r"^the\s+(?:easiest|best|only|main)\s+way\b",
            left,
            flags=re.IGNORECASE,
        )
    ):
        risk += 32
        reasons.append("way-clause subject separated from its predicate")

    if (
        head in {"is", "are", "was", "were"}
        and re.search(r"\b[A-Z][A-Za-z0-9'’.+-]*,?$", left)
        and (len(left_tokens) < 2 or left_tokens[-2] not in _DEPENDENT_RIGHT_HEADS)
    ):
        risk += 34
        reasons.append("proper-name subject separated from its predicate")

    if (
        head == "so"
        and right[:1].islower()
        and len(right_tokens) >= 2
        and (
            right_tokens[1].endswith("ly")
            or right_tokens[1] in {"far", "long", "many", "much", "strong", "well", "widespread"}
        )
    ):
        risk += 30
        reasons.append("degree complement separated from its predicate")

    if (
        tail == "already"
        and head in {"is", "are", "was", "were"}
        and re.search(r"\bthan\b[^.!?]*\balready$", left, re.IGNORECASE)
    ):
        risk += 38
        reasons.append("comparison auxiliary separated after 'already'")

    if (
        re.search(r"\b(?:a|an|the)\s+(?:19|20)\d{2}$", left, re.IGNORECASE)
        and (
            re.match(r"^[A-Z][A-Za-z0-9-]+\b", right)
            or re.match(
                r"^(?:acura|audi|bmw|cadillac|chevrolet|dodge|ford|gmc|honda|hyundai|"
                r"jeep|kia|lexus|lincoln|mazda|mercedes|nissan|porsche|ram|subaru|"
                r"tesla|toyota|volkswagen|volvo)\b",
                right,
                flags=re.IGNORECASE,
            )
        )
    ):
        risk += 38
        reasons.append("model year separated from vehicle name")

    if (
        re.match(r"^(?:and\s+)?what\b", left, re.IGNORECASE)
        and head == "and"
        and len(right_tokens) >= 2
        and right_tokens[1] in {"makes", "sets", "keeps", "gives", "gets", "drives"}
    ):
        risk += 34
        reasons.append("coordinated predicate separated inside a what-clause")

    # Keep location names such as "Ypsilanti, Michigan" in one cue.  This is
    # deliberately limited to a capitalized comma-separated left tail and a
    # known single-token state name to avoid treating ordinary sentence starts
    # as entities.
    left_name = re.search(r"\b([A-Z][A-Za-z'’-]+),\s*$", left)
    if (
        left_name
        and head in _US_STATE_NAMES
    ):
        risk += 38
        reasons.append("place name split between city and state")

    # Automotive speech commonly puts a trim before the model name ("RT392
    # Durango").  Moving this boundary is safe only for a compact alphanumeric
    # trim token followed by a capitalized model token.
    if re.fullmatch(r"(?:rt|srt|amg|rs|m)\d{1,3}", tail, flags=re.IGNORECASE) and re.match(
        r"^[A-Z][A-Za-z0-9-]+\b", right
    ):
        risk += 34
        reasons.append("vehicle trim separated from model name")

    if (
        head == "and"
        and len(right_tokens) >= 2
        and right_tokens[1] in {"go", "purchase", "buy"}
        and re.search(r"\b(?:after|before)\b[^,;.!?]*$", left, flags=re.IGNORECASE)
    ):
        risk += 34
        reasons.append("temporal phrase separated from its continuation")

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
        and (not left.speaker_id or not right.speaker_id or left.speaker_id == right.speaker_id)
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
            and "fixed phrase split inside 'do not get me wrong'" not in assessment.reasons
            and not _is_hard_boundary(left, right)
            and left.words
            and right.words
            and _word_count(words) <= hard_max_words
            and words[-1].end_time - words[0].start_time <= 8000
            and not re.search(r"[.!?][\"')\]]*\s+\S", combined_text)
        ):
            result[index : index + 2] = [_make_cue(words, left.speaker_id or right.speaker_id)]
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
    if head == "like" and re.search(r"\bso\s+much$", left, re.IGNORECASE):
        cost -= 12.0
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
                candidate = (
                    cost + segment_cost + boundary_cost,
                    path + [position],
                    risk + boundary_risk,
                )
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
                _is_hard_boundary(cues[index], cues[index + 1]) for index in range(len(cues) - 1)
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
                rebuilt.append(_make_cue(words[previous_position:position], fallback_speaker))
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
