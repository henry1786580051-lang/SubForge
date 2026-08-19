"""Global context helpers for LLM subtitle translation."""

from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from subforge.core.asr.asr_data import ASRData
from subforge.core.llm import (
    call_llm,
    get_response_text,
    parse_json_object,
)
from subforge.core.llm.client import ReasoningMode
from subforge.core.translate.types import TargetLanguage
from subforge.core.utils.cache import generate_cache_key
from subforge.core.utils.logger import setup_logger

logger = setup_logger("translation_context")

MAX_CONTEXT_CHARS = 12_000
MAX_TERMS = 48
MAX_TERMINOLOGY_CHARS = 4_000
CONTEXT_WINDOWS = 5
MAX_ENTITY_MENTIONS = 160
MAX_ENTITY_CONTEXTS = 64
MAX_ENTITY_CONTEXT_CHARS = 6_000
MAX_ENTITY_VARIANT_CANDIDATES = 32
MAX_ENTITY_ALIAS_GROUPS = 12
MAX_LEXICAL_VARIANT_CANDIDATES = 24
MAX_NUMERIC_CONTEXTS = 32
FOOTBALL_FIELD_AREA_SQUARE_METRES = 5351.0


@dataclass(frozen=True)
class TranslationContext:
    """Task-wide translation hints shared by all subtitle chunks."""

    summary: str = ""
    terminology: str = ""
    style: str = ""
    custom_prompt: str = ""

    def render(self) -> str:
        parts = []
        if self.summary.strip():
            parts.append(f"Video summary:\n{self.summary.strip()}")
        if self.terminology.strip():
            parts.append(f"Terminology and proper nouns:\n{self.terminology.strip()}")
        if self.style.strip():
            parts.append(f"Tone and style:\n{self.style.strip()}")
        if self.custom_prompt.strip():
            parts.append(f"User requirements:\n{self.custom_prompt.strip()}")
        return "\n\n".join(parts).strip()

    def fingerprint(self) -> str:
        return generate_cache_key(
            {
                "summary": self.summary,
                "terminology": self.terminology,
                "style": self.style,
                "custom_prompt": self.custom_prompt,
            }
        )


def _compact_transcript(segments: Iterable[str], limit: int = MAX_CONTEXT_CHARS) -> str:
    text = " ".join(seg.strip() for seg in segments if seg and seg.strip())
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text

    # Translation terminology can first appear anywhere in a long video. Sampling
    # several coherent windows gives the context pass whole-document coverage while
    # retaining a fixed token budget.
    separator = "\n...\n"
    window_size = max(
        16,
        (limit - len(separator) * (CONTEXT_WINDOWS - 1)) // CONTEXT_WINDOWS,
    )
    max_start = len(text) - window_size
    starts = [
        round(max_start * index / (CONTEXT_WINDOWS - 1))
        for index in range(CONTEXT_WINDOWS)
    ]
    windows = []
    for index, start in enumerate(starts):
        end = min(len(text), start + window_size)
        if index > 0:
            next_space = text.find(" ", start)
            if next_space != -1 and next_space < end:
                start = next_space + 1
        if index < len(starts) - 1:
            previous_space = text.rfind(" ", start, end)
            if previous_space > start:
                end = previous_space
        snippet = text[start:end].strip()
        if snippet and snippet not in windows:
            windows.append(snippet)
    return separator.join(windows)[:limit].strip()


def _canonical_name_from_asr_note(note: str) -> str:
    """Extract a Latin canonical name from common context-model note wording."""
    match = re.search(
        r"(?:(?i:canonical\s+(?:form|name|spelling))(?:\s+(?i:is)|\s*:)|"
        r"(?i:variant\s+of|phonetic\s+candidate\s+for))\s+"
        r"['\"]?([A-Z][A-Za-z'’.-]*(?:\s+[A-Z][A-Za-z'’.-]*){0,5})",
        str(note or ""),
    )
    return match.group(1).strip(" ,.;:!?-'’\"") if match else ""


def _asr_note_correction(
    source: str,
    target: str,
    note: str,
) -> str:
    """Recover an explicit correction that a context response hid in its note."""
    if not re.search(
        r"(?:asr|phonetic|mishear|recognition|spoken\s+self-correction|"
        r"self-correction|转录|听写|同音|口误|自我修正)",
        note,
        flags=re.IGNORECASE,
    ):
        return target

    match = re.search(
        r"(?:for|intended(?:\s+as)?|should\s+be|correct(?:ed)?(?:\s+as|\s+to)?)"
        r"\s+['\"]([^'\"]{2,80})['\"]",
        note,
        flags=re.IGNORECASE,
    )
    canonical = match.group(1).strip() if match else _canonical_name_from_asr_note(note)
    if not canonical:
        return target
    if canonical.casefold() == source.casefold():
        return target

    source_tokens = list(re.finditer(r"[A-Za-z0-9][A-Za-z0-9-]*", source))
    canonical_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", canonical)
    if len(canonical_tokens) != 1 or not source_tokens:
        return canonical

    canonical_token = canonical_tokens[0]
    closest = max(
        source_tokens,
        key=lambda token: difflib.SequenceMatcher(
            None,
            token.group().casefold(),
            canonical_token.casefold(),
        ).ratio(),
    )
    similarity = difflib.SequenceMatcher(
        None,
        closest.group().casefold(),
        canonical_token.casefold(),
    ).ratio()
    if similarity < 0.45:
        return canonical
    corrected = (
        source[: closest.start()] + canonical_token + source[closest.end() :]
    )
    return corrected


def _format_terms(value) -> str:
    if isinstance(value, list):
        prioritized_terms: list[tuple[int, int, str]] = []
        for position, item in enumerate(value):
            if isinstance(item, dict):
                source = str(item.get("source") or item.get("term") or "").strip()
                target = str(item.get("target") or item.get("translation") or "").strip()
                note = str(item.get("note") or "").strip()[:120]
                if not source:
                    continue
                target = _asr_note_correction(source, target, note)
                rendered = source
                if target:
                    rendered += f" -> {target}"
                if note:
                    rendered += f" ({note})"
                is_asr = bool(
                    re.search(
                        r"(?:asr|phonetic|mishear|recognition|spoken\s+self-correction|"
                        r"self-correction|转录|听写|同音|口误|自我修正)",
                        note,
                        flags=re.IGNORECASE,
                    )
                )
                is_nonliteral = bool(
                    re.search(
                        r"(?:idiom|figurative|irony|sarcasm|non-?literal|习语|反讽|讽刺|非字面)",
                        note,
                        flags=re.IGNORECASE,
                    )
                )
                is_identifier = bool(
                    re.search(r"[A-Z]{2,}|[A-Za-z]+\d|\d[A-Za-z]+", source)
                )
                priority = 0 if is_asr else 1 if is_nonliteral else 2 if is_identifier else 3
                prioritized_terms.append((priority, position, rendered))
            else:
                term = str(item).strip()
                if term:
                    prioritized_terms.append((3, position, term))
        # Keep corrections scoped to the complete heard phrase. A phonetic token
        # can refer to different names in different sentences, so promoting a
        # phrase-local edit into a global one-word replacement is unsafe.
        terms: list[str] = []
        terms.extend(
            rendered
            for _priority, _position, rendered in sorted(prioritized_terms)
            if rendered not in terms
        )
        terms = terms[:MAX_TERMS]
        return "\n".join(f"- {term}" for term in terms)[:MAX_TERMINOLOGY_CHARS].rstrip()
    return str(value or "").strip()[:MAX_TERMINOLOGY_CHARS].rstrip()


def _filter_acronym_wordplay_corrections(
    terms: list[Any],
    transcript_segments: Iterable[str],
) -> list[Any]:
    """Keep an explicit acronym pun from being collapsed into a similar acronym."""
    segments = [str(segment or "") for segment in transcript_segments]
    filtered: list[Any] = []
    for item in terms:
        if not isinstance(item, dict):
            filtered.append(item)
            continue
        source = str(item.get("source") or item.get("term") or "").strip()
        target = str(item.get("target") or item.get("translation") or "").strip()
        source_match = re.search(r"\b([A-Z]{2,})\b", source)
        target_match = re.search(r"\b([A-Z]{2,})\b", target)
        if not source_match or not target_match:
            filtered.append(item)
            continue
        source_acronym = source_match.group(1)
        target_acronym = target_match.group(1)
        if source_acronym == target_acronym:
            filtered.append(item)
            continue

        pun_evidence = False
        for index, segment in enumerate(segments):
            if not re.search(rf"\b{re.escape(source_acronym)}\b", segment):
                continue
            window = " ".join(segments[max(0, index - 1) : index + 3])
            if re.search(
                rf"not\b.{{0,80}}calling\s+it\b.{{0,80}}"
                rf"\b{re.escape(source_acronym.lower())}\b.{{0,160}}"
                r"\b(?:named|called)\s+it\b",
                window,
                flags=re.IGNORECASE,
            ):
                pun_evidence = True
                break
        if not pun_evidence:
            filtered.append(item)
    return filtered


def _document_entity_mentions(segments: Iterable[str]) -> list[str]:
    """Collect bounded whole-document name/model evidence missed by sampling."""
    mentions: dict[str, str] = {}
    pattern = re.compile(
        r"\b(?:[A-Z]{2,}[A-Za-z0-9-]*|[A-Za-z]+\d+[A-Za-z0-9-]*|"
        r"(?:Acura|Audi|BMW|Cadillac|Chevrolet|Dodge|Ford|GMC|Honda|Hyundai|Jeep|"
        r"Kia|Lexus|Lincoln|Mazda|Mercedes(?:-Benz)?|Nissan|Porsche|Ram|Subaru|"
        r"Tesla|Toyota|Volkswagen|Volvo|GR)"
        r"(?:\s+[A-Za-z0-9][A-Za-z0-9'’-]*){0,4})\b"
    )
    for segment in segments:
        for match in pattern.finditer(str(segment or "")):
            value = re.sub(r"\s+", " ", match.group()).strip(" ,.;:!?")
            if len(value) < 2:
                continue
            mentions.setdefault(value.casefold(), value)
            if len(mentions) >= MAX_ENTITY_MENTIONS:
                return list(mentions.values())
        # ASR-corrupted names can lose their model-code shape. Surface internal
        # capitalized tokens for task-wide reconciliation while excluding common
        # discourse words that are capitalized only by punctuation.
        for match in re.finditer(r"\b[A-Z][a-z]{3,}\b", str(segment or "")):
            if match.start() == 0 or match.group() in {
                "Actually",
                "Alright",
                "Because",
                "However",
                "Maybe",
                "Okay",
                "Otherwise",
                "Really",
                "Today",
            }:
                continue
            mentions.setdefault(match.group().casefold(), match.group())
            if len(mentions) >= MAX_ENTITY_MENTIONS:
                return list(mentions.values())
    return list(mentions.values())


def _document_entity_contexts(segments: Iterable[str]) -> list[str]:
    """Keep bounded local evidence for names that may be ASR-corrupted."""
    normalized_segments = [
        re.sub(r"\s+", " ", str(segment or "")).strip()
        for segment in segments
        if str(segment or "").strip()
    ]
    candidates: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    brand_pattern = re.compile(
        r"\b(?:Acura|Audi|BMW|Buick|Cadillac|Chevrolet|Dodge|Ford|GMC|Honda|"
        r"Hyundai|Jeep|Kia|Lexus|Lincoln|Mazda|Mercedes(?:-Benz)?|Nissan|"
        r"Porsche|Ram|Subaru|Tesla|Toyota|Volkswagen|Volvo)\b",
        flags=re.IGNORECASE,
    )
    identifier_pattern = re.compile(
        r"\b(?:[A-Z]{2,}[A-Za-z0-9-]*|[A-Za-z]+\d+[A-Za-z0-9-]*)\b"
    )
    internal_name_pattern = re.compile(r"(?<!^)\b[A-Z][a-z]{3,}(?:-[A-Za-z]+)?\b")

    for index, segment in enumerate(normalized_segments):
        score = 0
        if re.search(r"\b(?:or something|something like|sort of)\b", segment, re.I):
            score += 5
        if brand_pattern.search(segment):
            score += 3
        if identifier_pattern.search(segment):
            score += 3
        if internal_name_pattern.search(segment):
            score += 3
        if score < 3:
            continue

        start = max(0, index - 1)
        end = min(len(normalized_segments), index + 2)
        snippet = " | ".join(normalized_segments[start:end])
        fingerprint = snippet.casefold()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        candidates.append((score, index, snippet))

    contexts: list[str] = []
    used_chars = 0
    for _score, _index, snippet in sorted(candidates, key=lambda item: (-item[0], item[1])):
        additional_chars = len(snippet) + (1 if contexts else 0)
        if used_chars + additional_chars > MAX_ENTITY_CONTEXT_CHARS:
            continue
        contexts.append(snippet)
        used_chars += additional_chars
        if len(contexts) >= MAX_ENTITY_CONTEXTS:
            break
    return contexts


def _document_entity_variant_candidates(segments: Iterable[str]) -> list[dict[str, Any]]:
    """Shortlist noisy name forms against recurring document acronyms.

    This is intentionally evidence-only: it never changes source text. The
    context model receives bounded local excerpts and must still reject a pair
    unless the document topic makes one canonical entity unambiguous.
    """
    normalized_segments = [
        re.sub(r"\s+", " ", str(segment or "")).strip()
        for segment in segments
        if str(segment or "").strip()
    ]
    acronym_counts: dict[str, tuple[str, int]] = {}
    for segment in normalized_segments:
        for match in re.finditer(r"\b[A-Z][A-Z0-9&-]{2,9}\b", segment):
            value = match.group()
            compact = re.sub(r"[^a-z0-9]", "", value.casefold())
            if len(compact) < 3:
                continue
            previous = acronym_counts.get(compact, (value, 0))
            acronym_counts[compact] = (previous[0], previous[1] + 1)
    canonical_acronyms = {
        compact: (value, count)
        for compact, (value, count) in acronym_counts.items()
    }
    if not canonical_acronyms:
        return []

    discourse_words = {
        "actually",
        "as",
        "although",
        "because",
        "however",
        "maybe",
        "okay",
        "otherwise",
        "really",
        "today",
    }
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, segment in enumerate(normalized_segments):
        tokens = list(re.finditer(r"\b[A-Z][A-Za-z0-9&-]{1,11}\b", segment))
        for token in tokens:
            heard = token.group()
            heard_compact = re.sub(r"[^a-z0-9]", "", heard.casefold())
            if (
                len(heard_compact) < 2
                or heard.casefold() in discourse_words
                or canonical_acronyms.get(heard_compact, ("", 0))[1] >= 2
            ):
                continue
            heard_is_acronym_like = heard.isupper() or "&" in heard
            ranked = sorted(
                (
                    (
                        difflib.SequenceMatcher(None, heard_compact, canonical).ratio(),
                        canonical_value,
                        count,
                    )
                    for canonical, (canonical_value, count) in canonical_acronyms.items()
                    if canonical != heard_compact and (count >= 2 or heard_is_acronym_like)
                ),
                reverse=True,
            )
            if not ranked or ranked[0][0] < 0.72:
                continue
            best_score, canonical, canonical_count = ranked[0]
            if len(ranked) > 1 and best_score - ranked[1][0] < 0.08:
                continue
            fingerprint = (heard.casefold(), canonical.casefold())
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            start = max(0, index - 1)
            end = min(len(normalized_segments), index + 2)
            candidates.append(
                {
                    "heard": heard,
                    "possible_canonical": canonical,
                    "canonical_count": canonical_count,
                    "similarity": round(best_score, 3),
                    "context": " | ".join(normalized_segments[start:end]),
                }
            )
            if len(candidates) >= MAX_ENTITY_VARIANT_CANDIDATES:
                return candidates
    return candidates


def _document_entity_alias_groups(segments: Iterable[str]) -> list[dict[str, Any]]:
    """Group recurring phonetic proper-name variants without choosing a spelling.

    The result is evidence for the context model, not an automatic correction.
    Requiring repeated mentions and at least one multi-word form prevents common
    sentence-initial words from becoming speculative terminology.
    """
    normalized_segments = [
        re.sub(r"\s+", " ", str(segment or "")).strip()
        for segment in segments
        if str(segment or "").strip()
    ]
    ignored = {
        "actually",
        "also",
        "and",
        "because",
        "but",
        "completed",
        "however",
        "maybe",
        "now",
        "okay",
        "otherwise",
        "really",
        "so",
        "the",
        "this",
        "today",
        "well",
        "where",
    }
    records: dict[str, dict[str, Any]] = {}
    pattern = re.compile(
        r"\b[A-Z][A-Za-z'’.-]{2,}(?:\s+[A-Z][A-Za-z'’.-]{2,}){0,2}\b"
    )
    for index, segment in enumerate(normalized_segments):
        for match in pattern.finditer(segment):
            value = re.sub(r"\s+", " ", match.group()).strip(" ,.;:!?-'’")
            value = re.sub(r"['’]s$", "", value, flags=re.IGNORECASE)
            words = value.split()
            if not value or value.casefold() in ignored or all(
                word.casefold() in ignored for word in words
            ):
                continue
            compact = re.sub(r"[^a-z]", "", value.casefold())
            if len(compact) < 5:
                continue
            key = value.casefold()
            record = records.setdefault(
                key,
                {"text": value, "count": 0, "contexts": [], "word_count": len(words)},
            )
            record["count"] += 1
            snippet = " | ".join(
                normalized_segments[max(0, index - 1) : min(len(normalized_segments), index + 2)]
            )
            if snippet not in record["contexts"] and len(record["contexts"]) < 2:
                record["contexts"].append(snippet)

    values = list(records.values())
    if len(values) < 2:
        return []

    def _compact(value: str) -> str:
        return re.sub(r"[^a-z]", "", value.casefold())

    def _consonants(value: str) -> str:
        compact = re.sub(r"[aeiouy]", "", _compact(value))
        return re.sub(r"(.)\1+", r"\1", compact)

    def _same_alias_family(left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_compact = _compact(left["text"])
        right_compact = _compact(right["text"])
        for shorter, longer in (
            (left_compact, right_compact),
            (right_compact, left_compact),
        ):
            if longer.startswith(shorter) and longer[len(shorter) :] in {
                "er",
                "ers",
                "ian",
                "ians",
                "ite",
                "ites",
            }:
                return False
        left_consonants = _consonants(left["text"])
        right_consonants = _consonants(right["text"])
        if min(len(left_consonants), len(right_consonants)) < 4:
            return False
        if left["word_count"] != right["word_count"]:
            return left_consonants == right_consonants
        spelling_similarity = difflib.SequenceMatcher(
            None, left_compact, right_compact
        ).ratio()
        consonant_similarity = difflib.SequenceMatcher(
            None, left_consonants, right_consonants
        ).ratio()
        return spelling_similarity >= 0.77 and consonant_similarity >= 0.8

    neighbors: list[set[int]] = [set() for _ in values]
    for left_index, left in enumerate(values):
        for right_index in range(left_index + 1, len(values)):
            if _same_alias_family(left, values[right_index]):
                neighbors[left_index].add(right_index)
                neighbors[right_index].add(left_index)

    groups: list[dict[str, Any]] = []
    visited: set[int] = set()
    for start in range(len(values)):
        if start in visited or not neighbors[start]:
            continue
        component: list[int] = []
        stack = [start]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(neighbors[current] - visited)
        members = [values[index] for index in component]
        if (
            len(members) < 2
            or sum(int(member["count"]) for member in members) < 3
            or not any(int(member["word_count"]) > 1 for member in members)
        ):
            continue
        contexts: list[str] = []
        for member in members:
            for snippet in member["contexts"]:
                if snippet not in contexts:
                    contexts.append(snippet)
                if len(contexts) >= 4:
                    break
            if len(contexts) >= 4:
                break

        # A recurring proper name can occasionally be decoded as ordinary
        # lowercase words (for example, two words that merely sound like the
        # name). Surface only close phonetic n-grams to the resolver; they are
        # evidence candidates and never become corrections on their own.
        word_counts = {int(member["word_count"]) for member in members}
        known_variants = {str(member["text"]).casefold() for member in members}
        known_variant_token_lists = [
            re.findall(r"[a-z]+", value) for value in known_variants
        ]
        phonetic_candidates: list[dict[str, Any]] = []
        seen_candidates: set[str] = set()
        for segment_index, segment in enumerate(normalized_segments):
            words = re.findall(r"[A-Za-z][A-Za-z'’.-]*", segment)
            for word_count in word_counts:
                for word_index in range(0, len(words) - word_count + 1):
                    candidate = " ".join(words[word_index : word_index + word_count]).strip(
                        " .,-'’"
                    )
                    candidate_key = candidate.casefold()
                    if candidate != candidate.lower():
                        continue
                    if candidate_key in known_variants or candidate_key in seen_candidates:
                        continue
                    candidate_tokens = candidate_key.split()
                    if any(
                        candidate_tokens == known_tokens[start : start + len(candidate_tokens)]
                        for known_tokens in known_variant_token_lists
                        for start in range(
                            0, len(known_tokens) - len(candidate_tokens) + 1
                        )
                    ) or any(
                        token in known_variants for token in candidate_tokens
                    ):
                        continue
                    candidate_compact = _compact(candidate)
                    candidate_consonants = _consonants(candidate)
                    if len(candidate_consonants) < 4:
                        continue
                    similarities = [
                        (
                            difflib.SequenceMatcher(
                                None, candidate_compact, _compact(member["text"])
                            ).ratio(),
                            difflib.SequenceMatcher(
                                None, candidate_consonants, _consonants(member["text"])
                            ).ratio(),
                        )
                        for member in members
                    ]
                    viable = [
                        (spelling, consonants)
                        for spelling, consonants in similarities
                        if spelling >= 0.3 and consonants >= 0.78
                    ]
                    if not viable:
                        continue
                    similarity = max(
                        (spelling + consonants) / 2
                        for spelling, consonants in viable
                    )
                    seen_candidates.add(candidate_key)
                    phonetic_candidates.append(
                        {
                            "text": candidate,
                            "similarity": round(similarity, 3),
                            "context": " | ".join(
                                normalized_segments[
                                    max(0, segment_index - 1) : min(
                                        len(normalized_segments), segment_index + 2
                                    )
                                ]
                            ),
                        }
                    )
        phonetic_candidates.sort(
            key=lambda item: (-float(item["similarity"]), str(item["text"]).casefold())
        )
        groups.append(
            {
                "variants": [
                    {"text": member["text"], "count": member["count"]}
                    for member in sorted(
                        members,
                        key=lambda item: (-int(item["count"]), str(item["text"]).casefold()),
                    )
                ],
                "contexts": contexts,
                "phonetic_candidates": phonetic_candidates[:8],
            }
        )
        if len(groups) >= MAX_ENTITY_ALIAS_GROUPS:
            break
    return groups


def _document_lexical_variant_candidates(segments: Iterable[str]) -> list[dict[str, Any]]:
    """Surface rare long tokens that resemble recurring document terminology.

    These are context-model candidates, never deterministic corrections. The
    dual spelling/consonant threshold avoids broad fuzzy matching while still
    catching ASR renderings whose vowels or final syllable drifted.
    """
    normalized_segments = [
        re.sub(r"\s+", " ", str(segment or "")).strip()
        for segment in segments
        if str(segment or "").strip()
    ]
    occurrences: dict[str, list[int]] = {}
    display: dict[str, str] = {}
    for index, segment in enumerate(normalized_segments):
        for token in re.findall(r"[A-Za-z][A-Za-z'’-]{5,}", segment):
            key = token.casefold()
            occurrences.setdefault(key, []).append(index)
            display.setdefault(key, token)
    counts = Counter({key: len(indices) for key, indices in occurrences.items()})

    def compact(value: str) -> str:
        return re.sub(r"[^a-z]", "", value.casefold())

    def consonants(value: str) -> str:
        return re.sub(r"(.)\1+", r"\1", re.sub(r"[aeiouy]", "", compact(value)))

    def obvious_same_word(left: str, right: str) -> bool:
        left_compact = compact(left)
        right_compact = compact(right)
        if left_compact == right_compact:
            return True
        derivational_roots: dict[str, set[str]] = {}
        for value in (left_compact, right_compact):
            roots = {value}
            for suffix in (
                "ingly",
                "edly",
                "ally",
                "ness",
                "ment",
                "tion",
                "ure",
                "ing",
                "ed",
                "en",
                "ly",
                "es",
                "s",
            ):
                if value.endswith(suffix) and len(value) - len(suffix) >= 5:
                    roots.add(value[: -len(suffix)])
            if value.endswith("ble") and len(value) > 6:
                roots.add(value[:-3])
            if value.endswith("bly") and len(value) > 6:
                roots.add(value[:-3])
            if value.endswith("e") and len(value) > 6:
                roots.add(value[:-1])
            for prefix in ("dis", "non", "un", "im", "in", "ir"):
                if value.startswith(prefix) and len(value) - len(prefix) >= 6:
                    roots.add(value[len(prefix) :])
            derivational_roots[value] = roots
        if derivational_roots[left_compact] & derivational_roots[right_compact]:
            return True
        suffixes = (
            "'s",
            "ally",
            "edly",
            "ingly",
            "ly",
            "ness",
            "ions",
            "ion",
            "ies",
            "ers",
            "ing",
            "ed",
            "es",
            "s",
        )
        left_roots = {left_compact}
        right_roots = {right_compact}
        for value, roots in ((left_compact, left_roots), (right_compact, right_roots)):
            for suffix in suffixes:
                clean_suffix = compact(suffix)
                if value.endswith(clean_suffix) and len(value) - len(clean_suffix) >= 5:
                    roots.add(value[: -len(clean_suffix)])
        if left_roots & right_roots:
            return True
        return min(len(left_compact), len(right_compact)) >= 6 and (
            left_compact.endswith(right_compact)
            or right_compact.endswith(left_compact)
            or left_compact.startswith(right_compact)
            or right_compact.startswith(left_compact)
        )

    recurring = [key for key, count in counts.items() if count >= 2 and len(compact(key)) >= 7]
    candidates: list[dict[str, Any]] = []
    for heard, count in counts.items():
        if count != 1 or len(compact(heard)) < 7:
            continue
        ranked: list[tuple[float, str]] = []
        for canonical in recurring:
            if canonical == heard or obvious_same_word(heard, canonical):
                continue
            spelling = difflib.SequenceMatcher(None, compact(heard), compact(canonical)).ratio()
            consonant = difflib.SequenceMatcher(
                None, consonants(heard), consonants(canonical)
            ).ratio()
            if spelling >= 0.72 and consonant >= 0.8:
                ranked.append(((spelling + consonant) / 2, canonical))
        if not ranked:
            continue
        score, canonical = max(ranked)
        index = occurrences[heard][0]
        candidates.append(
            {
                "heard": display[heard],
                "possible_canonical": display[canonical],
                "canonical_count": counts[canonical],
                "similarity": round(score, 3),
                "context": " | ".join(
                    normalized_segments[
                        max(0, index - 1) : min(len(normalized_segments), index + 2)
                    ]
                ),
            }
        )
    candidates.sort(
        key=lambda item: (-float(item["similarity"]), str(item["heard"]).casefold())
    )
    return candidates[:MAX_LEXICAL_VARIANT_CANDIDATES]


def _document_lexical_context_hints(
    candidates: Iterable[dict[str, Any]],
) -> list[dict[str, str]]:
    """Keep only unusually close non-morphological candidates as translation hints.

    A hint is deliberately not labelled as a confirmed ASR correction, so it
    cannot rewrite locked source text. The translator still has to resolve it
    from the local sentence and neighboring cues.
    """

    def compact(value: str) -> str:
        return re.sub(r"[^a-z]", "", value.casefold())

    def same_derivation(left: str, right: str) -> bool:
        for adjective, adverb in ((left, right), (right, left)):
            if (
                adjective.endswith("ble")
                and adverb.endswith("bly")
                and adjective[:-3] == adverb[:-3]
            ):
                return True
            if (
                adjective.endswith("y")
                and adverb.endswith("ily")
                and adjective[:-1] == adverb[:-3]
            ):
                return True
        return False

    hints: list[dict[str, str]] = []
    for item in candidates:
        heard = str(item.get("heard") or "").strip()
        canonical = str(item.get("possible_canonical") or "").strip()
        heard_key = compact(heard)
        canonical_key = compact(canonical)
        if (
            not heard_key
            or not canonical_key
            or int(item.get("canonical_count") or 0) < 2
            or float(item.get("similarity") or 0) < 0.82
            or abs(len(heard_key) - len(canonical_key)) > 1
            or heard_key[:6] != canonical_key[:6]
            or same_derivation(heard_key, canonical_key)
        ):
            continue
        inflected = canonical
        if heard_key.endswith("s") and not canonical_key.endswith("s"):
            inflected += "s"
        hints.append(
            {
                "source": heard,
                "target": inflected,
                "note": (
                    "unconfirmed lexical similarity hint; use only when the local sentence "
                    "and recurring document subject prove this reading"
                ),
            }
        )
    return hints


def _extend_confirmed_alias_corrections(
    alias_groups: list[dict[str, Any]],
    terminology: list[Any],
) -> list[dict[str, str]]:
    """Extend an already confirmed alias family to lowercase phonetic mishears.

    The context model must independently map at least two listed proper-name
    variants to the same Latin canonical form. A phonetic candidate alone is
    never enough to create terminology.
    """
    parsed_mappings: list[tuple[str, str]] = []
    for item in terminology:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or item.get("term") or "").strip()
        target = str(item.get("target") or item.get("translation") or "").strip()
        note = str(item.get("note") or item.get("context") or "").strip()
        target = _asr_note_correction(source, target, note)
        if (
            source
            and target
            and re.search(
                r"(?:asr|phonetic|mishear|recognition|转录|听写|同音)",
                note,
                re.IGNORECASE,
            )
            and re.fullmatch(
                r"[A-Za-z][A-Za-z'’.-]*(?:\s+[A-Za-z][A-Za-z'’.-]*){0,5}",
                target,
            )
        ):
            parsed_mappings.append((source, target))

    corrections: list[dict[str, str]] = []
    seen_sources: set[str] = set()
    for group in alias_groups:
        variants = {
            str(item.get("text") or "").strip().casefold()
            for item in group.get("variants", [])
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        }
        canonical_sources: dict[str, set[str]] = {}
        canonical_spelling: dict[str, str] = {}
        for source, canonical in parsed_mappings:
            if source.casefold() not in variants:
                continue
            canonical_key = canonical.casefold()
            canonical_sources.setdefault(canonical_key, set()).add(source.casefold())
            canonical_spelling.setdefault(canonical_key, canonical)
        confirmed = [
            key for key, sources in canonical_sources.items() if len(sources) >= 2
        ]
        if len(confirmed) != 1:
            continue
        canonical = canonical_spelling[confirmed[0]]
        for candidate in group.get("phonetic_candidates", []):
            if not isinstance(candidate, dict):
                continue
            source = str(candidate.get("text") or "").strip()
            similarity = float(candidate.get("similarity") or 0)
            if not source or similarity < 0.68 or source.casefold() in seen_sources:
                continue
            seen_sources.add(source.casefold())
            corrections.append(
                {
                    "source": source,
                    "target": canonical,
                    "note": (
                        "probable ASR correction extended from two independently confirmed "
                        "proper-name variants in the same document"
                    ),
                }
            )
    return corrections


def _parse_small_spoken_number(value: str) -> float | None:
    normalized = re.sub(r"\s+", " ", str(value or "").strip().casefold())
    try:
        return float(normalized)
    except ValueError:
        pass
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }
    half = normalized.endswith(" and a half")
    base = normalized.removesuffix(" and a half")
    if base not in words:
        return None
    return words[base] + (0.5 if half else 0.0)


def _document_numeric_corrections(segments: Iterable[str]) -> list[dict[str, str]]:
    """Derive only arithmetic corrections proven by nearby source evidence."""
    normalized_segments = [
        re.sub(r"\s+", " ", str(segment or "")).strip()
        for segment in segments
        if str(segment or "").strip()
    ]
    corrections: list[dict[str, str]] = []
    seen: set[str] = set()

    for index, segment in enumerate(normalized_segments):
        following = " ".join(normalized_segments[index : index + 4])
        for match in re.finditer(
            r"(?P<first>\d+(?:\.\d+)?)\s*%\s*(?:,\s*)?or\s*"
            r"(?P<second>\d+(?:\.\d+)?)\s*%",
            segment,
            flags=re.IGNORECASE,
        ):
            remainder = re.search(
                r"\bremaining\s+(?P<value>\d+(?:\.\d+)?)\s*%",
                following,
                flags=re.IGNORECASE,
            )
            if not remainder:
                continue
            first = float(match.group("first"))
            second = float(match.group("second"))
            remaining = float(remainder.group("value"))
            if abs(second + remaining - 100.0) > 0.05 or abs(first + remaining - 100.0) <= 0.05:
                continue
            heard = match.group().strip()
            target = f"{match.group('second')}%"
            if heard.casefold() in seen:
                continue
            seen.add(heard.casefold())
            corrections.append(
                {
                    "source": heard,
                    "target": target,
                    "note": (
                        "probable ASR correction caused by a spoken self-correction; "
                        "the nearby remaining percentage proves the final value"
                    ),
                }
            )

        area_match = re.search(
            r"(?P<amount>\d[\d, ]*(?:\.\d+)?)\s+square\s+"
            r"(?P<unit>kilometres?|kilometers?)\b",
            segment,
            flags=re.IGNORECASE,
        )
        if not area_match or not re.search(
            r"football\s+fields?", following, flags=re.IGNORECASE
        ):
            continue
        field_match = re.search(
            r"approximately\s+(?P<count>\d+(?:\.\d+)?|"
            r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
            r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
            r"(?:\s+and\s+a\s+half)?)\s+(?:of\s+them|football\s+fields?)",
            following,
            flags=re.IGNORECASE,
        )
        if not field_match:
            continue
        field_count = _parse_small_spoken_number(field_match.group("count"))
        try:
            area = float(re.sub(r"[,\s]", "", area_match.group("amount")))
        except ValueError:
            continue
        if not field_count or not (0.5 <= area / (field_count * FOOTBALL_FIELD_AREA_SQUARE_METRES) <= 2.0):
            continue
        heard = area_match.group().strip()
        spelling = "meters" if "kilometer" in area_match.group("unit").casefold() else "metres"
        target = f"{area_match.group('amount').strip()} square {spelling}"
        if heard.casefold() in seen:
            continue
        seen.add(heard.casefold())
        corrections.append(
            {
                "source": heard,
                "target": target,
                "note": (
                    "probable ASR correction; the explicit nearby football-field "
                    "comparison proves the square-area unit"
                ),
            }
        )
    return corrections


def _document_entity_corrections(
    candidates: Iterable[dict[str, Any]],
) -> list[dict[str, str]]:
    """Promote acronym variants only when document evidence is unambiguous."""
    corrections: list[dict[str, str]] = []
    for item in candidates:
        heard = str(item.get("heard") or "").strip()
        canonical = str(item.get("possible_canonical") or "").strip()
        canonical_count = int(item.get("canonical_count") or 0)
        similarity = float(item.get("similarity") or 0.0)
        context = str(item.get("context") or "")
        canonical_is_acronym = bool(re.fullmatch(r"[A-Z][A-Z0-9-]{2,9}", canonical))
        ampersand_variant = "&" in heard and similarity >= 0.74
        recurring_acronym_variant = bool(
            re.fullmatch(r"[A-Z][A-Z0-9-]{2,9}", heard)
            and canonical_count >= 3
            and similarity >= 0.74
        )
        naming_wordplay = bool(
            re.search(
                rf"not\b.{{0,80}}calling\s+it\b.{{0,80}}"
                rf"\b{re.escape(heard.casefold())}\b",
                context,
                flags=re.IGNORECASE,
            )
        )
        if not canonical_is_acronym or naming_wordplay or not (
            ampersand_variant or recurring_acronym_variant
        ):
            continue
        corrections.append(
            {
                "source": heard,
                "target": canonical,
                "note": (
                    "probable ASR correction; the heard form closely matches a recurring "
                    "document-attested acronym"
                ),
            }
        )
    return corrections


def _document_numeric_contexts(segments: Iterable[str]) -> list[str]:
    """Collect local evidence for self-corrections and impossible unit comparisons."""
    normalized_segments = [
        re.sub(r"\s+", " ", str(segment or "")).strip()
        for segment in segments
        if str(segment or "").strip()
    ]
    numeric = re.compile(
        r"(?:\b\d[\d,.]*\s*(?:%|percent|percentage|square\s+(?:metres?|meters?|"
        r"kilometres?|kilometers?)|times|miles?|feet|foot)\b|"
        r"\b(?:remaining|remainder|football\s+fields?)\b)",
        re.IGNORECASE,
    )
    contexts: list[str] = []
    seen: set[str] = set()
    for index, segment in enumerate(normalized_segments):
        if not numeric.search(segment):
            continue
        start = max(0, index - 1)
        end = min(len(normalized_segments), index + 3)
        snippet = " | ".join(normalized_segments[start:end])
        if len(numeric.findall(snippet)) < 2:
            continue
        fingerprint = snippet.casefold()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        contexts.append(snippet)
        if len(contexts) >= MAX_NUMERIC_CONTEXTS:
            break
    return contexts


def build_translation_context(
    asr_data: ASRData,
    model: str,
    target_language: TargetLanguage,
    custom_prompt: str = "",
    use_cache: bool = True,
    llm_client: Any = None,
) -> TranslationContext:
    """Generate a task-wide summary and terminology list for LLM translation.

    The function is intentionally fail-open: translation should continue even
    if the provider rejects the context request or returns malformed JSON.
    """
    speaker_aliases: dict[str, str] = {}
    transcript_segments = []
    for segment in asr_data.segments:
        source = segment.text.strip()
        if not source:
            continue
        raw_speaker = str(segment.speaker_id or "").strip()
        if raw_speaker:
            alias = speaker_aliases.setdefault(raw_speaker, f"S{len(speaker_aliases) + 1}")
            transcript_segments.append(f"<{alias}> {source}")
        else:
            transcript_segments.append(source)
    transcript = _compact_transcript(transcript_segments)
    entity_mentions = _document_entity_mentions(transcript_segments)
    entity_contexts = _document_entity_contexts(transcript_segments)
    entity_variant_candidates = _document_entity_variant_candidates(transcript_segments)
    entity_alias_groups = _document_entity_alias_groups(transcript_segments)
    lexical_variant_candidates = _document_lexical_variant_candidates(transcript_segments)
    lexical_context_hints = _document_lexical_context_hints(lexical_variant_candidates)
    numeric_contexts = _document_numeric_contexts(transcript_segments)
    deterministic_corrections = [
        *_document_entity_corrections(entity_variant_candidates),
        *_document_numeric_corrections(transcript_segments),
    ]
    if not transcript:
        return TranslationContext(custom_prompt=custom_prompt)

    system_prompt = (
        "You prepare context for professional subtitle translation. "
        "Extract only information useful for consistent translation. "
        "Return pure JSON with keys: summary, terminology, style. "
        "terminology must be a list of {source, target, note}. "
        "For a probable ASR correction, source MUST be the complete heard form and target "
        "MUST be the complete corrected canonical form; never repeat the malformed source "
        "as target while mentioning a different correction only in note. "
        "Preserve proper nouns, model names, numbers, car trims, brands, and units. "
        "When surrounding transcript makes an ASR error unambiguous, include the heard form "
        "and intended form as a terminology item and label it probable ASR correction. Never "
        "guess from weak evidence. Treat punctuation, currency symbols, and number separators "
        "as potentially noisy ASR formatting when they make the utterance semantically "
        "impossible; infer the intended spoken unit only when the surrounding topic makes it "
        "unambiguous. Also include high-confidence idioms, figurative wording, ironic "
        "rephrasings, and sarcasm that would become misleading if translated literally. Use "
        "the complete source phrase, a concise natural target-language meaning, and label the "
        "note idiom/figurative/irony. Do not list ordinary compositional phrases or explain the "
        "speaker's intent beyond what is actually said. "
        "Record recurring spelling corrections and domain-specific word senses. "
        "Reconcile document_entity_mentions against the transcript: when several spellings "
        "clearly refer to one recurring name or product, keep the established canonical form "
        "and list noisy variants as probable ASR corrections. Do not merge merely similar "
        "names without strong document evidence. A spoken letter-by-letter spelling next to a "
        "name is high-confidence evidence: reconstruct the canonical name from those letters, "
        "record every heard variant that refers to it, and use one target rendering throughout "
        "the document. Never emit multiple transliterations for the same confirmed person. "
        "User requirements may contain the media title; "
        "treat a clearly named subject there as strong evidence for phonetic variants. "
        "A unique, widely established branded product name may also resolve a close phonetic "
        "ASR spelling when the brand and local technical facts agree; label that correction "
        "explicitly and do not preserve a malformed one-off spelling merely because it is rare. "
        "Use document_entity_contexts to inspect the sentence before and after a suspicious "
        "name. For a one-off branded model or trim, correct it only when the brand, nearby "
        "technical fact, and established product name point to one unique canonical form. "
        "Apply the same standard to a person's name used adjectivally for a trim or component. "
        "Otherwise retain uncertainty instead of guessing. "
        "document_entity_variant_candidates are similarity shortlists, never corrections. "
        "Confirm one only when its local context and the recurring document entity identify the "
        "same real name beyond reasonable doubt; then emit a probable ASR correction terminology "
        "item for the complete heard form. Reject ordinary people's names and coincidental "
        "spellings when that evidence is absent. "
        "document_entity_alias_groups contain recurring phonetic proper-name variants but do "
        "not choose a canonical spelling. Confirm a group only when its contexts identify one "
        "real entity. If a unique established canonical spelling is known with high confidence, "
        "emit a probable ASR correction for every listed malformed variant to that same complete "
        "canonical form. Never select the most frequent transcript spelling merely because it "
        "is frequent, and reject a group when the identity remains uncertain. The optional "
        "phonetic_candidates inside a group are lowercase phrases that only sound like the "
        "recurring name. Confirm one only when its own local context clearly uses it as that same "
        "entity; then emit the same probable ASR correction used for the confirmed capitalized "
        "variants. Never promote a candidate from sound similarity alone. "
        "document_lexical_variant_candidates are conservative similarity shortlists for a rare "
        "heard token and a recurring document term; they are never corrections by themselves. "
        "Evaluate every listed candidate rather than silently ignoring the list. Confirm one only "
        "when the recurring term has the same grammatical role and domain sense, "
        "the literal heard token makes the local sentence incoherent, and no other interpretation "
        "is reasonably plausible. Do not omit a candidate that satisfies all of those conditions. "
        "A literal everyday word can still be incoherent when its determiner, pronoun references, "
        "and the document's recurring subject all identify the listed domain term instead. "
        "Emit the complete heard token as a probable ASR correction. "
        "Reject ordinary inflection, derivation, related vocabulary, and coincidental spelling. "
        "Inspect document_numeric_contexts for a spoken self-correction or a unit contradicted by "
        "an explicit nearby comparison. When the speaker says alternatives such as '2% or 5%' "
        "and the following arithmetic uniquely confirms the latter value, record the complete "
        "heard phrase as a probable ASR correction caused by a spoken self-correction, mapping "
        "it to the final value. Likewise, "
        "correct an impossible unit only when a nearby quantity or comparison proves the intended "
        "unit. Never invent a correction from general plausibility alone. "
        "Prefer the domain-specific sense supported by adjacent source over a literal dictionary "
        "gloss, such as body adhesive rather than generic goop inside body panels. "
        "Prioritize ambiguous proper nouns, model trims, probable ASR corrections, and terms "
        "whose domain meaning differs from a literal dictionary gloss. Omit basic vocabulary "
        "such as brake pedal, manual transmission, or all-wheel drive when space is limited. "
        "List every confirmed entity or numeric correction before idioms. Do not spend the "
        "terminology budget on ordinary compositional metaphors or common expressions whose "
        "native meaning is obvious from their sentence. "
        "If the transcript explicitly jokes that a common-looking word is what an institution "
        "really named a product, treat the displayed all-caps form as a confirmed distinct name; "
        "never collapse it into a similar recurring acronym. "
        "If a speaker marks a one-off proper noun as uncertain with 'or something' and no "
        "canonical form is strongly supported, retain the brand and uncertainty but do not "
        "promote the malformed ASR spelling to canonical terminology. "
        "The summary must state the subject domain so later batches can disambiguate short "
        "fragments. The style must describe the speakers' actual register and concise native "
        "subtitle phrasing, not generic translation advice. For Chinese, preserve the source's "
        "existing imagery, contrast, irony, and rhetorical force with concise idiomatic wording, "
        "but never add decorative language or facts that are absent from the source. "
        "Tokens such as <S1> and <S2> are anonymous dialogue-turn metadata. Use them to "
        "understand roles and tone, but never include them as terminology or translated text."
    )
    user_payload = {
        "target_language": target_language.value,
        "user_requirements": custom_prompt,
        "transcript_excerpt": transcript,
        "document_entity_mentions": entity_mentions,
        "document_entity_contexts": entity_contexts,
        "document_entity_variant_candidates": entity_variant_candidates,
        "document_entity_alias_groups": entity_alias_groups,
        "document_lexical_variant_candidates": lexical_variant_candidates,
        "document_numeric_contexts": numeric_contexts,
    }

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        def request_context() -> dict[str, Any]:
            # Context extraction is structured terminology work, not a hard
            # semantic rewrite. Native reasoning can consume the full budget
            # before producing JSON, so keep both the initial request and its
            # malformed-response retry in the non-thinking path.
            attempts: tuple[tuple[ReasoningMode, int], ...] = (
                ("disabled", 4096),
                ("disabled", 4096),
            )
            last_error: Exception | None = None
            for reasoning_mode, max_output_tokens in attempts:
                try:
                    response = call_llm(
                        messages=messages,
                        model=model,
                        temperature=0.1,
                        use_cache=use_cache,
                        client=llm_client,
                        reasoning_mode=reasoning_mode,
                        max_output_tokens=max_output_tokens,
                    )
                    return parse_json_object(get_response_text(response))
                except Exception as error:
                    last_error = error
            assert last_error is not None
            raise last_error

        parsed = request_context()
        if not isinstance(parsed, dict):
            raise ValueError(f"context response must be dict, got {type(parsed).__name__}")
        parsed_terms = parsed.get("terminology")
        if not isinstance(parsed_terms, list):
            parsed_terms = []
        parsed_terms = _filter_acronym_wordplay_corrections(
            parsed_terms,
            transcript_segments,
        )
        alias_candidate_corrections = _extend_confirmed_alias_corrections(
            entity_alias_groups,
            parsed_terms,
        )
        return TranslationContext(
            summary=str(parsed.get("summary") or "").strip(),
            terminology=_format_terms(
                [
                    *deterministic_corrections,
                    *alias_candidate_corrections,
                    *lexical_context_hints,
                    *parsed_terms,
                ]
            ),
            style=str(parsed.get("style") or "").strip(),
            custom_prompt=custom_prompt,
        )
    except Exception as e:
        logger.warning("Translation context generation failed, continuing without it: %s", e)
        return TranslationContext(
            terminology=_format_terms(deterministic_corrections),
            custom_prompt=custom_prompt,
        )
