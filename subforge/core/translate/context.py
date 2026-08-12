"""Global context helpers for LLM subtitle translation."""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from subforge.core.asr.asr_data import ASRData
from subforge.core.llm import (
    call_llm,
    get_response_text,
    parse_json_object,
    prefers_native_reasoning,
)
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


def _asr_note_correction(
    source: str,
    target: str,
    note: str,
) -> str:
    """Recover an explicit correction that a context response hid in its note."""
    if not re.search(
        r"(?:asr|phonetic|mishear|recognition|转录|听写|同音)",
        note,
        flags=re.IGNORECASE,
    ):
        return target
    if target and target.casefold() != source.casefold():
        return target

    match = re.search(
        r"(?:for|intended(?:\s+as)?|should\s+be|correct(?:ed)?(?:\s+as|\s+to)?)"
        r"\s+['\"]([^'\"]{2,80})['\"]",
        note,
        flags=re.IGNORECASE,
    )
    if not match:
        return target
    canonical = match.group(1).strip()
    if not canonical or canonical.casefold() == source.casefold():
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
                        r"(?:asr|phonetic|mishear|recognition|转录|听写|同音)",
                        note,
                        flags=re.IGNORECASE,
                    )
                )
                is_identifier = bool(
                    re.search(r"[A-Z]{2,}|[A-Za-z]+\d|\d[A-Za-z]+", source)
                )
                priority = 0 if is_asr else 1 if is_identifier else 2
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
        "unambiguous. Record recurring spelling corrections and domain-specific word senses. "
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
        "Prefer the domain-specific sense supported by adjacent source over a literal dictionary "
        "gloss, such as body adhesive rather than generic goop inside body panels. "
        "Prioritize ambiguous proper nouns, model trims, probable ASR corrections, and terms "
        "whose domain meaning differs from a literal dictionary gloss. Omit basic vocabulary "
        "such as brake pedal, manual transmission, or all-wheel drive when space is limited. "
        "If a speaker marks a one-off proper noun as uncertain with 'or something' and no "
        "canonical form is strongly supported, retain the brand and uncertainty but do not "
        "promote the malformed ASR spelling to canonical terminology. "
        "The summary must state the subject domain so later batches can disambiguate short "
        "fragments. The style must describe the speakers' actual register and concise native "
        "subtitle phrasing, not generic translation advice. "
        "Tokens such as <S1> and <S2> are anonymous dialogue-turn metadata. Use them to "
        "understand roles and tone, but never include them as terminology or translated text."
    )
    user_payload = {
        "target_language": target_language.value,
        "user_requirements": custom_prompt,
        "transcript_excerpt": transcript,
        "document_entity_mentions": entity_mentions,
        "document_entity_contexts": entity_contexts,
    }

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        def request_context() -> dict[str, Any]:
            attempts = (
                (("enabled", 8192), ("disabled", 4096))
                if prefers_native_reasoning(model)
                else (("disabled", 4096),)
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
        return TranslationContext(
            summary=str(parsed.get("summary") or "").strip(),
            terminology=_format_terms(parsed.get("terminology")),
            style=str(parsed.get("style") or "").strip(),
            custom_prompt=custom_prompt,
        )
    except Exception as e:
        logger.warning("Translation context generation failed, continuing without it: %s", e)
        return TranslationContext(custom_prompt=custom_prompt)
