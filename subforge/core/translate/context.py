"""Global context helpers for LLM subtitle translation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from subforge.core.asr.asr_data import ASRData
from subforge.core.llm import call_llm, get_response_text, parse_json_object
from subforge.core.translate.types import TargetLanguage
from subforge.core.utils.cache import generate_cache_key
from subforge.core.utils.logger import setup_logger

logger = setup_logger("translation_context")

MAX_CONTEXT_CHARS = 12_000
MAX_TERMS = 48
MAX_TERMINOLOGY_CHARS = 4_000
CONTEXT_WINDOWS = 5


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


def _format_terms(value) -> str:
    if isinstance(value, list):
        terms = []
        for item in value[:MAX_TERMS]:
            if isinstance(item, dict):
                source = str(item.get("source") or item.get("term") or "").strip()
                target = str(item.get("target") or item.get("translation") or "").strip()
                note = str(item.get("note") or "").strip()[:120]
                if not source:
                    continue
                rendered = source
                if target:
                    rendered += f" -> {target}"
                if note:
                    rendered += f" ({note})"
                terms.append(rendered)
            else:
                term = str(item).strip()
                if term:
                    terms.append(term)
        return "\n".join(f"- {term}" for term in terms)[:MAX_TERMINOLOGY_CHARS].rstrip()
    return str(value or "").strip()[:MAX_TERMINOLOGY_CHARS].rstrip()


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
    if not transcript:
        return TranslationContext(custom_prompt=custom_prompt)

    system_prompt = (
        "You prepare context for professional subtitle translation. "
        "Extract only information useful for consistent translation. "
        "Return pure JSON with keys: summary, terminology, style. "
        "terminology must be a list of {source, target, note}. "
        "Preserve proper nouns, model names, numbers, car trims, brands, and units. "
        "When surrounding transcript makes an ASR error unambiguous, include the heard form "
        "and intended form as a terminology item and label it probable ASR correction. Never "
        "guess from weak evidence. Treat punctuation, currency symbols, and number separators "
        "as potentially noisy ASR formatting when they make the utterance semantically "
        "impossible; infer the intended spoken unit only when the surrounding topic makes it "
        "unambiguous. Record recurring spelling corrections and domain-specific word senses. "
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
    }

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        def request_context() -> dict[str, Any]:
            response = call_llm(
                messages=messages,
                model=model,
                temperature=0.1,
                use_cache=use_cache,
                client=llm_client,
                # Context extraction is structured summarization. DeepSeek thinking
                # adds substantial latency here and can exhaust the output budget
                # before emitting the required JSON.
                reasoning_mode="disabled",
                max_output_tokens=4096,
            )
            return parse_json_object(get_response_text(response))

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
