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
MAX_TERMS = 80


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
    head = text[: limit // 2].rsplit(" ", 1)[0]
    tail = text[-limit // 2 :].split(" ", 1)[-1]
    return f"{head}\n...\n{tail}"


def _format_terms(value) -> str:
    if isinstance(value, list):
        terms = []
        for item in value[:MAX_TERMS]:
            if isinstance(item, dict):
                source = str(item.get("source") or item.get("term") or "").strip()
                target = str(item.get("target") or item.get("translation") or "").strip()
                note = str(item.get("note") or "").strip()
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
        return "\n".join(f"- {term}" for term in terms)
    return str(value or "").strip()


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
    transcript = _compact_transcript(seg.text for seg in asr_data.segments)
    if not transcript:
        return TranslationContext(custom_prompt=custom_prompt)

    system_prompt = (
        "You prepare context for professional subtitle translation. "
        "Extract only information useful for consistent translation. "
        "Return pure JSON with keys: summary, terminology, style. "
        "terminology must be a list of {source, target, note}. "
        "Preserve proper nouns, model names, numbers, car trims, brands, and units."
    )
    user_payload = {
        "target_language": target_language.value,
        "user_requirements": custom_prompt,
        "transcript_excerpt": transcript,
    }

    try:
        response = call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            model=model,
            temperature=0.1,
            use_cache=use_cache,
            client=llm_client,
        )
        raw = get_response_text(response)
        parsed = parse_json_object(raw)
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
