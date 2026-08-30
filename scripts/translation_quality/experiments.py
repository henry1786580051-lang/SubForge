"""Opt-in behavioral candidates for the isolated evaluation process only.

Never imported by the application. Patches live only for the duration of a
single harness run and are restored even on failure. No provider settings change.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Iterator
from unittest.mock import patch

from subforge.core.translate.quality.preservation import exact_latin_spacing_spans

EXPERIMENTS = ("exact-name-spacing", "scoped-terminology", "unowned-fact-feedback")

TERM_SCOPE_GUIDANCE = (
    " A terminology target must be the lexical meaning of its complete source phrase, "
    "not a paraphrase of the surrounding episode. Do not insert a subject, direction of "
    "change, destination, number, tense, or situational outcome absent from that phrase. "
    "Put occurrence-specific explanations in note, explicitly conditional on the current "
    "subtitle supporting them. A domain term may have several contextual uses: keep its "
    "shared meaning neutral instead of turning one use into a document-wide instruction. "
    "Do not create mappings for ordinary descriptive vocabulary."
)


def unowned_fact_feedback(message: str) -> str:
    marker = "Cross-key duplicates: "
    if not message.startswith("A number or model fact was duplicated") or marker not in message:
        return message
    return (
        "These number/model facts are not supported by their current source keys. "
        "They may be borrowed facts OR extra explanations/unit conversions; a matching "
        "number elsewhere does not establish the cause. Remove unsupported additions, "
        "keep the original quantities and units, and do not append approximate conversions. "
        "Preserve all supported content and every subtitle key. Unsupported facts: "
        + message.split(marker, 1)[1]
    )


@contextmanager
def translation_experiments(names: tuple[str, ...] = ()) -> Iterator[None]:
    from subforge.core.translate.llm_translator import LLMTranslator

    unknown = set(names) - set(EXPERIMENTS)
    if unknown:
        raise ValueError(f"Unknown quality experiments: {sorted(unknown)}")
    if not names:
        yield
        return
    original = LLMTranslator._validate_no_unowned_latin_names

    def validate(self, response_dict, subtitle_dict, extract_text):
        masked = {}
        for key, source in subtitle_dict.items():
            source = self._all_source_by_index.get(int(key), source) if str(key).isdigit() else source
            target = str(extract_text(response_dict.get(key, "")) or "")
            characters = list(target)
            for start, end in exact_latin_spacing_spans(str(source or ""), target):
                characters[start:end] = " " * (end - start)
            masked[key] = "".join(characters)
        # Only this name-ownership check sees the masking. Every other validator,
        # returned translation, and immutable source sees the original content.
        return original(self, masked, subtitle_dict, lambda value: value)

    from subforge.core.translate import context

    context_call = context.call_llm
    ownership = LLMTranslator._validate_cross_key_boundaries

    def scoped_context(*args, **kwargs):
        messages = kwargs.get("messages", [])
        if messages and str(messages[0].get("content", "")).startswith(
            "You prepare context for professional subtitle translation."
        ):
            kwargs["messages"] = [
                {**messages[0], "content": messages[0]["content"] + TERM_SCOPE_GUIDANCE},
                *messages[1:],
            ]
        return context_call(*args, **kwargs)

    def ownership_feedback(self, *args, **kwargs):
        valid, message = ownership(self, *args, **kwargs)
        return valid, unowned_fact_feedback(message)

    with ExitStack() as stack:
        if "exact-name-spacing" in names:
            stack.enter_context(patch.object(LLMTranslator, "_validate_no_unowned_latin_names", validate))
        if "scoped-terminology" in names:
            stack.enter_context(patch.object(context, "call_llm", scoped_context))
        if "unowned-fact-feedback" in names:
            stack.enter_context(patch.object(LLMTranslator, "_validate_cross_key_boundaries", ownership_feedback))
        yield
