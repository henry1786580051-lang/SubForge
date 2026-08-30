"""Provider-independent completeness and target-script validation."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from subforge.core.entities import SubtitleProcessData
from subforge.core.translate.types import TargetLanguage

_REASONING_LEAK_RE = re.compile(
    r"(?:<think>|</think>|<analysis>|</analysis>|<reasoning>|</reasoning>|"
    r"作为\s*(?:一个|一名)?\s*AI|推理过程\s*[:：]|"
    r"(?:analysis|reasoning)\s*:|```(?:json|markdown)?)",
    flags=re.IGNORECASE,
)


def _compact_similarity_text(text: str) -> str:
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", str(text or "").casefold())


def contains_reasoning_leak(text: str) -> bool:
    """Detect high-confidence private reasoning or response-format residue."""
    return bool(_REASONING_LEAK_RE.search(str(text or "")))


def is_source_copy(output: str, source: str) -> bool:
    """Detect a substantial output that is effectively an unchanged source copy."""
    normalized_output = _compact_similarity_text(output)
    normalized_source = _compact_similarity_text(source)
    if not normalized_output or not normalized_source or len(normalized_output) < 4:
        return False
    similarity = SequenceMatcher(
        None,
        normalized_output,
        normalized_source,
        autojunk=False,
    ).ratio()
    return similarity >= 0.94


def is_placeholder_translation(text: str) -> bool:
    """Detect model notes that are not actual subtitle translations."""
    text = str(text or "").strip()
    if not text:
        return True
    compact = re.sub(r"\s+", "", text).strip("()（）[]【】<>《》“”\"'。，、；;：:！!?")
    previous_refs = r"上一句|上句|上一条|上条|前一句|前一条|前文|前面"
    patterns = [
        r"(?:此|本)句.*(?:合并|并入|省略|略去|无需翻译|不单独翻译).*",
        rf"(?:已)?(?:合并|并入|接上|延续|已译|包含).*(?:{previous_refs})",
        rf"(?:{previous_refs}).*(?:合并|包含|已译|并入|已经翻译)",
        r"(?:最终版本|最终字幕).*(?:合并|省略)",
        r"(?:内容)?(?:同上|见上|略|省略|无需翻译|不单独翻译)",
        r"merged(?:with|into)?(?:the)?(?:previous|above)",
        r"sameasabove",
        r"omitted",
    ]
    if any(re.fullmatch(pattern, compact, flags=re.IGNORECASE) for pattern in patterns):
        return True
    meta_note = re.compile(
        r"(?:\(|（|\[|【)\s*(?:应为|疑似|译注|注\s*[:：]|原文(?:应为)?|可能是)"
        r"[^\)）\]】]*(?:\)|）|\]|】)",
        flags=re.IGNORECASE,
    )
    return bool(meta_note.search(text))


def is_untranslated_output(
    output: str,
    source: str,
    target_language: TargetLanguage,
    source_language: str = "",
) -> bool:
    """Return whether output lacks the script required by an Asian target."""
    target_patterns = {
        TargetLanguage.SIMPLIFIED_CHINESE: r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]",
        TargetLanguage.TRADITIONAL_CHINESE: r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]",
        TargetLanguage.CANTONESE: r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]",
        TargetLanguage.JAPANESE: r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff]",
        TargetLanguage.KOREAN: r"[\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff]",
    }
    target_pattern = target_patterns.get(target_language)
    if target_pattern is None:
        return False
    normalized_source = re.sub(r"\s+", "", source).casefold()
    normalized_output = re.sub(r"\s+", "", output).casefold()
    if source_language and source_language != "zh" and normalized_output == normalized_source:
        return True
    if (
        target_language
        in {
            TargetLanguage.SIMPLIFIED_CHINESE,
            TargetLanguage.TRADITIONAL_CHINESE,
            TargetLanguage.CANTONESE,
        }
        and source_language in {"ja", "mixed"}
        and re.search(r"[\u3040-\u30ff\u31f0-\u31ff]", output)
    ):
        source_kana = re.findall(r"[\u3040-\u30ff\u31f0-\u31ff]", source)
        output_kana = re.findall(r"[\u3040-\u30ff\u31f0-\u31ff]", output)
        if source_kana and len(output_kana) >= max(2, len(source_kana) // 2):
            return True
    if re.search(target_pattern, output):
        return False
    if re.search(
        r"[\u3040-\u30ff\u31f0-\u31ff\u1100-\u11ff\u3130-\u318f"
        r"\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff\u3400-\u4dbf"
        r"\u4e00-\u9fff\uf900-\ufaff]",
        source,
    ):
        return True

    # A standalone web address is already language-neutral subtitle content.
    # Calls to action such as "visit example.com" still require translation.
    url_only = re.compile(
        r"^\s*(?:https?://)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
        r"(?:\s*(?:/|slash)\s*[A-Za-z0-9_-]+)*[.!]?\s*$",
        flags=re.IGNORECASE,
    )
    spoken_url_only = re.compile(
        r"^\s*[A-Za-z0-9-]+\s+(?:dot\s+)?(?:com|org|net|io)"
        r"(?:\s+(?:slash\s+)?[A-Za-z0-9_-]+)*[.!]?\s*$",
        flags=re.IGNORECASE,
    )
    if url_only.fullmatch(source) or spoken_url_only.fullmatch(source):
        return False
    source_words = re.findall(r"[A-Za-z]+", source)
    if not source_words:
        return False
    source_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.+#&/-]*", source)

    def is_identifier_like(token: str) -> bool:
        token = token.strip(".")
        letters = re.sub(r"[^A-Za-z]", "", token)
        return bool(
            re.search(r"\d", token)
            or (len(letters) >= 2 and letters.isupper())
            or re.search(r"[a-z][A-Z]", letters)
            or re.search(r"[.+#&/-]", token)
        )

    if (
        source_tokens
        and len(source_tokens) <= 3
        and all(is_identifier_like(token) for token in source_tokens)
    ):
        return False
    return bool(source_words)


@dataclass(slots=True)
class TranslationCompletenessReport:
    missing: list[str] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)
    untranslated: list[str] = field(default_factory=list)
    reasoning_leaks: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(
            (
                self.missing,
                self.empty,
                self.duplicates,
                self.placeholders,
                self.untranslated,
                self.reasoning_leaks,
            )
        )

    def error_detail(self) -> str:
        parts = []
        if self.missing:
            parts.append(f"missing indices: {self.missing[:20]}")
        if self.empty:
            parts.append(f"empty translations: {self.empty[:20]}")
        if self.duplicates:
            parts.append(f"duplicate indices: {self.duplicates[:20]}")
        if self.placeholders:
            parts.append(f"placeholder translations: {self.placeholders[:20]}")
        if self.untranslated:
            parts.append(f"untranslated indices: {self.untranslated[:20]}")
        if self.reasoning_leaks:
            parts.append(f"reasoning leaks: {self.reasoning_leaks[:20]}")
        return "; ".join(parts)


def inspect_translation_batch(
    source_list: Iterable[SubtitleProcessData],
    translated_list: Iterable[SubtitleProcessData],
    target_language: TargetLanguage,
) -> TranslationCompletenessReport:
    report = TranslationCompletenessReport()
    translated_by_index: dict[int, SubtitleProcessData] = {}
    for item in translated_list:
        if item.index in translated_by_index:
            report.duplicates.append(str(item.index))
        translated_by_index[item.index] = item

    for source in source_list:
        translated = translated_by_index.get(source.index)
        if translated is None:
            report.missing.append(str(source.index))
            continue
        output = translated.translated_text.strip()
        if not output:
            report.empty.append(str(source.index))
            continue
        if is_placeholder_translation(output):
            report.placeholders.append(str(source.index))
        if contains_reasoning_leak(output):
            report.reasoning_leaks.append(str(source.index))
        if is_untranslated_output(
            output,
            source.original_text,
            target_language,
            source.source_language,
        ):
            report.untranslated.append(str(source.index))
    return report
