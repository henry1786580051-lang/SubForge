"""Typed hard-invariant checks for provider translation responses."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from subforge.core.translate.quality.diagnostics import (
    DiagnosticCategory,
    DiagnosticSeverity,
    QualityDiagnostic,
    RepairStrategy,
)
from subforge.core.translate.quality.text import (
    contains_reasoning_leak,
    is_placeholder_translation,
    is_untranslated_output,
)
from subforge.core.translate.types import TargetLanguage


@dataclass(frozen=True, slots=True)
class PreservedTokenViolation:
    """One source fact that disappeared from a translated cue."""

    cue_key: str
    token: str
    kind: Literal["number", "entity"]

    @property
    def legacy_label(self) -> str:
        return f"{self.cue_key}:{self.token}"


def _sort_keys(keys: Iterable[str]) -> list[str]:
    return sorted(
        (str(key) for key in keys),
        key=lambda key: (0, int(key)) if key.isdigit() else (1, key),
    )


def _cue_keys(keys: Iterable[str]) -> tuple[int, ...]:
    return tuple(int(key) for key in _sort_keys(keys) if key.isdigit())


def _diagnostic(
    *,
    rule_id: str,
    category: DiagnosticCategory,
    cue_keys: Iterable[str] = (),
    evidence: Iterable[tuple[str, str]] = (),
    message: str,
) -> QualityDiagnostic:
    return QualityDiagnostic(
        rule_id=rule_id,
        category=category,
        severity=DiagnosticSeverity.ERROR,
        confidence=1.0,
        cue_keys=_cue_keys(cue_keys),
        evidence=tuple(evidence),
        repair_strategy=RepairStrategy.RETRY,
        message=message,
    )


def extract_translation_text(value: Any) -> str:
    """Read the accepted legacy response shapes without normalizing content."""
    if isinstance(value, dict):
        return str(value.get("native_translation", value.get("initial_translation", "")))
    return str(value)


def inspect_response_schema(
    response: Any,
    expected_keys: Iterable[str],
) -> tuple[QualityDiagnostic, ...]:
    """Validate the response container and exact key set."""
    if not isinstance(response, dict):
        return (
            _diagnostic(
                rule_id="schema.response.type",
                category=DiagnosticCategory.STRUCTURE,
                evidence=(("actual_type", type(response).__name__),),
                message=(
                    f"Output must be a dict, got {type(response).__name__}. "
                    "Use format: {'0': 'text', '1': 'text'}"
                ),
            ),
        )

    expected = {str(key) for key in expected_keys}
    actual = {str(key) for key in response}
    diagnostics: list[QualityDiagnostic] = []
    missing = expected - actual
    if missing:
        ordered = _sort_keys(missing)
        diagnostics.append(
            _diagnostic(
                rule_id="schema.missing_key",
                category=DiagnosticCategory.COMPLETENESS,
                cue_keys=ordered,
                evidence=(("missing_keys", repr(ordered)),),
                message=f"Missing keys {ordered} - you must translate these items",
            )
        )
    extra = actual - expected
    if extra:
        ordered = _sort_keys(extra)
        diagnostics.append(
            _diagnostic(
                rule_id="schema.extra_key",
                category=DiagnosticCategory.STRUCTURE,
                cue_keys=ordered,
                evidence=(("extra_keys", repr(ordered)),),
                message=f"Extra keys {ordered} - these keys are not in input, remove them",
            )
        )
    return tuple(diagnostics)


def inspect_target_script(
    response: Mapping[str, Any],
    source_by_key: Mapping[str, str],
    *,
    target_language: TargetLanguage,
    source_language_by_key: Mapping[str, str] | None = None,
) -> QualityDiagnostic | None:
    """Reject source-language output for targets that require another script."""
    if target_language.value not in {"简体中文", "繁体中文", "日本語", "韩语", "粤语"}:
        return None
    source_languages = source_language_by_key or {}
    untranslated = [
        key
        for key in _sort_keys(response)
        if key in source_by_key
        and is_untranslated_output(
            extract_translation_text(response[key]),
            source_by_key[key],
            target_language,
            source_languages.get(key, ""),
        )
    ]
    if not untranslated:
        return None
    return _diagnostic(
        rule_id="translation.untranslated",
        category=DiagnosticCategory.COMPLETENESS,
        cue_keys=untranslated,
        evidence=(("target_language", target_language.value),),
        message=(
            f"Translation to {target_language.value} failed: "
            f"{len(untranslated)}/{len(source_by_key)} entries are still in the source "
            f"language. You MUST translate ALL entries to {target_language.value}. "
            "Output target-language characters, not English. "
            f"Untranslated keys: {untranslated[:20]}"
        ),
    )


def inspect_placeholders(
    response: Mapping[str, Any],
    expected_keys: Iterable[str],
) -> tuple[QualityDiagnostic, ...]:
    """Classify empty output separately from model-authored placeholder notes."""
    empty: list[str] = []
    placeholders: list[str] = []
    for key in (str(item) for item in expected_keys):
        translated = extract_translation_text(response.get(key, ""))
        if not translated.strip():
            empty.append(key)
        elif is_placeholder_translation(translated):
            placeholders.append(key)

    all_invalid = [*empty, *placeholders]
    if not all_invalid:
        return ()
    message = (
        "Placeholder translations are not allowed. Every key must contain a real "
        "translation of its own source text. "
        f"Placeholder keys: {all_invalid[:20]}"
    )
    diagnostics: list[QualityDiagnostic] = []
    if empty:
        diagnostics.append(
            _diagnostic(
                rule_id="translation.empty",
                category=DiagnosticCategory.COMPLETENESS,
                cue_keys=empty,
                evidence=(("empty_keys", repr(empty)),),
                message=message,
            )
        )
    if placeholders:
        diagnostics.append(
            _diagnostic(
                rule_id="translation.placeholder",
                category=DiagnosticCategory.COMPLETENESS,
                cue_keys=placeholders,
                evidence=(("placeholder_keys", repr(placeholders)),),
                message=message,
            )
        )
    return tuple(diagnostics)


def inspect_reasoning_leaks(
    response: Mapping[str, Any],
    expected_keys: Iterable[str],
) -> QualityDiagnostic | None:
    """Reject private reasoning and response-format residue in subtitle values."""
    leaked = [
        key
        for key in (str(item) for item in expected_keys)
        if contains_reasoning_leak(extract_translation_text(response.get(key, "")))
    ]
    if not leaked:
        return None
    return _diagnostic(
        rule_id="translation.reasoning_leak",
        category=DiagnosticCategory.COMPLETENESS,
        cue_keys=leaked,
        evidence=(("reasoning_leak_keys", repr(leaked)),),
        message=(
            "Internal reasoning, analysis labels, and response-format markers must not "
            f"appear in subtitle text. Rewrite affected keys: {leaked[:20]}"
        ),
    )


def inspect_preserved_token_violations(
    violations: Iterable[PreservedTokenViolation],
) -> tuple[QualityDiagnostic, ...]:
    """Convert exact-value and identifier losses into stable diagnostics."""
    diagnostics: list[QualityDiagnostic] = []
    for violation in violations:
        rule_id = (
            "number.value_missing"
            if violation.kind == "number"
            else "entity.identifier_missing"
        )
        diagnostics.append(
            _diagnostic(
                rule_id=rule_id,
                category=DiagnosticCategory.COMPLETENESS,
                cue_keys=(violation.cue_key,),
                evidence=(
                    ("token", violation.token),
                    ("kind", violation.kind),
                    ("legacy_label", violation.legacy_label),
                ),
                message=(
                    "A source fact is missing from the translated cue. "
                    f"Preserve {violation.kind} token {violation.token!r}."
                ),
            )
        )
    return tuple(diagnostics)


def legacy_preserved_token_message(
    diagnostics: Iterable[QualityDiagnostic],
) -> str:
    """Recreate the historical provider feedback from typed diagnostics."""
    labels: list[str] = []
    for diagnostic in diagnostics:
        evidence = dict(diagnostic.evidence)
        label = evidence.get("legacy_label", "")
        if label:
            labels.append(label)
    if not labels:
        return ""
    return (
        "Likely dropped important source tokens. Preserve model names, years, specs, "
        f"and alphanumeric terms unless explicitly translated. Missing: {labels[:20]}"
    )


def inspect_reflect_value_schema(
    response: Mapping[str, Any],
) -> QualityDiagnostic | None:
    """Validate the legacy reflective response value shape."""
    for key, value in response.items():
        if isinstance(value, str) and value.strip():
            continue
        if not isinstance(value, dict):
            return _diagnostic(
                rule_id="schema.reflect.value_type",
                category=DiagnosticCategory.STRUCTURE,
                cue_keys=(str(key),),
                evidence=(("actual_type", type(value).__name__),),
                message=(
                    f"Key '{key}': value must be a translation string or a dict with "
                    f"'native_translation'. Got {type(value).__name__}."
                ),
            )
        if "native_translation" not in value:
            available_keys = list(value.keys())
            return _diagnostic(
                rule_id="schema.reflect.native_translation_missing",
                category=DiagnosticCategory.STRUCTURE,
                cue_keys=(str(key),),
                evidence=(("available_keys", repr(available_keys)),),
                message=(
                    f"Key '{key}': missing 'native_translation' field. Found keys: "
                    f"{available_keys}. Must include 'native_translation'."
                ),
            )
    return None


def legacy_diagnostic_message(diagnostics: Iterable[QualityDiagnostic]) -> str:
    """Preserve existing provider feedback while consumers migrate to rule IDs."""
    return "; ".join(diagnostic.message for diagnostic in diagnostics)
