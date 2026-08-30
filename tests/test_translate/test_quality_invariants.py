from subforge.core.translate.quality import (
    DiagnosticSeverity,
    PreservedTokenViolation,
    inspect_placeholders,
    inspect_preserved_token_violations,
    inspect_reasoning_leaks,
    inspect_reflect_value_schema,
    inspect_response_schema,
    inspect_target_script,
    legacy_diagnostic_message,
    legacy_preserved_token_message,
)
from subforge.core.translate.types import TargetLanguage


def test_response_schema_reports_mixed_missing_and_extra_keys_deterministically():
    diagnostics = inspect_response_schema(
        {"1": "一", "metadata": "ignored"},
        {"1", "2", "10"},
    )

    assert [item.rule_id for item in diagnostics] == [
        "schema.missing_key",
        "schema.extra_key",
    ]
    assert diagnostics[0].cue_keys == (2, 10)
    assert legacy_diagnostic_message(diagnostics) == (
        "Missing keys ['2', '10'] - you must translate these items; "
        "Extra keys ['metadata'] - these keys are not in input, remove them"
    )


def test_response_schema_rejects_non_mapping_with_legacy_feedback():
    diagnostics = inspect_response_schema([], {"1"})

    assert len(diagnostics) == 1
    assert diagnostics[0].rule_id == "schema.response.type"
    assert diagnostics[0].severity == DiagnosticSeverity.ERROR
    assert diagnostics[0].message == (
        "Output must be a dict, got list. Use format: {'0': 'text', '1': 'text'}"
    )


def test_target_script_diagnostic_preserves_untranslated_feedback():
    diagnostic = inspect_target_script(
        {"1": "Still English", "2": "已经翻译"},
        {"1": "Still English", "2": "Already translated"},
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
    )

    assert diagnostic is not None
    assert diagnostic.rule_id == "translation.untranslated"
    assert diagnostic.cue_keys == (1,)
    assert "Untranslated keys: ['1']" in diagnostic.message


def test_placeholder_diagnostics_distinguish_empty_from_model_note():
    diagnostics = inspect_placeholders(
        {"1": "", "2": "（此句合并至上一句）", "3": "正常译文"},
        ("1", "2", "3"),
    )

    assert [item.rule_id for item in diagnostics] == [
        "translation.empty",
        "translation.placeholder",
    ]
    assert diagnostics[0].cue_keys == (1,)
    assert diagnostics[1].cue_keys == (2,)
    assert diagnostics[0].message == diagnostics[1].message
    assert "Placeholder keys: ['1', '2']" in diagnostics[0].message


def test_reasoning_leak_diagnostic_identifies_only_affected_keys():
    diagnostic = inspect_reasoning_leaks(
        {"1": "正常译文", "2": "<analysis>private</analysis>最终译文"},
        ("1", "2"),
    )

    assert diagnostic is not None
    assert diagnostic.rule_id == "translation.reasoning_leak"
    assert diagnostic.cue_keys == (2,)


def test_preserved_token_diagnostics_distinguish_numbers_from_identifiers():
    diagnostics = inspect_preserved_token_violations(
        (
            PreservedTokenViolation("2", "2026", "number"),
            PreservedTokenViolation("3", "GTI", "entity"),
        )
    )

    assert [item.rule_id for item in diagnostics] == [
        "number.value_missing",
        "entity.identifier_missing",
    ]
    assert [item.cue_keys for item in diagnostics] == [(2,), (3,)]
    assert legacy_preserved_token_message(diagnostics) == (
        "Likely dropped important source tokens. Preserve model names, years, specs, "
        "and alphanumeric terms unless explicitly translated. Missing: ['2:2026', '3:GTI']"
    )


def test_reflect_value_schema_accepts_legacy_string_and_nested_shapes():
    assert (
        inspect_reflect_value_schema(
            {
                "1": "普通字符串也被旧流程接受",
                "2": {"native_translation": "嵌套译文"},
            }
        )
        is None
    )


def test_reflect_value_schema_reports_missing_native_translation_field():
    diagnostic = inspect_reflect_value_schema({"7": {"initial_translation": "初译"}})

    assert diagnostic is not None
    assert diagnostic.rule_id == "schema.reflect.native_translation_missing"
    assert diagnostic.cue_keys == (7,)
    assert diagnostic.message == (
        "Key '7': missing 'native_translation' field. Found keys: "
        "['initial_translation']. Must include 'native_translation'."
    )
