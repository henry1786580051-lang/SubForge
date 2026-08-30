import ast
from pathlib import Path

import pytest

from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.quality import (
    DiagnosticSeverity,
    RepairStrategy,
    boundary_diagnostic_from_legacy_message,
)
from subforge.core.translate.quality.boundary_registry import (
    BOUNDARY_RULES,
    BoundaryRuleKind,
    BoundaryRuleLevel,
    BoundaryRuleScope,
    boundary_rule_for_id,
    boundary_rule_for_message,
)
from subforge.core.translate.quality.diagnostics import registered_boundary_messages


def _literal_returns(function_names: set[str]) -> set[str]:
    path = Path("subforge/core/translate/llm_translator.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    messages: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in function_names:
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Return)
                and isinstance(child.value, ast.Constant)
                and isinstance(child.value.value, str)
                and child.value.value
            ):
                messages.add(child.value.value)
    return messages


def test_every_legacy_boundary_message_has_a_stable_rule_id() -> None:
    emitted = _literal_returns(
        {
            "_chinese_boundary_signal",
            "_long_gap_chinese_boundary_signal",
            "_source_boundary_signal",
        }
    )

    assert emitted <= registered_boundary_messages()
    diagnostics = [boundary_diagnostic_from_legacy_message(message) for message in emitted]
    rule_ids = [diagnostic.rule_id for diagnostic in diagnostics if diagnostic is not None]
    assert len(rule_ids) == len(set(rule_ids))


def test_typed_target_diagnostic_preserves_legacy_message() -> None:
    translator = object.__new__(LLMTranslator)
    translator._gap_after_index = {7: 500}

    diagnostic = translator._target_boundary_diagnostic(
        7,
        "在隧道内部",
        "会安装支撑结构",
    )

    assert diagnostic is not None
    assert diagnostic.rule_id == "translation.boundary.target.locative_predicate"
    assert diagnostic.severity == DiagnosticSeverity.ERROR
    assert diagnostic.repair_strategy == RepairStrategy.LOCAL_REWRITE
    assert diagnostic.cue_keys == (7, 8)
    assert (
        translator._target_boundary_signal(7, "在隧道内部", "会安装支撑结构") == diagnostic.message
    )


def test_unknown_legacy_boundary_message_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unregistered legacy boundary diagnostic"):
        boundary_diagnostic_from_legacy_message("new unregistered message")


def test_boundary_registry_has_unique_bidirectional_entries() -> None:
    assert len(BOUNDARY_RULES) == len(registered_boundary_messages())
    assert len({rule.rule_id for rule in BOUNDARY_RULES}) == len(BOUNDARY_RULES)
    assert all(boundary_rule_for_message(rule.legacy_message) is rule for rule in BOUNDARY_RULES)
    assert all(boundary_rule_for_id(rule.rule_id) is rule for rule in BOUNDARY_RULES)


def test_boundary_registry_preserves_legacy_rule_metadata() -> None:
    source_rule = boundary_rule_for_message("coordinate phrase crosses the subtitle boundary")
    duplicate_rule = boundary_rule_for_message("possible duplicated boundary phrase")
    display_rule = boundary_rule_for_message("number and unit are separated by a visible pause")

    assert source_rule is not None
    assert source_rule.scope == BoundaryRuleScope.SOURCE
    assert source_rule.level == BoundaryRuleLevel.SOFT
    assert source_rule.source_languages == ("en",)
    assert duplicate_rule is not None
    assert duplicate_rule.kind == BoundaryRuleKind.DUPLICATION
    assert duplicate_rule.level == BoundaryRuleLevel.SOFT
    assert display_rule is not None
    assert display_rule.scope == BoundaryRuleScope.DISPLAY
    assert display_rule.level == BoundaryRuleLevel.HARD
