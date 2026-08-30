"""Static inventory for incremental English boundary-rule migration."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from subforge.core.split.boundary_registry import BOUNDARY_SCORE_RULES


def _reason_template(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if not isinstance(node, ast.JoinedStr):
        return None
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif isinstance(value, ast.FormattedValue):
            parts.append("{" + ast.unparse(value.value) + "}")
        else:
            return None
    return "".join(parts)


def _reason_append(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (
        not isinstance(call.func, ast.Attribute)
        or call.func.attr != "append"
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "reasons"
        or len(call.args) != 1
    ):
        return None
    return _reason_template(call.args[0])


def _risk_increment(statement: ast.stmt) -> str | None:
    if (
        not isinstance(statement, ast.AugAssign)
        or not isinstance(statement.target, ast.Name)
        or statement.target.id != "risk"
        or not isinstance(statement.op, ast.Add)
    ):
        return None
    return ast.unparse(statement.value)


def _statement_blocks(node: ast.AST) -> Iterable[list[ast.stmt]]:
    for _field, value in ast.iter_fields(node):
        if isinstance(value, list) and value and all(isinstance(item, ast.stmt) for item in value):
            statements = list(value)
            yield statements
            for statement in statements:
                yield from _statement_blocks(statement)
        elif isinstance(value, ast.AST):
            yield from _statement_blocks(value)


def _family(reason: str) -> str:
    lowered = reason.casefold()
    if any(
        token in lowered
        for token in ("number", "numeric", "measurement", "year", "month", "percent")
    ):
        return "numeric"
    if any(
        token in lowered for token in ("proper name", "vehicle", "model", "brand", "place name")
    ):
        return "entity"
    if any(token in lowered for token in ("compar", "same", "than", "more and more")):
        return "comparison"
    if any(token in lowered for token in ("coordinat", "paired", "conjunction")):
        return "coordination"
    if any(token in lowered for token in ("connective", "discourse", "filler", "opinion")):
        return "discourse"
    if any(
        token in lowered
        for token in ("predicate", "subject", "clause", "complement", "object", "participle")
    ):
        return "predicate"
    return "grammar"


def _target_function(tree: ast.Module, function_name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise ValueError(f"Function not found: {function_name}")


def _score_stage_names(tree: ast.Module) -> tuple[str, ...]:
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            not isinstance(target, ast.Name)
            or target.id != "_ENGLISH_BOUNDARY_SCORE_STAGES"
            or not isinstance(node.value, (ast.List, ast.Tuple))
        ):
            continue
        names = tuple(
            item.id for item in node.value.elts if isinstance(item, ast.Name)
        )
        if len(names) != len(node.value.elts):
            raise ValueError("English boundary score stages must be direct function references")
        return names
    return ()


def _registered_calls(function: ast.FunctionDef) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for node in ast.walk(function):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id != "record_boundary_score"
        ):
            continue
        rule_id = (
            node.args[0].value
            if node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            else "<dynamic>"
        )
        calls.append(
            {
                "function": function.name,
                "line": node.lineno,
                "rule_id": rule_id,
            }
        )
    return calls


def audit_boundary_source(
    source: str,
    *,
    function_name: str = "assess_english_boundary",
) -> dict[str, Any]:
    tree = ast.parse(source)
    function = _target_function(tree, function_name)
    score_stage_names = _score_stage_names(tree)
    score_stages = [_target_function(tree, name) for name in score_stage_names]
    audited_functions = [*score_stages, function]
    legacy_sites: list[dict[str, Any]] = []
    for audited_function in audited_functions:
        for block in _statement_blocks(audited_function):
            for position, statement in enumerate(block):
                reason = _reason_append(statement)
                if reason is None:
                    continue
                weight = _risk_increment(block[position - 1]) if position else None
                legacy_sites.append(
                    {
                        "function": audited_function.name,
                        "line": statement.lineno,
                        "reason_template": reason,
                        "weight_expression": weight or "<unpaired>",
                        "family": _family(reason),
                    }
                )

    registered_calls = [
        call
        for audited_function in audited_functions
        for call in _registered_calls(audited_function)
    ]
    audited_names = {item.name for item in audited_functions}
    unscanned_registered_calls = [
        call
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name not in audited_names
        for call in _registered_calls(node)
    ]

    registered_ids = {rule.rule_id for rule in BOUNDARY_SCORE_RULES}
    static_call_ids = {
        item["rule_id"] for item in registered_calls if item["rule_id"] != "<dynamic>"
    }
    dynamic_call_count = sum(item["rule_id"] == "<dynamic>" for item in registered_calls)
    semantic_inventory = sorted(
        (
            item["reason_template"],
            item["weight_expression"],
            item["family"],
        )
        for item in legacy_sites
    )
    inventory_sha256 = hashlib.sha256(
        json.dumps(
            semantic_inventory,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "function": function_name,
        "score_stage_functions": list(score_stage_names),
        "score_stage_count": len(score_stage_names),
        "registered_definition_count": len(BOUNDARY_SCORE_RULES),
        "registered_call_count": len(registered_calls),
        "static_registered_call_count": len(static_call_ids),
        "dynamic_registered_call_count": dynamic_call_count,
        "unknown_registered_rule_ids": sorted(static_call_ids - registered_ids),
        "unreferenced_registered_rule_ids": sorted(registered_ids - static_call_ids),
        "legacy_site_count": len(legacy_sites),
        "legacy_reason_count": len({item["reason_template"] for item in legacy_sites}),
        "unpaired_legacy_site_count": sum(
            item["weight_expression"] == "<unpaired>" for item in legacy_sites
        ),
        "unscanned_registered_call_count": len(unscanned_registered_calls),
        "unscanned_registered_calls": unscanned_registered_calls,
        "legacy_family_counts": dict(
            sorted(Counter(item["family"] for item in legacy_sites).items())
        ),
        "inventory_sha256": inventory_sha256,
        "registered_calls": registered_calls,
        "legacy_sites": legacy_sites,
    }


def audit_boundary_file(path: Path) -> dict[str, Any]:
    return audit_boundary_source(path.read_text(encoding="utf-8"))


def write_boundary_audit(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
