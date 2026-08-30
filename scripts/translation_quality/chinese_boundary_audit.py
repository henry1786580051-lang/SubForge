"""Static inventory of the legacy Chinese boundary-quality flow."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from subforge.core.translate.quality.boundary_registry import BOUNDARY_RULES

_SIGNAL_FUNCTIONS = (
    "_chinese_boundary_signal",
    "_long_gap_chinese_boundary_signal",
    "_source_boundary_signal",
)
_FLOW_FUNCTIONS = (
    "_audit_reflective_alignment",
    "_target_boundary_diagnostic",
    "_chinese_fluency_candidates",
    "_mandatory_chinese_fluency_candidates",
    "_request_chinese_fluency_flags",
    "_chinese_fluency_windows",
    "_repair_chinese_boundary_fluency",
    "_repair_chinese_fluency_window_with_retries",
    "_should_reason_about_chinese_fluency_window",
)
_ALLOWED_SIGNAL_CALLERS = {
    "_chinese_boundary_signal": {
        "_target_boundary_diagnostic",
        "_mandatory_chinese_fluency_candidates",
    },
    "_long_gap_chinese_boundary_signal": {
        "_target_boundary_diagnostic",
        "_mandatory_chinese_fluency_candidates",
    },
    "_source_boundary_signal": {
        "_audit_reflective_alignment",
        "_chinese_fluency_candidates",
        "_request_chinese_fluency_flags",
        "_chinese_fluency_windows",
        "_should_reason_about_chinese_fluency_window",
    },
}


def _llm_translator_methods(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    translator = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LLMTranslator"
        ),
        None,
    )
    if translator is None:
        raise ValueError("LLMTranslator class not found")
    return {
        node.name: node
        for node in translator.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _literal_return_sites(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[dict[str, Any]]:
    sites: list[dict[str, Any]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Return):
            continue
        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        message = value.value.strip()
        if message:
            sites.append(
                {
                    "function": function.name,
                    "line": node.lineno,
                    "message": message,
                }
            )
    return sites


def _literal_message_assignments(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[dict[str, Any]]:
    sites: list[dict[str, Any]] = []
    for node in ast.walk(function):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if not any(isinstance(target, ast.Name) and target.id == "message" for target in targets):
            continue
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        message = value.value.strip()
        if message:
            sites.append(
                {
                    "function": function.name,
                    "line": node.lineno,
                    "message": message,
                }
            )
    return sites


def _dynamic_return_count(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    count = 0
    for node in ast.walk(function):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        if isinstance(node.value, ast.Constant):
            continue
        count += 1
    return count


def _literal_detector_sites(source_name: str, source: str) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    sites: list[dict[str, Any]] = []
    module_name = Path(source_name).stem
    for function in (
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        for node in ast.walk(function):
            if not isinstance(node, ast.Call) or _called_name(node) != "_match":
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            message = node.args[0].value
            if not isinstance(message, str) or not message.strip():
                continue
            sites.append(
                {
                    "function": f"{module_name}.{function.name}",
                    "line": node.lineno,
                    "message": message.strip(),
                }
            )
    return sites


def _called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def audit_chinese_boundary_source(
    source: str,
    *,
    detector_sources: dict[str, str] | None = None,
) -> dict[str, Any]:
    tree = ast.parse(source)
    methods = _llm_translator_methods(tree)
    missing_functions = sorted(
        (set(_SIGNAL_FUNCTIONS) | set(_FLOW_FUNCTIONS)) - methods.keys()
    )

    message_sites = [
        site
        for name in _SIGNAL_FUNCTIONS
        if (function := methods.get(name)) is not None
        for site in _literal_return_sites(function)
    ]
    target_diagnostic = methods.get("_target_boundary_diagnostic")
    if target_diagnostic is not None:
        message_sites.extend(_literal_message_assignments(target_diagnostic))
    for source_name, detector_source in sorted((detector_sources or {}).items()):
        message_sites.extend(_literal_detector_sites(source_name, detector_source))

    registry_by_message = {rule.legacy_message: rule for rule in BOUNDARY_RULES}
    for site in message_sites:
        rule = registry_by_message.get(site["message"])
        site["rule_id"] = rule.rule_id if rule is not None else "<unregistered>"

    emitted_messages = {site["message"] for site in message_sites}
    registered_messages = set(registry_by_message)
    function_message_counts = Counter(site["function"] for site in message_sites)
    unique_function_messages = {
        name: len({site["message"] for site in message_sites if site["function"] == name})
        for name in sorted(function_message_counts)
    }

    signal_calls: list[dict[str, Any]] = []
    diagnostic_adapter_calls: list[dict[str, Any]] = []
    for caller, function in methods.items():
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            callee = _called_name(node)
            if callee in _SIGNAL_FUNCTIONS:
                signal_calls.append({"caller": caller, "callee": callee, "line": node.lineno})
            if callee == "boundary_diagnostic_from_legacy_message":
                diagnostic_adapter_calls.append({"caller": caller, "line": node.lineno})

    unexpected_signal_calls = [
        call
        for call in signal_calls
        if call["caller"] not in _ALLOWED_SIGNAL_CALLERS[call["callee"]]
    ]
    dynamic_returns = {
        name: _dynamic_return_count(function)
        for name in _SIGNAL_FUNCTIONS
        if (function := methods.get(name)) is not None
    }
    flow_functions = {
        name: {
            "line": function.lineno,
            "line_count": (function.end_lineno or function.lineno) - function.lineno + 1,
            "if_count": sum(
                isinstance(node, (ast.If, ast.IfExp)) for node in ast.walk(function)
            ),
        }
        for name in (*_SIGNAL_FUNCTIONS, *_FLOW_FUNCTIONS)
        if (function := methods.get(name)) is not None
    }

    semantic_inventory = sorted(
        (
            message,
            rule_id,
            count,
        )
        for (message, rule_id), count in Counter(
            (site["message"], site["rule_id"]) for site in message_sites
        ).items()
    )
    layout_inventory = sorted(
        (
            function,
            message,
            rule_id,
            count,
        )
        for (function, message, rule_id), count in Counter(
            (site["function"], site["message"], site["rule_id"]) for site in message_sites
        ).items()
    )
    call_inventory = sorted(
        (caller, callee, count)
        for (caller, callee), count in Counter(
            (call["caller"], call["callee"]) for call in signal_calls
        ).items()
    )
    return {
        "signal_functions": list(_SIGNAL_FUNCTIONS),
        "flow_functions": flow_functions,
        "missing_functions": missing_functions,
        "registered_definition_count": len(BOUNDARY_RULES),
        "literal_message_site_count": len(message_sites),
        "emitted_message_count": len(emitted_messages),
        "function_message_site_counts": dict(sorted(function_message_counts.items())),
        "function_unique_message_counts": dict(sorted(unique_function_messages.items())),
        "dynamic_signal_return_counts": dynamic_returns,
        "unknown_emitted_messages": sorted(emitted_messages - registered_messages),
        "unreferenced_registered_messages": sorted(registered_messages - emitted_messages),
        "signal_call_count": len(signal_calls),
        "signal_calls": signal_calls,
        "unexpected_signal_call_count": len(unexpected_signal_calls),
        "unexpected_signal_calls": unexpected_signal_calls,
        "diagnostic_adapter_call_count": len(diagnostic_adapter_calls),
        "diagnostic_adapter_calls": diagnostic_adapter_calls,
        "inventory_sha256": hashlib.sha256(
            json.dumps(
                semantic_inventory,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "layout_sha256": hashlib.sha256(
            json.dumps(
                layout_inventory,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "call_flow_sha256": hashlib.sha256(
            json.dumps(call_inventory, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "message_sites": message_sites,
    }


def audit_chinese_boundary_file(path: Path) -> dict[str, Any]:
    detector_dir = path.parent / "quality" / "chinese_boundary_detectors"
    detector_sources = {
        detector_path.name: detector_path.read_text(encoding="utf-8")
        for detector_path in sorted(detector_dir.glob("*.py"))
        if detector_path.name != "__init__.py"
    }
    return audit_chinese_boundary_source(
        path.read_text(encoding="utf-8"),
        detector_sources=detector_sources,
    )


def write_chinese_boundary_audit(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
