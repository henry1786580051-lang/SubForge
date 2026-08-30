"""Typed translation diagnostics with legacy-message adapters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from subforge.core.translate.quality.boundary_registry import (
    BoundaryRuleKind,
    BoundaryRuleLevel,
    boundary_rule_for_message,
)
from subforge.core.translate.quality.boundary_registry import (
    registered_boundary_messages as registered_boundary_messages,
)


class DiagnosticCategory(str, Enum):
    STRUCTURE = "structure"
    COMPLETENESS = "completeness"
    OWNERSHIP = "ownership"
    DUPLICATION = "duplication"
    FLUENCY = "fluency"
    PROVIDER = "provider"


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class RepairStrategy(str, Enum):
    NONE = "none"
    DETERMINISTIC = "deterministic"
    LOCAL_REWRITE = "local_rewrite"
    BATCH_REBUILD = "batch_rebuild"
    RETRY = "retry"


@dataclass(frozen=True, slots=True)
class QualityDiagnostic:
    """Machine-readable quality finding independent of its log wording."""

    rule_id: str
    category: DiagnosticCategory
    severity: DiagnosticSeverity
    confidence: float
    cue_keys: tuple[int, ...]
    evidence: tuple[tuple[str, str], ...]
    repair_strategy: RepairStrategy
    message: str


def boundary_diagnostic_from_legacy_message(
    message: str,
    *,
    cue_keys: tuple[int, ...] = (),
    evidence: tuple[tuple[str, str], ...] = (),
) -> QualityDiagnostic | None:
    """Convert one legacy boundary signal while old callers still use its text."""
    normalized = str(message or "").strip()
    if not normalized:
        return None
    definition = boundary_rule_for_message(normalized)
    if definition is None:
        raise ValueError(f"Unregistered legacy boundary diagnostic: {normalized}")
    category = {
        BoundaryRuleKind.DUPLICATION: DiagnosticCategory.DUPLICATION,
        BoundaryRuleKind.FLUENCY: DiagnosticCategory.FLUENCY,
        BoundaryRuleKind.STRUCTURE: DiagnosticCategory.STRUCTURE,
    }[definition.kind]
    is_soft = definition.level == BoundaryRuleLevel.SOFT
    return QualityDiagnostic(
        rule_id=definition.rule_id,
        category=category,
        severity=DiagnosticSeverity.WARNING if is_soft else DiagnosticSeverity.ERROR,
        confidence=0.7 if is_soft else 0.95,
        cue_keys=cue_keys,
        evidence=evidence,
        repair_strategy=(
            RepairStrategy.DETERMINISTIC
            if category == DiagnosticCategory.DUPLICATION
            else RepairStrategy.LOCAL_REWRITE
        ),
        message=normalized,
    )
