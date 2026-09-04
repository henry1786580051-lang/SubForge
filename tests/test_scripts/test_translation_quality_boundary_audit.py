from pathlib import Path

from scripts.translation_quality.boundary_audit import (
    audit_boundary_file,
    audit_boundary_source,
)


def test_boundary_audit_extracts_static_and_formatted_legacy_rules() -> None:
    payload = audit_boundary_source(
        """
def assess_english_boundary(left, right):
    risk = 0
    reasons = []
    tail = "that"
    if left:
        risk += 32
        reasons.append(f"dangling function word '{tail}'")
    if right:
        risk += 40
        reasons.append("clause-final subject separated from its finite predicate")
    return risk, reasons
"""
    )

    assert payload["legacy_site_count"] == 2
    assert payload["legacy_reason_count"] == 2
    assert payload["unpaired_legacy_site_count"] == 0
    assert payload["legacy_sites"][0]["reason_template"] == ("dangling function word '{tail}'")
    assert payload["legacy_sites"][0]["weight_expression"] == "32"


def test_boundary_audit_follows_declared_score_stages() -> None:
    payload = audit_boundary_source(
        '''
def _score_stage(features, reasons, contributions):
    return record_boundary_score(
        "split.boundary.english.numeric.measurement_comparative",
        reasons=reasons,
        contributions=contributions,
    )

_ENGLISH_BOUNDARY_SCORE_STAGES = (_score_stage,)

def assess_english_boundary(left, right):
    return record_boundary_score(
        "split.boundary.english.observation.lowercase_continuation_bonus"
    )
'''
    )

    assert payload["score_stage_functions"] == ["_score_stage"]
    assert payload["registered_call_count"] == 2
    assert payload["unscanned_registered_call_count"] == 0


def test_project_boundary_audit_has_closed_registered_references() -> None:
    payload = audit_boundary_file(Path("subforge/core/split/boundary.py"))

    assert payload["registered_definition_count"] == 150
    assert payload["score_stage_count"] == 6
    assert payload["score_stage_functions"] == [
        "_score_english_boundary_foundation",
        "_score_english_boundary_relations",
        "_score_english_boundary_completions",
        "_score_english_boundary_discourse",
        "_score_english_boundary_clause_ownership",
        "_score_english_boundary_dependencies",
    ]
    assert payload["registered_call_count"] == 155
    assert payload["unknown_registered_rule_ids"] == []
    assert payload["unscanned_registered_call_count"] == 0
    assert payload["unpaired_legacy_site_count"] == 0
    assert payload["legacy_site_count"] == 0
    assert payload["legacy_reason_count"] == 0
    assert payload["legacy_family_counts"] == {}
