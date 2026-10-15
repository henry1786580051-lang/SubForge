from pathlib import Path

from scripts.translation_quality.chinese_boundary_audit import audit_chinese_boundary_file


def test_project_chinese_boundary_audit_has_closed_registry_and_call_flow() -> None:
    payload = audit_chinese_boundary_file(Path("subforge/core/translate/llm_translator.py"))

    assert payload["registered_definition_count"] == 82
    assert payload["literal_message_site_count"] == 113
    assert payload["emitted_message_count"] == 82
    assert payload["function_message_site_counts"] == {
        "_source_boundary_signal": 7,
        "_target_boundary_diagnostic": 1,
        "adverb_pronoun_attachment.detect_adverb_pronoun_attachment_boundary": 6,
        "clause_attachment.detect_clause_attachment_boundary": 7,
        "completion_frames.detect_completion_frame_boundary": 4,
        "consequence_predicate.detect_consequence_predicate_boundary": 1,
        "discourse_bridge.detect_discourse_bridge_boundary": 4,
        "foundation.detect_foundation_boundary": 5,
        "governing_attachment.detect_governing_attachment_boundary": 6,
        "incomplete_nominal_frames.detect_incomplete_nominal_frame_boundary": 4,
        "late_structural_frames.detect_late_structural_frame_boundary": 3,
        "nominal_attachment.detect_nominal_attachment_boundary": 9,
        "numeric_completion.detect_numeric_completion_boundary": 2,
        "predicate_completion.detect_predicate_completion_boundary": 2,
        "reason_construction.detect_reason_construction_boundary": 1,
        "semantic_attachment.detect_semantic_attachment_boundary": 3,
        "semantic_completion.detect_semantic_completion_boundary": 5,
        "subject_attachment.detect_subject_attachment_boundary": 6,
        "subject_nominal_completion.detect_subject_nominal_completion_boundary": 3,
        "structural_tail.detect_structural_tail_boundary": 5,
        "surface_fluency.detect_surface_fluency_boundary": 13,
        "temporal_locative_attachment.detect_temporal_locative_attachment_boundary": 4,
        "terminal_tokens.detect_terminal_token_boundary": 4,
        "unfinished_frames.detect_unfinished_frame_boundary": 5,
        "unfinished_predicate.detect_unfinished_predicate_boundary": 1,
        "visible_pause.detect_visible_pause_boundary": 2,
    }
    assert payload["unknown_emitted_messages"] == []
    assert payload["unreferenced_registered_messages"] == []
    assert payload["missing_functions"] == []
    assert payload["dynamic_signal_return_counts"] == {
        "_chinese_boundary_signal": 1,
        "_long_gap_chinese_boundary_signal": 1,
        "_source_boundary_signal": 1,
    }
    assert payload["signal_call_count"] == 9
    assert payload["unexpected_signal_call_count"] == 0
    assert payload["diagnostic_adapter_call_count"] == 1
    assert (
        payload["inventory_sha256"]
        == "7441664bc09f3810766292134e6c136ec46e561d795c73cf63cba9a5e262635e"
    )
    assert (
        payload["layout_sha256"]
        == "662ef657bcdc92a74bba2f2116b25597019248f492ba7e3b198b3f8f28773a4b"
    )
    assert (
        payload["call_flow_sha256"]
        == "a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38"
    )
