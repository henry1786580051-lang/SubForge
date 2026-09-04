"""Tests for source-anchored semantic action quality signals."""

import json
from types import SimpleNamespace

from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.quality import (
    detect_document_shortened_place,
    detect_semantic_action_mismatch,
)
from subforge.core.translate.types import TargetLanguage


def _translator() -> LLMTranslator:
    return LLMTranslator(
        thread_num=1,
        batch_num=1,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        model="test",
        custom_prompt="",
        is_reflect=True,
        update_callback=None,
    )


def test_parking_brake_down_rejects_literal_spatial_action() -> None:
    signal = detect_semantic_action_mismatch(
        "I've already got my parking brake down.",
        "手刹已经放下了",
    )

    assert signal is not None
    assert signal.rule_id == "translation.semantic.control.parking_brake_state"
    assert (
        detect_semantic_action_mismatch(
            "I've already got my parking brake down.",
            "驻车制动已经解除",
        )
        is None
    )


def test_named_drive_mode_requires_control_action_and_explicit_context() -> None:
    signal = detect_semantic_action_mismatch(
        "We're going to put it back into Bentley.",
        "我们把它调回宾利该有的样子",
        previous_source="All right, let's calm this down a bit.",
        next_source="I'm going to put this back into drive.",
    )

    assert signal is not None
    assert signal.rule_id == "translation.semantic.control.named_mode"
    assert (
        detect_semantic_action_mismatch(
            "We're going to put it back into Bentley.",
            "我们把它调回Bentley模式",
            previous_source="All right, let's calm this down a bit.",
            next_source="I'm going to put this back into drive.",
        )
        is None
    )
    assert (
        detect_semantic_action_mismatch(
            "Put it back into Bentley.",
            "把它放回Bentley里",
            previous_source="Return the brochure when you're done.",
            next_source="Close the box.",
        )
        is None
    )


def test_additive_reference_flags_only_unresolved_target() -> None:
    signal = detect_semantic_action_mismatch(
        "Okay. And it's also on the shift knob.",
        "好 换挡杆上也有",
        previous_source="You would find this tritone combination.",
    )

    assert signal is not None
    assert signal.rule_id == "translation.semantic.reference.additive_object"
    assert (
        detect_semantic_action_mismatch(
            "Okay. And it's also on the shift knob.",
            "好 换挡杆上也用了金色",
            previous_source="You would find this gold combination.",
        )
        is None
    )


def test_natural_event_take_down_rejects_intentional_attack_verb() -> None:
    signal = detect_semantic_action_mismatch(
        "and took down the Macon.",
        "并击落了梅肯号",
        previous_source="Then, in 1935, bad weather struck again",
    )

    assert signal is not None
    assert signal.rule_id == "translation.semantic.causation.non_agentive_take_down"
    assert (
        detect_semantic_action_mismatch(
            "and took down the Macon.",
            "并导致梅肯号坠毁",
            previous_source="Then, in 1935, bad weather struck again",
        )
        is None
    )
    assert (
        detect_semantic_action_mismatch(
            "It took down the fighter.",
            "它击落了那架战斗机",
            previous_source="A storm was reported earlier.",
        )
        is None
    )
    assert (
        detect_semantic_action_mismatch(
            "The fighter took down the hostile aircraft.",
            "战斗机击落了敌机",
        )
        is None
    )

def test_numbered_place_requires_unique_repeated_document_evidence() -> None:
    source = "This should get us back to 17 miles."
    evidence = [
        "There is only so much we can do here on 17 Mile Drive.",
        source,
    ]

    assert detect_document_shortened_place(source, evidence) == (
        "17 Mile Drive",
        "This should get us back to 17 Mile Drive.",
    )
    assert detect_document_shortened_place(
        source,
        [*evidence, "The detour follows 17 Mile Road."],
    ) is None
    assert detect_document_shortened_place(
        "We need to drive back 17 miles.",
        evidence,
    ) is None


def test_translator_selects_semantic_repairs_and_normalizes_unique_place() -> None:
    translator = _translator()
    translator._all_source_by_index = {
        1: "There is only so much we can do here on 17 Mile Drive.",
        2: "This should get us back to 17 miles.",
        3: "I've already got my parking brake down.",
    }

    hint = translator._alignment_asr_hint(
        translator._all_source_by_index[2],
        translator._all_source_by_index[1],
        translator._all_source_by_index[3],
    )
    candidates = translator._strong_asr_semantic_candidates(
        {"2": translator._all_source_by_index[2], "3": translator._all_source_by_index[3]},
        {"2": "这样开回去大概还剩17英里", "3": "手刹已经放下了"},
    )

    assert hint["kind"] == "document_repeated_place_variant"
    assert hint["canonical"] == "17 Mile Drive"
    assert hint["normalized_source"] == "This should get us back to 17 Mile Drive."
    assert candidates == ["2", "3"]


def test_unique_repeated_place_is_exposed_as_confirmed_canonical_name(monkeypatch) -> None:
    translator = _translator()
    translator._all_source_by_index = {
        1: "There is only so much we can do here on 17 Mile Drive.",
        2: "This should get us back to 17 miles.",
    }
    captured: dict = {}

    def fake_call(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="这样应该能带我们回到17 Mile Drive")
                )
            ]
        )

    monkeypatch.setattr(
        "subforge.core.translate.llm_translator.call_llm",
        fake_call,
    )

    result = translator._translate_alignment_item(
        translator._all_source_by_index[2],
        source_key="2",
        previous_source=translator._all_source_by_index[1],
        allow_reasoning=False,
    )

    payload = json.loads(captured["messages"][1]["content"])
    assert result == "这样应该能带我们回到17 Mile Drive"
    assert payload["current_source"] == "This should get us back to 17 Mile Drive."
    assert payload["confirmed_canonical_name"] == "17 Mile Drive"
