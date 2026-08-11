import json
from types import SimpleNamespace

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.translate import context as context_module
from subforge.core.translate.context import (
    MAX_TERMINOLOGY_CHARS,
    _compact_transcript,
    _document_entity_contexts,
    _document_entity_mentions,
    _format_terms,
    build_translation_context,
)
from subforge.core.translate.types import TargetLanguage


def test_compact_transcript_samples_the_middle_of_long_transcript():
    segments = [f"section-{index} " + (chr(65 + index) * 80) for index in range(9)]

    compact = _compact_transcript(segments, limit=500)

    assert len(compact) <= 500
    assert "section-0" in compact
    assert "section-4" in compact
    assert "section-8" in compact
    assert compact.count("\n...\n") == 4


def test_compact_transcript_keeps_short_transcript_unchanged():
    assert _compact_transcript([" first ", "second"], limit=100) == "first second"


def test_format_terms_bounds_long_context_payload():
    terms = [
        {"source": f"candidate-{index}", "target": f"候选人{index}", "note": "n" * 300}
        for index in range(80)
    ]

    rendered = _format_terms(terms)

    assert len(rendered) <= MAX_TERMINOLOGY_CHARS
    assert "candidate-0" in rendered
    assert "candidate-79" not in rendered


def test_format_terms_promotes_asr_corrections_and_recovers_canonical_note_target():
    terms = [
        {"source": f"basic-{index}", "target": f"基础{index}", "note": "common term"}
        for index in range(60)
    ]
    terms.append(
        {
            "source": "Grimina GR Corolla",
            "target": "Grimina GR Corolla",
            "note": "Likely ASR error for 'GRMN' (a Toyota performance variant).",
        }
    )

    rendered = _format_terms(terms)

    assert "Grimina -> GRMN (probable ASR correction derived from context)" not in rendered
    assert "Grimina GR Corolla -> GRMN GR Corolla" in rendered
    assert "basic-59" not in rendered


def test_document_entity_mentions_cover_model_evidence_outside_sampled_windows():
    mentions = _document_entity_mentions(
        [
            "Today we are driving the Toyota GR Corolla.",
            "The Lexus LBX Morizo RR uses the G16E-GTS engine.",
        ]
    )

    assert "Toyota GR Corolla" in mentions
    assert "Lexus LBX Morizo RR uses" in mentions
    assert "G16E-GTS" in mentions


def test_document_entity_mentions_surface_internal_phonetic_name_candidates():
    mentions = _document_entity_mentions(
        ["So if you want to make your Chiarco roll a little bit louder."]
    )

    assert "Chiarco" in mentions


def test_document_entity_contexts_keep_neighbors_for_uncertain_model_names():
    contexts = _document_entity_contexts(
        [
            "This is the same G16E engine used by Toyota.",
            "The Lexus LMXX Grimina or something has this engine as well.",
            "But under here there are no hood struts.",
            "You pay extra for the big Marizzo-style spoiler.",
        ]
    )

    rendered = "\n".join(contexts)
    assert "G16E engine" in rendered
    assert "Lexus LMXX Grimina or something" in rendered
    assert "Marizzo-style spoiler" in rendered


def test_document_entity_contexts_are_bounded():
    contexts = _document_entity_contexts(
        [f"Toyota Model{index} has technical detail {index}." for index in range(200)]
    )

    assert len(contexts) <= context_module.MAX_ENTITY_CONTEXTS
    assert sum(len(item) for item in contexts) <= context_module.MAX_ENTITY_CONTEXT_CHARS


def test_context_uses_bounded_native_reasoning_request_for_deepseek_v4(monkeypatch):
    calls = []
    response_text = json.dumps(
        {
            "summary": "Retirement policy",
            "terminology": [],
            "style": "Conversational",
        }
    )

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=response_text))
            ]
        )

    monkeypatch.setattr(context_module, "call_llm", fake_call_llm)
    data = ASRData([ASRDataSeg("Social Security is changing.", 0, 1000)])

    result = build_translation_context(
        data,
        model="deepseek-v4-flash",
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        use_cache=False,
    )

    assert result.summary == "Retirement policy"
    assert [call["reasoning_mode"] for call in calls] == ["enabled"]
    assert all(call["max_output_tokens"] == 8192 for call in calls)
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert "document_entity_mentions" in payload
    assert "document_entity_contexts" in payload


def test_context_retries_without_reasoning_when_native_answer_has_no_json(monkeypatch):
    calls = []
    responses = iter(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "summary": "Automotive review",
                                    "terminology": [],
                                    "style": "Conversational",
                                }
                            )
                        )
                    )
                ]
            ),
        ]
    )

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(context_module, "call_llm", fake_call_llm)
    data = ASRData([ASRDataSeg("A GR Corolla review.", 0, 1000)])

    result = build_translation_context(
        data,
        model="deepseek-v4-flash",
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        use_cache=False,
    )

    assert result.summary == "Automotive review"
    assert [call["reasoning_mode"] for call in calls] == ["enabled", "disabled"]
