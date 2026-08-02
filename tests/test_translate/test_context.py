import json
from types import SimpleNamespace

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.translate import context as context_module
from subforge.core.translate.context import (
    MAX_TERMINOLOGY_CHARS,
    _compact_transcript,
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


def test_context_uses_bounded_non_thinking_request(monkeypatch):
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
    assert [call["reasoning_mode"] for call in calls] == ["disabled"]
    assert all(call["max_output_tokens"] == 4096 for call in calls)
