from dataclasses import FrozenInstanceError

import pytest

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.translate.base import BaseTranslator
from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.quality import CueFeatures, build_translation_session
from subforge.core.translate.types import TargetLanguage


def test_cue_features_are_deterministic_and_preserve_identifiers() -> None:
    source = "  The F-150 uses 3.5L V6 power.  "

    first = CueFeatures.from_source(1, source)
    second = CueFeatures.from_source(1, source)

    assert first == second
    assert first.normalized_source == "The F-150 uses 3.5L V6 power."
    assert first.numeric_tokens == ("150", "3.5L")
    assert "F-150" in first.identifier_tokens
    assert "V6" in first.identifier_tokens
    assert first.scripts == frozenset({"latin"})
    assert first.ends_terminal_punctuation is True


def test_translation_session_copies_timing_language_and_speaker_context() -> None:
    source = ASRData(
        [
            ASRDataSeg(
                "Good morning.",
                100,
                900,
                speaker_id="host",
                language_code="en",
                timing_source="forced_alignment",
                timestamp_granularity="sentence",
            ),
            ASRDataSeg(
                "おはようございます。",
                1200,
                2100,
                speaker_id="guest",
                language_code="ja",
                timing_source="forced_alignment",
                timestamp_granularity="sentence",
            ),
        ]
    )

    session = build_translation_session(
        source,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        model="deepseek-v4-flash",
    )

    assert session.speaker_count == 2
    assert session.is_multispeaker is True
    assert session.cue(1).speaker == "S1"
    assert session.cue(1).gap_after_ms == 300
    assert session.cue(2).speaker == "S2"
    assert session.cue(2).gap_before_ms == 300
    assert session.feature(2).scripts == frozenset({"kana"})
    assert session.cue(3) is None
    assert source.segments[0].speaker_id == "host"


def test_translation_session_is_immutable() -> None:
    session = build_translation_session(
        ASRData([ASRDataSeg("Hello.", 0, 500)]),
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        model="deepseek-v4-flash",
    )

    with pytest.raises(FrozenInstanceError):
        session.model = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        session.cues[0].source = "Changed"  # type: ignore[misc]


def test_llm_translator_builds_session_without_changing_legacy_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ASRData(
        [
            ASRDataSeg("Hello.", 0, 500, speaker_id="host", language_code="en"),
            ASRDataSeg("Welcome.", 700, 1200, speaker_id="guest", language_code="en"),
        ]
    )
    observed: dict[str, object] = {}

    def observe_legacy_context(self: BaseTranslator, subtitle_data: ASRData) -> ASRData:
        translator = self
        observed["session"] = translator._translation_session  # type: ignore[attr-defined]
        observed["source"] = dict(translator._all_source_by_index)  # type: ignore[attr-defined]
        observed["speaker"] = dict(translator._all_speaker_by_index)  # type: ignore[attr-defined]
        observed["language"] = dict(translator._all_language_by_index)  # type: ignore[attr-defined]
        observed["gaps"] = dict(translator._gap_after_index)  # type: ignore[attr-defined]
        return subtitle_data

    monkeypatch.setattr(BaseTranslator, "translate_subtitle", observe_legacy_context)
    translator = LLMTranslator(
        thread_num=1,
        batch_num=1,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        model="deepseek-v4-flash",
        custom_prompt="",
        is_reflect=True,
        update_callback=None,
        llm_client=object(),
    )

    result = translator.translate_subtitle(source)

    assert result is source
    assert observed["source"] == {1: "Hello.", 2: "Welcome."}
    assert observed["speaker"] == {1: "S1", 2: "S2"}
    assert observed["language"] == {1: "en", 2: "en"}
    assert observed["gaps"] == {1: 200}
    session = observed["session"]
    assert session is not None
    assert session.cues[0].source == "Hello."  # type: ignore[union-attr]
    assert translator._translation_session is None
    assert translator._all_source_by_index == {}
