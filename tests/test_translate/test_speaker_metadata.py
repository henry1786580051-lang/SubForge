import json
from types import SimpleNamespace

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.translate.base import BaseTranslator
from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.types import TargetLanguage


class _CapturingTranslator(BaseTranslator):
    def __init__(self):
        super().__init__(
            thread_num=1,
            batch_num=10,
            target_language=TargetLanguage.SIMPLIFIED_CHINESE,
            update_callback=None,
            use_cache=False,
        )
        self.source_payload: list[str] = []

    def _translate_chunk(self, subtitle_chunk):
        self.source_payload.extend(item.original_text for item in subtitle_chunk)
        for item in subtitle_chunk:
            item.translated_text = f"译文:{item.original_text}"
        return subtitle_chunk


def test_speaker_metadata_is_not_sent_as_translation_text():
    data = ASRData(
        [
            ASRDataSeg("How is it?", 0, 1000, speaker_id="Speaker 1"),
            ASRDataSeg("Very quick.", 1000, 2000, speaker_id="Speaker 2"),
        ]
    )
    translator = _CapturingTranslator()

    result = translator.translate_subtitle(data)

    assert translator.source_payload == ["How is it?", "Very quick."]
    assert all("Speaker" not in text for text in translator.source_payload)
    assert [segment.speaker_id for segment in result.segments] == [
        "Speaker 1",
        "Speaker 2",
    ]


def test_llm_translation_receives_anonymous_dialogue_metadata(monkeypatch):
    captured = {}

    def fake_call_llm(**kwargs):
        captured["messages"] = kwargs["messages"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"1": "你觉得怎么样？", "2": "非常快。"}, ensure_ascii=False)
                    )
                )
            ]
        )

    monkeypatch.setattr("subforge.core.translate.llm_translator.call_llm", fake_call_llm)
    data = ASRData(
        [
            ASRDataSeg("How is it?", 0, 1000, speaker_id="Speaker 4"),
            ASRDataSeg("Very quick.", 1000, 2000, speaker_id="Speaker 9"),
        ]
    )
    translator = LLMTranslator(
        thread_num=1,
        batch_num=10,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        model="MiniMax-M3",
        custom_prompt="",
        is_reflect=False,
        update_callback=None,
        use_cache=False,
    )

    result = translator.translate_subtitle(data)

    user_payload = json.loads(captured["messages"][1]["content"])
    assert user_payload["current_subtitles"] == {
        "1": {"speaker": "S1", "source": "How is it?"},
        "2": {"speaker": "S2", "source": "Very quick."},
    }
    assert "Never translate, repeat, rename, or output speaker labels" in captured["messages"][0]["content"]
    assert [segment.translated_text for segment in result.segments] == ["你觉得怎么样？", "非常快。"]
    assert [segment.speaker_id for segment in result.segments] == ["Speaker 4", "Speaker 9"]


def test_single_speaker_translation_keeps_legacy_text_payload(monkeypatch):
    captured = {}

    def fake_call_llm(**kwargs):
        captured["messages"] = kwargs["messages"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"1":"你好"}'))]
        )

    monkeypatch.setattr("subforge.core.translate.llm_translator.call_llm", fake_call_llm)
    translator = LLMTranslator(
        thread_num=1,
        batch_num=10,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        model="MiniMax-M3",
        custom_prompt="",
        is_reflect=False,
        update_callback=None,
        use_cache=False,
    )

    translator.translate_subtitle(ASRData([ASRDataSeg("Hello", 0, 1000)]))

    user_payload = json.loads(captured["messages"][1]["content"])
    assert user_payload["current_subtitles"] == {"1": "Hello"}
    assert "<dialogue_metadata>" not in captured["messages"][0]["content"]


def test_dialogue_translation_rejects_speaker_labels_in_output():
    translator = LLMTranslator(
        thread_num=1,
        batch_num=10,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        model="MiniMax-M3",
        custom_prompt="",
        is_reflect=False,
        update_callback=None,
        use_cache=False,
    )
    translator._all_speaker_by_index = {1: "S1", 2: "S2"}

    valid, error = translator._validate_llm_response(
        {"1": "[S1] 你觉得怎么样？", "2": "Speaker 2：非常快"},
        {"1": "How is it?", "2": "Very quick."},
    )

    translator.stop()
    assert valid is False
    assert "Speaker identifiers are metadata only" in error
