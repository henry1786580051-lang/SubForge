from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.translate.base import BaseTranslator
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
