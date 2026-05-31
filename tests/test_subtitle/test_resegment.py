from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.subtitle.resegment import resegment_subtitles


def test_resegment_preserves_bilingual_segment_alignment():
    data = ASRData(
        [
            ASRDataSeg(
                text="Under the hood, the 2.4-liter FA24 boxer engine with 260 horsepower and 277 pound-feet of torque.",
                start_time=0,
                end_time=6000,
                translated_text="引擎盖下是2.4升FA24水平对置发动机，260马力，277磅-英尺扭矩。",
            )
        ]
    )

    result = resegment_subtitles(data, max_chars_en=50, max_chars_cjk=18)

    assert len(result.segments) == 1
    assert result.segments[0].text == data.segments[0].text
    assert result.segments[0].translated_text == data.segments[0].translated_text


def test_resegment_does_not_split_bilingual_when_counts_differ():
    data = ASRData(
        [
            ASRDataSeg(
                text="and of course, this has symmetrical all-wheel drive.",
                start_time=0,
                end_time=4000,
                translated_text="当然还有斯巴鲁招牌的对称式全时四驱系统",
            )
        ]
    )

    result = resegment_subtitles(data, max_chars_en=50, max_chars_cjk=18)

    assert len(result.segments) == 1
    assert result.segments[0].text == data.segments[0].text
    assert result.segments[0].translated_text == data.segments[0].translated_text


def test_resegment_keeps_translated_sentence_with_original_sentence():
    data = ASRData(
        [
            ASRDataSeg(
                text="Today we are driving the all new 2026 Lexus ES 350h.",
                start_time=0,
                end_time=4000,
                translated_text="今天来看看2026款雷克萨斯ES 350h。",
            )
        ]
    )

    result = resegment_subtitles(data, max_chars_en=30, max_chars_cjk=12)

    assert len(result.segments) == 1
    assert result.segments[0].text == "Today we are driving the all new 2026 Lexus ES 350h."
    assert result.segments[0].translated_text == "今天来看看2026款雷克萨斯ES 350h。"


def test_resegment_still_splits_monolingual_long_segment():
    data = ASRData(
        [
            ASRDataSeg(
                text="Today we are driving the all new 2026 Lexus ES 350h.",
                start_time=0,
                end_time=4000,
            )
        ]
    )

    result = resegment_subtitles(data, max_chars_en=30, max_chars_cjk=12)

    assert len(result.segments) > 1
    assert "".join(seg.text.replace(" ", "") for seg in result.segments) == data.segments[0].text.replace(" ", "")
