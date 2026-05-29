from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.subtitle.resegment import resegment_subtitles


def test_resegment_keeps_bilingual_parts_in_sequence():
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

    # 关键：技术参数不应被拆散
    all_zh = " ".join(seg.translated_text for seg in result.segments)
    assert "260马力" in all_zh, "260马力 should not be split"
    assert "277磅" in all_zh or "277磅-英尺" in all_zh, "277磅-英尺 should not be split"
    assert "FA24" in all_zh, "FA24 should not be split"

    # 每段不超过字符限制
    for seg in result.segments:
        assert len(seg.translated_text) <= 22, f"Chinese too long: {seg.translated_text!r}"
        if seg.text:
            assert len(seg.text) <= 60, f"English too long: {seg.text!r}"


def test_resegment_does_not_duplicate_one_side_when_counts_differ():
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

    assert len(result.segments) == 2
    assert [seg.translated_text for seg in result.segments] == [
        "当然还有斯巴鲁招牌的",
        "对称式全时四驱系统",
    ]
    assert result.segments[0].text != result.segments[1].text
    assert all(seg.translated_text != "统" for seg in result.segments)
