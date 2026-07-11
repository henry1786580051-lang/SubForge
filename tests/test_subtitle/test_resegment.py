import pytest

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.subtitle.resegment import resegment_subtitles


@pytest.mark.parametrize(
    ("source", "translation", "part_count"),
    [
        (
            "in this interior, it's a pretty nice overall well-thought-out package.",
            "整个设计考虑得非常周到",
            2,
        ),
        (
            "I don't really think there's any point in going into sport, "
            "especially since we're trying to be efficient.",
            "我觉得没必要切到运动模式 毕竟咱们现在是奔着省油去的",
            3,
        ),
        (
            "A lot of these magazine companies get on these midsize sedans for "
            "needing to have razor sharp handling.",
            "很多车评媒体老揪着中型轿车 非要人家操控像刀切一样犀利",
            3,
        ),
    ],
)
def test_resegment_does_not_recreate_known_bilingual_clause_mismatches(
    source,
    translation,
    part_count,
):
    data = ASRData([ASRDataSeg(source, 0, part_count * 2000, translation)])

    result = resegment_subtitles(data, max_chars_en=50, max_chars_cjk=18)

    assert len(result.segments) == 1
    assert result.segments[0].text == source
    assert result.segments[0].translated_text == translation


def test_resegment_keeps_long_bilingual_segment_locked():
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
    assert (result.segments[0].start_time, result.segments[0].end_time) == (0, 6000)


def test_resegment_keeps_short_bilingual_when_only_translation_is_long():
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


def test_resegment_keeps_long_duration_bilingual_segment_locked():
    data = ASRData(
        [
            ASRDataSeg(
                text="This is the all-wheel drive, so it retains the six-speed automatic transmission and the 3.5-liter V6 engine.",
                start_time=0,
                end_time=9000,
                translated_text="这是四驱版，所以保留了六速自动变速箱和3.5升V6发动机。",
            )
        ]
    )

    result = resegment_subtitles(data, max_chars_en=50, max_chars_cjk=18)

    assert len(result.segments) == 1
    assert result.segments[0].text == data.segments[0].text
    assert result.segments[0].translated_text == data.segments[0].translated_text


def test_resegment_keeps_short_bilingual_sentence_unchanged():
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

    result = resegment_subtitles(data, max_chars_en=60, max_chars_cjk=30)

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


def test_resegment_does_not_create_numeric_only_translation_fragment():
    data = ASRData(
        [
            ASRDataSeg(
                text="But I don't really feel the need to be going 160 up the left lane in this car.",
                start_time=0,
                end_time=5000,
                translated_text="但我觉得没必要在左车道飙到160",
            )
        ]
    )

    result = resegment_subtitles(data, max_chars_en=42, max_chars_cjk=16)

    assert len(result.segments) == 1
    assert result.segments[0].translated_text == data.segments[0].translated_text


def test_resegment_does_not_create_latin_only_translation_fragment():
    data = ASRData(
        [
            ASRDataSeg(
                text="That green frozen Tampa Bay green individual M3 Competition.",
                start_time=0,
                end_time=5000,
                translated_text="那款冰冻坦帕湾绿个性化M3 Competition",
            )
        ]
    )

    result = resegment_subtitles(data, max_chars_en=42, max_chars_cjk=16)

    assert len(result.segments) == 1
    assert result.segments[0].translated_text == data.segments[0].translated_text
