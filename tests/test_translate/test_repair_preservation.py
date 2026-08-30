"""Adversarial checks: a repair must not invent facts from a familiar phrase."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from subforge.core.entities import SubtitleProcessData
from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.types import TargetLanguage


@pytest.fixture
def translator():
    value = LLMTranslator(
        thread_num=1,
        batch_num=20,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        model="test",
        custom_prompt="",
        is_reflect=True,
        update_callback=None,
        use_cache=False,
    )
    yield value
    value.stop()


def pairs(sources, targets):
    source = [SubtitleProcessData(index=i, original_text=text) for i, text in enumerate(sources, 1)]
    current = [replace(item, translated_text=text) for item, text in zip(source, targets)]
    return source, current


@pytest.mark.parametrize(
    "sources,targets",
    [
        (["Do not stick the car in the car; the toy will break."], ["别把玩具车塞进车里 它会坏的"]),
        (["They removed the dine-in bits, not the engine, for 200 dollars."], ["他们花200美元拆掉了就餐配件 而不是发动机"]),
        (["It is not an up badge in this game.", "It means you lost two points."], ["游戏里这不是升级徽章", "它表示你丢了两分"]),
        (["I'm such a", "good car person."], ["我真是个", "懂车的人"]),
        (["I bought this."], ["我买了这台"]),
        (["I took this."], ["我拿了这张"]),
        (["Cobalt is older than Cruze."], ["Cobalt比科鲁兹更老"]),
        (["I wanted better for the staff.", "They needed higher wages."], ["我希望员工得到更好的", "他们需要更高的工资"]),
        (["They support us. We", "need help."], ["他们支持我们", "需要帮助"]),
        (["They chose me. I", "John Smith."], ["他们选择了我", "我是John Smith"]),
        (["My Cobalt that I bought in 2012."], ["我的Cobalt是2012年买的"]),
        (["Can we go for a short", "walk, not a drive?"], ["我们能不能走一小段", "路 而不是开车"]),
        (["I remember the weather in the last", "two years, not the coming two."], ["我记得天气 在过去的", "两年 而不是未来两年"]),
        (["I can arrange this", "like a triangle."], ["我能把它摆成", "像三角形一样"]),
        (["They kept the original plan. That's what", "I think is wrong."], ["他们保留了原方案 那正是", "我觉得不对的地方"]),
    ],
)
def test_semantic_fallback_does_not_replace_unknown_meaning(translator, sources, targets):
    source, current = pairs(sources, targets)
    result = {item.index: item for item in current}
    translator._repair_high_confidence_semantic_asr_fallbacks(source, result)
    assert [result[item.index].translated_text for item in source] == targets


@pytest.mark.parametrize(
    "sources,targets,multispeaker",
    [
        (
            ["You cannot put one unless you just sort of set", "it in your cargo area, where it costs 200 dollars", "and accepted that neither worker can lift the engine."],
            ["除非直接把它", "放进载物区 要花200美元", "还得接受两名工人都搬不动发动机的事实"],
            False,
        ),
        (
            ["I disagree about the hyperlinks as much, but this is easier to focus on.", "And so", "it does seem that people read better on this screen."],
            ["关于超链接 我有同样强烈的异议 但这个更容易让人专注", "所以", "人们似乎在这块屏幕上读得更好"],
            True,
        ),
        (
            ["Is it sort of like", "we're just back to where we started, except everyone is safer?"],
            ["是不是有点像", "我们又回到了起点 但所有人都更安全了"],
            True,
        ),
        (
            ["The lights only flash for the first 10", "or 20 seconds, not for an hour."],
            ["灯光只闪烁10秒", "或20秒 并不是一小时"],
            True,
        ),
        (
            ["It would need to be a really", "large-scale shift for people to make any money at all."],
            ["要想让人们赚到钱", "就需要一次真正大规模的转变"],
            True,
        ),
        (
            ["This is better quality than most other", "products that I bought in 2020, except the steel ones."],
            ["这个质量比大多数其他", "我在2020年买过的产品都好 但钢制品除外"],
            False,
        ),
        (
            ["The 200 dollar option is unsafe; in many ways, it is actually", "sometimes worse than the old one."],
            ["200美元的选项并不安全 从很多方面来说 它其实", "有时比旧款还糟"],
            False,
        ),
        (
            ["I think 90% of what you're going to use this truck for, like the steering rack,", "is a good thing because it is cheaper to replace, not faster."],
            ["我觉得在90%的用途里 这辆卡车的转向系统", "都是个优点 因为更换成本更低 而不是反应更快"],
            False,
        ),
        (
            ["Like it would just make this thing.", "So much more excellent than it already is; it costs 500 dollars, don't get me wrong."],
            ["这样就能让它", "在现有基础上更上一层楼 要花500美元 可别误会"],
            False,
        ),
    ],
)
def test_fluency_fallback_preserves_facts_and_qualification(sources, targets, multispeaker):
    source, current = pairs(sources, targets)
    result = LLMTranslator._deterministic_chinese_fluency_fallback(
        source, current, multispeaker=multispeaker
    )
    assert result is None or [item.translated_text for item in result] == targets


@pytest.mark.parametrize(
    "source,target",
    [
        ("Play the first video, not the second one.", "播放第一个视频 而不是第二个"),
        ("The earliest video in the archive is missing.", "档案里首个视频不见了"),
    ],
)
def test_ordinal_video_is_not_automatically_a_launch_video(translator, source, target):
    assert translator._deterministic_chinese_prose_fallback(source, target) == target
    assert not translator._chinese_prose_repair_hint(source, target)


def test_nuclear_neighbor_does_not_turn_manufacturing_plants_into_power_stations(translator):
    source, current = pairs(
        ["These plants make parts for nuclear power stations."],
        ["这些工厂为核电站生产部件"],
    )
    result = {item.index: item for item in current}
    translator._repair_contextual_nuclear_plant_terms(source, result)
    assert result[1].translated_text == current[0].translated_text


def test_an_explicit_nuclear_translation_does_not_rename_another_facility(translator):
    source, current = pairs(
        ["The nuclear plant is near a carmaker."],
        ["核电站在一家汽车工厂旁边"],
    )
    result = {item.index: item for item in current}
    translator._repair_contextual_nuclear_plant_terms(source, result)
    assert result == {item.index: item for item in current}


@pytest.mark.parametrize(
    "sources,targets",
    [
        (["I support you.", "Can we talk?"], ["我支持你", "我们能谈谈吗"]),
        (["It was the two of us who you", "could trust."], ["那是你能信任的我们", "可以信任"]),
        (["They supported me. I", "am ready."], ["他们支持了 我", "我准备好了"]),
        (["You only trusted yourself. You", "could change that."], ["你只相信 你", "你可以改变这一点"]),
        (["I've always trusted myself. I", "could change that."], ["我一直只相信 我", "我可以改变这一点"]),
    ],
)
def test_subject_cleanup_does_not_delete_an_object_or_part_of_a_pronoun(translator, sources, targets):
    source, current = pairs(sources, targets)
    result = {item.index: item for item in current}
    translator._remove_stranded_chinese_subject_tails(source, result)
    assert [result[item.index].translated_text for item in source] == targets


@pytest.mark.parametrize("boundary", ["speaker", "language", "pause", "nonadjacent"])
def test_local_repairs_preserve_turn_language_and_display_boundaries(translator, boundary):
    source, current = pairs(
        ["You insert the key. You", "can see the logo."],
        ["你插入钥匙 你", "你能看见标志"],
    )
    if boundary == "speaker":
        translator._all_speaker_by_index = {1: "A", 2: "B"}
    elif boundary == "language":
        source[0].source_language = "en"
        source[1].source_language = "ja"
    elif boundary == "pause":
        translator._gap_after_index = {1: 900}
    else:
        source[1].index = 3
        current[1].index = 3
    result = {item.index: item for item in current}
    translator._remove_stranded_chinese_subject_tails(source, result)
    assert list(result.values()) == current


def test_subject_cleanup_requires_exact_duplicate_and_is_idempotent(translator):
    source, current = pairs(
        ["You insert the key. You", "can see the logo."],
        ["你插入钥匙 你", "你们能看见标志"],
    )
    result = {item.index: item for item in current}
    translator._remove_stranded_chinese_subject_tails(source, result)
    assert result[1] == current[0]
    result[2] = replace(current[1], translated_text="你能看见标志")
    translator._remove_stranded_chinese_subject_tails(source, result)
    assert result[1].translated_text == "你插入钥匙"
    once = dict(result)
    translator._remove_stranded_chinese_subject_tails(source, result)
    assert result == once


def test_explicit_lexical_corrections_keep_all_other_words(translator):
    source, current = pairs(
        ["The nuclear plant is not closed.", "My Cobalt has 200,000 miles."],
        ["这家工厂并未关闭", "我的科鲁兹开了200,000英里"],
    )
    result = {item.index: item for item in current}
    translator._repair_contextual_nuclear_plant_terms(source, result)
    translator._repair_high_confidence_semantic_asr_fallbacks(source, result)
    assert result[1].translated_text == "这家核电站并未关闭"
    assert result[2].translated_text == "我的Cobalt开了200,000英里"


@pytest.mark.parametrize("multispeaker", [False, True])
def test_semantic_repair_still_uses_validated_model_route(translator, monkeypatch, multispeaker):
    source, current = pairs(
        ["It would need to be a really", "large-scale shift for people to make any money at all."],
        ["这需要人们作出一次", "真正大规模的转变 才有可能赚到钱"],
    )
    candidate = [replace(current[0], translated_text="这需要人们真正改变"), current[1]]
    monkeypatch.setattr(translator, "_is_multispeaker_document", lambda: multispeaker)
    monkeypatch.setattr(translator, "_should_reason_about_chinese_fluency_window", lambda *a, **k: False)
    rewrite = Mock(return_value=candidate)
    validate = Mock()
    monkeypatch.setattr(translator, "_rewrite_chinese_fluency_window", rewrite)
    monkeypatch.setattr(translator, "_validate_chinese_fluency_repair", validate)
    window, result, error = translator._repair_chinese_fluency_window_with_retries(source, current)
    assert window == source and result == candidate and error is None
    rewrite.assert_called_once()
    validate.assert_called_once_with(source, current, candidate)


def test_protected_fluency_boundary_does_not_move_a_connector():
    source, current = pairs(["This is true, but", "I disagree."], ["这是事实 但是", "我不认同"])
    assert LLMTranslator._deterministic_chinese_fluency_fallback(
        source, current, multispeaker=True, protected_boundaries=frozenset({1})
    ) is None


def test_gear_deduplication_does_not_erase_a_different_gear():
    source, current = pairs(
        ["Even here in fourth", "gear, unlike second gear."],
        ["即使挂上四挡", "二挡则不同"],
    )
    assert LLMTranslator._deterministic_chinese_fluency_fallback(source, current) is None
