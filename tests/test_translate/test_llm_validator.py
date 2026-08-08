"""Tests for LLM translation response validation."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from subforge.core.entities import SubtitleProcessData
from subforge.core.prompts import get_prompt
from subforge.core.translate.context import TranslationContext
from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.types import TargetLanguage


def _make_translator(is_reflect=False):
    """Create a translator instance for testing."""
    return LLMTranslator(
        thread_num=1,
        batch_num=1,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        model="test",
        custom_prompt="",
        is_reflect=is_reflect,
        update_callback=None,
    )


def _make_minimax_reflect_translator():
    translator = _make_translator(is_reflect=True)
    translator.model = "MiniMax-M3"
    return translator


def _llm_response(payload):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))
            )
        ]
    )


def _text_response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class TestValidateLLmResponse:
    """Test _validate_llm_response with standard and reflect modes."""

    def test_reflect_prompt_prioritizes_fidelity_and_boundary_ownership(self):
        prompt = get_prompt(
            "translate/reflect",
            target_language="简体中文",
            custom_prompt="",
        )

        assert "locked semantic boundary" in prompt
        assert "Do not convert currencies" in prompt
        assert "Do not invent context" in prompt
        assert "appears exactly once under the correct key" in prompt
        assert "semantically incoherent" in prompt
        assert "currency symbols" in prompt
        assert "semantic plausibility" in prompt

    def test_translation_prompts_do_not_encourage_unrequested_conversions(self):
        for name in ("translate/standard", "translate/reflect"):
            prompt = get_prompt(
                name,
                target_language="简体中文",
                custom_prompt="",
            )
            assert "Do not convert currencies" in prompt
            assert "miles → kilometers" not in prompt
            assert "dollars → local currency" not in prompt

    def test_minimax_reflect_chunk_always_runs_independent_alignment_audit(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        chunk = [
            SubtitleProcessData(index=23, original_text="And how could an alliance"),
            SubtitleProcessData(
                index=24,
                original_text="between white-collar and blue-collar workers",
            ),
            SubtitleProcessData(index=25, original_text="transform American politics?"),
        ]
        initial = {
            "23": {"native_translation": "而这样的联盟会如何"},
            "24": {"native_translation": "连接白领与蓝领工人"},
            "25": {"native_translation": "改变美国政治？"},
        }
        audited = {
            "23": "而这样的联盟会如何",
            "24": "连接白领与蓝领工人",
            "25": "改变美国政治？",
        }
        calls = []

        monkeypatch.setattr(translator, "_agent_loop", lambda *_args: initial)

        def fake_audit(source, translated, **kwargs):
            calls.append((source, translated, kwargs))
            return audited

        monkeypatch.setattr(translator, "_audit_reflective_alignment", fake_audit)

        result = translator._translate_chunk(chunk)

        assert len(calls) == 1
        assert [item.translated_text for item in result] == list(audited.values())

    def test_valid_standard_response(self):
        t = _make_translator()
        resp = {"0": "你好", "1": "世界"}
        inp = {"0": "hello", "1": "world"}
        ok, msg = t._validate_llm_response(resp, inp)
        assert ok is True
        assert msg == ""

    def test_not_a_dict(self):
        t = _make_translator()
        ok, msg = t._validate_llm_response("not a dict", {"0": "hello"})
        assert ok is False
        assert "must be a dict" in msg

    def test_not_a_dict_list(self):
        t = _make_translator()
        ok, msg = t._validate_llm_response(["a", "b"], {"0": "hello"})
        assert ok is False
        assert "must be a dict" in msg

    def test_missing_keys(self):
        t = _make_translator()
        resp = {"0": "你好"}
        inp = {"0": "hello", "1": "world"}
        ok, msg = t._validate_llm_response(resp, inp)
        assert ok is False
        assert "Missing keys" in msg
        assert "1" in msg

    def test_extra_keys(self):
        t = _make_translator()
        resp = {"0": "你好", "1": "世界", "99": "多余的"}
        inp = {"0": "hello", "1": "world"}
        ok, msg = t._validate_llm_response(resp, inp)
        assert ok is False
        assert "Extra keys" in msg
        assert "99" in msg

    def test_missing_and_extra_keys(self):
        t = _make_translator()
        resp = {"0": "你好", "99": "多余"}
        inp = {"0": "hello", "1": "world"}
        ok, msg = t._validate_llm_response(resp, inp)
        assert ok is False
        assert "Missing keys" in msg
        assert "Extra keys" in msg

    def test_empty_response(self):
        t = _make_translator()
        ok, msg = t._validate_llm_response({}, {"0": "hello"})
        assert ok is False
        assert "Missing keys" in msg

    def test_cjk_target_rejects_all_english(self):
        """100% English output for CJK target should fail."""
        t = _make_translator()
        resp = {"0": "hello", "1": "world", "2": "foo", "3": "bar"}
        inp = {"0": "a", "1": "b", "2": "c", "3": "d"}
        ok, msg = t._validate_llm_response(resp, inp)
        assert ok is False
        assert "still in the source language" in msg

    def test_cjk_target_rejects_any_full_english_entry(self):
        """A CJK batch cannot contain a fully untranslated sentence."""
        t = _make_translator()
        resp = {
            "0": "你好",
            "1": "世界",
            "2": "美好的",
            "3": "天气",
            "4": "今天",
            "5": "不错",
            "6": "OK",
        }
        inp = {str(i): f"This is sentence {i}" for i in range(7)}
        ok, msg = t._validate_llm_response(resp, inp)
        assert ok is False
        assert "Untranslated keys" in msg

    def test_cjk_target_allows_brand_model_only_latin_text(self):
        """Short model-name captions may legitimately remain in Latin script."""
        t = _make_translator()
        resp = {"0": "BMW M2 CS", "1": "你好"}
        inp = {"0": "BMW M2 CS", "1": "hello"}
        ok, msg = t._validate_llm_response(resp, inp)
        assert ok is True
        assert msg == ""

    @pytest.mark.parametrize("source", ["Area.", "Okay", "Welcome"])
    def test_cjk_target_rejects_untranslated_titlecase_words(self, source):
        t = _make_translator()

        ok, msg = t._validate_llm_response({"0": source}, {"0": source})

        assert ok is False
        assert "Untranslated keys" in msg

    def test_cjk_target_rejects_threshold_boundary_english(self):
        """The old percentage threshold must not allow untranslated full lines."""
        t = _make_translator()
        resp = {"0": "你好", "1": "世界", "2": "好的", "3": "hello"}
        inp = {"0": "a", "1": "b", "2": "c", "3": "d"}
        ok, msg = t._validate_llm_response(resp, inp)
        assert ok is False
        assert "Untranslated keys" in msg

    def test_chinese_target_rejects_unchanged_korean_source(self):
        t = _make_translator()
        resp = {"0": "교황이 일시적인 호흡 곤란을 겪었습니다"}
        inp = {"0": "교황이 일시적인 호흡 곤란을 겪었습니다"}

        ok, msg = t._validate_llm_response(resp, inp)

        assert ok is False
        assert "Untranslated keys" in msg

    def test_chinese_target_accepts_chinese_translation_of_korean(self):
        t = _make_translator()
        resp = {"0": "教皇一度出现呼吸困难"}
        inp = {"0": "교황이 일시적인 호흡 곤란을 겪었습니다"}

        ok, msg = t._validate_llm_response(resp, inp)

        assert ok is True
        assert msg == ""

    def test_rejects_placeholder_merge_translation(self):
        t = _make_translator()
        resp = {
            "0": "这句正常翻译",
            "1": "(此句内容在最终版本中与上一句合并)",
        }
        inp = {
            "0": "This one is translated.",
            "1": "of 3 series and 4 series",
        }

        ok, msg = t._validate_llm_response(resp, inp)

        assert ok is False
        assert "Placeholder translations" in msg
        assert "1" in msg

    def test_rejects_translation_meta_annotation(self):
        t = _make_translator()
        resp = {"0": "是我在Couth and Mayo...（应为Mayor）的朋友"}
        inp = {"0": "My friend from Couth and Mayor"}

        ok, msg = t._validate_llm_response(resp, inp)

        assert ok is False
        assert "Placeholder translations" in msg

    def test_fatal_provider_error_opens_circuit_without_single_item_fallback(self, monkeypatch):
        translator = _make_translator()
        fallback_called = False

        class PaymentRequiredError(Exception):
            status_code = 402

        def fail_agent_loop(*_args, **_kwargs):
            raise PaymentRequiredError("Insufficient Balance")

        def fail_if_fallback_called(_chunk):
            nonlocal fallback_called
            fallback_called = True
            return []

        monkeypatch.setattr(translator, "_agent_loop", fail_agent_loop)
        monkeypatch.setattr(translator, "_translate_chunk_single", fail_if_fallback_called)
        chunk = [SubtitleProcessData(index=1, original_text="Hello")]

        with pytest.raises(RuntimeError, match="HTTP 402"):
            translator._translate_chunk(chunk)
        with pytest.raises(RuntimeError, match="HTTP 402"):
            translator._translate_chunk(chunk)

        assert fallback_called is False

    def test_reflect_failure_recovers_as_locked_batch(self, monkeypatch):
        translator = _make_translator(is_reflect=True)
        translator._all_source_by_index = {
            1: "That's fine. That's actually better than",
            2: "I felt in many similar cars.",
        }
        chunk = [
            SubtitleProcessData(index=1, original_text=translator._all_source_by_index[1]),
            SubtitleProcessData(index=2, original_text=translator._all_source_by_index[2]),
        ]
        calls = []

        def fake_call_llm(**kwargs):
            calls.append(kwargs)
            return _llm_response({"1": "这没问题 其实表现还更好", "2": "很多同类车型都是如此"})

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            fake_call_llm,
        )

        recovered = translator._translate_chunk_single(chunk)

        assert [item.translated_text for item in recovered] == [
            "这没问题 其实表现还更好",
            "很多同类车型都是如此",
        ]
        assert len(calls) == 1
        assert "boundary-safe recovery pass" in calls[0]["messages"][0]["content"]

    def test_locked_batch_fatal_error_does_not_fall_back_to_single_items(self, monkeypatch):
        translator = _make_translator(is_reflect=True)
        chunk = [
            SubtitleProcessData(index=1, original_text="First subtitle."),
            SubtitleProcessData(index=2, original_text="Second subtitle."),
        ]
        calls = 0

        class PaymentRequiredError(Exception):
            status_code = 402

        def fail_call_llm(**_kwargs):
            nonlocal calls
            calls += 1
            raise PaymentRequiredError("Insufficient Balance")

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            fail_call_llm,
        )

        with pytest.raises(RuntimeError, match="HTTP 402"):
            translator._translate_chunk_single(chunk)

        assert calls == 1

    @pytest.mark.parametrize(
        "placeholder",
        [
            "（本句已并入前一句）",
            "此句无需翻译，见上",
            "内容同上",
            "（合并至上一条）",
            "本句已经合并到上条字幕",
            "此句在最终字幕中省略",
        ],
    )
    def test_rejects_additional_placeholder_variants(self, placeholder):
        t = _make_translator()
        ok, msg = t._validate_llm_response(
            {"0": placeholder},
            {"0": "A complete sentence."},
        )
        assert ok is False
        assert "Placeholder" in msg

    @pytest.mark.parametrize(
        "translation",
        [
            "我觉得很多厂家都忽略了这一点。",
            "这项战略调整非常重要。",
            "下面先做一个简略介绍。",
        ],
    )
    def test_placeholder_validation_allows_normal_words_containing_lue(
        self,
        translation,
    ):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"0": translation},
            {"0": "This is a complete source sentence."},
        )

        assert ok is True
        assert msg == ""

    def test_final_validation_rejects_untranslated_and_placeholder_items(self):
        t = _make_translator()
        source = [SubtitleProcessData(index=1, original_text="This is a complete sentence.")]

        with pytest.raises(RuntimeError, match="placeholder translations"):
            t._validate_translated_list(
                source,
                [
                    SubtitleProcessData(
                        index=1,
                        original_text=source[0].original_text,
                        translated_text="本句已并入前一句",
                    )
                ],
            )

        with pytest.raises(RuntimeError, match="untranslated indices"):
            t._validate_translated_list(
                source,
                [
                    SubtitleProcessData(
                        index=1,
                        original_text=source[0].original_text,
                        translated_text="This is a complete sentence.",
                    )
                ],
            )

    def test_chunk_validation_rejects_placeholder_before_progress_callback(self, monkeypatch):
        progress = []
        t = _make_translator()
        t.use_cache = False
        t.update_callback = progress.append
        chunk = [SubtitleProcessData(index=1, original_text="A complete sentence.")]

        monkeypatch.setattr(
            t,
            "_translate_chunk",
            lambda _chunk: [
                SubtitleProcessData(
                    index=1,
                    original_text="A complete sentence.",
                    translated_text="（合并至上一条）",
                )
            ],
        )

        with pytest.raises(RuntimeError, match="placeholder translations"):
            t._safe_translate_chunk(chunk)

        assert progress == []

    def test_reflect_mode_valid(self):
        t = _make_translator(is_reflect=True)
        resp = {
            "0": {"native_translation": "你好", "initial_translation": "hi"},
            "1": {"native_translation": "世界", "initial_translation": "globe"},
        }
        inp = {"0": "hello", "1": "world"}
        ok, msg = t._validate_llm_response(resp, inp)
        assert ok is True

    def test_reflect_mode_missing_native_translation(self):
        t = _make_translator(is_reflect=True)
        resp = {
            "0": {"initial_translation": "你好"},
            "1": {"initial_translation": "世界"},
        }
        inp = {"0": "hello", "1": "world"}
        ok, msg = t._validate_llm_response(resp, inp)
        assert ok is False
        assert "native_translation" in msg

    def test_reflect_mode_value_not_dict(self):
        t = _make_translator(is_reflect=True)
        resp = {"0": "你好", "1": "世界"}
        inp = {"0": "hello", "1": "world"}
        ok, msg = t._validate_llm_response(resp, inp)
        assert ok is False
        assert "must be a dict" in msg

    def test_reflect_mode_extra_keys(self):
        t = _make_translator(is_reflect=True)
        resp = {
            "0": {"native_translation": "你好"},
            "1": {"native_translation": "世界"},
            "99": {"native_translation": "多余"},
        }
        inp = {"0": "hello", "1": "world"}
        ok, msg = t._validate_llm_response(resp, inp)
        assert ok is False
        assert "Extra keys" in msg

    def test_numeric_string_keys(self):
        t = _make_translator()
        resp = {"0": "你好", "1": "世界", "2": "好的"}
        inp = {"0": "hello", "1": "world", "2": "ok"}
        ok, msg = t._validate_llm_response(resp, inp)
        assert ok is True

    def test_extract_text_from_reflect_dict(self):
        """Standard mode should extract text from reflect-style dicts."""
        t = _make_translator(is_reflect=False)
        # Even in standard mode, if values are dicts, _extract_text handles it
        resp = {"0": {"native_translation": "你好"}, "1": {"native_translation": "世界"}}
        inp = {"0": "hello", "1": "world"}
        ok, _ = t._validate_llm_response(resp, inp)
        # Should pass because keys match and CJK content is present
        assert ok is True

    def test_rejects_reflection_that_moves_content_between_keys(self):
        t = _make_translator(is_reflect=True)
        resp = {
            "1": {
                "native_translation": "先说到这里",
                "reflection": "把后半句合并到下一条字幕会更自然",
            },
            "2": {
                "native_translation": "下一句",
                "reflection": "保持原意",
            },
        }

        ok, msg = t._validate_llm_response(resp, {"1": "first", "2": "second"})

        assert ok is False
        assert "redistribute meaning" in msg

    def test_rejects_numeric_fact_duplicated_into_neighbor_key(self):
        t = _make_translator()
        resp = {"1": "还额外配了20英寸轮毂", "2": "配的是20英寸轮毂"}
        source = {"1": "But anyway,", "2": "It has the 20-inch wheels."}

        ok, msg = t._validate_llm_response(resp, source)

        assert ok is False
        assert "Cross-key duplicates" in msg

    def test_rejects_translated_small_quantity_anticipated_from_next_key(self):
        t = _make_translator()
        resp = {
            "425": "只要手离开方向盘不到两秒 它就会提醒你",
            "426": "超过两秒左右 它就会开始报警",
        }
        source = {
            "425": "It gets annoyed if your hands leave the wheel for, I don't know,",
            "426": "anything longer than two seconds or so.",
        }

        ok, msg = t._validate_llm_response(resp, source)

        assert ok is False
        assert "Repeated quantities" in msg

    def test_allows_same_quantity_when_both_source_keys_repeat_it(self):
        t = _make_translator()
        resp = {"1": "等两秒", "2": "再等两秒"}
        source = {"1": "Wait two seconds.", "2": "Wait another two seconds."}

        ok, msg = t._validate_llm_response(resp, source)

        assert ok is True
        assert msg == ""

    def test_rejects_condition_anticipated_from_following_key(self):
        t = _make_translator()
        resp = {
            "64": "这里是供消防员在必要时切断混动线缆的位置",
            "65": "如果需要 就在这块面板后面",
        }
        source = {
            "64": "This is where a firefighter would cut the hybrid cables",
            "65": "if they needed to, behind this panel.",
        }

        ok, msg = t._validate_llm_response(resp, source)

        assert ok is False
        assert "Anticipated conditions" in msg

    def test_rejects_i_mean_marker_moved_to_following_key(self):
        translator = _make_translator()

        ok, message = translator._validate_llm_response(
            {
                "559": "我不明白为什么没更多人开这种舒适的老式美国车",
                "560": "我的意思是 与其花八万美元买一辆新的Expedition",
            },
            {
                "559": "I don't see why more people don't drive old American cars. I mean,",
                "560": "instead of spending eighty thousand dollars on a new Expedition.",
            },
        )

        assert ok is False
        assert "I mean" in message
        assert "560" in message

    def test_rejects_if_condition_omitted_from_its_own_key(self):
        t = _make_translator()
        resp = {"65": "就在这块面板后面"}
        source = {"65": "If they needed to, behind this panel."}

        ok, msg = t._validate_llm_response(resp, source)

        assert ok is False
        assert "Missing conditions" in msg

    def test_rejects_repeated_chinese_conclusion_without_speaker_metadata(self):
        t = _make_translator()
        resp = {"135": "成年人坐这里完全没问题", "136": "完全没问题"}
        source = {
            "135": "You could put an adult in this middle seat",
            "136": "and you would be completely fine.",
        }

        ok, msg = t._validate_llm_response(resp, source)

        assert ok is False
        assert "Repeated endings" in msg

    def test_numeric_boundary_check_does_not_match_larger_number_or_model(self):
        t = _make_translator()
        resp = {
            "1": "综合油耗25 mpg",
            "2": "动力是255马力",
            "3": "这台是RS3",
        }
        source = {
            "1": "It gets 25 mpg.",
            "2": "It has 255 horsepower.",
            "3": "This is the RS3.",
        }

        ok, msg = t._validate_llm_response(resp, source)

        assert ok is True
        assert msg == ""

    @pytest.mark.parametrize(
        ("source", "translation"),
        [
            ("Five to 10 years would be quick.", "五到十年就算很快了"),
            ("It was completed after 15 years.", "它历时十五年才建成"),
            ("It happened in 2026.", "这件事发生在二零二六年"),
        ],
    )
    def test_preserved_numbers_accept_exact_chinese_numerals(self, source, translation):
        t = _make_translator()

        ok, msg = t._validate_llm_response({"1": translation}, {"1": source})

        assert ok is True
        assert msg == ""

    def test_preserved_numbers_rejects_a_different_chinese_number(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"1": "它历时十年才建成"},
            {"1": "It was completed after 15 years."},
        )

        assert ok is False
        assert "1:15" in msg

    def test_compound_model_separator_variants_are_preserved_and_not_leaked(self):
        t = _make_translator()
        source = {
            "212": "Let's pop the hood and show you the 392.",
            "218": "It's back for 26 in this new RT392 trim.",
        }
        response = {
            "212": "打开引擎盖 看看这台392",
            "218": "它在26款以全新的R/T 392配置回归",
        }

        ok, msg = t._validate_llm_response(response, source)

        assert ok is True
        assert msg == ""

    def test_rejects_model_token_leak_even_when_source_owns_token_in_multiple_keys(self):
        t = _make_translator()
        source = {
            "1": "One last look at this 760i.",
            "2": "I hope that gives you a good idea of",
            "3": "what it is like to drive this 760i.",
        }
        response = {
            "1": "最后看一眼这台760i",
            "2": "希望大家了解这台760i开起来怎么样",
            "3": "这台760i的驾驶感受",
        }

        ok, msg = t._validate_llm_response(response, source)

        assert ok is False
        assert "2:760i" in msg

    def test_rejects_adjacent_target_duplication_not_present_in_source(self):
        t = _make_translator()
        source = {
            "402": "and that's the highest praise",
            "403": "you can give something in this class.",
        }
        response = {
            "402": "这是我能给出的最高评价",
            "403": "这是同级别里我能给出的最高评价",
        }

        ok, msg = t._validate_llm_response(response, source)

        assert ok is False
        assert "Suspicious pairs" in msg

    def test_rejects_long_neighbor_phrase_anticipation_below_ratio_threshold(self):
        t = _make_translator()
        source = {
            "398": "controller right here, but we can change",
            "399": "what color we want this to be in here. It's quite simple.",
        }
        response = {
            "398": "就在这儿 不过想换什么颜色都行",
            "399": "想换什么颜色都行 操作非常简单",
        }

        ok, msg = t._validate_llm_response(response, source)

        assert ok is False
        assert "398-399" in msg

    def test_rejects_short_model_phrase_repeated_at_adjacent_boundary(self):
        t = _make_translator()
        source = {
            "246": "But hey, you touch the turn signal all the time, and it's the same",
            "247": "as an S class, so that's cool.",
        }
        response = {
            "246": "但你经常会碰转向灯 而它和S级是一样的",
            "247": "和S级一样 所以这点挺酷",
        }

        ok, msg = t._validate_llm_response(response, source)

        assert ok is False
        assert "246-247" in msg

    def test_rejects_repeated_restart_qualification_across_keys(self):
        t = _make_translator()
        source = {
            "187": "There's potential in restarts, but there's only so many plants,",
            "188": "I think, that are even in a condition or in the space to restart.",
        }
        response = {
            "187": "重启确实有潜力 但适合重启的核电站也就那么多",
            "188": "适合重启的核电站数量有限 还得处于合适状态和位置",
        }

        ok, msg = t._validate_llm_response(response, source)

        assert ok is False
        assert "187-188" in msg

    def test_rejects_short_translation_repeated_from_previous_key(self):
        t = _make_translator()
        source = {
            "421": "Seeing how he does it up close, I understand",
            "422": "a little more why.",
        }
        response = {
            "421": "近距离看到他的工作方式 我也更明白了为什么会是这样",
            "422": "为什么会是这样",
        }

        ok, msg = t._validate_llm_response(response, source)

        assert ok is False
        assert "421-422" in msg

    def test_minimax_alignment_audit_applies_only_sparse_valid_corrections(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        source = {
            "33": "made a dramatic film from them. The issue",
            "34": "over the years as we use them more and more",
        }
        translated = {
            "33": "不过麻烦也跟着来了 这些年我们越用越多",
            "34": "也越来越想把这项技术整合到电影里",
        }
        responses = iter(
            [
                _llm_response({"misaligned_keys": ["33", "34"]}),
                _llm_response({"misaligned_keys": ["33", "34"]}),
                _text_response("我们首次用这些摄影机拍了剧情片 但问题也随之而来"),
                _text_response("这些年来我们用得越来越多"),
                _llm_response({"misaligned_keys": []}),
            ]
        )
        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **_kwargs: next(responses),
        )

        result = translator._audit_reflective_alignment(source, translated)

        assert result["33"].endswith("问题也随之而来")
        assert result["34"] == "这些年来我们用得越来越多"

    def test_alignment_item_marks_neighbor_source_as_read_only_context(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        captured = {}

        def fake_call(**kwargs):
            captured.update(kwargs)
            return _text_response("从很多方面来说")

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            fake_call,
        )

        result = translator._translate_alignment_item(
            "in so many ways",
            previous_source="Because it was so challenging",
            next_source="not only for me, for everybody",
        )

        assert result == "从很多方面来说"
        payload = json.loads(captured["messages"][1]["content"])
        assert payload["current_source"] == "in so many ways"
        assert payload["previous_source"] == "Because it was so challenging"
        assert payload["next_source"] == "not only for me, for everybody"
        assert "read-only" in captured["messages"][0]["content"]
        assert "currency formatting" in captured["messages"][0]["content"]
        assert "translation" not in captured["messages"][1]["content"]

    def test_alignment_audit_requests_an_exhaustive_per_key_verdict(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        captured = {}

        def fake_call(**kwargs):
            captured.update(kwargs)
            return _llm_response(
                {
                    "alignment": {"1": True, "2": False},
                    "misaligned_keys": ["2"],
                }
            )

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            fake_call,
        )

        result = translator._request_alignment_flags(
            {
                "1": {"source": "one", "translation": "一"},
                "2": {"source": "two", "translation": "错位"},
            }
        )

        assert result == ["2"]
        assert "evaluate every input key" in captured["messages"][0]["content"]
        assert "literal meaning is impossible" in captured["messages"][0]["content"]
        assert captured["reasoning_mode"] == "disabled"
        assert captured["max_output_tokens"] == 4096

        translator._request_alignment_flags(
            {
                "1": {"source": "one", "translation": "一"},
                "2": {"source": "two", "translation": "错位"},
            },
            focused=True,
        )
        assert captured["reasoning_mode"] == "enabled"
        assert captured["max_output_tokens"] == 4096

    def test_focused_alignment_audit_falls_back_when_thinking_has_no_verdict(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        calls = []
        responses = iter(
            [
                _text_response("<think>unfinished reasoning"),
                _llm_response({"alignment": {"1": True}, "misaligned_keys": []}),
            ]
        )

        def fake_call(**kwargs):
            calls.append(kwargs)
            return next(responses)

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            fake_call,
        )

        assert (
            translator._request_alignment_flags(
                {"1": {"source": "one", "translation": "一"}},
                focused=True,
            )
            == []
        )
        assert [call["reasoning_mode"] for call in calls] == ["enabled", "disabled"]

    def test_alignment_audit_receives_read_only_global_context(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        translator.translation_context = TranslationContext(summary="Automotive road test")
        captured = {}

        def fake_call(**kwargs):
            captured.update(kwargs)
            return _llm_response({"alignment": {"91": False}, "misaligned_keys": ["91"]})

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            fake_call,
        )

        result = translator._request_alignment_flags(
            {
                "91": {
                    "source": "I got $18,000 and we were doing $85,000.",
                    "translation": "我有18000美元 我们一直卖85000美元",
                }
            }
        )

        assert result == ["91"]
        assert "Automotive road test" in captured["messages"][0]["content"]

    @pytest.mark.parametrize(
        ("source", "translation"),
        [
            (
                "I got $18,000, and we were doing $85,000 the whole time.",
                "我跑出了18 mpg 而且一路基本都开着85 mph",
            ),
            (
                "I got $18,000, and we were doing $85,000 the whole time.",
                "油耗是每加仑18英里 而且全程时速85英里",
            ),
            ("It took a break for 25.", "它在2025款短暂停产"),
        ],
    )
    def test_accepts_twice_audited_asr_number_format_repairs(self, source, translation):
        translator = _make_minimax_reflect_translator()

        ok, message = translator._validate_llm_response(
            {"1": translation},
            {"1": source},
            require_reflect=False,
        )

        assert ok is True
        assert message == ""

    def test_real_currency_amount_still_requires_full_number(self):
        translator = _make_minimax_reflect_translator()

        ok, message = translator._validate_llm_response(
            {"1": "它的价格是18美元"},
            {"1": "It costs $18,000."},
            require_reflect=False,
        )

        assert ok is False
        assert "18000" in message

    @pytest.mark.parametrize(
        ("source", "translation"),
        [
            ("because you spent 20 grand on it", "因为你为它花了2万美元"),
            ("because you spent 20 grand on it", "因为你为它花了20,000美元"),
            ("It costs 53K.", "它售价5.3万美元"),
            ("A 50K Hemi masterpiece.", "一台价值5万美元的HEMI杰作"),
        ],
    )
    def test_accepts_equivalent_numeric_magnitude_notation(self, source, translation):
        translator = _make_minimax_reflect_translator()

        ok, message = translator._validate_llm_response(
            {"1": translation},
            {"1": source},
            require_reflect=False,
        )

        assert ok is True
        assert message == ""

    @pytest.mark.parametrize(
        ("source", "translation"),
        [
            (
                "We had 20-some-odd thousand people sign a pledge to live a healthier life.",
                "有两万多人签署了承诺 要过上更健康的生活",
            ),
            (
                "The town has 20 thousand residents.",
                "这座小镇有两万名居民",
            ),
            (
                "because most of it is health promotion 101.",
                "因为其中大部分只是健康促进的基础常识",
            ),
        ],
    )
    def test_accepts_natural_equivalents_for_thousands_and_101_idiom(
        self,
        source,
        translation,
    ):
        translator = _make_minimax_reflect_translator()

        ok, message = translator._validate_llm_response(
            {"1": translation},
            {"1": source},
            require_reflect=False,
        )

        assert ok is True
        assert message == ""

    @pytest.mark.parametrize(
        ("source", "translation", "missing_token"),
        [
            ("The town has 20 thousand residents.", "这座小镇有很多居民", "20"),
            ("Take Route 101 north.", "沿这条公路向北行驶", "101"),
            ("It makes 101 horsepower.", "它具备基础动力", "101"),
        ],
    )
    def test_numeric_equivalents_do_not_relax_real_missing_facts(
        self,
        source,
        translation,
        missing_token,
    ):
        translator = _make_minimax_reflect_translator()

        ok, message = translator._validate_llm_response(
            {"1": translation},
            {"1": source},
            require_reflect=False,
        )

        assert ok is False
        assert missing_token in message

    def test_rejects_lost_grand_magnitude(self):
        translator = _make_minimax_reflect_translator()

        ok, message = translator._validate_llm_response(
            {"1": "因为这玩意儿你花了20张富兰克林"},
            {"1": "because you spent 20 grand on it"},
            require_reflect=False,
        )

        assert ok is False
        assert "numeric magnitude" in message

    def test_alignment_audit_rejects_incomplete_per_key_verdict(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **_kwargs: _llm_response({"alignment": {"1": True}, "misaligned_keys": []}),
        )

        with pytest.raises(ValueError, match="evaluate every input key"):
            translator._request_alignment_flags(
                {
                    "1": {"source": "one", "translation": "一"},
                    "2": {"source": "two", "translation": "二"},
                }
            )

    def test_alignment_repair_fills_single_key_hole_in_confirmed_shift(self):
        translator = _make_minimax_reflect_translator()
        ordered = ["616", "617", "618", "619", "620"]

        result = translator._expand_confirmed_alignment_keys(
            ordered,
            {"616", "618", "619", "620"},
        )

        assert result == ordered

    def test_alignment_repair_does_not_expand_across_speaker_boundary(self):
        translator = _make_minimax_reflect_translator()
        translator._all_speaker_by_index = {616: "S1", 617: "S2", 618: "S2"}

        result = translator._expand_confirmed_alignment_keys(
            ["616", "617", "618"],
            {"616", "618"},
        )

        assert result == ["616", "618"]

    def test_alignment_repair_skips_short_repeated_asr_fragment(self):
        translator = _make_minimax_reflect_translator()

        assert translator._is_disfluent_alignment_fragment("of that of that struggle")
        assert not translator._is_disfluent_alignment_fragment(
            "that can actually just create a podcast"
        )

    def test_alignment_length_outlier_only_flags_extreme_chinese_expansion(self):
        translator = _make_minimax_reflect_translator()
        source = {
            "618": "who end up supporting the president there",
            "619": "who led to the nationwide strike a couple years ago",
        }
        translated = {
            "618": "最终支持了那位工会主席由其带领在几年前发起了一场全国性罢工",
            "619": "几年前领导了全国性罢工",
            "620": "对吧",
            "621": "好谢谢你",
        }
        source.update(
            {
                "620": "in Minnesota called Elina right",
                "621": "Area",
            }
        )

        assert translator._strong_alignment_length_outliers(source, translated) == [
            "618",
            "620",
            "621",
        ]

    def test_semantic_asr_candidates_require_explicit_local_contradiction(self):
        translator = _make_minimax_reflect_translator()
        source = {
            "90": "It gets like 18 mpg.",
            "91": "I got $18,000, and we were doing $85,000 the whole time.",
            "92": "The fuel economy was good.",
            "216": "This would have been the SRT Durango.",
            "217": "It took a break for 25.",
            "218": "It's back for 26 in this new RT392 trim.",
        }
        translated = {
            "90": "油耗大约18 mpg",
            "91": "我有18000美元 全程花了85000美元",
            "92": "油耗表现不错",
            "216": "它原本是SRT Durango",
            "217": "它停了25年",
            "218": "它在26款以R/T 392配置回归",
        }

        assert translator._strong_asr_semantic_candidates(source, translated) == [
            "91",
            "217",
        ]

        translated["91"] = "我加了18000英里的油 全程开了85000英里"
        assert translator._strong_asr_semantic_candidates(source, translated) == [
            "91",
            "217",
        ]

        assert (
            translator._strong_asr_semantic_candidates(
                {"1": "The car costs $18,000."},
                {"1": "这台车售价18000美元"},
            )
            == []
        )

    def test_semantic_asr_candidate_reads_across_batch_boundary(self):
        translator = _make_minimax_reflect_translator()
        translator._all_source_by_index = {
            90: "It gets like 18 mpg.",
            91: "I got $18,000, and we were doing $85,000 the whole time.",
            92: "That's right.",
        }

        assert translator._strong_asr_semantic_candidates(
            {"91": translator._all_source_by_index[91], "92": "That's right."},
            {"91": "我花了18000美元 全程花了85000美元", "92": "没错"},
        ) == ["91"]

    def test_semantic_asr_candidate_catches_baselift_as_base_trim(self):
        translator = _make_minimax_reflect_translator()

        assert translator._strong_asr_semantic_candidates(
            {"537": "Baselift 540 with the clear taillights."},
            {"537": "基础款540 配透明尾灯"},
        ) == ["537"]

    def test_semantic_asr_candidate_catches_reverse_camera_disappears_error(self):
        translator = _make_minimax_reflect_translator()

        assert translator._strong_asr_semantic_candidates(
            {
                "297": "Let's show you the reverse camera.",
                "298": "It disappears from 2014 when it was introduced.",
            },
            {"297": "看看倒车影像", "298": "它从2014年推出后就消失了"},
        ) == ["298"]

    def test_strict_semantic_candidate_does_not_depend_on_a_second_model_vote(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        translator._all_source_by_index = {
            90: "It gets like 18 mpg.",
            91: "I got $18,000, and we were doing $85,000 the whole time.",
            92: "That's right.",
        }
        monkeypatch.setattr(
            translator,
            "_request_alignment_flags",
            lambda *_a, **_k: ["91"],
        )
        monkeypatch.setattr(
            translator,
            "_translate_alignment_item",
            lambda *_a, **_k: "我跑出了18 mpg 全程保持85 mph",
        )

        result = translator._audit_reflective_alignment(
            {
                "91": translator._all_source_by_index[91],
                "92": translator._all_source_by_index[92],
            },
            {"91": "我花了18000美元 全程花了85000美元", "92": "没错"},
        )

        assert result["91"] == "我跑出了18 mpg 全程保持85 mph"

    def test_alignment_role_hint_uses_explicit_union_context_only(self):
        translator = _make_minimax_reflect_translator()

        assert "工会主席" in translator._alignment_role_hint(
            "who end up supporting the president there",
            "graduate students who joined the UAW",
            "who led a nationwide strike",
        )
        assert (
            translator._alignment_role_hint(
                "who end up supporting the president there",
                "graduate students attended the university",
                "at the event",
            )
            == ""
        )
        hint = "The role is president of the union (工会主席), not a head of state or school."
        assert translator._apply_alignment_role_hint("最终支持当地主席的人", hint) == (
            "最终支持工会主席的人"
        )
        assert translator._alignment_reference_hint(
            "who led to the nationwide strike",
            "who supported the president there",
        ).endswith("a person.")
        assert (
            translator._alignment_reference_hint(
                "which led to the nationwide strike",
                "who supported the president there",
            )
            == ""
        )
        title_hint = translator._alignment_title_fragment_hint(
            "Area.",
            "thank you for coming on The Gray",
        )
        assert "The Gray Area" in title_hint
        assert "not a reply" in title_hint

    def test_alignment_asr_hint_is_narrow_and_machine_verifiable(self):
        translator = _make_minimax_reflect_translator()

        quantity_hint = translator._alignment_asr_hint(
            "I got $18,000, and we were doing $85,000 the whole time.",
            "It gets like 18 mpg.",
            "That's right.",
        )
        assert quantity_hint["kind"] == "grouped_quantity_units"
        assert quantity_hint["normalized_source"] == (
            "I got 18 mpg, and we were doing 85 mph the whole time."
        )
        assert (
            translator._validate_alignment_asr_hint(
                "我跑出了18 mpg 全程保持85 mph",
                quantity_hint,
            )
            == ""
        )
        assert (
            translator._validate_alignment_asr_hint(
                "油耗达到每加仑18英里 全程时速85英里",
                quantity_hint,
            )
            == ""
        )
        assert "requires" in translator._validate_alignment_asr_hint(
            "我花了18000美元 全程花了85000美元",
            quantity_hint,
        )

        year_hint = translator._alignment_asr_hint(
            "It took a break for 25.",
            "This would have been the SRT Durango.",
            "It's back for 26 in the new RT392 trim.",
        )
        assert year_hint["kind"] == "model_year_shorthand"
        assert "model year 2025" in year_hint["normalized_source"]
        assert (
            translator._validate_alignment_asr_hint(
                "它在25款上短暂停产",
                year_hint,
            )
            == ""
        )
        assert (
            translator._alignment_asr_hint(
                "It costs $18,000.",
                "The base price is affordable.",
                "Another trim costs more.",
            )
            == {}
        )

        process_hint = translator._alignment_asr_hint(
            "production to turn on a heated seat was",
            "The one bad thing about that car.",
            "a death-defying project behind the wheel on a highway.",
        )
        assert process_hint["kind"] == "process_homophone"
        assert (
            translator._validate_alignment_asr_hint(
                "开启座椅加热的过程",
                process_hint,
            )
            == ""
        )
        assert "next_source" in translator._validate_alignment_asr_hint(
            "在高速上开启座椅加热简直是要命的挑战",
            process_hint,
        )

        camera_hint = translator._alignment_asr_hint(
            "It disappears from 2014 when it was introduced.",
            "Let's show you the reverse camera.",
            "Watch out for this guy.",
        )
        assert camera_hint["kind"] == "reverse_camera_age_homophone"
        assert (
            translator._validate_alignment_asr_hint(
                "这套倒车影像看起来还是2014年刚推出时的样子",
                camera_hint,
            )
            == ""
        )

    def test_alignment_item_translates_verified_normalized_source(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        captured = {}

        def fake_call(**kwargs):
            captured.update(kwargs)
            return _text_response("我跑出了18 mpg 全程保持85 mph")

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            fake_call,
        )

        result = translator._translate_alignment_item(
            "I got $18,000, and we were doing $85,000 the whole time.",
            previous_source="It gets like 18 mpg.",
            next_source="That's right.",
        )

        payload = json.loads(captured["messages"][1]["content"])
        assert result == "我跑出了18 mpg 全程保持85 mph"
        assert payload["current_source"] == (
            "I got 18 mpg, and we were doing 85 mph the whole time."
        )
        assert payload["original_asr_source"].startswith("I got $18,000")

    def test_alignment_audit_focuses_on_neighbors_of_detected_shift(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        source = {str(i): f"source {i}" for i in range(1, 7)}
        translated = {str(i): f"译文{i}" for i in range(1, 7)}
        responses = iter(
            [
                _llm_response({"misaligned_keys": ["3"]}),
                _llm_response({"misaligned_keys": ["3", "4", "5"]}),
                _text_response("正确三"),
                _llm_response({"misaligned_keys": []}),
            ]
        )
        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **_kwargs: next(responses),
        )

        result = translator._audit_reflective_alignment(source, translated)

        assert result["2"] == "译文2"
        assert result["3"] == "正确三"
        assert result["4"] == "译文4"
        assert result["5"] == "译文5"
        assert result["6"] == "译文6"

    def test_alignment_audit_rechecks_confirmed_dialogue_shift(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        translator._all_speaker_by_index = {1: "S1", 2: "S1", 3: "S2", 4: "S2"}
        source = {
            "1": "I cannot find an example.",
            "2": "But the blood and soil—",
            "3": "He certainly delivered a very",
            "4": "strong message to the government.",
        }
        translated = {
            "1": "我找不到这样的例子",
            "2": "他当然传递了非常",
            "3": "血与土那一套",
            "4": "向政府发出了强烈信号",
        }
        payloads = []
        responses = iter(
            [
                _llm_response({"misaligned_keys": ["2", "3"]}),
                _llm_response({"misaligned_keys": ["2", "3"]}),
                _text_response("但是血与土那一套——"),
                _text_response("他当然传递了非常"),
                _llm_response({"misaligned_keys": []}),
            ]
        )

        def fake_call(**kwargs):
            payloads.append(kwargs["messages"][1]["content"])
            return next(responses)

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            fake_call,
        )

        result = translator._audit_reflective_alignment(source, translated)

        assert result["2"] == "但是血与土那一套——"
        assert result["3"] == "他当然传递了非常"
        assert '"speaker": "S2"' in payloads[0]
        assert len(payloads) == 5

    def test_alignment_audit_uses_locked_recovery_when_sparse_fixes_repeat(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        source = {
            "1": "He certainly delivered a very",
            "2": "strong message to the government.",
        }
        translated = {
            "1": "他当然传递了非常",
            "2": "向政府发出了强烈信号",
        }
        responses = iter(
            [
                _llm_response({"misaligned_keys": ["1", "2"]}),
                _llm_response({"misaligned_keys": ["1", "2"]}),
                _text_response("向政府发出了强烈信号"),
                _text_response("向政府发出了强烈信号"),
                _llm_response(
                    {
                        "1": "他当然传递了一个非常",
                        "2": "强烈的对政府表态",
                    }
                ),
                _llm_response({"misaligned_keys": []}),
            ]
        )
        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **_kwargs: next(responses),
        )

        result = translator._audit_reflective_alignment(source, translated)

        assert result == {
            "1": "他当然传递了一个非常",
            "2": "强烈的对政府表态",
        }

    @pytest.mark.parametrize(
        "model",
        ["gpt-4o-mini", "deepseek-v4-pro", "MiniMax-M3", "nvidia/llama-3.3-70b"],
    )
    def test_alignment_audit_is_enabled_for_every_model_in_reflective_mode(self, model):
        translator = _make_translator(is_reflect=True)
        translator.model = model

        assert translator._needs_alignment_audit() is True

    def test_alignment_audit_is_disabled_outside_reflective_mode(self):
        translator = _make_translator(is_reflect=False)
        translator.model = "MiniMax-M3"

        assert translator._needs_alignment_audit() is False

    def test_alignment_audit_requires_two_matching_flags(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        source = {
            "23": "And how could an alliance",
            "24": "between white-collar and blue-collar workers",
            "25": "transform American politics?",
        }
        translated = {
            "23": "而一旦白领和蓝领工人联手",
            "24": "会怎样改变美国政治的格局",
            "25": "诺姆 欢迎来到灰色地带",
        }
        responses = iter(
            [
                _llm_response({"misaligned_keys": ["25"]}),
                _llm_response({"misaligned_keys": []}),
            ]
        )
        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **_kwargs: next(responses),
        )

        result = translator._audit_reflective_alignment(
            source,
            translated,
            initial_focus_keys=["25"],
        )

        assert result == translated

    def test_alignment_audit_corrects_twice_confirmed_shift(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        source = {
            "23": "And how could an alliance",
            "24": "between white-collar and blue-collar workers",
            "25": "transform American politics?",
        }
        translated = {
            "23": "而一旦结成联盟",
            "24": "白领与蓝领工人之间",
            "25": "诺姆 欢迎来到灰色地带",
        }
        responses = iter(
            [
                _llm_response({"misaligned_keys": ["25"]}),
                _llm_response({"misaligned_keys": ["25"]}),
                _text_response("会怎样改变美国政治？"),
                _llm_response({"misaligned_keys": []}),
            ]
        )
        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **_kwargs: next(responses),
        )

        result = translator._audit_reflective_alignment(
            source,
            translated,
            initial_focus_keys=["25"],
        )

        assert result["25"] == "会怎样改变美国政治？"

    def test_alignment_audit_repairs_confirmed_multi_key_shift_from_manual_benchmark(
        self, monkeypatch
    ):
        translator = _make_minimax_reflect_translator()
        source = {
            "119": "The kind of bohemian bourgeois was",
            "120": "this idea that it",
            "121": "really kind of epitomized the 90s",
            "122": "when tech was becoming ascendant",
        }
        translated = {
            "119": "所谓波西米亚资产阶级 就是这个概念",
            "120": "它几乎概括了90年代",
            "121": "当时科技产业开始崛起",
            "122": "自由贸易协定也接连签署",
        }
        responses = iter(
            [
                _llm_response({"misaligned_keys": ["120", "121", "122"]}),
                _llm_response({"misaligned_keys": ["120", "121", "122"]}),
                _text_response("这个概念认为"),
                _text_response("它几乎就是90年代的缩影"),
                _text_response("当时科技产业开始崛起"),
                _llm_response({"misaligned_keys": []}),
            ]
        )
        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **_kwargs: next(responses),
        )

        result = translator._audit_reflective_alignment(source, translated)

        assert result == {
            "119": "所谓波西米亚资产阶级 就是这个概念",
            "120": "这个概念认为",
            "121": "它几乎就是90年代的缩影",
            "122": "当时科技产业开始崛起",
        }

    def test_alignment_audit_discards_repair_that_still_borrows_neighbor_meaning(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        source = {
            "844": "that can actually just create a podcast",
            "845": "from a set of academic articles.",
        }
        translated = {
            "844": "可以直接生成一档播客",
            "845": "素材是一组学术论文",
        }
        responses = iter(
            [
                _llm_response({"misaligned_keys": ["844"]}),
                _llm_response({"misaligned_keys": ["844"]}),
                _text_response("可以根据一组学术论文生成播客"),
                _llm_response({"misaligned_keys": ["844"]}),
                _text_response("可以根据一组学术论文生成播客"),
                _llm_response({"misaligned_keys": ["844"]}),
            ]
        )
        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **_kwargs: next(responses),
        )

        result = translator._audit_reflective_alignment(source, translated)

        assert result == translated

    def test_alignment_audit_trims_borrowed_context_from_contextual_repair(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        source = {
            "844": "that can actually just create a podcast",
            "845": "from a set of academic articles.",
        }
        translated = {
            "844": "素材是一组学术论文",
            "845": "素材是一组学术论文",
        }
        payloads = []
        responses = iter(
            [
                _llm_response({"misaligned_keys": ["844"]}),
                _llm_response({"misaligned_keys": ["844"]}),
                _text_response("可以根据一组学术论文生成播客"),
                _llm_response({"misaligned_keys": ["844"]}),
                _text_response("可以直接生成一档播客"),
                _llm_response({"misaligned_keys": []}),
            ]
        )

        def fake_call(**kwargs):
            payloads.append(kwargs["messages"][1]["content"])
            return next(responses)

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            fake_call,
        )

        result = translator._audit_reflective_alignment(source, translated)

        assert result["844"] == "可以直接生成一档播客"
        fallback_payload = json.loads(payloads[-2])
        assert fallback_payload["current_source"] == source["844"]
        assert fallback_payload["candidate_translation"] == "可以根据一组学术论文生成播客"
        assert "previous_source" not in fallback_payload
        assert "next_source" not in fallback_payload

    def test_alignment_audit_keeps_repair_verified_in_isolation(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        source = {
            "844": "that can actually just create a podcast",
            "845": "from a set of academic articles.",
        }
        translated = {
            "844": "素材是一组学术论文",
            "845": "素材是一组学术论文",
        }
        responses = iter(
            [
                _llm_response({"misaligned_keys": ["844"]}),
                _llm_response({"misaligned_keys": ["844"]}),
                _text_response("可以直接生成一档播客"),
                _llm_response({"misaligned_keys": []}),
            ]
        )
        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **_kwargs: next(responses),
        )

        result = translator._audit_reflective_alignment(source, translated)

        assert result["844"] == "可以直接生成一档播客"
        assert result["845"] == translated["845"]

    def test_allows_repeated_translation_when_source_is_also_repeated(self):
        t = _make_translator()
        source = {"118": "Stop.", "119": "Stop."}
        response = {"118": "赶快停下来", "119": "赶快停下来"}

        ok, msg = t._validate_llm_response(response, source)

        assert ok is True
        assert msg == ""

    def test_allows_repeated_named_subject_when_source_repeats_it(self):
        translator = _make_translator()

        ok, message = translator._validate_llm_response(
            {
                "9": "当然 这并非史上第一辆Charger",
                "10": "Charger已经存在了几十年",
            },
            {
                "9": "Of course, this isn't the first ever Charger.",
                "10": "The Charger existed for many decades.",
            },
        )

        assert ok is True
        assert message == ""

    def test_rejects_repeated_connector_at_same_speaker_boundary(self):
        translator = _make_translator()
        translator._all_speaker_by_index = {1: "S1", 2: "S1"}

        ok, message = translator._validate_llm_response(
            {"1": "但他的重要性真的怎么说都", "2": "都不为过"},
            {"1": "it matters greatly", "2": "the importance cannot be overstated"},
        )

        assert ok is False
        assert "1-2:都" in message

    def test_allows_same_connector_across_different_speakers(self):
        translator = _make_translator()
        translator._all_speaker_by_index = {1: "S1", 2: "S2"}

        ok, message = translator._validate_llm_response(
            {"1": "我觉得也是", "2": "是的"},
            {"1": "I think so too", "2": "Yes"},
        )

        assert ok is True
        assert message == ""

    def test_rejects_repeated_chinese_conclusion_for_same_speaker(self):
        translator = _make_translator()
        translator._all_speaker_by_index = {1: "S1", 2: "S1"}

        ok, message = translator._validate_llm_response(
            {"1": "但这一点再怎么强调都不为过", "2": "真的不为过"},
            {"1": "it matters greatly", "2": "the importance cannot be overstated"},
        )

        assert ok is False
        assert "1-2:不为过" in message

    def test_rejects_equivalent_repeated_chinese_conclusions(self):
        translator = _make_translator()
        translator._all_speaker_by_index = {1: "S1", 2: "S1"}

        ok, message = translator._validate_llm_response(
            {"1": "怎么强调都不过分", "2": "怎么强调都不为过"},
            {"1": "it matters greatly", "2": "the importance cannot be overstated"},
        )

        assert ok is False
        assert "不为过" in message

    def test_allows_repeated_chinese_conclusion_when_source_repeats_predicate(self):
        translator = _make_translator()
        translator._all_speaker_by_index = {1: "S1", 2: "S1"}

        ok, message = translator._validate_llm_response(
            {"1": "怎么强调都不过分", "2": "怎么强调都不为过"},
            {"1": "but it really cannot", "2": "the importance cannot be overstated"},
        )

        assert ok is True
        assert message == ""

    def test_allows_repeated_ending_when_source_repeats_last_word(self):
        translator = _make_translator()
        translator._all_speaker_by_index = {1: "S1", 2: "S1"}

        ok, message = translator._validate_llm_response(
            {"1": "这就是特朗普", "2": "他还是特朗普"},
            {"1": "That is Trump", "2": "He is still Trump"},
        )

        assert ok is True
        assert message == ""

    def test_rejects_lowercase_english_function_word_in_chinese_translation(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"285": "into 表现、放松、影院模式之后"},
            {"285": "into expressive, relax, theater modes"},
        )

        assert ok is False
        assert "285:into" in msg

    def test_allows_english_word_inside_proper_product_name(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"1": "配的是Bowers and Wilkins音响"},
            {"1": "It has the Bowers and Wilkins sound system."},
        )

        assert ok is True
        assert msg == ""

    def test_agent_loop_strips_minimax_thinking_and_normalizes_keys(self, monkeypatch):
        t = _make_translator()

        class _Message:
            content = '<think>analysis</think>\n```json\n{1: "你好"}\n```'

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **_kwargs: _Response(),
        )

        assert t._agent_loop("prompt", {"1": "hello"}) == {"1": "你好"}

    def test_reflect_agent_loop_uses_structured_reflection_without_internal_thinking(
        self, monkeypatch
    ):
        t = _make_translator(is_reflect=True)
        calls = []

        def fake_call_llm(**kwargs):
            calls.append(kwargs)
            return _llm_response({"1": {"native_translation": "你好"}})

        monkeypatch.setattr("subforge.core.translate.llm_translator.call_llm", fake_call_llm)

        assert t._agent_loop("prompt", {"1": "hello"}) == {"1": {"native_translation": "你好"}}
        assert [call["reasoning_mode"] for call in calls] == ["disabled"]
        assert [call["max_output_tokens"] for call in calls] == [4096]

    def test_single_fallback_rejects_untranslated_cjk_result(self, monkeypatch):
        t = _make_translator()
        calls = []

        class _Message:
            content = "hello"

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **kwargs: calls.append(kwargs) or _Response(),
        )

        with pytest.raises(RuntimeError, match="Single item translation failed"):
            t._translate_chunk_single([SubtitleProcessData(index=1, original_text="hello")])
        assert len(calls) == t.SINGLE_FALLBACK_MAX_ATTEMPTS

    def test_single_fallback_retries_invalid_output_and_keeps_valid_result(self, monkeypatch):
        t = _make_translator()
        contents = iter(["That's fine.", "没关系 这样其实更好"])
        captured = []

        class _Message:
            content = ""

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        def fake_call_llm(**kwargs):
            captured.append(kwargs["messages"])
            _Message.content = next(contents)
            return _Response()

        monkeypatch.setattr("subforge.core.translate.llm_translator.call_llm", fake_call_llm)

        result = t._translate_chunk_single(
            [SubtitleProcessData(index=1, original_text="That's fine.")]
        )

        assert result[0].translated_text == "没关系 这样其实更好"
        assert len(captured) == 2
        assert "previous answer was invalid" in captured[1][-1]["content"]

    def test_single_fallback_retries_literal_resultative_degree_calque(self, monkeypatch):
        t = _make_translator()
        contents = iter(
            [
                "这就是安静模式下它有多安静",
                "静音模式下 它的声音就是这么轻",
            ]
        )
        calls = []

        def fake_call_llm(**kwargs):
            calls.append(kwargs)
            return _text_response(next(contents))

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            fake_call_llm,
        )

        result = t._translate_chunk_single(
            [
                SubtitleProcessData(
                    index=389,
                    original_text="That is how quiet this gets in quiet mode.",
                )
            ]
        )

        assert result[0].translated_text == "静音模式下 它的声音就是这么轻"
        assert len(calls) == 2

    def test_single_fallback_keeps_individually_valid_repeated_phrasing(self, monkeypatch):
        t = _make_translator()
        chunk = [
            SubtitleProcessData(index=1, original_text="This proposal changed everything."),
            SubtitleProcessData(index=2, original_text="Nobody expected the backlash."),
        ]
        responses = iter(["这个提案彻底改变了一切", "这场反弹彻底改变了一切"])

        monkeypatch.setattr(
            t,
            "_translate_locked_batch",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("retry singly")),
        )
        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **_kwargs: _text_response(next(responses)),
        )

        result = t._translate_chunk_single(chunk)

        assert [item.translated_text for item in result] == [
            "这个提案彻底改变了一切",
            "这场反弹彻底改变了一切",
        ]

    def test_single_fallback_does_not_discard_all_isolated_results_when_recheck_fails(
        self, monkeypatch
    ):
        t = _make_translator()
        chunk = [
            SubtitleProcessData(index=211, original_text="It does."),
            SubtitleProcessData(index=212, original_text="Let's pop the hood."),
        ]
        responses = iter(["确实如此", "打开引擎盖看看"])

        monkeypatch.setattr(
            t,
            "_translate_locked_batch",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("locked recheck failed")),
        )
        monkeypatch.setattr(
            t,
            "_validate_llm_response",
            lambda *_args, **_kwargs: (False, "cross-key recheck failed"),
        )
        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **_kwargs: _text_response(next(responses)),
        )

        result = t._translate_chunk_single(chunk)

        assert [item.translated_text for item in result] == [
            "确实如此",
            "打开引擎盖看看",
        ]

    def test_finalizer_repairs_repetition_across_batch_boundary(self, monkeypatch):
        t = _make_minimax_reflect_translator()
        t.batch_num = 2
        source = [
            SubtitleProcessData(index=1, original_text="First line."),
            SubtitleProcessData(index=2, original_text="Open the hood."),
            SubtitleProcessData(index=3, original_text="Check the rear brakes."),
            SubtitleProcessData(index=4, original_text="Last line."),
        ]
        translated = [
            SubtitleProcessData(index=1, original_text="First line.", translated_text="第一句"),
            SubtitleProcessData(
                index=2,
                original_text="Open the hood.",
                translated_text="打开引擎盖检查刹车",
            ),
            SubtitleProcessData(
                index=3,
                original_text="Check the rear brakes.",
                translated_text="打开引擎盖检查刹车",
            ),
            SubtitleProcessData(index=4, original_text="Last line.", translated_text="最后一句"),
        ]
        calls = []

        def fake_repair(pair, initial_feedback=""):
            calls.append((pair, initial_feedback))
            return [
                SubtitleProcessData(
                    index=2,
                    original_text="Open the hood.",
                    translated_text="打开引擎盖",
                ),
                SubtitleProcessData(
                    index=3,
                    original_text="Check the rear brakes.",
                    translated_text="检查后轮刹车",
                ),
            ]

        monkeypatch.setattr(t, "_translate_locked_batch", fake_repair)

        result = t._finalize_translated_list(source, translated)

        assert len(calls) == 1
        assert [item.translated_text for item in result] == [
            "第一句",
            "打开引擎盖",
            "检查后轮刹车",
            "最后一句",
        ]

    def test_finalizer_repairs_repetition_inside_a_batch(self, monkeypatch):
        t = _make_minimax_reflect_translator()
        t.batch_num = 10
        source = [
            SubtitleProcessData(index=1, original_text="First line."),
            SubtitleProcessData(index=2, original_text="Open the hood."),
            SubtitleProcessData(index=3, original_text="Check the rear brakes."),
            SubtitleProcessData(index=4, original_text="Last line."),
        ]
        translated = [
            replace(source[0], translated_text="第一句"),
            replace(source[1], translated_text="打开引擎盖检查刹车"),
            replace(source[2], translated_text="打开引擎盖检查刹车"),
            replace(source[3], translated_text="最后一句"),
        ]

        monkeypatch.setattr(
            t,
            "_translate_locked_batch",
            lambda _pair, initial_feedback="": [
                replace(source[1], translated_text="打开引擎盖"),
                replace(source[2], translated_text="检查后轮刹车"),
            ],
        )

        result = t._finalize_translated_list(source, translated)

        assert [item.translated_text for item in result[1:3]] == [
            "打开引擎盖",
            "检查后轮刹车",
        ]

    def test_finalizer_rebuilds_entire_batch_containing_queued_alignment(self, monkeypatch):
        t = _make_minimax_reflect_translator()
        t.batch_num = 4
        source = [
            SubtitleProcessData(index=1, original_text="Before."),
            SubtitleProcessData(index=2, original_text="I had to cough"),
            SubtitleProcessData(index=3, original_text="and sniff, but let's continue."),
            SubtitleProcessData(index=4, original_text="Now the interior."),
        ]
        translated = [
            replace(source[0], translated_text="前一句"),
            replace(source[1], translated_text="我咳了咳也擤了鼻子"),
            replace(source[2], translated_text="接下来看看内饰"),
            replace(source[3], translated_text="现在看内饰"),
        ]
        t._pending_alignment_repair_keys.update({2, 3})

        repaired_batches = []

        def repair_batch(items, initial_feedback=""):
            repaired_batches.append([item.index for item in items])
            return [
                replace(source[0], translated_text="前一句"),
                replace(source[1], translated_text="我不得不停下来咳嗽"),
                replace(source[2], translated_text="还擤了鼻子 不过我们继续"),
                replace(source[3], translated_text="现在看内饰"),
            ]

        monkeypatch.setattr(t, "_translate_locked_batch", repair_batch)

        result = t._finalize_translated_list(source, translated)

        assert [item.translated_text for item in result] == [
            "前一句",
            "我不得不停下来咳嗽",
            "还擤了鼻子 不过我们继续",
            "现在看内饰",
        ]
        assert repaired_batches == [[1, 2, 3, 4]]

    def test_finalizer_revalidates_a_still_repeated_repair(self, monkeypatch):
        t = _make_minimax_reflect_translator()
        t.batch_num = 2
        source = [
            SubtitleProcessData(index=1, original_text="First line."),
            SubtitleProcessData(index=2, original_text="Open the hood."),
            SubtitleProcessData(index=3, original_text="Check the rear brakes."),
            SubtitleProcessData(index=4, original_text="Last line."),
        ]
        translated = [
            replace(source[0], translated_text="第一句"),
            replace(source[1], translated_text="打开引擎盖检查刹车"),
            replace(source[2], translated_text="打开引擎盖检查刹车"),
            replace(source[3], translated_text="最后一句"),
        ]
        repeated = [
            replace(source[1], translated_text="打开引擎盖检查刹车"),
            replace(source[2], translated_text="打开引擎盖检查刹车"),
        ]
        corrected = [
            replace(source[1], translated_text="打开引擎盖"),
            replace(source[2], translated_text="检查后轮刹车"),
        ]
        locked_calls = []

        monkeypatch.setattr(t, "_translate_chunk_single", lambda _pair: repeated)

        def locked_retry(_pair, initial_feedback=""):
            locked_calls.append(initial_feedback)
            return corrected

        monkeypatch.setattr(t, "_translate_locked_batch", locked_retry)

        result = t._finalize_translated_list(source, translated)

        assert len(locked_calls) == 1
        assert "repeat" in locked_calls[0].lower()
        assert [item.translated_text for item in result[1:3]] == [
            "打开引擎盖",
            "检查后轮刹车",
        ]

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            (
                "Another virtue for nuclear power plants",
                "is that it tends to be one of the safest sources of electricity.",
            ),
            (
                "There's potential in restarts, but there's only so many plants,",
                "I think, that are even in a condition to restart.",
            ),
        ],
    )
    def test_finalizer_retranslates_dependent_batch_boundaries(self, monkeypatch, left, right):
        t = _make_minimax_reflect_translator()
        t.batch_num = 1
        source = [
            SubtitleProcessData(index=1, original_text=left),
            SubtitleProcessData(index=2, original_text=right),
        ]
        translated = [
            replace(source[0], translated_text="重复的前半句"),
            replace(source[1], translated_text="重复的后半句"),
        ]
        corrected = [
            replace(source[0], translated_text="这种方案确实有其优势"),
            replace(source[1], translated_text="但实际可行的对象十分有限"),
        ]
        locked_calls = []

        def locked_retry(_pair, initial_feedback=""):
            locked_calls.append(initial_feedback)
            return corrected

        monkeypatch.setattr(t, "_translate_locked_batch", locked_retry)

        result = t._finalize_translated_list(source, translated)

        assert len(locked_calls) == 1
        assert [item.translated_text for item in result] == [
            "这种方案确实有其优势",
            "但实际可行的对象十分有限",
        ]

    def test_finalizer_keeps_clean_dependent_batch_boundary(self, monkeypatch):
        t = _make_minimax_reflect_translator()
        t.batch_num = 1
        source = [
            SubtitleProcessData(
                index=1,
                original_text="I went to Oswego to learn why it wants another plant",
            ),
            SubtitleProcessData(
                index=2,
                original_text="and what would happen next if it got one.",
            ),
        ]
        translated = [
            replace(source[0], translated_text="我前往奥斯威戈了解它为何想再建一座核电站"),
            replace(source[1], translated_text="以及建成后会发生什么"),
        ]

        monkeypatch.setattr(
            t,
            "_translate_locked_batch",
            lambda *_args, **_kwargs: pytest.fail("clean boundary must not be retranslated"),
        )

        result = t._finalize_translated_list(source, translated)

        assert [item.translated_text for item in result] == [
            "我前往奥斯威戈了解它为何想再建一座核电站",
            "以及建成后会发生什么",
        ]

    def test_finalizer_repairs_repeated_meaning_after_sort_of(self, monkeypatch):
        t = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(
                index=720,
                original_text="echelon than they've been in and they're",
            ),
            SubtitleProcessData(
                index=721,
                original_text=(
                    "sort of revamping the lineup they've got the Wagoneer which is very luxurious"
                ),
            ),
        ]
        translated = [
            replace(source[0], translated_text="正在重塑产品线"),
            replace(source[1], translated_text="算是重新整理了产品线 他们有豪华的Wagoneer"),
        ]
        repaired = [
            replace(source[0], translated_text="进入了比以往更高端的市场 而且他们正在"),
            replace(source[1], translated_text="调整产品线 其中Wagoneer非常豪华"),
        ]

        monkeypatch.setattr(
            t,
            "_translate_locked_batch",
            lambda _pair, initial_feedback="": repaired,
        )

        result = t._finalize_translated_list(source, translated)

        assert [item.translated_text for item in result] == [
            "进入了比以往更高端的市场 而且他们正在",
            "调整产品线 其中Wagoneer非常豪华",
        ]

    def test_finalizer_keeps_term_repeated_in_dependent_source(self, monkeypatch):
        t = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(
                index=334,
                original_text="This does not have air suspension,",
            ),
            SubtitleProcessData(
                index=335,
                original_text="but it has been making air suspension noises.",
            ),
        ]
        translated = [
            replace(source[0], translated_text="这辆车没有配备空气悬挂"),
            replace(source[1], translated_text="但一直在发出很像空气悬挂的声响"),
        ]

        monkeypatch.setattr(
            t,
            "_translate_locked_batch",
            lambda *_args, **_kwargs: pytest.fail(
                "a term repeated in the source must not be repaired"
            ),
        )

        result = t._finalize_translated_list(source, translated)

        assert [item.translated_text for item in result] == [
            "这辆车没有配备空气悬挂",
            "但一直在发出很像空气悬挂的声响",
        ]

    def test_finalizer_expands_a_repeated_fragment_to_its_unfinished_predecessor(self, monkeypatch):
        t = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(
                index=719,
                original_text="Jeep is aiming for this higher",
            ),
            SubtitleProcessData(
                index=720,
                original_text="echelon than before and they're",
            ),
            SubtitleProcessData(
                index=721,
                original_text="sort of revamping the lineup with the Grand",
            ),
            SubtitleProcessData(
                index=722,
                original_text="Cherokee in the top trim.",
            ),
        ]
        translated = [
            replace(source[0], translated_text="Jeep瞄准了比以往更高的市场 而且他们"),
            replace(source[1], translated_text="正在重塑产品线"),
            replace(source[2], translated_text="正在调整产品线 其中包括Grand Cherokee"),
            replace(source[3], translated_text="Cherokee的高配车型"),
        ]
        captured = []

        def repair(items, initial_feedback=""):
            captured.append([item.index for item in items])
            return [
                replace(source[0], translated_text="Jeep正瞄准比以往更高端的市场"),
                replace(source[1], translated_text="并且正在"),
                replace(source[2], translated_text="调整产品阵容 其中包括Grand"),
                replace(source[3], translated_text="Cherokee的高配车型"),
            ]

        monkeypatch.setattr(t, "_translate_locked_batch", repair)

        result = t._finalize_translated_list(source, translated)

        assert captured == [[719, 720, 721, 722]]
        assert [item.translated_text for item in result] == [
            "Jeep正瞄准比以往更高端的市场",
            "并且正在",
            "调整产品阵容 其中包括Grand",
            "Cherokee的高配车型",
        ]

    def test_finalizer_repairs_semantic_asr_errors_after_single_item_fallback(self, monkeypatch):
        t = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(index=295, original_text="Show the reverse camera."),
            SubtitleProcessData(index=296, original_text="There it is."),
            SubtitleProcessData(index=297, original_text="That's all."),
            SubtitleProcessData(
                index=298,
                original_text="It disappears from 2014 when it was introduced.",
            ),
        ]
        translated = [
            replace(item, translated_text=text)
            for item, text in zip(
                source,
                ["看看倒车影像", "就在那", "就这些", "它从2014年推出后就消失了"],
            )
        ]
        t._all_source_by_index = {item.index: item.original_text for item in source}
        monkeypatch.setattr(
            t,
            "_translate_alignment_item",
            lambda *_args, **_kwargs: "这套倒车影像看起来还是2014年刚推出时的样子",
        )

        result = t._finalize_translated_list(source, translated)

        assert result[-1].translated_text == ("这套倒车影像看起来还是2014年刚推出时的样子")

    def test_finalizer_resolves_ambiguous_plant_from_nuclear_context(self):
        t = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(
                index=34,
                original_text="The fourth nuclear power plant would bring jobs.",
            ),
            SubtitleProcessData(
                index=35,
                original_text="Most people know somebody who works at the plants.",
            ),
        ]
        translated = [
            replace(source[0], translated_text="第四座核电站会带来就业"),
            replace(source[1], translated_text="大多数人都认识在那些工厂工作的人"),
        ]

        result = t._finalize_translated_list(source, translated)

        assert result[1].translated_text == "大多数人都认识在那些核电站工作的人"

    def test_finalizer_keeps_factory_plant_outside_nuclear_context(self):
        t = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(
                index=1,
                original_text="These car plants employ thousands of workers.",
            )
        ]
        translated = [replace(source[0], translated_text="这些汽车工厂雇用了数千名工人")]

        result = t._finalize_translated_list(source, translated)

        assert result[0].translated_text == "这些汽车工厂雇用了数千名工人"

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ("但问题是他们", "把尾灯设计成一条灯带", True),
            ("采用锯齿一样的", "边缘方便装载物品", True),
            ("过弯时车身很稳 所以", "你不会觉得车身失控", True),
            ("后排空间很宽敞", "坐起来也很舒服", False),
        ],
    )
    def test_chinese_boundary_signal_shortlists_only_structural_breaks(self, left, right, expected):
        signal = LLMTranslator._chinese_boundary_signal(left, right)

        assert bool(signal) is expected

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("但实际上 这一代新时代Charger身上", "有不少欧洲影响"),
            ("想想现在48,000美元几乎", "就能买到一辆RT392 Durango"),
        ],
    )
    def test_chinese_boundary_signal_catches_full_run_failures(self, left, right):
        assert LLMTranslator._chinese_boundary_signal(left, right)

    @pytest.mark.parametrize(
        ("left", "right", "expected_signal"),
        [
            (
                "我觉得你90%的使用场景里 这个转向机",
                "都是好的 因为它让响应更快了一些",
                "percentage use-case predicate is stranded",
            ),
            (
                "就像会让这车",
                "比现在还要出色得多 而且别误会",
                "resultative predicate is stranded",
            ),
        ],
    )
    def test_chinese_boundary_signal_catches_raptor_full_run_failures(
        self, left, right, expected_signal
    ):
        assert LLMTranslator._chinese_boundary_signal(left, right) == expected_signal

    def test_chinese_boundary_signal_trims_chinese_ellipsis(self):
        assert LLMTranslator._chinese_boundary_signal(
            "就像它会让这台车变得……",
            "比现在还要出色得多",
        ) == "unfinished Chinese grammatical structure"

    def test_deterministic_fallback_repairs_percentage_use_case_pair(self):
        source = [
            SubtitleProcessData(
                index=268,
                original_text=(
                    "I think 90% of what you're going to use this truck for, "
                    "like the steering rack,"
                ),
            ),
            SubtitleProcessData(
                index=269,
                original_text="is a good thing because it quickened things up a bit.",
            ),
        ]
        current = [
            replace(source[0], translated_text="我认为90%的场景 比如转向机"),
            replace(source[1], translated_text="都是好事 因为响应更快"),
        ]

        repaired = LLMTranslator._deterministic_chinese_fluency_fallback(source, current)

        assert repaired is not None
        assert [item.translated_text for item in repaired] == [
            "我觉得在90%的使用场景里",
            "这套转向系统都更好用 因为响应更快了",
        ]

    def test_deterministic_fallback_repairs_resultative_pair(self):
        source = [
            SubtitleProcessData(
                index=360,
                original_text="Like it would just make this thing.",
            ),
            SubtitleProcessData(
                index=361,
                original_text=(
                    "So much more excellent than it already is and don't get me wrong."
                ),
            ),
        ]
        current = [
            replace(source[0], translated_text="就像它会让这台车变得……"),
            replace(source[1], translated_text="比现在还要出色得多 别误会"),
        ]

        repaired = LLMTranslator._deterministic_chinese_fluency_fallback(source, current)

        assert repaired is not None
        assert [item.translated_text for item in repaired] == [
            "它仿佛能让这台车更上一层楼",
            "尽管它本来就已经很优秀了 但别误会",
        ]

    def test_chinese_boundary_signal_catches_stranded_i_mean(self):
        assert LLMTranslator._chinese_boundary_signal(
            "我不明白为什么没人开这种老式美国车 我是说",
            "与其花八万美元买新车",
        )

    @pytest.mark.parametrize(
        ("left", "right", "expected_signal"),
        [
            (
                "也许蓝色地带之所以能引起如此强烈共鸣的部分原因",
                "我们之所以如此扎根美国 是因为现代美国生活与长寿背道而驰",
                "unfinished Chinese reason construction",
            ),
            ("而孤独和孤立变得", "如此普遍", "unfinished Chinese grammatical structure"),
        ],
    )
    def test_chinese_boundary_signal_catches_blue_zone_failures(self, left, right, expected_signal):
        assert LLMTranslator._chinese_boundary_signal(left, right) == expected_signal

    def test_chinese_boundary_signal_does_not_treat_lexical_le_as_particle(self):
        assert not LLMTranslator._chinese_boundary_signal(
            "请前往Patreon获取更多故事",
            "了解我们的最新工作",
        )

    @pytest.mark.parametrize("tail", ["我觉得可以买到一辆", "就接近程度来说"])
    def test_chinese_boundary_signal_catches_incomplete_charger_tails(self, tail):
        assert LLMTranslator._chinese_boundary_signal(tail, "下一条内容")

    @pytest.mark.parametrize("complete", ["作为通勤车来说 算我一个", "就是这一款"])
    def test_chinese_boundary_signal_does_not_treat_complete_quantifiers_as_dangling(
        self, complete
    ):
        assert not LLMTranslator._chinese_boundary_signal(complete, "下一条内容")

    def test_complete_demonstrative_remains_soft_not_mandatory(self):
        assert (
            LLMTranslator._chinese_boundary_signal("看看这个", "下一条内容")
            == "possible demonstrative split"
        )

    def test_complete_source_sentence_is_not_a_mandatory_fluency_repair(self):
        translator = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(index=1, original_text="I should just have one."),
            SubtitleProcessData(index=2, original_text="Like an SRT Chrysler 300."),
        ]
        translated = {
            1: replace(source[0], translated_text="我应该就买一辆"),
            2: replace(source[1], translated_text="比如SRT版克莱斯勒300"),
        }

        assert translator._mandatory_chinese_fluency_candidates(source, translated) == []

    def test_subject_ba_tail_is_a_structural_boundary(self):
        assert (
            LLMTranslator._chinese_boundary_signal(
                "还要感谢这家经销商 他们把",
                "这辆车的钥匙交给了我",
            )
            == "unfinished Chinese grammatical structure"
        )

    def test_mandatory_fluency_candidates_require_deterministic_breaks(self):
        translator = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(index=1, original_text="You insert the key you"),
            SubtitleProcessData(index=2, original_text="can see the Dodge logo"),
            SubtitleProcessData(index=3, original_text="The next sentence is complete."),
        ]
        translated = {
            1: replace(source[0], translated_text="你把钥匙插进去 你"),
            2: replace(source[1], translated_text="可以看到道奇标志"),
            3: replace(source[2], translated_text="下一句话是完整的"),
        }

        assert translator._mandatory_chinese_fluency_candidates(source, translated) == [1]

    def test_removes_only_stranded_subject_before_following_auxiliary(self):
        source = [
            SubtitleProcessData(index=1, original_text="You insert the key you"),
            SubtitleProcessData(index=2, original_text="can see the Dodge logo"),
            SubtitleProcessData(index=3, original_text="I support you"),
            SubtitleProcessData(index=4, original_text="when the work is difficult"),
        ]
        translated = {
            1: replace(source[0], translated_text="你把钥匙插进去 你"),
            2: replace(source[1], translated_text="可以看到道奇标志"),
            3: replace(source[2], translated_text="我支持你"),
            4: replace(source[3], translated_text="当工作遇到困难时"),
        }

        LLMTranslator._remove_stranded_chinese_subject_tails(source, translated)

        assert translated[1].translated_text == "你把钥匙插进去"
        assert translated[3].translated_text == "我支持你"

    @pytest.mark.parametrize(
        ("left_source", "right_source", "left_translation", "right_translation"),
        [
            (
                "We actually saw this evolve",
                "before our very eyes, from SRT8 to Hellcat.",
                "我们确实看到它演变",
                "在我们眼前 从SRT8一路发展到Hellcat",
            ),
            (
                "Maybe you will go out after this video",
                "and go purchase it.",
                "也许你看完视频后",
                "就去买下它",
            ),
            (
                "So excited to take you on a city commute today",
                "because I want to see what it is like to daily drive.",
                "所以非常兴奋今天带大家城市通勤",
                "因为我想看看它日常驾驶怎么样",
            ),
            (
                "I think 90% of what you're going to use",
                "this truck for, like the steering rack, is a good thing",
                "我觉得你90%的使用场景",
                "比如用这辆卡车 转向齿条是件好事",
            ),
            (
                "And then if you're having trouble trying to follow this little puny",
                "RPM gauge over here on the left, you can change the view.",
                "然后如果你很难看清这个小小的",
                "左边的RPM转速表 你可以切换显示模式",
            ),
        ],
    )
    def test_source_boundary_signal_shortlists_translation_order_risks(
        self, left_source, right_source, left_translation, right_translation
    ):
        assert LLMTranslator._source_boundary_signal(
            left_source,
            right_source,
            left_translation,
            right_translation,
        )

    def test_source_boundary_signal_shortlists_degree_complement(self):
        assert LLMTranslator._source_boundary_signal(
            "And maybe part of the reason blue zones resonate",
            "so deeply in America is because modern life is designed against longevity.",
            "也许蓝色地带之所以能引起如此强烈共鸣的部分原因",
            "我们之所以如此扎根美国 是因为现代生活与长寿背道而驰",
        )

    def test_blue_zone_failures_are_mandatory_fluency_candidates(self):
        translator = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(
                index=130,
                original_text="And maybe part of the reason blue zones resonate",
            ),
            SubtitleProcessData(
                index=131,
                original_text=(
                    "so deeply in America is because so much of modern American life "
                    "is designed against longevity."
                ),
            ),
            SubtitleProcessData(
                index=132,
                original_text="Most Americans do not live in walkable communities.",
            ),
            SubtitleProcessData(
                index=133,
                original_text="Ultra processed food makes up much of the American diet.",
            ),
            SubtitleProcessData(
                index=134,
                original_text="And loneliness and isolation became",
            ),
            SubtitleProcessData(
                index=135,
                original_text=(
                    "so widespread that the Surgeon General considered it a public health "
                    "epidemic in 2023."
                ),
            ),
        ]
        translated = {
            130: replace(
                source[0],
                translated_text="也许蓝色地带之所以能引起如此强烈共鸣的部分原因",
            ),
            131: replace(
                source[1],
                translated_text="我们之所以如此扎根美国 是因为现代美国生活与长寿背道而驰",
            ),
            132: replace(source[2], translated_text="大多数美国人并不住在适合步行的社区"),
            133: replace(source[3], translated_text="超加工食品占美国饮食的很大一部分"),
            134: replace(source[4], translated_text="而孤独和孤立变得"),
            135: replace(source[5], translated_text="如此普遍 以至于这被视为公共卫生流行病"),
        }

        assert translator._mandatory_chinese_fluency_candidates(source, translated) == [130, 134]

    def test_pronoun_ending_remains_a_soft_contextual_signal(self):
        assert (
            LLMTranslator._chinese_boundary_signal(
                "我会立刻选他",
                "他是个很有吸引力的赢家",
            )
            == "possible pronoun boundary"
        )

    def test_chinese_fluency_rewrite_uses_non_thinking_request(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(index=1, original_text="I'd pick him in a second"),
            SubtitleProcessData(index=2, original_text="as an appealing winner"),
        ]
        current = [
            replace(source[0], translated_text="我会立刻选他"),
            replace(source[1], translated_text="作为一个有吸引力的赢家"),
        ]
        captured = {}

        def fake_call(**kwargs):
            captured.update(kwargs)
            return _llm_response(
                {
                    "translations": {
                        "1": "我会立刻选他",
                        "2": "他是个很有吸引力的赢家",
                    }
                }
            )

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            fake_call,
        )

        repaired = translator._rewrite_chinese_fluency_window(source, current)

        assert [item.translated_text for item in repaired] == [
            "我会立刻选他",
            "他是个很有吸引力的赢家",
        ]
        assert captured["reasoning_mode"] == "disabled"

    def test_chinese_fluency_rewrite_uses_thinking_for_deepseek_v4(self, monkeypatch):
        translator = _make_translator(is_reflect=True)
        translator.model = "deepseek-v4-flash"
        source = [
            SubtitleProcessData(index=1, original_text="We saw it evolve"),
            SubtitleProcessData(index=2, original_text="before our eyes."),
        ]
        current = [
            replace(source[0], translated_text="我们看到它演变"),
            replace(source[1], translated_text="在我们眼前"),
        ]
        captured = {}

        def fake_call(**kwargs):
            captured.update(kwargs)
            return _llm_response({"translations": {"1": "我们亲眼见证了它的演变", "2": "整个过程"}})

        monkeypatch.setattr("subforge.core.translate.llm_translator.call_llm", fake_call)

        translator._rewrite_chinese_fluency_window(source, current)

        assert captured["reasoning_mode"] == "enabled"
        assert captured["max_output_tokens"] == 8192

    def test_chinese_fluency_audit_disables_deepseek_reasoning(self, monkeypatch):
        translator = _make_translator(is_reflect=True)
        translator.model = "deepseek-v4-flash"
        source = [
            SubtitleProcessData(index=1, original_text="The problem is they"),
            SubtitleProcessData(index=2, original_text="made it too wide"),
        ]
        translated = {
            1: replace(source[0], translated_text="但问题是他们"),
            2: replace(source[1], translated_text="把它做得太宽了"),
        }
        captured = {}

        def fake_call(**kwargs):
            captured.update(kwargs)
            return _llm_response({"awkward_boundaries": ["1-2"]})

        monkeypatch.setattr("subforge.core.translate.llm_translator.call_llm", fake_call)

        assert translator._request_chinese_fluency_flags([1], source, translated) == [1]
        assert captured["reasoning_mode"] == "disabled"
        assert captured["max_output_tokens"] == 4096

    def test_chinese_window_fidelity_accepts_complete_local_reordering(self, monkeypatch):
        translator = _make_translator(is_reflect=True)
        source = [
            SubtitleProcessData(index=1, original_text="We saw it evolve"),
            SubtitleProcessData(index=2, original_text="before our eyes."),
        ]
        repaired = [
            replace(source[0], translated_text="我们亲眼见证了它的演变"),
            replace(source[1], translated_text="整个过程"),
        ]
        captured = {}

        def fake_call(**kwargs):
            captured.update(kwargs)
            return _llm_response({"valid": True, "issues": []})

        monkeypatch.setattr("subforge.core.translate.llm_translator.call_llm", fake_call)

        translator._validate_chinese_window_fidelity(source, repaired)

        assert captured["reasoning_mode"] == "disabled"
        assert captured["max_output_tokens"] == 2048

    def test_chinese_window_fidelity_rejects_missing_meaning(self, monkeypatch):
        translator = _make_translator(is_reflect=True)
        source = [
            SubtitleProcessData(index=1, original_text="It has 340 horsepower"),
            SubtitleProcessData(index=2, original_text="and 390 pound-feet of torque."),
        ]
        repaired = [
            replace(source[0], translated_text="它有340马力"),
            replace(source[1], translated_text="动力很强"),
        ]
        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **_kwargs: _llm_response({"valid": False, "issues": ["遗漏390磅英尺扭矩"]}),
        )

        with pytest.raises(ValueError, match="390磅英尺扭矩"):
            translator._validate_chinese_window_fidelity(source, repaired)

    def test_chinese_fluency_candidates_stop_at_speaker_changes(self):
        t = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(index=1, original_text="The problem is they"),
            SubtitleProcessData(index=2, original_text="made it too wide"),
        ]
        translated = {
            1: replace(source[0], translated_text="但问题是他们"),
            2: replace(source[1], translated_text="把它做得太宽了"),
        }
        t._all_speaker_by_index = {1: "S1", 2: "S2"}

        assert t._chinese_fluency_candidates(source, translated) == []

    def test_finalizer_repairs_confirmed_chinese_boundary_without_touching_timeline(
        self, monkeypatch
    ):
        t = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(index=1, original_text="The problem is they"),
            SubtitleProcessData(index=2, original_text="made the taillights too narrow"),
            SubtitleProcessData(index=3, original_text="There is plenty of cargo room."),
        ]
        translated = [
            replace(source[0], translated_text="但问题是他们"),
            replace(source[1], translated_text="把尾灯设计得太窄"),
            replace(source[2], translated_text="后备厢空间很充足"),
        ]
        repaired = [
            replace(source[0], translated_text="但设计上的问题在于"),
            replace(source[1], translated_text="尾灯做得太窄了"),
        ]
        t._all_source_by_index = {item.index: item.original_text for item in source}
        audit_calls = []

        def flags(indices, *_args):
            audit_calls.append(indices)
            return [1] if len(audit_calls) == 1 else []

        monkeypatch.setattr(t, "_request_chinese_fluency_flags", flags)
        monkeypatch.setattr(t, "_validate_chinese_window_fidelity", lambda *_args: None)
        monkeypatch.setattr(
            t,
            "_rewrite_chinese_fluency_window",
            lambda source_items, _current, **_kwargs: repaired
            if [item.index for item in source_items] == [1, 2]
            else pytest.fail("repair window must stay narrow"),
        )

        result = t._finalize_translated_list(source, translated)

        assert [item.index for item in result] == [1, 2, 3]
        assert [item.original_text for item in result] == [item.original_text for item in source]
        assert [item.translated_text for item in result] == [
            "但设计上的问题在于",
            "尾灯做得太窄了",
            "后备厢空间很充足",
        ]

    def test_finalizer_rejects_chinese_fluency_repair_that_remains_broken(self, monkeypatch):
        t = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(index=1, original_text="It is a very"),
            SubtitleProcessData(index=2, original_text="comfortable seat."),
        ]
        translated = [
            replace(source[0], translated_text="这是一个非常"),
            replace(source[1], translated_text="舒适的座椅"),
        ]
        t._all_source_by_index = {item.index: item.original_text for item in source}
        audit_calls = 0

        def flags(*_args):
            nonlocal audit_calls
            audit_calls += 1
            return [1]

        monkeypatch.setattr(t, "_request_chinese_fluency_flags", flags)
        monkeypatch.setattr(t, "_request_alignment_flags", lambda *_args: [])
        rewrite_calls = []
        monkeypatch.setattr(
            t,
            "_rewrite_chinese_fluency_window",
            lambda *_args, **kwargs: rewrite_calls.append(kwargs.get("feedback", ""))
            or [
                replace(source[0], translated_text="这辆座椅仍然非常"),
                replace(source[1], translated_text="舒适平稳"),
            ],
        )

        result = t._finalize_translated_list(source, translated)

        assert audit_calls == 1
        assert len(rewrite_calls) == t.MAX_STEPS
        assert "structural boundary signals" in rewrite_calls[-1]
        assert [item.translated_text for item in result] == [
            "这是一个非常",
            "舒适的座椅",
        ]

    def test_fluency_repair_reaudits_source_driven_boundary(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(
                index=1,
                original_text="The easiest way to tell if it was a V8",
            ),
            SubtitleProcessData(index=2, original_text="was by the exhaust tips."),
        ]
        current = [
            replace(source[0], translated_text="判断它是不是V8"),
            replace(source[1], translated_text="最简单的方法是看排气管口"),
        ]
        monkeypatch.setattr(
            translator,
            "_request_chinese_fluency_flags",
            lambda indices, *_args: list(indices),
        )
        monkeypatch.setattr(translator, "_validate_chinese_window_fidelity", lambda *_args: None)

        with pytest.raises(ValueError, match="confirmed soft boundary signals"):
            translator._validate_chinese_fluency_repair(source, current, current)

    def test_single_fallback_strips_reasoning_and_sends_neighbor_context(self, monkeypatch):
        t = _make_translator()
        t._all_source_by_index = {1: "before", 2: "in a Q3", 3: "after"}
        captured = {}

        class _Message:
            content = "<think>Q3 is the Audi model</think>\n在Q3里"

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        def fake_call_llm(**kwargs):
            captured.update(kwargs)
            return _Response()

        monkeypatch.setattr("subforge.core.translate.llm_translator.call_llm", fake_call_llm)

        result = t._translate_chunk_single([SubtitleProcessData(index=2, original_text="in a Q3")])

        assert result[0].translated_text == "在Q3里"
        user_content = captured["messages"][1]["content"]
        assert '"previous_context": [{"index": "1", "source": "before"}]' in user_content
        assert '"next_context": [{"index": "3", "source": "after"}]' in user_content

    def test_single_context_ownership_allows_numeric_alias_owned_by_plural_source(self):
        t = _make_translator()
        t._all_source_by_index = {
            191: "Let's show you these tires.",
            192: "Now, these are the 37s.",
            193: "These are available in the Raptor 37 pack.",
        }

        t._validate_single_context_ownership(
            {"192": t._all_source_by_index[192]},
            "这些是37英寸轮胎",
        )

    def test_single_context_ownership_still_rejects_number_owned_only_by_neighbor(self):
        t = _make_translator()
        t._all_source_by_index = {
            191: "Let's show you these tires.",
            192: "Now, these are the tires.",
            193: "These are available in the Raptor 37 pack.",
        }

        with pytest.raises(RuntimeError, match=r"borrowed.*37"):
            t._validate_single_context_ownership(
                {"192": t._all_source_by_index[192]},
                "这些是37英寸轮胎",
            )

    def test_batch_context_ownership_allows_numeric_alias_owned_by_plural_source(self):
        t = _make_translator()
        source = {
            "192": "Now, these are the 37s.",
            "193": "These are available in the Raptor 37 pack.",
        }
        response = {
            "192": "这些是37英寸轮胎",
            "193": "这些也可通过Raptor 37套件选装",
        }

        ok, message = t._validate_cross_key_boundaries(response, source, str)

        assert ok is True
        assert message == ""

    def test_rejects_dropped_alphanumeric_model_tokens(self):
        t = _make_translator()
        resp = {"0": "今天来试试新款雷克萨斯。"}
        inp = {"0": "Today we drive the 2026 Lexus IS 350 F Sport."}

        ok, msg = t._validate_llm_response(resp, inp)

        assert ok is False
        assert "2026" in msg
        assert "350" in msg

    def test_allows_exact_chinese_ten_thousand_number_equivalent(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"567": "这辆车的售价是11.7万美元"},
            {"567": "This truck costs $117,000."},
        )

        assert ok is True
        assert msg == ""

    def test_rejects_wrong_chinese_ten_thousand_number_magnitude(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"567": "这辆车的售价是117万美元"},
            {"567": "This truck costs $117,000."},
        )

        assert ok is False
        assert "117000" in msg

    def test_allows_standard_brand_translation_for_preserved_tokens(self):
        t = _make_translator()
        resp = {"0": "宝马给这辆车做了专属调校。"}
        inp = {"0": "BMW gave this car a unique tune."}

        ok, msg = t._validate_llm_response(resp, inp)

        assert ok is True
        assert msg == ""

    def test_allows_standard_rem_translation_for_preserved_tokens(self):
        t = _make_translator()
        resp = {"0": "也会出现在非快速眼动睡眠阶段。"}
        inp = {"0": "It also occurs during non-REM sleep."}

        ok, msg = t._validate_llm_response(resp, inp)

        assert ok is True
        assert msg == ""

    def test_allows_world_war_ii_to_be_translated_semantically(self):
        t = _make_translator()
        resp = {"865": "一直到二战及40年代后期"}
        inp = {"865": "up through World War II, the late 40s"}

        ok, msg = t._validate_llm_response(resp, inp)

        assert ok is True
        assert msg == ""

    def test_rejects_literal_chinese_resultative_degree_calque(self):
        t = _make_translator()

        ok, message = t._validate_llm_response(
            {"389": "这就是安静模式下它有多安静"},
            {"389": "That is how quiet this gets in quiet mode."},
        )

        assert ok is False
        assert "静音模式下 它的声音就是这么轻" in message

    def test_accepts_natural_chinese_resultative_degree_translation(self):
        t = _make_translator()

        ok, message = t._validate_llm_response(
            {"389": "静音模式下 它的声音就是这么轻"},
            {"389": "That is how quiet this gets in quiet mode."},
        )

        assert ok is True
        assert message == ""

    def test_degree_calque_rule_does_not_reject_normal_how_much_expression(self):
        t = _make_translator()

        ok, message = t._validate_llm_response(
            {"1": "这就是我有多爱它"},
            {"1": "This is how much I love it."},
        )

        assert ok is True
        assert message == ""

    def test_chinese_style_prompt_explicitly_avoids_resultative_degree_calque(self):
        t = _make_translator()

        assert "这就是安静模式下它有多安静" in t._target_language_style_rules()
        assert "在90%的使用场景里" in t._target_language_style_rules()
        assert "福特推出了720马力版本" in t._target_language_style_rules()

        t.target_language = TargetLanguage.ENGLISH
        assert t._target_language_style_rules() == ""

    def test_rejects_vehicle_use_case_translation_with_wrong_chinese_subject(self):
        t = _make_translator()
        source = (
            "I think 90% of what you're going to use this truck for, like the steering "
            "rack, is a good thing."
        )

        ok, message = t._validate_llm_response(
            {"271": "你使用这辆卡车90%的场景 比如转向齿条 都是好事"},
            {"271": source},
        )

        assert ok is False
        assert "在90%的使用场景里" in message

        ok, message = t._validate_llm_response(
            {"271": "我认为在90%的使用场景里 这个转向齿条都是好东西"},
            {"271": source},
        )

        assert ok is False
        assert "转向响应都有帮助" in message

    def test_accepts_vehicle_use_case_translation_with_feature_as_subject(self):
        t = _make_translator()
        source = (
            "I think 90% of what you're going to use this truck for, like the steering "
            "rack, is a good thing."
        )

        ok, message = t._validate_llm_response(
            {"271": "我觉得在90%的使用场景里 更快的转向响应都有帮助"},
            {"271": source},
        )

        assert ok is True
        assert message == ""

    def test_rejects_incomplete_chinese_numeric_shorthand_noun(self):
        t = _make_translator()

        ok, message = t._validate_llm_response(
            {"294": "然后福特推出了这款720马力的"},
            {"294": "and then Ford came in with this 720,"},
        )

        assert ok is False
        assert "720马力版本" in message

    def test_accepts_completed_chinese_numeric_shorthand_noun(self):
        t = _make_translator()

        ok, message = t._validate_llm_response(
            {"294": "然后福特推出了720马力版本"},
            {"294": "and then Ford came in with this 720,"},
        )

        assert ok is True
        assert message == ""

    def test_rejects_temporal_translation_for_standalone_now_discourse_marker(self):
        t = _make_translator()

        ok, message = t._validate_llm_response({"449": "现在"}, {"449": "Now."})

        assert ok is False
        assert "接下来" in message

    def test_accepts_discourse_translation_for_standalone_now(self):
        t = _make_translator()

        ok, message = t._validate_llm_response({"449": "那么"}, {"449": "Now."})

        assert ok is True
        assert message == ""

    def test_allows_colloquial_plural_price_band_in_chinese(self):
        t = _make_translator()
        resp = {"500": "SRT8的价格大概在1.8万到2万美元之间"}
        inp = {
            "500": (
                "They range more expensive. I think an SRT8 would probably be "
                "somewhere in the 18s to 20s."
            )
        }

        ok, msg = t._validate_llm_response(resp, inp)

        assert ok is True
        assert msg == ""

    def test_allows_price_band_when_price_context_is_only_in_translation(self):
        t = _make_translator()
        resp = {"514": "我觉得SRT8大概在1.8万到两万美元之间"}
        inp = {"514": "I think an SRT8 would probably be somewhere in the 18s to 20s."}

        ok, msg = t._validate_llm_response(resp, inp)

        assert ok is True
        assert msg == ""

    def test_plural_decade_still_requires_decade_meaning(self):
        t = _make_translator()
        resp = {"1": "那是一段很久以前的历史"}
        inp = {"1": "It happened in the late 40s."}

        ok, msg = t._validate_llm_response(resp, inp)

        assert ok is False
        assert "40s" in msg

    def test_price_band_does_not_accept_decade_translation(self):
        t = _make_translator()
        resp = {"500": "SRT8的价格大概处于18年代到20年代"}
        inp = {
            "500": (
                "They range more expensive. I think an SRT8 would probably be "
                "somewhere in the 18s to 20s."
            )
        }

        ok, msg = t._validate_llm_response(resp, inp)

        assert ok is False
        assert "18s" in msg or "20s" in msg

    def test_price_band_rejects_unnatural_literal_thousands_wording(self):
        translator = _make_translator()

        ok, message = translator._validate_llm_response(
            {"520": "我想SRT8大概会在18到20多千美元这个价位"},
            {"520": "The SRT8 price would range somewhere in the 18s to 20s."},
        )

        assert ok is False
        assert "1.8万到2万美元" in message

    def test_price_band_rejects_redundant_source_suffix(self):
        translator = _make_translator()

        ok, message = translator._validate_llm_response(
            {"511": "SRT8大概在1.8万到2万美元（18s到20s）之间"},
            {"511": "The SRT8 price would range somewhere in the 18s to 20s."},
        )

        assert ok is False
        assert "1.8万到2万美元" in message

    def test_pending_alignment_chunk_is_not_stable_for_recovery(self):
        t = _make_minimax_reflect_translator()
        t._pending_alignment_repair_keys.update({74, 75})

        assert (
            t._is_chunk_result_stable(
                [
                    SubtitleProcessData(
                        index=74,
                        original_text="European influence here with this first",
                        translated_text="第一代车型受到欧洲影响",
                    ),
                    SubtitleProcessData(
                        index=75,
                        original_text="generation of the new era Charger.",
                        translated_text="这是新一代Charger",
                    ),
                ]
            )
            is False
        )
        assert (
            t._is_chunk_result_stable(
                [
                    SubtitleProcessData(
                        index=76,
                        original_text="And that's because Chrysler...",
                        translated_text="这是因为克莱斯勒",
                    )
                ]
            )
            is True
        )

    def test_pending_alignment_chunk_is_not_published_to_progress(self, monkeypatch):
        progress = []
        t = _make_minimax_reflect_translator()
        t.use_cache = False
        t.update_callback = progress.extend
        t._pending_alignment_repair_keys.add(74)
        result = [
            SubtitleProcessData(
                index=74,
                original_text="European influence here with this first",
                translated_text="第一代车型受到欧洲影响",
            )
        ]
        monkeypatch.setattr(t, "_translate_chunk", lambda _chunk: result)

        assert t._safe_translate_chunk(result) == result
        assert progress == []

    def test_still_requires_roman_numeral_for_product_model(self):
        t = _make_translator()
        resp = {"1": "这是新款捷豹Mark车型"}
        inp = {"1": "This is the new Jaguar Mark II."}

        ok, msg = t._validate_llm_response(resp, inp)

        assert ok is False
        assert "II" in msg

    def test_does_not_treat_uppercase_pronoun_as_model_token(self):
        t = _make_translator()
        resp = {"0": "这给了我们一个理解梦境的模型。"}
        inp = {"0": "Does that give US a model for understanding dreams?"}

        ok, msg = t._validate_llm_response(resp, inp)

        assert ok is True
        assert msg == ""

    def test_cache_key_includes_prompt_reflect_and_context(self):
        base = _make_translator()
        prompt = _make_translator()
        prompt.custom_prompt = "keep Lexus terms"
        reflect = _make_translator(is_reflect=True)
        context = _make_translator()
        context.translation_context = TranslationContext(summary="car review")
        chunk = [SubtitleProcessData(index=1, original_text="hello")]

        keys = {
            base._get_cache_key(chunk),
            prompt._get_cache_key(chunk),
            reflect._get_cache_key(chunk),
            context._get_cache_key(chunk),
        }

        assert len(keys) == 4

    def test_neighbor_context_uses_adjacent_source_only(self):
        t = _make_translator()
        t._all_source_by_index = {
            1: "before one",
            2: "before two",
            3: "current",
            4: "after one",
            5: "after two",
            6: "after three",
        }
        current = {"3": "current"}

        assert t._neighbor_context(current, before=True) == [
            {"index": "1", "source": "before one"},
            {"index": "2", "source": "before two"},
        ]
        assert t._neighbor_context(current, before=False) == [
            {"index": "4", "source": "after one"},
            {"index": "5", "source": "after two"},
        ]
