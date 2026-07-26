"""Tests for LLM translation response validation."""

import json
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
            SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))
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

    def test_minimax_reflect_chunk_always_runs_independent_alignment_audit(
        self, monkeypatch
    ):
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

    def test_alignment_audit_rejects_incomplete_per_key_verdict(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **_kwargs: _llm_response(
                {"alignment": {"1": True}, "misaligned_keys": []}
            ),
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

    def test_alignment_role_hint_uses_explicit_union_context_only(self):
        translator = _make_minimax_reflect_translator()

        assert "工会主席" in translator._alignment_role_hint(
            "who end up supporting the president there",
            "graduate students who joined the UAW",
            "who led a nationwide strike",
        )
        assert translator._alignment_role_hint(
            "who end up supporting the president there",
            "graduate students attended the university",
            "at the event",
        ) == ""
        hint = "The role is president of the union (工会主席), not a head of state or school."
        assert translator._apply_alignment_role_hint("最终支持当地主席的人", hint) == (
            "最终支持工会主席的人"
        )
        assert translator._alignment_reference_hint(
            "who led to the nationwide strike",
            "who supported the president there",
        ).endswith("a person.")
        assert translator._alignment_reference_hint(
            "which led to the nationwide strike",
            "who supported the president there",
        ) == ""
        title_hint = translator._alignment_title_fragment_hint(
            "Area.",
            "thank you for coming on The Gray",
        )
        assert "The Gray Area" in title_hint
        assert "not a reply" in title_hint

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
        assert '\"speaker\": \"S2\"' in payloads[0]
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

    def test_legacy_alignment_rewrite_is_disabled_for_all_models(self):
        translator = _make_translator(is_reflect=True)
        assert translator._needs_alignment_audit() is False
        translator.model = "MiniMax-M3"
        assert translator._needs_alignment_audit() is True

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

    def test_alignment_audit_discards_repair_that_still_borrows_neighbor_meaning(
        self, monkeypatch
    ):
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

    def test_alignment_audit_trims_borrowed_context_from_contextual_repair(
        self, monkeypatch
    ):
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

    def test_alignment_audit_keeps_repair_verified_in_isolation(
        self, monkeypatch
    ):
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

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm", fake_call_llm
        )

        result = t._translate_chunk_single(
            [SubtitleProcessData(index=1, original_text="That's fine.")]
        )

        assert result[0].translated_text == "没关系 这样其实更好"
        assert len(captured) == 2
        assert "previous answer was invalid" in captured[1][-1]["content"]

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

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm", fake_call_llm
        )

        result = t._translate_chunk_single(
            [SubtitleProcessData(index=2, original_text="in a Q3")]
        )

        assert result[0].translated_text == "在Q3里"
        user_content = captured["messages"][1]["content"]
        assert '"previous_context": [{"index": "1", "source": "before"}]' in user_content
        assert '"next_context": [{"index": "3", "source": "after"}]' in user_content

    def test_rejects_dropped_alphanumeric_model_tokens(self):
        t = _make_translator()
        resp = {"0": "今天来试试新款雷克萨斯。"}
        inp = {"0": "Today we drive the 2026 Lexus IS 350 F Sport."}

        ok, msg = t._validate_llm_response(resp, inp)

        assert ok is False
        assert "2026" in msg
        assert "350" in msg

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
