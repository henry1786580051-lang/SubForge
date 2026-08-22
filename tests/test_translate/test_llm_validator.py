"""Tests for LLM translation response validation."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from subforge.core.entities import SubtitleProcessData
from subforge.core.prompts import get_prompt
from subforge.core.translate.context import TranslationContext
from subforge.core.translate.guidance import repair_mode_guidance, target_language_style_rules
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

    def test_context_asr_mapping_does_not_treat_explanation_as_canonical_name(self):
        mapping = LLMTranslator._parse_context_asr_mapping(
            "- Marabba Vale -> 马拉巴谷 (probable ASR correction; canonical form "
            "confirmed by multiple mentions and context)"
        )

        assert mapping is None

    def test_validator_rejects_traditional_script_for_simplified_target(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"1": "別以為這開起來會像小一號的 GR Corolla"},
            {"1": "Don't think this will feel like a baby GR Corolla."},
            require_reflect=False,
        )

        assert valid is False
        assert "Simplified Chinese" in error

    def test_validator_allows_simplified_script_for_simplified_target(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"1": "别以为它开起来会像小一号的 GR Corolla"},
            {"1": "Don't think this will feel like a baby GR Corolla."},
            require_reflect=False,
        )

        assert valid is True
        assert error == ""

    def test_validator_resolves_audio_demonstration_medium_from_context(self):
        translator = _make_translator()
        translator._all_source_by_index = {
            8: "This has the standard sound system.",
            9: "We are going to run a sound test.",
            10: "And we'll show you what that's all about.",
        }
        source = {"10": "And we'll show you what that's all about."}

        invalid, error = translator._validate_llm_response(
            {"10": "我们会给你展示这是怎么回事"},
            source,
            require_reflect=False,
        )
        valid, _ = translator._validate_llm_response(
            {"10": "也让大家听听它的表现"},
            source,
            require_reflect=False,
        )

        assert invalid is False
        assert "audio demonstration" in error
        assert valid is True

    @pytest.mark.parametrize(
        ("source", "bad_target", "good_target", "error_fragment"),
        [
            (
                "It is actually a pretty quiet vent.",
                "这个出风口挺安静的",
                "这个出风口的风噪很小",
                "wind noise",
            ),
            (
                "Use our auto-down window.",
                "用一下自动降窗",
                "用一下车窗一键下降",
                "one-touch-down",
            ),
        ],
    )
    def test_validator_requires_established_automotive_control_terms(
        self,
        source,
        bad_target,
        good_target,
        error_fragment,
    ):
        translator = _make_translator()

        invalid, error = translator._validate_llm_response(
            {"1": bad_target},
            {"1": source},
            require_reflect=False,
        )
        valid, _ = translator._validate_llm_response(
            {"1": good_target},
            {"1": source},
            require_reflect=False,
        )

        assert invalid is False
        assert error_fragment in error
        assert valid is True

    def test_validator_rejects_unowned_latin_name_from_global_context(self):
        translator = _make_translator()
        source = {"1": "Physical volume controls are on the left side of the wheel."}

        invalid, error = translator._validate_llm_response(
            {"1": "方向盘左侧有实体音量控制的 Corollas"},
            source,
            require_reflect=False,
        )
        valid, _ = translator._validate_llm_response(
            {"1": "方向盘左侧有实体音量控制键"},
            source,
            require_reflect=False,
        )

        assert invalid is False
        assert "Corollas" in error
        assert valid is True

    def test_validator_allows_owned_lowercase_source_identifier_in_target(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"1": "丰田称之为 Sport Touring 座椅"},
            {"1": "Toyota calls these the sport touring seats."},
            require_reflect=False,
        )

        assert valid is True
        assert error == ""

    def test_context_asr_mapping_accepts_explicit_canonical_name(self):
        mapping = LLMTranslator._parse_context_asr_mapping(
            "- rubber veil -> 马拉巴谷 (probable ASR correction; canonical form is Maraba Vale)"
        )

        assert mapping == ("rubber veil", "Maraba Vale")

    def test_context_asr_mapping_rejects_generic_variant_description(self):
        mapping = LLMTranslator._parse_context_asr_mapping(
            "- Maraba Vale -> 马拉巴谷 (probable ASR correction; variant of the same tower name)"
        )

        assert mapping is None

    def test_confirmed_context_canonical_exposes_only_supported_name(self):
        translator = _make_translator()
        translator.translation_context = TranslationContext(
            terminology=(
                "- Marabba Vale -> Maraba Vale (probable ASR correction)\n"
                "- Moorabbah Vale -> Maraba Vale (probable ASR correction)"
            )
        )
        translator._all_source_by_index = {
            1: "This is Maraba Vale.",
            2: "Moorabbah Vale uses hydraulic dampers.",
        }

        assert (
            translator._confirmed_context_canonical("Moorabbah Vale uses hydraulic dampers.")
            == "Maraba Vale"
        )
        assert translator._confirmed_context_canonical("Dubai uses dampers.") == ""

    def test_validator_requires_document_confirmed_manufacturer_identifier(self):
        translator = _make_translator()
        translator.translation_context = TranslationContext(
            terminology=(
                "- sport touring seats -> Sport Touring "
                "(official manufacturer identifier introduced by Toyota)"
            )
        )
        source = {"1": "what Toyota call the sport touring seats"}

        invalid, error = translator._validate_llm_response(
            {"1": "丰田称之为运动旅行座椅"},
            source,
            require_reflect=False,
        )
        valid, _ = translator._validate_llm_response(
            {"1": "丰田称之为 Sport Touring 座椅"},
            source,
            require_reflect=False,
        )

        assert invalid is False
        assert "Sport Touring" in error
        assert valid is True

    @pytest.mark.parametrize(
        ("source", "bad_target", "good_target", "error_fragment"),
        [
            (
                "It feels biblically accurate for a compact car.",
                "它很符合圣经里的紧凑型车",
                "它很符合紧凑型车本该有的样子",
                "Biblically accurate",
            ),
            (
                "It feels on par with the rest of the segment.",
                "它在同级中表现相当不错",
                "它和同级车型基本处于同一水平",
                "On par with",
            ),
            (
                "In a traffic situation it is natural to get through it.",
                "在堵车时通过车辆非常自然",
                "在拥堵路况中穿梭起来很自然",
                "traffic situation",
            ),
        ],
    )
    def test_validator_rejects_contextual_idiom_calques(
        self,
        source,
        bad_target,
        good_target,
        error_fragment,
    ):
        translator = _make_translator()

        invalid, error = translator._validate_llm_response(
            {"1": bad_target},
            {"1": source},
            require_reflect=False,
        )
        valid, _ = translator._validate_llm_response(
            {"1": good_target},
            {"1": source},
            require_reflect=False,
        )

        assert invalid is False
        assert error_fragment in error
        assert valid is True

    def test_context_lexical_correction_does_not_require_english_in_chinese(self):
        translator = _make_translator()
        error = translator._validate_alignment_asr_hint(
            "这些超高层建筑并非为普通纽约人设计",
            {
                "kind": "context_confirmed_asr_variant",
                "canonical": "supertall",
                "normalized_source": "But these supertalls aren't designed for New Yorkers.",
            },
        )

        assert error is None

    def test_context_proper_name_correction_still_requires_canonical_latin(self):
        translator = _make_translator()
        error = translator._validate_alignment_asr_hint(
            "这是马拉巴谷",
            {
                "kind": "context_confirmed_asr_variant",
                "canonical": "Maraba Vale",
                "normalized_source": "This is Maraba Vale.",
            },
        )

        assert "Maraba Vale" in str(error)

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

    def test_translation_prompts_forbid_added_stage_directions(self):
        for name in ("translate/standard", "translate/reflect"):
            prompt = get_prompt(
                name,
                target_language="简体中文",
                custom_prompt="",
            )
            assert "stage directions" in prompt
            assert "IRL" in prompt

    def test_validator_rejects_added_sarcasm_stage_direction(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"1": "[讽刺地] 这地方最棒了"},
            {"1": "Our favorite place."},
            require_reflect=False,
        )

        assert valid is False
        assert "stage directions" in error

    def test_validator_allows_stage_direction_present_in_source(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"1": "[讽刺地] 这地方最棒了"},
            {"1": "[sarcastically] Our favorite place."},
            require_reflect=False,
        )

        assert valid is True, error

    def test_validator_rejects_subtitle_key_leaked_into_translation(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"670": "这些年我也看了很多670:TV节目"},
            {"670": "I've watched so much TV over time."},
            require_reflect=False,
        )

        assert valid is False
        assert "key labels" in error

    @pytest.mark.parametrize(
        ("source", "translation", "error_fragment"),
        [
            (
                "but it does feel like it's maybe 20 softer for 26 from the 25 model",
                "但感觉比25款软了大概20% 针对26款来说",
                "model-year comparison",
            ),
            ("Merging IRL.", "实际道路汇入IRL", "in real life"),
            (
                "Reading was fundamental to the way I grew up.",
                "阅读对我的成长方式来说太根本了",
                "fundamental to",
            ),
        ],
    )
    def test_validator_rejects_confirmed_contextual_calques(
        self, source, translation, error_fragment
    ):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"1": translation},
            {"1": source},
            require_reflect=False,
        )

        assert valid is False
        assert error_fragment in error

    def test_validator_accepts_semantic_chinese_equivalent_for_irl(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"1": "现实路况下汇入高速"},
            {"1": "Merging IRL."},
            require_reflect=False,
        )

        assert valid is True, error

    def test_validator_treats_spaced_thousands_separator_as_one_number(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"1": "它是一辆价值5万美元的丰田Corolla"},
            {"1": "but it's also a $50, 000 Toyota Corolla."},
            require_reflect=False,
        )

        assert valid is True, error

    def test_validator_rejects_adjective_substituted_for_direct_yes_answer(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"1": "答案是 容易"},
            {"1": "And the answer here is yes."},
            require_reflect=False,
        )

        assert valid is False
        assert "yes/no answer" in error

    @pytest.mark.parametrize(
        ("source", "translation", "error_fragment"),
        [
            (
                "They had this fresh slate to make a hot hatch.",
                "他们有了这个全新的平台来打造小钢炮",
                "clean starting point",
            ),
            (
                "Try and take as much of a racing line as we can here.",
                "尽量走一条接近赛道的路线",
                "motorsport",
            ),
            (
                "You could argue for your $50,000.",
                "你可以为你的5万美元争辩一下",
                "reasonably expect",
            ),
            (
                "You could argue for your $50,000.",
                "就冲这5万美元 你确实有得争",
                "reasonably expect",
            ),
            (
                "It is stiff, bouncy, and crashy.",
                "它有点硬 有点颠 有点颠簸",
                "distinct ride qualities",
            ),
            (
                "Trip average 20, a 20.8.",
                "行程平均油耗20 还有一个20.8",
                "self-correction",
            ),
            (
                "so if you have a smaller this is a pro mac,",
                "所以如果你有个小一点的 这是Pro Max",
                "phone-fit aside",
            ),
        ],
    )
    def test_validator_rejects_confirmed_full_run_chinese_defects(
        self, source, translation, error_fragment
    ):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"1": translation},
            {"1": source},
            require_reflect=False,
        )

        assert valid is False
        assert error_fragment in error

    def test_validator_accepts_only_final_value_of_spoken_numeric_self_correction(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"1": "行程平均油耗是20.8"},
            {"1": "Trip average 20, a 20.8."},
            require_reflect=False,
        )

        assert valid is True, error

    def test_validator_rejects_invitation_anticipated_from_next_key(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {
                "1": "趁我们堵在车流里 不如给你看看这个新改版的",
                "2": "不如我们给你展示一下",
            },
            {
                "1": "Actually, while we're just kind of chilling here in traffic.",
                "2": "Why don't we show you this newly revised",
            },
            require_reflect=False,
        )

        assert valid is False
        assert "Anticipated invitations" in error

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

    def test_preserved_tokens_accepts_standard_chinese_qr_equivalent(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"108": "点击下方链接 或扫描屏幕上的二维码"},
            {"108": "Click the link below or scan the QR code on screen."},
        )

        assert valid is True, error

    def test_preserved_tokens_accepts_natural_chinese_ordinal(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"68": "这栋楼坐落在第五大道"},
            {"68": "The building stands on 5th Avenue."},
        )

        assert valid is True, error

    def test_preserved_tokens_accepts_compact_decimal_currency_magnitude(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"87": "这座耗资125亿美元的新机场"},
            {"87": "a $12.5BN new airport"},
        )

        assert valid is True, error

    def test_preserved_tokens_accepts_compact_chinese_decade_range(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"161": "很多七八十年代的车都没有副驾外后视镜"},
            {"161": "A lot of cars in the 1970s and 80s didn't have passenger mirrors."},
        )

        assert valid is True, error

    def test_context_ownership_accepts_localized_month_number(self):
        assert LLMTranslator._ownership_token_belongs_to_source(
            "12",
            "The land needs levelling by December.",
            "这片土地需要在12月前完成平整",
        )

    def test_context_ownership_accepts_localized_large_number_coefficient(self):
        assert LLMTranslator._ownership_token_belongs_to_source(
            "35",
            "It covers 350, 000 square metres.",
            "占地35万平方米",
        )

    def test_context_ownership_accepts_full_year_decade_for_short_decade_token(self):
        assert LLMTranslator._ownership_token_belongs_to_source(
            "80",
            "This became common in the late 1980s.",
            "这到1980年代末才变得常见",
        )

    def test_preserved_tokens_still_rejects_missing_ordinal(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"68": "这栋楼坐落在曼哈顿"},
            {"68": "The building stands on 5th Avenue."},
        )

        assert valid is False
        assert "5th" in error

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

    def test_reflect_mode_accepts_plain_translation_values(self):
        t = _make_translator(is_reflect=True)
        resp = {"0": "你好", "1": "世界"}
        inp = {"0": "hello", "1": "world"}
        ok, msg = t._validate_llm_response(resp, inp)
        assert ok is True
        assert msg == ""

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

    def test_allows_i_mean_filler_to_move_for_natural_chinese(self):
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

        assert ok is True
        assert message == ""

    def test_rejects_if_condition_omitted_from_its_own_key(self):
        t = _make_translator()
        resp = {"65": "就在这块面板后面"}
        source = {"65": "If they needed to, behind this panel."}

        ok, msg = t._validate_llm_response(resp, source)

        assert ok is False
        assert "Missing conditions" in msg

    def test_accepts_concessive_if_rendered_as_natural_chinese_concession(self):
        t = _make_translator()
        resp = {"67": "尽管仍令人咋舌 造价为900亿澳元"}
        source = {"67": "if still eye-watering, $90 billion."}

        ok, msg = t._validate_llm_response(resp, source)

        assert ok is True
        assert msg == ""

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

    @pytest.mark.parametrize(
        ("source", "translation"),
        [
            (
                "That's around 350 to 450 million litres of jet fuel every single year.",
                "这相当于每年消耗约3.5亿至4.5亿升航空燃油",
            ),
            (
                "50 to 100,000 new dwellings in that corridor.",
                "该走廊预计可新增5万至10万套住房",
            ),
            (
                "Of those 194km, 115km are going to be in tunnels,",
                "在这194公里中 有115公里将建在隧道内",
            ),
            (
                "These trains will reach maximum speeds of up to 320km an hour,",
                "这些列车最高时速可达320公里",
            ),
            (
                "although that's going to be limited to 200kph in tunnels.",
                "尽管在隧道内最高时速将限制在200公里",
            ),
        ],
    )
    def test_preserved_numbers_accepts_exact_localized_range_and_units(self, source, translation):
        t = _make_translator()

        ok, msg = t._validate_llm_response({"1": translation}, {"1": source})

        assert ok is True
        assert msg == ""

    @pytest.mark.parametrize(
        "translation", ["其中有114公里将建在隧道内", "其中有1150公里将建在隧道内"]
    )
    def test_preserved_measurement_still_rejects_wrong_localized_value(self, translation):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"1": translation},
            {"1": "115km are going to be in tunnels."},
        )

        assert ok is False
        assert "1:115km" in msg

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

    def test_rejects_neighbor_paraphrase_that_drops_source_object(self):
        t = _make_translator()
        source = {
            "1": "it was capable of shooting out protons",
            "2": "at much higher energies than previously possible.",
        }
        response = {
            "1": "能够以远高于以往的能量",
            "2": "以远超以往可能的能量射出",
        }

        ok, msg = t._validate_llm_response(response, source)

        assert ok is False
        assert "Suspicious pairs" in msg

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

    def test_alignment_item_marks_prior_object_evaluation_as_separate(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        captured = {}

        def fake_call(**kwargs):
            captured.update(kwargs)
            return _text_response("那辆真不错 有辆Corolla开过来了")

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            fake_call,
        )

        result = translator._translate_alignment_item(
            "That is quite good, fellow Corolla coming up.",
            previous_source="Oh, look at this tasty Ford Excursion.",
        )

        assert result == "那辆真不错 有辆Corolla开过来了"
        payload = json.loads(captured["messages"][1]["content"])
        assert "previous_source" in payload["reference_hint"]
        assert "separate observation" in payload["reference_hint"]

    def test_deepseek_alignment_rewrite_uses_reasoning_and_records_acceptance(self, monkeypatch):
        translator = _make_translator(is_reflect=True)
        translator.model = "deepseek-v4-flash"
        calls = []

        def fake_call(**kwargs):
            calls.append(kwargs)
            return _text_response("从很多方面来说")

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            fake_call,
        )

        assert translator._translate_alignment_item("in so many ways") == "从很多方面来说"
        assert [call["reasoning_mode"] for call in calls] == ["enabled"]
        assert [call["max_output_tokens"] for call in calls] == [6144]
        assert [call["reasoning_effort"] for call in calls] == ["low"]
        assert "priority is fidelity first" in calls[0]["messages"][0]["content"]
        metrics = translator.reasoning_metrics()
        assert metrics["rewrite_requests"] == 1
        assert metrics["final_answers"] == 1
        assert metrics["accepted_repairs"] == 1
        assert metrics["fallback_requests"] == 0

    def test_deepseek_alignment_rewrite_falls_back_after_reasoning_only_response(self, monkeypatch):
        translator = _make_translator(is_reflect=True)
        translator.model = "deepseek-v4-flash"
        responses = iter(
            [
                _text_response("<think>unfinished reasoning"),
                _text_response("从很多方面来说"),
            ]
        )
        calls = []

        def fake_call(**kwargs):
            calls.append(kwargs)
            return next(responses)

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            fake_call,
        )

        assert translator._translate_alignment_item("in so many ways") == "从很多方面来说"
        assert [call["reasoning_mode"] for call in calls] == ["enabled", "disabled"]
        metrics = translator.reasoning_metrics()
        assert metrics["rewrite_requests"] == 1
        assert metrics["no_final_answers"] == 1
        assert metrics["rejected_repairs"] == 1
        assert metrics["fallback_requests"] == 1

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
        assert "surrounding parallel list" in captured["messages"][0]["content"]
        assert captured["reasoning_mode"] == "disabled"
        assert captured["max_output_tokens"] == 4096

        translator._request_alignment_flags(
            {
                "1": {"source": "one", "translation": "一"},
                "2": {"source": "two", "translation": "错位"},
            },
            focused=True,
        )
        assert captured["reasoning_mode"] == "disabled"
        assert captured["max_output_tokens"] == 4096

    def test_focused_alignment_audit_uses_one_non_thinking_request(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        calls = []

        def fake_call(**kwargs):
            calls.append(kwargs)
            return _llm_response({"alignment": {"1": True}, "misaligned_keys": []})

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
        assert [call["reasoning_mode"] for call in calls] == ["disabled"]
        assert translator.reasoning_metrics()["audit_requests"] == 1

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
        "translation",
        [
            "但它也是一辆价值5万美元的丰田Corolla",
            "但它也是一辆价值五万美元的丰田Corolla",
        ],
    )
    def test_accepts_equivalent_chinese_ten_thousand_currency(self, translation):
        translator = _make_minimax_reflect_translator()

        ok, message = translator._validate_llm_response(
            {"451": translation},
            {"451": "but it's also a $50,000 Toyota Corolla."},
            require_reflect=False,
        )

        assert ok is True
        assert message == ""

    def test_rejects_wrong_chinese_ten_thousand_currency(self):
        translator = _make_minimax_reflect_translator()

        ok, message = translator._validate_llm_response(
            {"451": "但它也是一辆价值五十万美元的丰田Corolla"},
            {"451": "but it's also a $50,000 Toyota Corolla."},
            require_reflect=False,
        )

        assert ok is False
        assert "50000" in message

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

    def test_semantic_asr_candidate_catches_redundant_numeric_magnitude(self):
        translator = _make_minimax_reflect_translator()

        assert translator._strong_asr_semantic_candidates(
            {"108": "This weighs about 4,700 hundred pounds."},
            {"108": "它的重量约为四十七万磅"},
        ) == ["108"]

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

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("We bump this up into support.", "into Sport mode"),
            ("I want you to love here.", "want you to look here"),
            ("This dealer accessory matte is useful.", "dealer accessory mat"),
            ("A hot hat should do two things well.", "hot hatch"),
            (
                "We should get a move on and exit the city into reverse.",
                "shift into reverse",
            ),
        ],
    )
    def test_alignment_asr_hint_repairs_unambiguous_local_vehicle_homophones(
        self, source, expected
    ):
        translator = _make_translator()

        hint = translator._alignment_asr_hint(source, "", "")

        assert expected in hint["normalized_source"]

    def test_reverse_control_hint_rejects_reversing_out_of_the_city(self):
        translator = _make_translator()
        hint = translator._alignment_asr_hint(
            "We should get a move on and exit the city into reverse.",
            "",
            "You can see the reverse camera.",
        )

        error = translator._validate_alignment_asr_hint(
            "我们该动身 倒车离开市区",
            hint,
        )

        assert "selecting reverse gear" in error

    def test_alignment_asr_hint_resolves_body_adhesive_from_following_panel_context(self):
        translator = _make_translator()

        hint = translator._alignment_asr_hint(
            "They added 45 more feet of sort of goop",
            "",
            "within the body panels to make it more rigid.",
        )

        assert hint["kind"] == "body_adhesive_colloquialism"
        assert "structural body adhesive" in hint["normalized_source"]
        assert translator._validate_alignment_asr_hint("45英尺车身结构胶", hint) == ""
        assert "structural body adhesive" in translator._validate_alignment_asr_hint(
            "45英尺胶状物", hint
        )

    def test_alignment_asr_hint_uses_only_explicitly_labelled_context_variants(self):
        translator = _make_translator()
        translator.translation_context = TranslationContext(
            terminology=(
                "- Chiarco roll -> GR Corolla (probable ASR phonetic correction)\n"
                "- Sport mode -> 运动模式 (preferred translation)"
            )
        )

        hint = translator._alignment_asr_hint(
            "Make your Chiarco roll louder.",
            "",
            "",
        )

        assert hint["kind"] == "context_confirmed_asr_variant"
        assert "GR Corolla" in hint["normalized_source"]
        assert translator._context_asr_variant("Select Sport mode.") == {}

    @pytest.mark.parametrize(
        ("heard, canonical, translation"),
        [
            (
                "Honda Civic Type are last year",
                "Honda Civic Type R last year",
                "去年我们测试过本田思域Type R",
            ),
            ("Buick Riata", "Buick Reatta", "快看那辆别克Reatta"),
        ],
    )
    def test_context_asr_variant_accepts_translated_brand_and_trailing_time_context(
        self, heard, canonical, translation
    ):
        translator = _make_translator()
        translator.translation_context = TranslationContext(
            terminology=f"- {heard} -> {canonical} (probable ASR correction)"
        )

        hint = translator._alignment_asr_hint(heard, "", "")

        assert hint["kind"] == "context_confirmed_asr_variant"
        assert translator._validate_alignment_asr_hint(translation, hint) == ""

    def test_context_asr_variant_rejects_unsupported_one_off_name_guess(self):
        translator = _make_translator()
        translator.translation_context = TranslationContext(
            terminology=(
                "- Lexus LMXX Grimina or something -> Lexus LMXX GRMN or something "
                "(probable ASR correction)"
            )
        )
        translator._all_source_by_index = {
            1: "The GR Corolla uses this engine.",
            2: "And the Lexus LMXX Grimina or something has it too.",
        }

        assert (
            translator._context_asr_variant("And the Lexus LMXX Grimina or something has it too.")
            == {}
        )

    def test_context_asr_variant_accepts_model_correction_supported_by_document_subject(self):
        translator = _make_translator()
        translator.translation_context = TranslationContext(
            terminology=("- Grimina GR Corolla -> GRMN GR Corolla (probable ASR correction)")
        )
        translator._all_source_by_index = {
            1: "Today we drive the GR Corolla.",
            2: "The GR Corolla is a hot hatch.",
            3: "The new Grimina GR Corolla costs more.",
        }

        hint = translator._context_asr_variant("The new Grimina GR Corolla costs more.")

        assert hint["canonical"] == "GRMN GR Corolla"

    def test_source_for_translation_uses_confirmed_context_without_mutating_source(self):
        translator = _make_translator()
        translator.translation_context = TranslationContext(
            terminology=(
                "- The known particles explain 2% or 5% -> "
                "The known particles explain 5% "
                "(probable ASR correction caused by a spoken self-correction)"
            )
        )
        source = "The known particles explain 2% or 5% of the universe."
        translator._all_source_by_index = {
            1: source,
            2: "The remaining 95% is still unknown.",
        }

        normalized = translator._source_for_translation(source)

        assert normalized == "The known particles explain 5% of the universe."
        assert source == "The known particles explain 2% or 5% of the universe."

    def test_source_for_translation_rejects_unsupported_entity_guess(self):
        translator = _make_translator()
        translator.translation_context = TranslationContext(
            terminology="- Rick -> RHIC (probable ASR correction)"
        )
        translator._all_source_by_index = {1: "Rick is nearby."}

        assert translator._source_for_translation("Rick is nearby.") == "Rick is nearby."

    def test_context_epithet_target_rejects_a_literal_calque(self):
        translator = _make_translator()
        translator.translation_context = TranslationContext(
            terminology=(
                "- Great White North -> 北方雪国 "
                "(confirmed cultural or geographic epithet)"
            )
        )

        valid, error = translator._validate_llm_response(
            {"1": "为什么大白北决定现在建造超高层建筑"},
            {"1": "Why has the Great White North decided to build a supertall now?"},
        )

        assert not valid
        assert "1:北方雪国" in error

    def test_context_epithet_target_accepts_reviewed_rendering(self):
        translator = _make_translator()
        translator.translation_context = TranslationContext(
            terminology=(
                "- Great White North -> 北方雪国 "
                "(confirmed cultural or geographic epithet)"
            )
        )

        valid, error = translator._validate_llm_response(
            {"1": "为什么北方雪国决定现在建造超高层建筑"},
            {"1": "Why has the Great White North decided to build a supertall now?"},
        )

        assert valid, error

    def test_context_acronym_expansion_removes_adjacent_redundancy(self):
        translator = _make_translator()
        translator.translation_context = TranslationContext(
            terminology="- TMD -> 调谐质量阻尼器 (technical acronym)"
        )
        translator._all_source_by_index = {1: "This TMD is too large to lift in one go."}
        response = {"1": "这个TMD调谐质量阻尼器太大了"}

        translator._normalize_chinese_response_connectives(response)

        assert response["1"] == "这个调谐质量阻尼器太大了"

    def test_rejects_vague_target_for_concrete_technical_compound(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"1": "只是为了特定的墙体系统"},
            {"1": "just for the curtain wall system."},
        )

        assert not valid
        assert "concrete technical compound" in error

    def test_allows_vague_target_when_source_is_explicitly_general(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"1": "这只是某种墙体系统"},
            {"1": "This is just some kind of wall system."},
        )

        assert valid, error

    def test_context_mapping_rejects_translated_display_target_as_source_correction(self):
        translator = _make_translator()
        translator.translation_context = TranslationContext(
            terminology=(
                "- EPIC detector -> EPIC探测器 "
                "(probable ASR correction: context describes the detector)"
            )
        )

        assert translator._source_for_translation("The EPIC detector is here.") == (
            "The EPIC detector is here."
        )

    def test_preserved_tokens_rejects_dropped_spelled_number_fact(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"1": "事实上 AGS非常成功 而且至今仍在使用"},
            {"1": "AGS was so successful that three Nobel Prizes were awarded."},
        )

        assert valid is False
        assert "1:three" in error

    def test_preserved_tokens_accepts_chinese_spelled_number_fact(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"1": "AGS的成果催生了三项诺贝尔奖"},
            {"1": "AGS was so successful that three Nobel Prizes were awarded."},
        )

        assert valid is True, error

    def test_spelled_number_fact_guard_ignores_fractional_phrase(self):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"1": "面积约相当于6.5个美式足球场"},
            {"1": "It is approximately six and a half American football fields."},
        )

        assert valid is True, error

    def test_context_asr_variant_parses_nested_note_from_cached_context(self):
        translator = _make_translator()
        translator.translation_context = TranslationContext(
            terminology=(
                "- Infinity -> 英菲尼迪 (Probable ASR correction: 'Infinity' "
                "should be 'Infiniti' (brand name).)"
            )
        )
        translator._all_source_by_index = {
            1: "Today we drive the Infiniti QX65.",
            2: "This is your Infinity QX65.",
        }

        hint = translator._context_asr_variant("This is your Infinity QX65.")

        assert hint["canonical"] == "Infiniti"
        assert hint["normalized_source"] == "This is your Infiniti QX65."
        assert translator._validate_alignment_asr_hint("这是你的英菲尼迪QX65", hint) == ""

    def test_alignment_asr_hint_prefers_grmn_over_an_unrelated_document_model(self):
        translator = _make_translator()
        translator.custom_prompt = "2026 Toyota GR Corolla review"
        translator.translation_context = TranslationContext(
            terminology="- Morizo GR Corolla (track trim)"
        )
        translator._all_source_by_index = {
            1: "The Morizo GR Corolla was offered previously.",
            2: "The new Grimina GR Corolla costs $65,000.",
        }

        hint = translator._alignment_asr_hint(
            "The new Grimina GR Corolla costs $65,000.",
            "",
            "",
        )

        assert hint["canonical"] == "GRMN GR Corolla"
        assert "GRMN GR Corolla" in hint["normalized_source"]

    @pytest.mark.parametrize(
        ("source", "translation", "valid"),
        [
            (
                "That is quite good, fellow Corolla coming up.",
                "那辆Corolla很不错 正开过来",
                False,
            ),
            (
                "That is quite good, fellow Corolla coming up.",
                "那辆真不错 有辆Corolla开过来了",
                True,
            ),
            (
                "you are able to do your limo stops in this",
                "你能用它做出平稳的礼宾式停车",
                False,
            ),
            (
                "you are able to do your limo stops in this",
                "你能像豪华轿车司机那样平稳刹停",
                True,
            ),
            (
                "they've added 45 more feet of goop within the body panels",
                "他们在车身面板里又加了45英尺的胶状物",
                False,
            ),
            (
                "they've added 45 more feet of goop within the body panels",
                "他们在车身面板里又加了45英尺长的车身结构胶",
                True,
            ),
            (
                "Well, now it can run even cooler.",
                "现在它能跑得更凉快了",
                False,
            ),
            (
                "Well, now it can run even cooler.",
                "现在它的运行温度能更低",
                True,
            ),
            (
                "I'm going to downshift a second.",
                "我要降一挡",
                False,
            ),
            (
                "I'm going to downshift a second.",
                "我要稍微降一下挡",
                True,
            ),
            (
                "if your driver is doing the bat out of hell portion of the commute",
                "如果司机像地狱蝙蝠一样开",
                False,
            ),
            (
                "if your driver is doing the bat out of hell portion of the commute",
                "如果司机一路开得飞快",
                True,
            ),
            (
                "rolling down this parking structure in first gear",
                "挂着一挡慢慢驶出停车楼",
                False,
            ),
            (
                "rolling down this parking structure in first gear",
                "挂着一挡在停车楼里慢慢往下开",
                True,
            ),
            (
                "Rev match first? Yes, indeed it will.",
                "降挡补油 先挂一挡 是的 确实会",
                False,
            ),
            (
                "Rev match first? Yes, indeed it will.",
                "先说降挡补油 是的 它确实会自动补油",
                True,
            ),
            (
                "I don't know that you're putting somebody in the middle.",
                "我不确定你会让谁坐中间",
                False,
            ),
            (
                "I don't know that you're putting somebody in the middle.",
                "不过我觉得不会有人坐中间",
                True,
            ),
            (
                "We're now on the fourth model year of this thing.",
                "这车现在已经是第四代了",
                False,
            ),
            (
                "We're now on the fourth model year of this thing.",
                "这辆车现在已经到第四个车型年份了",
                False,
            ),
            (
                "We're now on the fourth model year of this thing.",
                "这车现在已经进入第四个年款了",
                True,
            ),
        ],
    )
    def test_validator_handles_contextual_driving_expressions(self, source, translation, valid):
        translator = _make_translator()

        ok, _message = translator._validate_llm_response(
            {"1": translation},
            {"1": source},
            require_reflect=False,
        )

        assert ok is valid

    def test_validator_uses_document_context_for_fuel_economy_and_revised_component(self):
        translator = _make_translator()
        translator._all_source_by_index = {
            26: "This is rated at 21 mpg in the city.",
            27: "so we'll have to see how we do",
            341: "Why don't we show you this newly revised",
            342: "nine speaker JBL sound system",
        }

        ok, message = translator._validate_llm_response(
            {
                "27": "所以这台车开起来如何 我们还得再看看",
                "341": "不如展示一下这款新改款车型",
            },
            {
                "27": translator._all_source_by_index[27],
                "341": translator._all_source_by_index[341],
            },
            require_reflect=False,
        )

        assert ok is False
        assert "fuel economy" in message
        assert "audio/component" in message

    @pytest.mark.parametrize(
        ("source", "canonical"),
        [
            ("The Lexus LMXX Grimina has this engine.", "Lexus LBX Morizo RR"),
            ("It has a Marizzo-style spoiler.", "Morizo"),
        ],
    )
    def test_alignment_asr_hint_normalizes_confirmed_automotive_names(self, source, canonical):
        translator = _make_translator()

        hint = translator._alignment_asr_hint(source, "", "")

        assert hint["canonical"] == canonical
        assert canonical in hint["normalized_source"]
        assert translator._validate_alignment_asr_hint(canonical, hint) == ""

    def test_known_automotive_name_hint_rejects_partial_canonical_name(self):
        translator = _make_translator()
        hint = translator._alignment_asr_hint(
            "The Lexus LMXX Grimina has this engine.",
            "",
            "",
        )

        error = translator._validate_alignment_asr_hint(
            "雷克萨斯LMXX Morizo RR也搭载这台发动机",
            hint,
        )

        assert "Lexus LBX Morizo RR" in error

    def test_alignment_repair_validates_confirmed_normalized_source(self, monkeypatch):
        translator = _make_translator()
        translator.model = "deepseek-v4-flash"
        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **_kwargs: _text_response("雷克萨斯LBX Morizo RR也搭载这台发动机"),
        )

        translated = translator._translate_alignment_item(
            "The Lexus LMXX Grimina has this engine as well.",
        )

        assert translated == "雷克萨斯LBX Morizo RR也搭载这台发动机"

    def test_alignment_asr_hint_uses_repeated_elantra_n_to_correct_one_off_m(self):
        translator = _make_translator()
        translator._all_source_by_index = {
            1: "The Elantra N is more aggressive.",
            2: "and the Bose and the Elantra M.",
        }

        hint = translator._alignment_asr_hint(translator._all_source_by_index[2], "", "")

        assert hint["canonical"] == "Elantra N"
        assert "Elantra N" in hint["normalized_source"]
        assert "Bose system in the Elantra N" in hint["normalized_source"]

        ok, message = translator._validate_llm_response(
            {"2": "也跟Elantra M上的Bose音响差不多"},
            {"2": translator._all_source_by_index[2]},
            require_reflect=False,
        )
        assert ok is False
        assert "Elantra N" in message

    @pytest.mark.parametrize(
        "source",
        [
            "Make your Chiarco roll a little louder.",
            "We feel it here in this 26 GR Cruel.",
        ],
    )
    def test_alignment_asr_hint_reconciles_close_model_variant_with_repeated_subject(self, source):
        translator = _make_translator()
        translator.custom_prompt = "2026 Toyota GR Corolla real-world review"
        translator._all_source_by_index = {
            1: "Today we drive the GR Corolla.",
            2: "The GR Corolla is a hot hatch.",
        }

        hint = translator._alignment_asr_hint(source, "", "")

        assert hint["kind"] == "document_repeated_model_variant"
        assert "GR Corolla" in hint["normalized_source"]
        assert translator._validate_alignment_asr_hint("这辆GR Corolla", hint) == ""
        assert "canonical" in translator._validate_alignment_asr_hint("这辆车", hint)

    def test_document_model_variant_reconciles_short_trim_suffix(self):
        translator = _make_translator()
        translator._all_source_by_index = {
            1: "The Elantra N is the wild one.",
            2: "Compared with the Elantra N.",
        }

        hint = translator._alignment_asr_hint(
            "and the Bose and the Elantra M.",
            "",
            "",
        )

        assert hint["kind"] == "document_repeated_model_variant"
        assert "Elantra N" in hint["normalized_source"]

    def test_document_model_variant_uses_context_terminology_as_repeat_evidence(self):
        translator = _make_translator()
        translator.translation_context = TranslationContext(
            terminology="- Hyundai Elantra N -> Hyundai Elantra N (proper noun)"
        )
        translator._all_source_by_index = {
            1: "The Bose system in the Elantra N is similar.",
            2: "and the Bose and the Elantra M.",
        }

        hint = translator._alignment_asr_hint(
            "and the Bose and the Elantra M.",
            "",
            "",
        )

        assert hint["canonical"] == "Elantra N"

    def test_document_model_variant_overrides_weaker_context_guess(self):
        translator = _make_translator()
        translator.custom_prompt = "2026 Toyota GR Corolla review"
        translator._all_source_by_index = {
            1: "Today we drive the GR Corolla.",
            2: "The GR Corolla is a hot hatch.",
        }
        translator.translation_context = TranslationContext(
            terminology="- Chiarco roll -> Chiaro (probable ASR correction)"
        )

        hint = translator._alignment_asr_hint(
            "Make your Chiarco roll louder.",
            "",
            "",
        )

        assert hint["kind"] == "document_repeated_model_variant"
        assert "GR Corolla" in hint["normalized_source"]

    def test_document_model_variant_never_replaces_an_already_canonical_name(self):
        translator = _make_translator()
        translator.custom_prompt = "2026 Toyota GR Corolla review"
        translator.translation_context = TranslationContext(
            terminology=("- GR Cruel -> GR Corolla (probable ASR correction)")
        )
        translator._all_source_by_index = {
            1: "Today we drive the GR Corolla.",
            2: "The GR Corolla is a hot hatch.",
            3: "We feel it here in this 26 GR Cruel.",
        }

        assert (
            translator._document_model_asr_variant(
                "Before living with this GR Corolla, let's inspect the changes."
            )
            == {}
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

    def test_alignment_audit_keeps_successful_repairs_when_one_key_fails(self, monkeypatch):
        translator = _make_translator(is_reflect=True)
        source = {"1": "The steering is precise.", "2": "The ride is firm."}
        translated = {"1": "转向很准", "2": "行驶很稳"}
        flags = iter([["1", "2"], ["1", "2"], []])
        monkeypatch.setattr(
            translator,
            "_request_alignment_flags",
            lambda *_args, **_kwargs: next(flags),
        )

        def repair(text, **_kwargs):
            if "ride" in text.lower():
                raise ValueError("temporary malformed answer")
            return "转向精准"

        monkeypatch.setattr(translator, "_translate_alignment_item", repair)

        result = translator._audit_reflective_alignment(source, translated)

        assert result == {"1": "转向精准", "2": "行驶很稳"}
        assert translator._pending_alignment_repair_keys == {2}

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

    def test_discourse_filler_does_not_hide_a_shift_terminating_duplicate(self):
        translator = _make_translator()

        ok, message = translator._validate_llm_response(
            {
                "78": "我在文章中提到退休人员、大学毕业生和女性",
                "79": "我在文章里提到退休人员、大学毕业生和女性",
            },
            {
                "78": "This is something that, you know, stretches across demographics.",
                "79": "You know, I note in the piece that retirees and college graduates and women.",
            },
        )

        assert ok is False
        assert "repeat" in message.lower()

    def test_batch_tail_duplicate_widens_repair_to_original_batch(self):
        translator = _make_translator()
        translator.batch_num = 20
        source = [
            SubtitleProcessData(index=index, original_text=f"Distinct source {index}.")
            for index in range(1, 21)
        ]
        source[17] = replace(
            source[17],
            original_text="This stretches across demographics.",
        )
        source[18] = replace(
            source[18],
            original_text="Retirees and college graduates read the most.",
        )
        response = {str(index): f"不同译文{index}" for index in range(1, 21)}
        response["18"] = "我在文章中提到退休人员大学毕业生和女性"
        response["19"] = "我在文章里提到退休人员大学毕业生和女性"

        widened = translator._widen_batch_tail_shift_repair(
            source,
            18,
            {"18": response["18"], "19": response["19"]},
            source[17:19],
            repetition_failure=True,
        )

        assert [item.index for item in widened] == list(range(1, 21))

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
        assert "Perform the draft and audit internally" in calls[0]["messages"][0]["content"]
        assert '"initial_translation"' not in calls[0]["messages"][0]["content"]

    def test_reflect_validation_accepts_plain_final_translation_values(self):
        translator = _make_translator(is_reflect=True)

        ok, message = translator._validate_llm_response(
            {"1": "你好"},
            {"1": "Hello"},
        )

        assert ok is True
        assert message == ""

    def test_deepseek_v4_reflect_agent_keeps_native_reasoning_disabled(self, monkeypatch):
        translator = _make_translator(is_reflect=True)
        translator.model = "deepseek-v4-flash"
        calls = []

        def fake_call_llm(**kwargs):
            calls.append(kwargs)
            return _llm_response({"1": {"native_translation": "你好"}})

        monkeypatch.setattr("subforge.core.translate.llm_translator.call_llm", fake_call_llm)

        assert translator._agent_loop("prompt", {"1": "hello"}) == {
            "1": {"native_translation": "你好"}
        }
        assert [call["reasoning_mode"] for call in calls] == ["disabled"]
        assert [call["max_output_tokens"] for call in calls] == [4096]
        system_prompt = calls[0]["messages"][0]["content"]
        assert "Perform the draft and audit internally" in system_prompt
        assert '"initial_translation"' not in system_prompt

    def test_deepseek_v4_large_batch_keeps_reasoning_disabled(self, monkeypatch):
        translator = _make_translator(is_reflect=True)
        translator.model = "deepseek-v4-flash"
        source = {str(index): f"source {index}" for index in range(1, 21)}
        response = {str(index): {"native_translation": f"译文{index}"} for index in range(1, 21)}
        calls = []

        def fake_call_llm(**kwargs):
            calls.append(kwargs)
            return _llm_response(response)

        monkeypatch.setattr("subforge.core.translate.llm_translator.call_llm", fake_call_llm)

        assert translator._agent_loop("prompt", source) == response
        assert [call["reasoning_mode"] for call in calls] == ["disabled"]
        assert [call["max_output_tokens"] for call in calls] == [4096]

    def test_deepseek_v4_agent_never_enables_reasoning_for_format_retry(self, monkeypatch):
        translator = _make_translator(is_reflect=True)
        translator.model = "deepseek-v4-flash"
        responses = iter(
            [
                _text_response("<think>reasoning without final JSON"),
                _llm_response({"1": {"native_translation": "你好"}}),
            ]
        )
        calls = []

        def fake_call_llm(**kwargs):
            calls.append(kwargs)
            return next(responses)

        monkeypatch.setattr("subforge.core.translate.llm_translator.call_llm", fake_call_llm)

        assert translator._agent_loop("prompt", {"1": "hello"}) == {
            "1": {"native_translation": "你好"}
        }
        assert [call["reasoning_mode"] for call in calls] == ["disabled", "disabled"]
        assert all("reasoning_effort" not in call for call in calls)

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

    def test_single_fallback_accepts_language_neutral_numeric_reply(self, monkeypatch):
        t = _make_translator()
        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **_kwargs: _text_response("100%"),
        )

        result = t._translate_chunk_single([SubtitleProcessData(index=637, original_text="100%.")])

        assert result[0].translated_text == "100%"

    @pytest.mark.parametrize(
        ("source", "translation"),
        [
            ("especially if you're in the UK", "尤其是如果你身处英国"),
            ("without producing CO2 emissions", "同时不产生二氧化碳排放"),
        ],
    )
    def test_single_fallback_accepts_standard_chinese_token_equivalents(
        self, monkeypatch, source, translation
    ):
        translator = _make_translator()
        calls = []
        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **kwargs: calls.append(kwargs) or _text_response(translation),
        )

        result = translator._translate_chunk_single(
            [SubtitleProcessData(index=1, original_text=source)]
        )

        assert result[0].translated_text == translation
        assert len(calls) == 1

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

    def test_finalizer_repairs_cross_key_acronym_ownership(self, monkeypatch):
        translator = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(index=1, original_text="The reactor is ready."),
            SubtitleProcessData(index=2, original_text="SMR testing starts next year."),
        ]
        translated = [
            replace(source[0], translated_text="这座SMR已经准备就绪"),
            replace(source[1], translated_text="SMR测试将于明年开始"),
        ]
        repair_calls = []

        def repair_batch(items, initial_feedback=""):
            repair_calls.append(initial_feedback)
            return [
                replace(source[0], translated_text="这座反应堆已经准备就绪"),
                replace(source[1], translated_text="SMR测试将于明年开始"),
            ]

        monkeypatch.setattr(translator, "_translate_locked_batch", repair_batch)

        result = translator._finalize_translated_list(source, translated)

        assert repair_calls
        assert "Cross-key duplicates" in repair_calls[0]
        assert [item.translated_text for item in result] == [
            "这座反应堆已经准备就绪",
            "SMR测试将于明年开始",
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

    def test_pending_alignment_repair_uses_local_window_not_full_large_batch(self):
        t = _make_minimax_reflect_translator()
        t.batch_num = 20
        source = [
            SubtitleProcessData(index=index, original_text=f"Source {index}.")
            for index in range(1, 21)
        ]
        translated = {
            item.index: replace(item, translated_text=f"译文{item.index}") for item in source
        }

        windows = t._pending_alignment_repair_windows(source, translated, [10])

        assert [[item.index for item in window] for window in windows] == [[9, 10, 11]]

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
        assert (
            LLMTranslator._chinese_boundary_signal(
                "就像它会让这台车变得……",
                "比现在还要出色得多",
            )
            == "unfinished Chinese grammatical structure"
        )

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

    def test_deterministic_fallback_repairs_copula_and_additive_connector(self):
        source = [
            SubtitleProcessData(
                index=510,
                original_text="or send you flying into the ditch or something. It is just",
            ),
            SubtitleProcessData(
                index=511,
                original_text="arrow straight when you get on it. And then",
            ),
            SubtitleProcessData(
                index=512,
                original_text="the gears aren't so long that you can't go through them all.",
            ),
        ]
        current = [
            replace(source[0], translated_text="或者把你甩进沟里之类的 它就是"),
            replace(source[1], translated_text="一踩油门就笔直向前 而且"),
            replace(source[2], translated_text="挡位也不是那么长 让你没法全过一遍"),
        ]

        repaired = LLMTranslator._deterministic_chinese_fluency_fallback(source, current)

        assert repaired is not None
        assert [item.translated_text for item in repaired] == [
            "或者把你甩进沟里之类的",
            "一踩油门就笔直向前",
            "挡位也不是那么长 让你没法全过一遍",
        ]

    def test_deterministic_fallback_moves_adversative_before_discourse_filler(self):
        source = [
            SubtitleProcessData(
                index=565,
                original_text="is going to be the best idea but",
            ),
            SubtitleProcessData(
                index=566,
                original_text="well we'll give it a shot I guess",
            ),
        ]
        current = [
            replace(source[0], translated_text="是不是最好的主意 但是"),
            replace(source[1], translated_text="好吧 我想我们就试试看吧"),
        ]

        repaired = LLMTranslator._deterministic_chinese_fluency_fallback(source, current)

        assert repaired is not None
        assert [item.translated_text for item in repaired] == [
            "是不是最好的主意",
            "不过 我想我们就试试看吧",
        ]

    def test_deterministic_fallback_moves_trailing_document_connector(self):
        source = [
            SubtitleProcessData(
                index=183,
                original_text="The government isn't paying. On the other hand,",
            ),
            SubtitleProcessData(
                index=184,
                original_text="supporters expect the project to transform the country.",
            ),
        ]
        current = [
            replace(source[0], translated_text="政府并未出资 另一方面"),
            replace(source[1], translated_text="支持者认为该项目将改变这个国家"),
        ]

        repaired = LLMTranslator._deterministic_chinese_fluency_fallback(source, current)

        assert repaired is not None
        assert [item.translated_text for item in repaired] == [
            "政府并未出资",
            "另一方面 支持者认为该项目将改变这个国家",
        ]

    def test_deterministic_fallback_moves_connector_before_terminal_punctuation(self):
        source = [
            SubtitleProcessData(
                index=330,
                original_text="Because, of course, it's not adjustable, so.",
            ),
            SubtitleProcessData(
                index=331,
                original_text="I have to get used to how this car handles.",
            ),
        ]
        current = [
            replace(source[0], translated_text="毕竟它无法调节 所以。"),
            replace(source[1], translated_text="我只能适应这辆车的操控感"),
        ]

        repaired = LLMTranslator._deterministic_chinese_fluency_fallback(source, current)

        assert repaired is not None
        assert [item.translated_text for item in repaired] == [
            "毕竟它无法调节",
            "所以 我只能适应这辆车的操控感",
        ]

    @pytest.mark.parametrize(
        ("left", "right", "signal"),
        [
            (
                "要把声誉提升到新高度 就必须大力",
                "改造现有机场",
                "unfinished Chinese predicate or governing word",
            ),
            (
                "项目获得的资金由",
                "国家航空公司提供",
                "unfinished Chinese predicate or governing word",
            ),
            (
                "航站楼希望能",
                "每年处理数百万旅客",
                "unfinished Chinese predicate or governing word",
            ),
            (
                "成熟枢纽每年接待",
                "2000万至3000万名乘客",
                "transitive predicate is split from its quantified object",
            ),
            ("该机场年吞吐量超过2000万人次", "每年", "standalone Chinese temporal fragment"),
            ("这是整个设计中", "至关重要的一环", "locative frame is separated from its complement"),
            (
                "计划在既有优势的基础上",
                "进一步发展航空市场",
                "locative phrase is separated from its predicate",
            ),
            (
                "机场位于大约35公里之外",
                "新址人口密度较低",
                "distance modifier is separated from its noun",
            ),
        ],
    )
    def test_chinese_boundary_signal_catches_general_incomplete_units(self, left, right, signal):
        assert LLMTranslator._chinese_boundary_signal(left, right) == signal

    @pytest.mark.parametrize(
        ("source", "token", "translated"),
        [
            ("closing $8BN in a year", "8BN", "一年内完成80亿美元融资"),
            ("a $12.5BN airport", "12.5BN", "一座耗资125亿美元的机场"),
            ("about $4.5BN of funding", "4.5BN", "约45亿美元资金"),
        ],
    )
    def test_compact_monetary_magnitude_accepts_natural_chinese(self, source, token, translated):
        assert LLMTranslator._localized_magnitude_rendered(source, token, translated)

    def test_compact_magnitude_rejects_nonfinancial_model_suffix(self):
        assert not LLMTranslator._localized_magnitude_rendered(
            "The 8B processor is available now.",
            "8B",
            "这款80亿处理器现已上市",
        )

    def test_chinese_boundary_signal_accepts_complete_design_aspect(self):
        assert (
            LLMTranslator._chinese_boundary_signal(
                "这是设计中的一个关键方面",
                "因为大多数旅客只在这里转机",
            )
            == ""
        )

    def test_deterministic_fallback_repairs_cross_speaker_comparison(self):
        source = [
            SubtitleProcessData(
                index=627,
                original_text="They're much better quality than most other",
            ),
            SubtitleProcessData(
                index=628,
                original_text="socks that I've found.",
            ),
        ]
        current = [
            replace(source[0], translated_text="它们的质量比大多数其他"),
            replace(source[1], translated_text="我找到的袜子要好得多"),
        ]

        repaired = LLMTranslator._deterministic_chinese_fluency_fallback(source, current)

        assert repaired is not None
        assert [item.translated_text for item in repaired] == [
            "它们的质量远胜于大多数同类袜子",
            "至少在我找到的袜子中是这样",
        ]

    def test_deterministic_fallback_repairs_perspective_frame(self):
        source = [
            SubtitleProcessData(
                index=671,
                original_text="In many ways, it's actually, you know,",
            ),
            SubtitleProcessData(
                index=672,
                original_text="sometimes a more effective way to get information",
            ),
        ]
        current = [
            replace(source[0], translated_text="在很多方面 它实际上"),
            replace(source[1], translated_text="有时候是获取信息更有效的方式"),
        ]

        repaired = LLMTranslator._deterministic_chinese_fluency_fallback(source, current)

        assert repaired is not None
        assert [item.translated_text for item in repaired] == [
            "其实换个角度看",
            "有时候是获取信息更有效的方式",
        ]

    @pytest.mark.parametrize(
        ("source_texts", "current_texts", "expected"),
        [
            (
                ["Is it sort of like", "We're just back to where we started?"],
                ["是不是有点像", "我们只是回到了起点吗？"],
                ["这算不算某种倒退？", "仿佛我们只是回到了起点？"],
            ),
            (
                [
                    "People put more emphasis on grabbing attention in the first 10",
                    "or 15 seconds.",
                ],
                ["人们会更注重在最初的10", "到15秒内抓住注意力"],
                [
                    "人们会更加注重在最初10秒内抓住注意力",
                    "有时甚至会把这个窗口放宽到15秒",
                ],
            ),
            (
                [
                    "And I think that would need to be a really",
                    "large scale shift for people to make.",
                ],
                ["我认为这需要人们做出一个", "非常大规模的转变"],
                ["我认为 这会要求人们真正行动起来", "共同推动一次大规模转变"],
            ),
        ],
    )
    def test_multispeaker_deterministic_fallback_repairs_confirmed_dialogue_breaks(
        self,
        source_texts,
        current_texts,
        expected,
    ):
        source = [
            SubtitleProcessData(index=index, original_text=text)
            for index, text in enumerate(source_texts, 1)
        ]
        current = [replace(item, translated_text=text) for item, text in zip(source, current_texts)]

        repaired = LLMTranslator._deterministic_chinese_fluency_fallback(
            source,
            current,
            multispeaker=True,
        )

        assert repaired is not None
        assert [item.translated_text for item in repaired] == expected

    def test_multispeaker_deterministic_fallback_repairs_standalone_connector(self):
        source_texts = [
            "about or the hyperlinks as much so that's going to be easier to focus on.",
            "And so",
            "it does seem that people read better on e-readers than on phones.",
        ]
        source = [
            SubtitleProcessData(index=index, original_text=text)
            for index, text in enumerate(source_texts, 1)
        ]
        current = [
            replace(item, translated_text=text)
            for item, text in zip(
                source,
                ["超链接也没那么多 所以更容易集中注意力", "所以", "电子阅读器更好"],
            )
        ]

        repaired = LLMTranslator._deterministic_chinese_fluency_fallback(
            source,
            current,
            multispeaker=True,
        )

        assert repaired is not None
        assert [item.translated_text for item in repaired] == [
            "超链接也没那么多 因此更容易集中注意力",
            "这也会带来实际差异",
            "电子阅读器更好",
        ]

    def test_dialogue_fallback_does_not_change_single_speaker_path(self):
        source = [
            SubtitleProcessData(index=1, original_text="Is it sort of like"),
            SubtitleProcessData(
                index=2,
                original_text="We're just back to where we started?",
            ),
        ]
        current = [
            replace(source[0], translated_text="是不是有点像"),
            replace(source[1], translated_text="我们只是回到了起点吗？"),
        ]

        repaired = LLMTranslator._deterministic_chinese_fluency_fallback(
            source,
            current,
            multispeaker=False,
        )

        assert repaired is None

    @pytest.mark.parametrize(
        ("sources", "translations", "expected"),
        [
            (
                [
                    "Are we just talking kids?",
                    "Are we just talking adults? Are we talking everybody?",
                    "So I think we're talking everybody.",
                ],
                [
                    "还是只是在说成年人？我们说的是所有人吗？",
                    "我觉得我们说的是所有人",
                    "所以我觉得我们说的是所有人",
                ],
                [
                    "还是只是在说孩子？",
                    "还是只是在说成年人？我们说的是所有人吗？",
                    "我觉得我们说的是所有人",
                ],
            ),
            (
                [
                    "And then at the same time that we've seen books leaving the classroom,",
                    "we've definitely seen,",
                    "tablets and Chromebooks and laptops entering the classroom much more.",
                ],
                ["而就在我们看到书籍离开课堂的同时", "我们也确实看到", "更多设备进入课堂"],
                ["与此同时 书籍正逐渐退出课堂", "我们也明显看到了另一种变化", "更多设备进入课堂"],
            ),
            (
                [
                    "I think at the same time,",
                    "an example that was really illustrative to me",
                    "and that I talk about in the piece is this work by Plato, the Phaedrus.",
                ],
                ["不过与此同时", "有一个例子让我印象很深", "我在文章里谈到了柏拉图"],
                ["不过 我也想到了另一个角度", "有个例子尤其能说明问题", "我在文章里谈到了柏拉图"],
            ),
            (
                [
                    "They didn't have the same experimental standards we would now in the 1930s",
                    "when this research was being done.",
                    "But I think Ong's larger point that",
                ],
                ["显然 在20世纪30年代", "当这项研究进行时", "但Ong的观点是"],
                [
                    "当然 这项研究开展于20世纪30年代",
                    "当时的实验标准与今天并不完全相同",
                    "但Ong的观点是",
                ],
            ),
            (
                [
                    "But I think Ong's larger point that",
                    "Literate cultures value sustained linear argumentation with evidence",
                    "and counterpoints, does stand.",
                ],
                [
                    "但我认为Ong更宏观的观点 即",
                    "文字文化重视持续论证和证据",
                    "和反驳观点 这一点成立",
                ],
                [
                    "但我认为 Ong 更宏观的观点仍然成立",
                    "文字文化重视持续的线性论证和证据",
                    "也重视对不同观点的反驳",
                ],
            ),
        ],
    )
    def test_repairs_confirmed_multispeaker_dialogue_sequences(
        self,
        sources,
        translations,
        expected,
    ):
        source_items = [
            SubtitleProcessData(index=index, original_text=text)
            for index, text in enumerate(sources, 1)
        ]
        translated = {
            item.index: replace(item, translated_text=text)
            for item, text in zip(source_items, translations)
        }

        LLMTranslator._repair_multispeaker_dialogue_sequences(
            source_items,
            translated,
        )

        assert [translated[index].translated_text for index in sorted(translated)] == expected

    @pytest.mark.parametrize(
        ("sources", "expected"),
        [
            (
                [
                    "what's unique about post-literacy",
                    "and the broader phenomenon that's occurring",
                    "text is no longer the main way",
                    "people transmit information, news, entertainment and cultural connection",
                    "the change is that it shifted from text to video",
                ],
                [
                    "但我认为 后文字时代的独特之处在于这种转变",
                    "它也反映了当下更广泛的趋势",
                    "文字已不再是人们传递信息的主要方式",
                    "新闻、娱乐和文化联系也不再以文字为主",
                    "真正的变化 是重心从文字转向了视频",
                ],
            ),
            (
                [
                    "So I think that, you know, I think,",
                    "I don't know that I asked that exact question in my reporting,",
                    "but I think that, you know,",
                    "it wasn't something that was specific to text that allowed for revolutions.",
                ],
                [
                    "关于这个问题 我得先说明",
                    "我在报道中没有直接追问这一点",
                    "但我的判断是",
                    "促成革命的并不是文字本身",
                ],
            ),
            (
                [
                    "Like, that brings in new voices",
                    "and that's always destabilizing and uncomfortable",
                    "to people who previously had a monopoly on the ability to share information",
                ],
                [
                    "这会让更多新的声音进入公共讨论",
                    "这种变化总会让既得利益者感到不安",
                    "尤其是那些原本垄断信息传播的人",
                ],
            ),
            (
                [
                    "And I think that would need to be a really, you know,",
                    "large scale shift, you know, for people to make.",
                ],
                ["我认为 这会要求人们真正行动起来", "共同推动一次大规模转变"],
            ),
        ],
    )
    def test_repairs_long_multispeaker_dialogue_sequences(self, sources, expected):
        source_items = [
            SubtitleProcessData(index=index, original_text=text)
            for index, text in enumerate(sources, 1)
        ]
        translated = {
            item.index: replace(item, translated_text=f"旧译文{item.index}")
            for item in source_items
        }

        LLMTranslator._repair_multispeaker_dialogue_sequences(source_items, translated)

        assert [translated[index].translated_text for index in sorted(translated)] == expected

    def test_deterministic_fallback_repairs_resultative_pair(self):
        source = [
            SubtitleProcessData(
                index=360,
                original_text="Like it would just make this thing.",
            ),
            SubtitleProcessData(
                index=361,
                original_text=("So much more excellent than it already is and don't get me wrong."),
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

    def test_deterministic_fallback_repairs_ordinal_gear_split(self):
        translator = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(
                index=299,
                original_text="and you do have the torque, even here in fourth",
            ),
            SubtitleProcessData(
                index=300,
                original_text="gear at 30 miles an hour, half throttle,",
            ),
            SubtitleProcessData(
                index=301,
                original_text="It's able to pull itself along.",
            ),
        ]
        current = [
            replace(source[0], translated_text="而且你确实有扭矩 即使在四挡"),
            replace(source[1], translated_text="挂上挡 在时速30英里、半油门时"),
            replace(source[2], translated_text="它自己就能轻松往前行进"),
        ]

        repaired = translator._deterministic_chinese_fluency_fallback(source, current)

        assert repaired is not None
        assert [item.translated_text for item in repaired] == [
            "而且你确实有扭矩 即使在四挡",
            "在时速30英里、半油门时",
            "它自己就能轻松往前行进",
        ]

    def test_deterministic_fallback_repairs_rev_matching_split(self):
        translator = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(
                index=289,
                original_text=(
                    "This has Toyota's IMT intelligent manual transmission which is "
                    "just their fancy way of saying rev"
                ),
            ),
            SubtitleProcessData(
                index=290,
                original_text="matching which also aids smooth city driving",
            ),
        ]
        current = [
            replace(
                source[0],
                translated_text=(
                    "这车搭载了丰田的IMT智能手动变速箱 其实就是他们花哨的说法 指的是降挡"
                ),
            ),
            replace(
                source[1],
                translated_text="补油 这也能帮你在城市里开得更平顺",
            ),
        ]

        repaired = translator._deterministic_chinese_fluency_fallback(source, current)

        assert repaired is not None
        assert [item.translated_text for item in repaired] == [
            "这车搭载了丰田的IMT智能手动变速箱",
            "说白了就是降挡补油 这也能帮你在城市里开得更平顺",
        ]

    def test_deterministic_fallback_moves_revised_modifier_to_audio_component(self):
        translator = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(
                index=341,
                original_text="Why don't we show you this newly revised",
            ),
            SubtitleProcessData(
                index=342,
                original_text="nine speaker JBL sound system",
            ),
        ]
        current = [
            replace(source[0], translated_text="不如趁现在给你看看这个新改版的"),
            replace(source[1], translated_text="九扬声器JBL音响"),
        ]

        repaired = translator._deterministic_chinese_fluency_fallback(source, current)

        assert repaired is not None
        assert [item.translated_text for item in repaired] == [
            "不如趁现在给你看看",
            "这套新改版的九扬声器JBL音响",
        ]

    def test_deterministic_fallback_repairs_spare_wheel_cargo_sequence(self):
        translator = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(
                index=98,
                original_text=("and I don't think you could put one unless you just sort of set"),
            ),
            SubtitleProcessData(
                index=99,
                original_text="it in your cargo area back here",
            ),
            SubtitleProcessData(
                index=100,
                original_text=(
                    "and accepted that neither you nor your passengers are able to take "
                    "anything in the back"
                ),
            ),
        ]
        current = [
            replace(source[0], translated_text="我觉得你也放不下一个 除非直接把它"),
            replace(source[1], translated_text="放在后面的载物区里"),
            replace(source[2], translated_text="然后接受你和乘客没法在后面放任何东西"),
        ]

        repaired = translator._deterministic_chinese_fluency_fallback(source, current)

        assert repaired is not None
        assert [item.translated_text for item in repaired] == [
            "我觉得也放不下备胎 除非直接把备胎",
            "放在后面的载物区里",
            "那样一来 你和乘客就无法再往后备厢放其他东西",
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

    @pytest.mark.parametrize(
        ("left", "right", "expected_signal"),
        [
            (
                "我们把管片",
                "一块块地装入隧道",
                "ba construction is separated from its predicate",
            ),
            (
                "要把管片这些构件",
                "在我们开挖出的隧道内",
                "ba construction is separated from its predicate",
            ),
            (
                "一块块地安装到",
                "开挖出的隧道壁上",
                "predicate is separated from its required complement",
            ),
            (
                "依靠这些管片来承受",
                "隧道所受到的土压力和水压力",
                "predicate is separated from its required complement",
            ),
            (
                "而这",
                "正是难度极高的施工内容",
                "demonstrative subject is stranded",
            ),
            (
                "将既有混凝土墙",
                "用盾构机穿透",
                "disposal construction is separated from its predicate",
            ),
            (
                "用盾构机在世界首次",
                "实施此类穿越",
                "superlative modifier is separated from its predicate",
            ),
            (
                "这也是整项工程中",
                "非常困难的工程内容",
                "literal Japanese difficulty construction",
            ),
            (
                "在已经掘好的隧道里",
                "不断贴到土壁上",
                "locative phrase is separated from its predicate",
            ),
            (
                "沿着已经掘好的隧道",
                "不断压贴在土体上",
                "locative phrase is separated from its predicate",
            ),
            (
                "这也是",
                "难度极高的工程内容",
                "literal Japanese difficulty construction",
            ),
            (
                "该项工程即为如此推进的施工",
                "这正是一项难度极高的工程",
                "duplicated construction nominalization",
            ),
            (
                "这在世界盾构隧道工程中尚属首次",
                "是一项穿越施工工程",
                "duplicated construction nominalization",
            ),
            (
                "依靠这些来承受隧道所受的",
                "隧道所受到的土压力和水压力",
                "possible duplicated boundary phrase",
            ),
        ],
    )
    def test_chinese_boundary_signal_catches_mixed_japanese_translation_fragments(
        self, left, right, expected_signal
    ):
        assert LLMTranslator._chinese_boundary_signal(left, right) == expected_signal

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("将穿过既有混凝土墙", "这是世界首例"),
            ("将穿越既有混凝土墙", "这是世界首例"),
            ("由这些管片来承受", "下一条内容"),
        ],
    )
    def test_chinese_boundary_signal_accepts_completed_mixed_japanese_phrases(self, left, right):
        assert not LLMTranslator._chinese_boundary_signal(left, right)

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

    def test_display_boundary_metadata_classifies_visible_pause(self):
        translator = _make_translator()
        translator._gap_after_index = {1: 90, 2: 220, 3: 900}

        assert translator._display_boundary_metadata(1) == {
            "gap_after_ms": 90,
            "display_continuity": "continuous",
        }
        assert translator._display_boundary_metadata(2)["display_continuity"] == "brief_pause"
        assert translator._display_boundary_metadata(3)["display_continuity"] == "separated"
        assert translator._display_boundary_metadata(4) == {}

    def test_long_gap_numeric_atom_is_a_mandatory_fluency_candidate(self):
        translator = _make_minimax_reflect_translator()
        translator._gap_after_index = {1: 1944}
        source = [
            SubtitleProcessData(index=1, original_text="It costs a whopping $12.5"),
            SubtitleProcessData(index=2, original_text="billion, but it is urgently needed."),
        ]
        translated = {
            1: replace(source[0], translated_text="它的造价高达125"),
            2: replace(source[1], translated_text="亿美元 但这项工程迫在眉睫"),
        }

        assert translator._mandatory_chinese_fluency_candidates(source, translated) == [1]

    def test_long_gap_predicate_complement_is_a_mandatory_fluency_candidate(self):
        translator = _make_minimax_reflect_translator()
        translator._gap_after_index = {1: 2468}
        source = [
            SubtitleProcessData(index=1, original_text="The project is supported"),
            SubtitleProcessData(index=2, original_text="by the national airline."),
        ]
        translated = {
            1: replace(source[0], translated_text="这个项目获得了"),
            2: replace(source[1], translated_text="国家航空公司的支持"),
        }

        assert translator._mandatory_chinese_fluency_candidates(source, translated) == [1]

    def test_long_gap_trailing_connector_is_a_mandatory_fluency_candidate(self):
        translator = _make_minimax_reflect_translator()
        translator._gap_after_index = {1: 4700}
        source = [
            SubtitleProcessData(
                index=1,
                original_text="Because, of course, it's not adjustable, so.",
            ),
            SubtitleProcessData(
                index=2,
                original_text="I have to get used to the ergonomics of this car.",
            ),
        ]
        translated = {
            1: replace(source[0], translated_text="因为它无法调节 所以"),
            2: replace(source[1], translated_text="我得适应这辆车的驾驶姿势"),
        }

        assert translator._mandatory_chinese_fluency_candidates(source, translated) == [1]

    @pytest.mark.parametrize(
        ("left_source", "right_source", "left_translation", "right_translation"),
        [
            (
                "It costs a whopping $12.5",
                "billion, but it is urgently needed.",
                "它的造价高达125亿美元",
                "但这项工程迫在眉睫",
            ),
            (
                "The airport is supported",
                "by about $4.5BN from the airline.",
                "这座机场资金来源明确",
                "其中约45亿美元来自航空公司的资助",
            ),
        ],
    )
    def test_long_gap_source_dependency_does_not_rewrite_complete_chinese(
        self,
        left_source,
        right_source,
        left_translation,
        right_translation,
    ):
        translator = _make_minimax_reflect_translator()
        translator._gap_after_index = {1: 1900}
        source = [
            SubtitleProcessData(index=1, original_text=left_source),
            SubtitleProcessData(index=2, original_text=right_source),
        ]
        translated = {
            1: replace(source[0], translated_text=left_translation),
            2: replace(source[1], translated_text=right_translation),
        }

        assert translator._mandatory_chinese_fluency_candidates(source, translated) == []

    def test_long_gap_complete_locative_does_not_force_rewrite(self):
        translator = _make_minimax_reflect_translator()
        translator._gap_after_index = {1: 650}
        source = [
            SubtitleProcessData(index=1, original_text="Over in East Africa,"),
            SubtitleProcessData(index=2, original_text="Ethiopia is building a new airport."),
        ]
        translated = {
            1: replace(source[0], translated_text="而在东非"),
            2: replace(source[1], translated_text="埃塞俄比亚正在建设一座新机场"),
        }

        assert translator._mandatory_chinese_fluency_candidates(source, translated) == []

    @pytest.mark.parametrize("left_translation", ["它拥有出色性能", "这是一场发布会", "这是我的"])
    def test_long_gap_complete_chinese_word_endings_do_not_force_rewrite(
        self,
        left_translation,
    ):
        translator = _make_minimax_reflect_translator()
        translator._gap_after_index = {1: 900}
        source = [
            SubtitleProcessData(index=1, original_text="That statement is complete."),
            SubtitleProcessData(index=2, original_text="A new sentence starts here."),
        ]
        translated = {
            1: replace(source[0], translated_text=left_translation),
            2: replace(source[1], translated_text="下一句从这里开始"),
        }

        assert translator._mandatory_chinese_fluency_candidates(source, translated) == []

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
            (
                "But what I really wanted to show was mostly the ease of use in day-to-day",
                "life for one of these Toyota electric vehicles,",
                "但我真正想展示的主要是日常使用的便利性",
                "这些丰田电动汽车的使用寿命",
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

    def test_lexical_and_phrasal_splits_are_mandatory_fluency_candidates(self):
        translator = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(
                index=289,
                original_text="which is their fancy way of saying rev",
            ),
            SubtitleProcessData(
                index=290,
                original_text="matching which helps in city driving",
            ),
            SubtitleProcessData(
                index=378,
                original_text="because this car sort of takes that ability",
            ),
            SubtitleProcessData(
                index=379,
                original_text="away from you and encourages you to push it",
            ),
        ]
        translated = {
            289: replace(source[0], translated_text="也就是他们所谓的降挡"),
            290: replace(source[1], translated_text="补油 这有助于市区驾驶"),
            378: replace(source[2], translated_text="因为这车把那种能力"),
            379: replace(source[3], translated_text="它从你手里夺走 还鼓励你压榨它"),
        }

        assert translator._mandatory_chinese_fluency_candidates(source, translated) == [
            289,
            378,
        ]

    def test_revised_component_split_is_a_mandatory_fluency_candidate(self):
        translator = _make_minimax_reflect_translator()
        source = [
            SubtitleProcessData(
                index=341,
                original_text="Why don't we show you this newly revised",
            ),
            SubtitleProcessData(
                index=342,
                original_text="nine speaker JBL sound system",
            ),
        ]
        translated = {
            341: replace(source[0], translated_text="不如展示一下这款新改款车型"),
            342: replace(source[1], translated_text="九扬声器JBL音响"),
        }

        assert translator._mandatory_chinese_fluency_candidates(source, translated) == [341]

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
        translator._all_source_by_index = {
            0: "Earlier context.",
            1: "We saw it evolve",
            2: "before our eyes.",
            3: "Later context.",
        }
        translator._gap_after_index = {1: 700}
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
        assert captured["max_output_tokens"] == 6144
        assert captured["reasoning_effort"] == "low"
        user_payload = json.loads(captured["messages"][1]["content"])
        assert user_payload["readonly_context"] == {
            "previous_source": "Earlier context.",
            "next_source": "Later context.",
        }
        assert user_payload["items"]["1"]["gap_after_ms"] == 700
        assert user_payload["items"]["1"]["display_continuity"] == "separated"

    def test_deepseek_fluency_rewrite_skips_thinking_for_routine_coordination(self, monkeypatch):
        translator = _make_translator(is_reflect=True)
        translator.model = "deepseek-v4-flash"
        source = [
            SubtitleProcessData(index=1, original_text="There's plenty of headroom"),
            SubtitleProcessData(index=2, original_text="and plenty of legroom."),
        ]
        current = [
            replace(source[0], translated_text="头部空间很充裕"),
            replace(source[1], translated_text="腿部空间也很宽敞"),
        ]
        captured = {}

        def fake_call(**kwargs):
            captured.update(kwargs)
            return _llm_response({"translations": {"1": "头部空间很充裕", "2": "腿部空间也很宽敞"}})

        monkeypatch.setattr("subforge.core.translate.llm_translator.call_llm", fake_call)

        translator._rewrite_chinese_fluency_window(source, current)

        assert captured["reasoning_mode"] == "disabled"
        assert translator.reasoning_metrics()["rewrite_requests"] == 0

    def test_chinese_fluency_audit_disables_deepseek_reasoning(self, monkeypatch):
        translator = _make_translator(is_reflect=True)
        translator.model = "deepseek-v4-flash"
        translator._gap_after_index = {1: 700}
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
        assert "material noun-list subject" in captured["messages"][0]["content"]
        user_payload = json.loads(captured["messages"][1]["content"])
        assert user_payload["1-2"]["gap_after_ms"] == 700
        assert user_payload["1-2"]["display_continuity"] == "separated"

    def test_chinese_window_fidelity_accepts_complete_local_reordering(self, monkeypatch):
        translator = _make_translator(is_reflect=True)
        translator._gap_after_index = {1: 700}
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
        assert "material coordinated noun subject" in captured["messages"][0]["content"]
        payload = json.loads(captured["messages"][1]["content"])
        assert payload["1"]["gap_after_ms"] == 700
        assert payload["1"]["display_continuity"] == "separated"

    def test_chinese_window_fidelity_uses_confirmed_asr_name_not_literal_mishear(self, monkeypatch):
        translator = _make_translator(is_reflect=True)
        translator.translation_context = TranslationContext(
            terminology=(
                "- rubber veil -> Muraba Veil (probable ASR correction)\n"
                "- Marabba Vale -> Muraba Veil (phonetic ASR variant)"
            )
        )
        source = [
            SubtitleProcessData(
                index=1,
                original_text="This is what the core looks like on a rubber veil.",
            )
        ]
        repaired = [replace(source[0], translated_text="这是Muraba Veil的核心筒")]
        captured = {}

        def fake_call(**kwargs):
            captured.update(kwargs)
            return _llm_response({"valid": True, "issues": []})

        monkeypatch.setattr("subforge.core.translate.llm_translator.call_llm", fake_call)

        translator._validate_chinese_window_fidelity(source, repaired)

        payload = json.loads(captured["messages"][1]["content"])
        assert payload["1"]["source"] == ("This is what the core looks like on a Muraba Veil.")

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

    def test_fluency_repair_uses_window_fidelity_for_minimal_reordering(self, monkeypatch):
        translator = _make_translator(is_reflect=True)
        source = [
            SubtitleProcessData(index=1, original_text="This matters"),
            SubtitleProcessData(index=2, original_text="when people choose repeatedly."),
        ]
        current = [
            replace(source[0], translated_text="这件事很重要"),
            replace(source[1], translated_text="当人们反复选择时"),
        ]
        repaired = [
            replace(source[0], translated_text="当人们反复作出选择时"),
            replace(source[1], translated_text="这件事就很重要"),
        ]
        calls = []

        def validate_response(*_args, **kwargs):
            calls.append(kwargs)
            return True, ""

        monkeypatch.setattr(translator, "_validate_llm_response", validate_response)
        monkeypatch.setattr(translator, "_request_chinese_fluency_flags", lambda *_args: [])
        monkeypatch.setattr(translator, "_validate_chinese_window_fidelity", lambda *_args: None)

        translator._validate_chinese_fluency_repair(source, current, repaired)

        assert calls == [
            {
                "require_reflect": False,
                "check_adjacent_repetition": True,
            }
        ]

    def test_window_fidelity_rejects_reversed_effort_contrast(self):
        translator = _make_translator(is_reflect=True)
        source = [
            SubtitleProcessData(index=1, original_text="Reading and writing are not something"),
            SubtitleProcessData(
                index=2,
                original_text="we can continue without effort in the same way as speaking is.",
            ),
        ]
        repaired = [
            replace(source[0], translated_text="阅读和写作不一样"),
            replace(source[1], translated_text="像说话那样需要付出努力"),
        ]

        with pytest.raises(ValueError, match="reversed the effort contrast"):
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
        fresh_calls = []
        broken = [
            replace(source[0], translated_text="这辆座椅仍然非常"),
            replace(source[1], translated_text="舒适平稳"),
        ]
        monkeypatch.setattr(
            t,
            "_rewrite_chinese_fluency_window",
            lambda *_args, **kwargs: rewrite_calls.append(kwargs.get("feedback", "")) or broken,
        )
        monkeypatch.setattr(
            t,
            "_rewrite_chinese_fluency_window_fresh",
            lambda *_args, **kwargs: fresh_calls.append(kwargs.get("feedback", "")) or broken,
        )

        result = t._finalize_translated_list(source, translated)

        assert audit_calls == 1
        assert len(rewrite_calls) == t.CHINESE_FLUENCY_ANCHORED_MAX_ATTEMPTS
        assert len(fresh_calls) == t.CHINESE_FLUENCY_FRESH_MAX_ATTEMPTS
        assert fresh_calls[-1] == ""
        assert [item.translated_text for item in result] == [
            "这是一个非常",
            "舒适的座椅",
        ]

    def test_confirmed_semantic_retry_uses_reasoning_only_on_first_attempt(self, monkeypatch):
        translator = _make_translator(is_reflect=True)
        translator.model = "deepseek-v4-flash"
        source = [
            SubtitleProcessData(index=1, original_text="Reading is not something we could"),
            SubtitleProcessData(index=2, original_text="expect to continue without effort."),
        ]
        current = [
            replace(source[0], translated_text="阅读并不是我们能够"),
            replace(source[1], translated_text="不费力气继续的事情"),
        ]
        reasoning_overrides = []
        fresh_reasoning = []

        def rewrite(_source, _current, **kwargs):
            reasoning_overrides.append(kwargs["reasoning_override"])
            return current

        monkeypatch.setattr(translator, "_rewrite_chinese_fluency_window", rewrite)
        monkeypatch.setattr(
            translator,
            "_rewrite_chinese_fluency_window_fresh",
            lambda *_args, **kwargs: fresh_reasoning.append(kwargs.get("reasoning", False))
            or current,
        )
        monkeypatch.setattr(
            translator,
            "_validate_chinese_fluency_repair",
            lambda *_args: (_ for _ in ()).throw(
                ValueError("fluency repair left structural boundary signals")
            ),
        )

        _window, repaired, error = translator._repair_chinese_fluency_window_with_retries(
            source,
            current,
        )

        assert repaired is None
        assert error is not None
        assert fresh_reasoning == [True, False]
        assert reasoning_overrides == [False]

    def test_fluency_repair_uses_fidelity_as_final_soft_boundary_arbiter(self, monkeypatch):
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
        audit_calls = []
        fidelity_calls = []
        monkeypatch.setattr(
            translator,
            "_request_chinese_fluency_flags",
            lambda *_args: audit_calls.append(True) or [1],
        )
        monkeypatch.setattr(
            translator,
            "_validate_chinese_window_fidelity",
            lambda *_args: fidelity_calls.append(True),
        )

        translator._validate_chinese_fluency_repair(source, current, current)

        assert audit_calls == []
        assert fidelity_calls == [True]

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

    def test_context_ownership_allows_article_fused_into_owned_identifier(self):
        t = _make_translator()
        t._all_source_by_index = {
            116: "visit incogni.com slash the B1M",
            117: "use my code theB1M to get 60% off",
        }

        t._validate_single_context_ownership(
            {"116": t._all_source_by_index[116]},
            "访问incogni.com/theB1M",
        )

        ok, message = t._validate_cross_key_boundaries(
            {
                "116": "访问incogni.com/theB1M",
                "117": "使用代码theB1M可享60%折扣",
            },
            {
                "116": t._all_source_by_index[116],
                "117": t._all_source_by_index[117],
            },
            str,
        )

        assert ok is True
        assert message == ""

    def test_context_ownership_allows_localized_unit_owned_by_current_key(self):
        t = _make_translator()
        source = {
            "68": "The route is 194 miles in total.",
            "69": "Of those 194km, 115km are going to be in tunnels,",
        }
        response = {
            "68": "这条线路全长194英里",
            "69": "在这194公里中 有115公里将建在隧道内",
        }

        ok, message = t._validate_cross_key_boundaries(response, source, str)

        assert ok is True
        assert message == ""

    def test_context_ownership_rejects_localized_magnitude_borrowed_from_neighbor(self):
        t = _make_translator()
        t._all_source_by_index = {
            190: "deliver a $250 billion boost",
            191: "to the economy over the next half century.",
        }

        with pytest.raises(RuntimeError, match=r"borrowed.*250"):
            t._validate_single_context_ownership(
                {"191": t._all_source_by_index[191]},
                "未来半个世纪将为经济带来2500亿澳元的推动",
            )

        ok, message = t._validate_cross_key_boundaries(
            {
                "190": "将为经济带来2500亿澳元的推动",
                "191": "未来半个世纪将为经济带来2500亿澳元的推动",
            },
            {
                "190": t._all_source_by_index[190],
                "191": t._all_source_by_index[191],
            },
            str,
        )

        assert ok is False
        assert "191:250" in message

    def test_context_ownership_rejects_localized_acronym_borrowed_from_neighbor(self):
        translator = _make_translator()
        source = {
            "89": "Nuclear power generates huge amounts of energy without",
            "90": "CO2 emissions while remaining continuously available.",
        }
        response = {
            "89": "核电能提供巨量能源 同时不产生二氧化碳排放",
            "90": "不产生二氧化碳排放 而且能够持续供电",
        }

        valid, error = translator._validate_cross_key_boundaries(response, source, str)

        assert not valid
        assert "89:co2" in error.lower()

    def test_context_ownership_allows_acronym_and_expanded_source_owners(self):
        translator = _make_translator()
        source = {
            "172": "Its methods are aligned with the IAEA.",
            "173": "That stands for the International Atomic Energy Agency.",
        }
        response = {
            "172": "其做法遵循IAEA准则",
            "173": "IAEA即国际原子能机构",
        }

        valid, error = translator._validate_cross_key_boundaries(response, source, str)

        assert valid
        assert error == ""

    def test_context_ownership_recognizes_plural_acronym_owner(self):
        translator = _make_translator()
        source = {"184": "Other SMRs do already exist."}
        response = {"184": "其他小型模块化反应堆确实已经存在"}

        valid, error = translator._validate_cross_key_boundaries(response, source, str)

        assert valid
        assert error == ""

    def test_context_ownership_locks_hyphenated_alphanumeric_model(self):
        translator = _make_translator()
        source = {
            "183": "Linglong-1 is the first land-based reactor.",
            "184": "It is set to come online.",
        }
        response = {
            "183": "这是首座陆基反应堆",
            "184": "玲龙一号 Linglong-1 即将投入运行",
        }

        valid, error = translator._validate_cross_key_boundaries(response, source, str)

        assert not valid
        assert "184:linglong-1" in error.lower()

    def test_chinese_translation_rejects_lowercase_english_phrase_residue(self):
        translator = _make_translator()

        valid, error = translator._validate_unexpected_latin_residue(
            {"183": "全球首座 fully commercial 陆基反应堆"},
            {"183": "the world's first fully commercial land-based reactor"},
            str,
        )

        assert not valid
        assert "fully commercial" in error

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

    def test_allows_sentence_initial_uppercase_interjection_to_be_translated(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"12": "好 我现在好奇了"},
            {"12": "OK, I'm curious now."},
            require_reflect=False,
        )

        assert ok is True
        assert msg == ""

    def test_still_rejects_dropped_technical_acronym(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"12": "这台对撞机已经投入运行"},
            {"12": "The LHC is now operational."},
            require_reflect=False,
        )

        assert ok is False
        assert "LHC" in msg

    def test_allows_exact_chinese_ten_thousand_number_equivalent(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"567": "这辆车的售价是11.7万美元"},
            {"567": "This truck costs $117,000."},
        )

        assert ok is True
        assert msg == ""

    def test_allows_exact_chinese_spoken_magnitude_equivalent(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"620": "买一捐一，累计捐赠已超过两亿件"},
            {"620": "One purchased equals one donated with over 200 million donations."},
        )

        assert ok is True
        assert msg == ""

    def test_allows_natural_chinese_age_decade_equivalent(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"598": "她二十多岁的女儿因一种罕见疾病去世了"},
            {"598": "her daughter in her 20s passed away from a rare disease."},
            require_reflect=False,
        )

        assert ok is True
        assert msg == ""

    def test_age_decade_equivalent_does_not_relax_historical_decade(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"598": "这件事发生在二十多岁时"},
            {"598": "This happened in the 1920s."},
            require_reflect=False,
        )

        assert ok is False
        assert "1920s" in msg

    def test_allows_standalone_100_percent_as_affirmative_reply(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"637": "完全如此"},
            {"637": "100%."},
            require_reflect=False,
        )

        assert ok is True
        assert msg == ""

    def test_standalone_percentage_equivalent_does_not_relax_quantity_in_sentence(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"637": "性能提升非常明显"},
            {"637": "Performance improved by 100%."},
            require_reflect=False,
        )

        assert ok is False
        assert "100" in msg

    def test_rejects_wrong_chinese_spoken_magnitude(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"620": "买一捐一，累计捐赠已超过两百万件"},
            {"620": "One purchased equals one donated with over 200 million donations."},
        )

        assert ok is False
        assert "620:200" in msg

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

    @pytest.mark.parametrize(
        ("source", "translation"),
        [
            (
                "It works like GM's Super Cruise or Ford's Blue Cruise.",
                "它的工作方式类似通用汽车的Super Cruise或福特的Blue Cruise",
            ),
            (
                "Steering assist is limited while the HD map is under maintenance.",
                "高清地图维护期间 转向辅助功能受限",
            ),
            (
                "The project involves partners in the UK and the EU.",
                "该项目有英国和欧盟的合作伙伴参与",
            ),
            (
                "It provides power without producing CO2 emissions.",
                "它能在不产生二氧化碳排放的情况下供电",
            ),
        ],
    )
    def test_allows_standard_chinese_equivalents_for_common_acronyms(self, source, translation):
        t = _make_translator()

        ok, msg = t._validate_llm_response({"0": translation}, {"0": source})

        assert ok is True
        assert msg == ""

    @pytest.mark.parametrize(
        ("source", "translation", "missing_token"),
        [
            ("The project is based in the UK.", "该项目设在欧洲", "UK"),
            ("The process releases CO2.", "这一过程会产生排放", "CO2"),
        ],
    )
    def test_standard_chinese_equivalents_do_not_hide_missing_facts(
        self, source, translation, missing_token
    ):
        translator = _make_translator()

        valid, error = translator._validate_llm_response(
            {"0": translation},
            {"0": source},
        )

        assert valid is False
        assert missing_token in error

    def test_rejects_literal_you_know_after_completed_sentence(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"0": "这是一套很好的音响系统 你知道"},
            {"0": "It's a very good sound system. You know,"},
        )

        assert ok is False
        assert "speech filler" in msg

    @pytest.mark.parametrize(
        "source",
        [
            "But you know I like to compare it with a BMW X6.",
            "They still have very light and you know luxurious steering.",
        ],
    )
    def test_rejects_literal_you_know_inside_discourse_frame(self, source):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"0": "但你知道 这套转向很有豪华感"},
            {"0": source},
        )

        assert ok is False
        assert "speech filler" in msg

    def test_allows_semantic_you_know_predicate(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"0": "你知道答案吗"},
            {"0": "Do you know the answer?"},
        )

        assert ok is True
        assert msg == ""

    def test_allows_jfk_translated_as_kennedy(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"467": "肯尼迪就是这样彻底击败尼克松的"},
            {"467": "That's how JFK mopped the floor with Nixon."},
        )

        assert ok is True
        assert msg == ""

    def test_allows_natural_chinese_casual_numeric_range(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"451": "尤其是在开头十几秒"},
            {"451": "attention in the first 10 or 15 seconds."},
        )

        assert ok is True
        assert msg == ""
        assert msg == ""

    def test_allows_standard_rem_translation_for_preserved_tokens(self):
        t = _make_translator()
        resp = {"0": "也会出现在非快速眼动睡眠阶段。"}
        inp = {"0": "It also occurs during non-REM sleep."}

        ok, msg = t._validate_llm_response(resp, inp)

        assert ok is True
        assert msg == ""

    def test_allows_tv_to_be_translated_semantically(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"670": "这些年我也看了很多电视节目"},
            {"670": "I watched a lot of TV shows over the years."},
        )

        assert ok is True
        assert msg == ""

    def test_allows_percent_off_as_exact_chinese_discount(self):
        t = _make_translator()

        ok, msg = t._validate_llm_response(
            {"622": "首次购买可享八折优惠"},
            {"622": "Use code grayarea for 20% off your first purchase."},
        )

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

    def test_chinese_style_prompt_selects_only_relevant_guidance(self):
        t = _make_translator()

        generic = t._target_language_style_rules(["This is a general interview."])
        automotive = t._target_language_style_rules(
            ["That is how quiet this truck gets at 2,000 RPM."]
        )
        infrastructure = t._target_language_style_rules(
            [
                "The project is an exercise in controlling structural loads.",
                "It transfers forces down the height of the building.",
                "The city created more land by drilling it out from the sea.",
            ]
        )
        reading = t._target_language_style_rules(["What would it take to become literate again?"])

        assert "English degree constructions" not in generic
        assert "automotive Chinese" not in generic
        assert "English degree constructions" in automotive
        assert "automotive Chinese" in automotive
        assert "civil-aviation and construction Chinese" in infrastructure
        assert "concentrated undertaking" in infrastructure
        assert "land-reclamation" in infrastructure
        assert "automotive Chinese" not in infrastructure
        assert "literacy from general culture" in reading
        assert "automotive Chinese" not in reading
        assert "Map the complete source clause" in generic
        assert "spoken self-corrections" in generic
        assert "contrastive references" in generic
        assert "same Chinese head noun twice" in generic
        assert "bare pronoun, demonstrative" in generic
        assert "interaction and medium" in generic
        assert "base/standard equipment from bass" in automotive

        official_feature = t._target_language_style_rules(
            ["These are what Toyota calls the Sport Touring seats."]
        )
        assert "official identifier" in official_feature

        performance_corner = t._target_language_style_rules(
            ["We braked late for the hot left-hander."]
        )
        assert "not temperature" in performance_corner

        audited_idioms = t._target_language_style_rules(
            [
                "This feels biblically accurate for a compact car.",
                "It is on par with the rest of the segment.",
                "In a traffic situation it is natural to get through it.",
            ]
        )
        assert "Do not introduce the Bible" in audited_idioms
        assert "neutral equality comparison" in audited_idioms
        assert "threading through congestion" in audited_idioms

        pragmatic = t._target_language_style_rules(
            [
                "Talk about a lesson in form.",
                "I don't know if I think no boundaries are good.",
                "The word is synonymous with frustration.",
            ]
        )
        assert "actual stance and polarity" in pragmatic
        assert "emphatic example marker" in pragmatic
        assert "让人联想到" in pragmatic

        t.target_language = TargetLanguage.ENGLISH
        assert t._target_language_style_rules(["vehicle"]) == ""

    def test_chinese_boundary_signal_detects_split_copular_result(self):
        signal = LLMTranslator._chinese_boundary_signal(
            "或者把你甩进沟里之类的 它就是",
            "一踩油门就笔直往前冲",
        )

        assert signal == "copular frame is separated from its result"

    def test_single_and_multispeaker_batch_prompts_are_isolated(self):
        translator = _make_translator(is_reflect=True)

        assert translator._batch_translation_prompt_name(reflect=False) == "translate/standard"
        assert translator._batch_translation_prompt_name(reflect=True) == "translate/reflect"

        translator._all_speaker_by_index = {1: "speaker-1", 2: "speaker-2"}

        assert (
            translator._batch_translation_prompt_name(reflect=False) == "translate/standard_multi"
        )
        assert translator._batch_translation_prompt_name(reflect=True) == "translate/reflect_multi"

    def test_rejects_lost_elliptical_percentage_in_vehicle_tuning(self):
        translator = _make_translator()
        source = {"290": "The dampers are 20 softer than before."}

        ok, message = translator._validate_llm_response(
            {"290": "减震器比以前更软"},
            source,
        )

        assert ok is False
        assert "20%" in message

        ok, message = translator._validate_llm_response(
            {"290": "减震器比以前软约20%"},
            source,
        )
        assert ok is True
        assert message == ""

    def test_rejects_missing_thousand_scale_in_elliptical_rpm_range(self):
        translator = _make_translator()
        source = {"336": "It pulls between 1 and 2,000 RPM."}

        ok, message = translator._validate_llm_response(
            {"336": "它在1到2000转之间开始发力"},
            source,
        )

        assert ok is False
        assert "1000到2000" in message

        ok, message = translator._validate_llm_response(
            {"336": "它在1000到2000转/分之间开始发力"},
            source,
        )
        assert ok is True
        assert message == ""

    def test_chinese_boundary_signal_catches_split_negative_comparison(self):
        assert (
            LLMTranslator._chinese_boundary_signal(
                "它并不那么",
                "像伊兰特一样幼稚又疯狂",
            )
            == "negated comparison is split from its complement"
        )

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

    def test_pending_alignment_chunk_is_not_stable_for_cache(self):
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

    def test_pending_alignment_chunk_is_checkpointed_but_not_cached(self, monkeypatch):
        progress = []
        t = _make_minimax_reflect_translator()
        t.use_cache = False
        t.update_callback = progress.extend
        t._pending_alignment_repair_keys.add(133)
        result = [
            SubtitleProcessData(
                index=index,
                original_text=f"Source subtitle {index}",
                translated_text=f"第{index}条候选译文",
            )
            for index in range(121, 141)
        ]
        monkeypatch.setattr(t, "_translate_chunk", lambda _chunk: result)

        assert t._safe_translate_chunk(result) == result
        assert progress == result

    def test_accepts_established_chinese_name_for_zf_token(self):
        translator = _make_translator()

        ok, message = translator._validate_llm_response(
            {"91": "这辆车用的是采埃孚9速自动变速箱 具体型号是9HP"},
            {"91": "It's actually a ZF 9-speed, a 9HP in this car."},
        )

        assert ok is True
        assert message == ""

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

    def test_cache_key_isolated_by_provider_base_url(self, monkeypatch):
        first = _make_translator()
        second = _make_translator()
        chunk = [SubtitleProcessData(index=1, original_text="hello")]

        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
        first_key = first._get_cache_key(chunk)
        monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
        second_key = second._get_cache_key(chunk)

        assert first_key != second_key

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


def _dialogue_items(left_source, right_source, left_translation, right_translation):
    source = [
        SubtitleProcessData(index=1, original_text=left_source),
        SubtitleProcessData(index=2, original_text=right_source),
    ]
    translated = {
        1: replace(source[0], translated_text=left_translation),
        2: replace(source[1], translated_text=right_translation),
    }
    return source, translated


def test_multispeaker_candidates_audit_same_speaker_continuation():
    source, translated = _dialogue_items(
        "We talked about reading",
        "and what it means.",
        "我们谈到了阅读",
        "以及阅读的意义",
    )
    single = _make_translator(is_reflect=True)
    multi = _make_translator(is_reflect=True)
    multi._all_speaker_by_index = {1: "S1", 2: "S1", 3: "S2"}

    assert single._chinese_fluency_candidates(source, translated) == [1]
    assert multi._chinese_fluency_candidates(source, translated) == [1]


def test_multispeaker_candidate_repairs_auxiliary_participle_boundary_without_reasoning():
    source, translated = _dialogue_items(
        "and college graduates and women have historically",
        "been the groups that read the most and declined as well.",
        "和大学毕业生以及女性历来",
        "是读书最多的群体 但他们也经历了大幅下降",
    )
    translator = _make_translator(is_reflect=True)
    translator.model = "deepseek-v4-flash"
    translator._all_speaker_by_index = {1: "S1", 2: "S1", 3: "S2"}

    assert translator._chinese_fluency_candidates(source, translated) == [1]
    assert not translator._should_reason_about_chinese_fluency_window(
        source,
        list(translated.values()),
    )
    assert "minimum collective reference" in repair_mode_guidance(True)
    assert "Map the complete source clause" in repair_mode_guidance(True)
    assert "spoken self-correction" in repair_mode_guidance(False)


def test_multispeaker_candidates_audit_visible_soft_chinese_break():
    source, translated = _dialogue_items(
        "Reading and writing are not something that we could expect to always",
        "continue without effort in the same way as speaking.",
        "阅读和写作并不是我们可以指望它",
        "像说话一样不费力气就能一直延续下去的",
    )
    translator = _make_translator(is_reflect=True)
    translator._all_speaker_by_index = {1: "S1", 2: "S1", 3: "S2"}

    assert translator._chinese_fluency_candidates(source, translated) == [1]


@pytest.mark.parametrize(
    ("left", "right", "signal"),
    [
        ("我真正想表达的重点", "是我们反复做出的选择", "possible copular bridge"),
        ("人们可以作出任何", "价值判断", "unfinished Chinese grammatical structure"),
    ],
)
def test_chinese_boundary_signal_catches_general_incomplete_frames(left, right, signal):
    assert LLMTranslator._chinese_boundary_signal(left, right) == signal


@pytest.mark.parametrize(
    ("left", "right", "signal"),
    [
        ("这项技术向其他厂商证明仍然", "可以保留备胎", "unfinished Chinese adverbial predicate"),
        ("而这款一路", "涨到4.85万美元", "unfinished Chinese degree phrase"),
        ("它不像一场荒谬的灾难", "那样难以收拾", "comparison phrase is stranded"),
    ],
)
def test_chinese_boundary_signal_catches_general_degree_and_comparison_breaks(left, right, signal):
    assert LLMTranslator._chinese_boundary_signal(left, right) == signal


def test_style_guidance_disambiguates_adverse_thanks_to_and_collapsed_addresses():
    guidance = target_language_style_rules(
        "简体中文",
        ["Thanks in no small part to the recession, 525th Avenue stalled."],
    )

    assert "harmful or unwanted result" in guidance
    assert "collapse a building number and an ordinal street name" in guidance


def test_multispeaker_candidate_keeps_confirmed_chinese_structure_break():
    source, translated = _dialogue_items(
        "In many ways, it's actually,",
        "sometimes a more effective way to get information.",
        "在很多方面 它实际上",
        "有时候是获取信息更有效的方式",
    )
    translator = _make_translator(is_reflect=True)
    translator.model = "deepseek-v4-flash"
    translator._all_speaker_by_index = {1: "S1", 2: "S1", 3: "S2"}

    assert translator._chinese_fluency_candidates(source, translated) == [1]
    assert translator._should_reason_about_chinese_fluency_window(
        source,
        list(translated.values()),
    )


def test_multispeaker_candidate_keeps_coordinated_subject_predicate_boundary():
    source, translated = _dialogue_items(
        "I note in the piece that retirees and college graduates and women",
        "have historically been the groups that read the most and declined as well.",
        "历来是阅读最多的群体 而他们的阅读量也大幅下降",
        "所以这并不局限于某一个群体",
    )
    translator = _make_translator(is_reflect=True)
    translator.model = "deepseek-v4-flash"
    translator._all_speaker_by_index = {1: "S1", 2: "S1", 3: "S2"}

    assert translator._chinese_fluency_candidates(source, translated) == [1]
    assert translator._should_reason_about_chinese_fluency_window(
        source,
        list(translated.values()),
    )


def test_correct_reported_topic_handoff_skips_audit_and_rewrite(
    monkeypatch,
):
    source, translated = _dialogue_items(
        "I note in the piece that retirees and college graduates and women",
        "have historically been the groups that read the most and declined as well.",
        "我在文章中提到 退休人士 大学毕业生和女性",
        "历来是阅读最多的群体 他们的阅读量也同样大幅下降",
    )
    translator = _make_translator(is_reflect=True)
    translator._all_source_by_index = {item.index: item.original_text for item in source}
    translator._all_speaker_by_index = {1: "S1", 2: "S1", 3: "S2"}
    audit_calls = []

    def audit(indices, *_args):
        audit_calls.append(indices)
        pytest.fail("readable reported topic must not spend an audit request")

    monkeypatch.setattr(translator, "_request_chinese_fluency_flags", audit)
    monkeypatch.setattr(
        translator,
        "_repair_chinese_fluency_window_with_retries",
        lambda *_args, **_kwargs: pytest.fail("unconfirmed boundary must not be rewritten"),
    )

    translator._repair_chinese_boundary_fluency(source, translated)

    assert audit_calls == []
    assert translated[1].translated_text == "我在文章中提到 退休人士 大学毕业生和女性"
    assert translated[2].translated_text == "历来是阅读最多的群体 他们的阅读量也同样大幅下降"


def test_multispeaker_edited_handoff_is_audited_without_relabeling_speakers():
    source, translated = _dialogue_items(
        "They're much better quality than most other",
        "Socks that I've found.",
        "它们的质量比大多数其他",
        "我找到的袜子",
    )
    translator = _make_translator(is_reflect=True)
    translator.model = "deepseek-v4-flash"
    translator._all_speaker_by_index = {1: "S1", 2: "S2"}
    translator._gap_after_index = {1: 322}

    assert translator._is_edited_speaker_handoff(source[0], source[1])
    assert translator._chinese_fluency_candidates(source, translated) == [1]
    assert translator._should_reason_about_chinese_fluency_window(
        source,
        list(translated.values()),
    )
    assert translator._all_speaker_by_index == {1: "S1", 2: "S2"}


def test_multispeaker_distant_turn_is_not_treated_as_edited_handoff():
    source, translated = _dialogue_items(
        "They're much better quality than most other",
        "Socks that I've found.",
        "它们的质量比大多数其他",
        "我找到的袜子",
    )
    translator = _make_translator(is_reflect=True)
    translator._all_speaker_by_index = {1: "S1", 2: "S2"}
    translator._gap_after_index = {1: 900}

    assert not translator._is_edited_speaker_handoff(source[0], source[1])
    assert translator._chinese_fluency_candidates(source, translated) == []


def test_multispeaker_duplicate_boundary_uses_audit_without_native_reasoning():
    source, translated = _dialogue_items(
        "Podcasts and videos, there's no",
        "problem with getting information that way.",
        "播客和视频可以通过那种方式获取信息",
        "通过这种方式获取信息没什么问题",
    )
    translator = _make_translator(is_reflect=True)
    translator.model = "deepseek-v4-flash"
    translator._all_speaker_by_index = {1: "S1", 2: "S1", 3: "S2"}

    assert (
        translator._chinese_boundary_signal(
            translated[1].translated_text,
            translated[2].translated_text,
        )
        == "possible duplicated boundary phrase"
    )
    assert translator._chinese_fluency_candidates(source, translated) == [1]
    assert not translator._should_reason_about_chinese_fluency_window(
        source,
        list(translated.values()),
    )


def test_multispeaker_short_repeated_predicate_uses_nonthinking_audit():
    source, translated = _dialogue_items(
        "So I spoke with",
        "experts who study politics.",
        "所以我采访了",
        "我采访了研究政治的专家",
    )
    translator = _make_translator(is_reflect=True)
    translator.model = "deepseek-v4-flash"
    translator._all_speaker_by_index = {1: "S1", 2: "S1", 3: "S2"}

    assert (
        translator._chinese_boundary_signal(
            translated[1].translated_text,
            translated[2].translated_text,
        )
        == "possible duplicated boundary phrase"
    )
    assert translator._chinese_fluency_candidates(source, translated) == [1]
    assert not translator._should_reason_about_chinese_fluency_window(
        source,
        list(translated.values()),
    )


def test_multispeaker_dangling_pronoun_uses_selective_reasoning():
    source, translated = _dialogue_items(
        "It is a change that I",
        "call post-literacy.",
        "这是一种变化 我",
        "称之为后文字时代",
    )
    translator = _make_translator(is_reflect=True)
    translator.model = "deepseek-v4-flash"
    translator._all_speaker_by_index = {1: "S1", 2: "S1", 3: "S2"}

    assert translator._should_reason_about_chinese_fluency_window(
        source,
        list(translated.values()),
    )


def test_chinese_boundary_signal_accepts_complete_numbered_aspect_phrase():
    assert not LLMTranslator._chinese_boundary_signal(
        "嗯 我觉得这有两方面",
        "一方面是文化传承",
    )
    assert (
        LLMTranslator._chinese_boundary_signal(
            "我们关注变化的很多方面",
            "人们开始改变习惯",
        )
        == "unfinished Chinese locative subject"
    )
    assert not LLMTranslator._chinese_boundary_signal(
        "作为人类 我们长久以来都处于口头传统之中",
        "现在大家都在谈论《奥德赛》",
    )
    assert not LLMTranslator._chinese_boundary_signal(
        "我想谈谈它改变我们社会的另一个方面",
        "那就是对政治的影响",
    )
    assert not LLMTranslator._chinese_boundary_signal(
        "他不会是最后一位",
        "未来的传播会更重视吸引注意力",
    )
    assert not LLMTranslator._chinese_boundary_signal(
        "他不会是我们最后一个",
        "未来的传播会更重视吸引注意力",
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("有人谈到 当", "你试图安装某样东西时"),
        ("学校一面在应对这一趋势 同时", "也在做出改变"),
        ("他说话", "时会自相矛盾"),
    ],
)
def test_chinese_boundary_signal_catches_multicue_dialogue_fragments(left, right):
    assert (
        LLMTranslator._chinese_boundary_signal(
            left,
            right,
        )
        == "unfinished Chinese grammatical structure"
    )


@pytest.mark.parametrize(
    ("left", "right", "signal"),
    [
        ("他们预测会出现像", "唐纳德·特朗普这样的人", "comparison example is stranded"),
        ("数量在零本", "到四本之间", "numeric range is split"),
        ("这件事的重点更多是", "我们反复做出的选择", "semantic frame is incomplete"),
        ("我认为我们看到", "小学到高中都发生了变化", "possible reporting frame"),
        ("这需要一场真正的", "大规模转变", "nominal modifier is stranded"),
        ("它们比大多数其他", "我找到的袜子质量更好", "comparative noun modifier is stranded"),
        ("这当然也不是一幅", "完全美好的图景", "classifier phrase is stranded"),
        ("而在过去", "而在过去保存资料很困难", "possible duplicated boundary phrase"),
    ],
)
def test_chinese_boundary_signal_catches_dialogue_readability_failures(left, right, signal):
    assert LLMTranslator._chinese_boundary_signal(left, right) == signal


def test_multispeaker_repair_prompt_is_dialogue_specific_and_compact(monkeypatch):
    source, translated = _dialogue_items(
        "They're much better quality than most other",
        "Socks that I've found.",
        "它们的质量比大多数其他",
        "我找到的袜子",
    )
    translator = _make_translator(is_reflect=True)
    translator._all_speaker_by_index = {1: "S1", 2: "S2"}
    translator._gap_after_index = {1: 322}
    captured = {}

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return _llm_response(
            {
                "translations": {
                    "1": "它们的品质远胜大多数同类袜子",
                    "2": "至少在我找到的袜子里是这样",
                }
            }
        )

    monkeypatch.setattr(
        "subforge.core.translate.llm_translator.call_llm",
        fake_call_llm,
    )

    repaired = translator._rewrite_chinese_fluency_window(
        source,
        list(translated.values()),
    )

    prompt = captured["messages"][0]["content"]
    assert "tightly edited handoff" in prompt
    assert "Speaker values are read-only metadata" in prompt
    assert "continuous single-speaker passage" not in prompt
    assert "Mercedes was involved" not in prompt
    assert captured["reasoning_mode"] == "disabled"
    assert [item.translated_text for item in repaired] == [
        "它们的品质远胜大多数同类袜子",
        "至少在我找到的袜子里是这样",
    ]


def test_multispeaker_repair_prompt_handles_fillers_without_changing_single_prompt(
    monkeypatch,
):
    source, translated = _dialogue_items(
        "He contradicts himself as though there's no",
        "record of his previous words.",
        "他说话自相矛盾 仿佛没有",
        "他之前言论的任何记录",
    )
    prompts = []

    def fake_call_llm(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return _llm_response(
            {
                "translations": {
                    "1": "他不断自相矛盾 却似乎毫不受影响",
                    "2": "仿佛过去的言论从未留下记录",
                }
            }
        )

    monkeypatch.setattr(
        "subforge.core.translate.llm_translator.call_llm",
        fake_call_llm,
    )

    multi = _make_translator(is_reflect=True)
    multi._all_speaker_by_index = {1: "S1", 2: "S1", 3: "S2"}
    multi._rewrite_chinese_fluency_window(source, list(translated.values()))

    single = _make_translator(is_reflect=True)
    single._rewrite_chinese_fluency_window(source, list(translated.values()))

    assert "Speaker values are read-only metadata" in prompts[0]
    assert "continuous single-speaker passage" not in prompts[0]
    assert "continuous single-speaker passage" in prompts[1]
    assert "Speaker values are read-only metadata" not in prompts[1]


@pytest.mark.parametrize(
    ("left", "right", "signal"),
    [
        (
            "我们已经看到学校两方面都在 你知道",
            "应对这种趋势",
            "unfinished Chinese grammatical structure",
        ),
        (
            "如果你过去几个月",
            "过去几个月一直穿着旧T恤",
            "possible duplicated boundary phrase",
        ),
        (
            "当时不到",
            "不到一半的人读过书",
            "possible duplicated boundary phrase",
        ),
        (
            "所以 但我确实认为",
            "我们能从中吸取教训",
            "stacked discourse connectives",
        ),
    ],
)
def test_chinese_boundary_signal_catches_full_audit_failures(left, right, signal):
    assert LLMTranslator._chinese_boundary_signal(left, right) == signal


@pytest.mark.parametrize(
    ("left", "right", "signal"),
    [
        (
            "政治人物会把自己塑造成可信的局外人 而美国选民",
            "几乎只凭候选人给人的印象来选择领导人",
            "material subject may be stranded",
        ),
        (
            "社交媒体算法偏爱民粹主义的",
            "简单化、情感共鸣强的信息",
            "coordinated modifier may be stranded",
        ),
    ],
)
def test_chinese_boundary_signal_catches_subject_and_modifier_stranding(left, right, signal):
    assert LLMTranslator._chinese_boundary_signal(left, right) == signal


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("还是只是在说孩子？", "还是只是在说成年人？"),
        ("但现在 他们更多是在教学生", "应该如何阅读"),
    ],
)
def test_chinese_boundary_signal_ignores_material_noun_used_as_object(left, right):
    assert LLMTranslator._chinese_boundary_signal(left, right) == ""


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("我们把它关上", "然后继续往前开"),
        ("你可以把手肘搭在扶手上", "坐姿会轻松不少"),
        ("我不知道他们从哪儿找到这些东西的。", "设计、材料和做工都很特别"),
    ],
)
def test_chinese_boundary_signal_ignores_complete_ba_and_sentence_final_de(left, right):
    assert LLMTranslator._chinese_boundary_signal(left, right) == ""


def test_chinese_boundary_signal_keeps_incomplete_ba_construction_detection():
    assert LLMTranslator._chinese_boundary_signal(
        "我们把管片",
        "一块块地装入隧道",
    ) == "ba construction is separated from its predicate"


def test_chinese_boundary_signal_accepts_complete_ba_return_action():
    assert LLMTranslator._chinese_boundary_signal(
        "好了 我们把窗户装回去",
        "再次感谢本田",
    ) == ""


def test_single_speaker_material_subject_signal_requires_llm_confirmation(monkeypatch):
    source, translated = _dialogue_items(
        "Political figures portray themselves as authentic outsiders and American voters,",
        "almost choose leaders based only on how they appear.",
        "政治人物会把自己塑造成可信的局外人 而美国选民",
        "几乎只凭候选人给人的印象来选择领导人",
    )
    translator = _make_translator(is_reflect=True)
    translator._all_source_by_index = {item.index: item.original_text for item in source}
    audit_calls = []

    def audit(indices, *_args):
        audit_calls.append(indices)
        return []

    monkeypatch.setattr(translator, "_request_chinese_fluency_flags", audit)
    monkeypatch.setattr(
        translator,
        "_repair_chinese_fluency_window_with_retries",
        lambda *_args, **_kwargs: pytest.fail("unconfirmed boundary must not be rewritten"),
    )

    translator._repair_chinese_boundary_fluency(source, translated)

    assert audit_calls == [[1]]
    assert translated[1].translated_text.endswith("美国选民")


@pytest.mark.parametrize(
    ("source", "translation", "error_fragment"),
    [
        (
            "And so, you know, this matters.",
            "所以 你知道 这很重要",
            "speech filler",
        ),
        (
            "And so, but I do think it matters.",
            "所以但我确实认为这很重要",
            "stacked combinations",
        ),
        (
            "Almost two dozen states have banned phones.",
            "将近二十多个州已经禁止手机",
            "close to 24",
        ),
        (
            "What would it take for us to become literate again?",
            "怎样才能让我们重新变得有文化",
            "restoring literacy",
        ),
        (
            "We should not slide into post literacy.",
            "我们不应该滑向后读写时代",
            "后文字时代",
        ),
        (
            "Drop us a line at thegrayareaatvox.com.",
            "请通过 thegrayareaatvox.com 给我们留言",
            "email address",
        ),
    ],
)
def test_rejects_low_quality_chinese_patterns_from_full_audit(
    source,
    translation,
    error_fragment,
):
    translator = _make_translator(is_reflect=False)

    valid, error = translator._validate_llm_response(
        {"1": translation},
        {"1": source},
        require_reflect=False,
    )

    assert not valid
    assert error_fragment in error


@pytest.mark.parametrize(
    "left",
    [
        "所以有趣的是 阅读和写作并不是",
        "这当然不是美好图景 他们传播的不只是",
        "有意思的是",
    ],
)
def test_chinese_boundary_signal_catches_new_unfinished_predicates(left):
    assert (
        LLMTranslator._chinese_boundary_signal(left, "下一条承接谓语或宾语")
        == "unfinished Chinese grammatical structure"
    )


@pytest.mark.parametrize(
    ("left", "right", "signal"),
    [
        (
            "学校正在做出可能进一步",
            "让学生减少阅读的改变",
            "unfinished Chinese grammatical structure",
        ),
        ("人们可以对此有任何价值", "判断", "unfinished Chinese grammatical structure"),
        ("这种新的", "传播媒介也会带来取舍", "unfinished Chinese grammatical structure"),
        ("而且这很讽刺 我写", "一篇近九千字的文章", "unfinished Chinese grammatical structure"),
        (
            "所以我觉得确实如此",
            "阅读深刻影响了我的成长",
            "unfinished Chinese grammatical structure",
        ),
        ("印刷术发挥了重要作用 开国元勋们", "使用报纸传播主张", "material subject may be stranded"),
    ],
)
def test_chinese_boundary_signal_catches_latest_full_run_failures(left, right, signal):
    assert LLMTranslator._chinese_boundary_signal(left, right) == signal


@pytest.mark.parametrize(
    ("left", "right", "signal"),
    [
        ("我们确实看到了", "平板电脑正在进入课堂", "possible reporting frame"),
        ("所以我想指出的更多是", "选择会逐渐累积", "semantic frame is incomplete"),
        ("这项报道真正想说的", "是选择会逐渐累积", "semantic frame is incomplete"),
        ("我们看到书籍", "和文字变得更容易获取", "coordinated subject may be stranded"),
        ("信息会通过音频传播", "而成为一种新事物", "predicate fragment starts at next subtitle"),
        (
            "人们不再重视或主动获取",
            "那些触手可及的知识",
            "transitive predicate is split from its object",
        ),
    ],
)
def test_chinese_boundary_signal_catches_remaining_full_run_failures(left, right, signal):
    assert LLMTranslator._chinese_boundary_signal(left, right) == signal


@pytest.mark.parametrize(
    ("left", "right", "signal"),
    [
        ("但我觉得吧", "阅读深刻影响了我的成长", "vague filler-only frame"),
        ("而且我觉得真正想说的重点并非如此", "选择会逐渐累积", "semantic frame is incomplete"),
        ("但他确实非常适合", "而且他懂得如何传播观点", "adjective complement is missing"),
        ("而且这篇将近九千字", "但希望读者理解", "classifier phrase is stranded"),
        ("而是在数字时代", "同时兼顾印刷时代", "unfinished Chinese locative frame"),
        ("是的 我认为阅读", "阅读影响了我的成长", "possible duplicated boundary phrase"),
    ],
)
def test_chinese_boundary_signal_catches_final_quality_audit_failures(left, right, signal):
    assert LLMTranslator._chinese_boundary_signal(left, right) == signal


def test_chinese_boundary_signal_catches_semantically_duplicated_neighbor():
    assert (
        LLMTranslator._chinese_boundary_signal(
            "因为这让他感觉 他在个人生活中和工作中的是同一个人",
            "他在个人生活和工作中是同一个人",
        )
        == "possible duplicated boundary meaning"
    )


def test_chinese_boundary_signal_catches_repeated_short_locative_topic():
    assert (
        LLMTranslator._chinese_boundary_signal(
            "他在个人生活中和工作中是同一个人",
            "在工作中也是如此",
        )
        == "possible duplicated boundary meaning"
    )


@pytest.mark.parametrize(
    ("left", "right", "signal"),
    [
        (
            "这是一本很有趣的书 我觉得",
            "它挑战了基本直觉",
            "unfinished Chinese grammatical structure",
        ),
        ("你觉得在约束和障碍之间", "是否存在区别", "unfinished Chinese locative frame"),
        ("我要长途步行 带着一个", "小婴儿在身上", "classifier phrase is stranded"),
    ],
)
def test_chinese_boundary_signal_catches_general_residual_dependency(
    left,
    right,
    signal,
):
    assert LLMTranslator._chinese_boundary_signal(left, right) == signal


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("这么说很恰当", "我们就聊到这里"),
        ("强迫自己使用另一种东西", "他随后改变了方法"),
    ],
)
def test_chinese_boundary_signal_avoids_embedded_character_false_positives(left, right):
    assert LLMTranslator._chinese_boundary_signal(left, right) == ""


def test_chinese_boundary_signal_does_not_flag_short_parallel_contrast():
    assert (
        LLMTranslator._chinese_boundary_signal(
            "还是只是在说孩子",
            "还是只是在说成年人",
        )
        == ""
    )


def test_multispeaker_same_turn_audits_strong_contrast_dependency():
    source, translated = _dialogue_items(
        "Do you draw a meaningful distinction between constraints on the one hand",
        "and impediments on the other?",
        "你是否在约束一方面做出了有意义的区分",
        "和障碍之间有明显的区别吗",
    )
    translator = _make_translator(is_reflect=True)
    translator._all_speaker_by_index = {1: "S1", 2: "S1", 3: "S2"}

    assert translator._chinese_fluency_candidates(source, translated) == [1]


def test_chinese_fluency_window_expands_across_pronoun_dependency_chain():
    translator = _make_translator(is_reflect=True)
    source = [
        SubtitleProcessData(index=720, original_text="These ideas are useless"),
        SubtitleProcessData(index=721, original_text="if you do not find ways to concretize"),
        SubtitleProcessData(index=722, original_text="them"),
        SubtitleProcessData(index=723, original_text="in your own life so they create change."),
    ]
    translator._all_speaker_by_index = {item.index: "S1" for item in source}

    windows = translator._chinese_fluency_windows(source, [0])

    assert [[item.index for item in window] for window in windows] == [[720, 721, 722, 723]]


def test_chinese_fluency_window_expands_to_full_source_dependency_chain():
    translator = _make_translator(is_reflect=True)
    source = [
        SubtitleProcessData(index=4, original_text="but that he's really well suited"),
        SubtitleProcessData(
            index=5,
            original_text="and he's figured out how to get his message out in this",
        ),
        SubtitleProcessData(index=6, original_text="current information environment."),
    ]
    translator._all_speaker_by_index = {4: "S1", 5: "S1", 6: "S1"}

    windows = translator._chinese_fluency_windows(source, [1])

    assert [[item.index for item in window] for window in windows] == [[4, 5, 6]]


def test_single_speaker_candidates_audit_every_open_source_clause_boundary():
    translator = _make_translator()
    source = [
        SubtitleProcessData(
            index=18,
            original_text=(
                "Now, with Frankfurt's long-awaited new terminal finally up and running "
                "after experiencing"
            ),
        ),
        SubtitleProcessData(
            index=19,
            original_text=(
                "some turbulence of its own, the question is has Germany achieved a smooth "
                "touchdown this time around?"
            ),
        ),
    ]
    translated = {
        18: replace(source[0], translated_text="法兰克福期待已久的新航站楼终于启用"),
        19: replace(source[1], translated_text="德国这次实现平稳落地了吗"),
    }

    candidates = translator._chinese_fluency_candidates(source, translated)

    assert candidates == [18]


def test_multispeaker_candidates_audit_strong_open_boundary_risk():
    translator = _make_translator()
    translator._all_speaker_by_index = {
        18: "speaker-1",
        19: "speaker-1",
        20: "speaker-2",
    }
    source = [
        SubtitleProcessData(
            index=18,
            original_text=(
                "Now, with Frankfurt's long-awaited new terminal finally up and running "
                "after experiencing"
            ),
        ),
        SubtitleProcessData(
            index=19,
            original_text=(
                "some turbulence of its own, the question is has Germany achieved a smooth "
                "touchdown this time around?"
            ),
        ),
    ]
    translated = {
        18: replace(source[0], translated_text="法兰克福期待已久的新航站楼终于启用"),
        19: replace(source[1], translated_text="德国这次实现平稳落地了吗"),
    }

    candidates = translator._chinese_fluency_candidates(source, translated)

    assert candidates == [18]


def test_open_source_boundary_stops_before_clear_new_sentence_without_period():
    assert not LLMTranslator._is_open_source_boundary(
        "There is plenty of cargo room",
        "The rear seats also fold flat.",
    )


def test_open_source_boundary_keeps_hyphenated_modifier_with_capitalized_acronym():
    assert LLMTranslator._is_open_source_boundary(
        "Linglong-1 is the world's first fully commercial land-based",
        "SMR, and it is set to come online.",
    )


def test_fluency_window_expands_across_generic_open_clause_run():
    translator = _make_translator()
    source = [
        SubtitleProcessData(index=810, original_text="The opening was delayed"),
        SubtitleProcessData(index=811, original_text="after suppliers struggled"),
        SubtitleProcessData(index=812, original_text="to bring equipment to the site."),
        SubtitleProcessData(index=813, original_text="The terminal later opened."),
    ]

    windows = translator._chinese_fluency_windows(source, [0])

    assert [[item.index for item in window] for window in windows] == [[810, 811, 812]]


def test_rejects_double_negative_that_reverses_source_meaning():
    translator = _make_translator(is_reflect=False)

    valid, error = translator._validate_llm_response(
        {"1": "而且 不是不重视或不愿去获取"},
        {"1": "And not kind of valuing or choosing to access"},
        require_reflect=False,
    )

    assert not valid
    assert "double negative" in error


def test_accepts_direct_negative_valuation_translation():
    translator = _make_translator(is_reflect=False)

    valid, error = translator._validate_llm_response(
        {"1": "而是不再重视 也不愿主动获取"},
        {"1": "And not kind of valuing or choosing to access"},
        require_reflect=False,
    )

    assert valid
    assert error == ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("所以但我觉得这件事很重要", "不过我觉得这件事很重要"),
        ("而且不过这并非坏事", "不过这并非坏事"),
        ("但是所以我们需要继续", "所以我们需要继续"),
        ("不过是的 我同意", "是的 不过 我同意"),
        ("正常的连接词不应改变", "正常的连接词不应改变"),
    ],
)
def test_normalizes_only_sentence_initial_stacked_connectives(raw, expected):
    assert LLMTranslator._normalize_stacked_chinese_connectives(raw) == expected


def test_normalizes_nested_reflective_translation_in_place():
    translator = _make_translator(is_reflect=True)
    response = {"1": {"native_translation": "所以但我仍然会阅读"}}

    translator._normalize_chinese_response_connectives(response)

    assert response["1"]["native_translation"] == "不过我仍然会阅读"


def test_chinese_boundary_signal_allows_complete_degree_phrase_ending_in_some():
    assert (
        LLMTranslator._chinese_boundary_signal(
            "这样学生在阅读理解上会更容易一些",
            "所以趋势可能开始变化",
        )
        == ""
    )


def test_chinese_boundary_signal_catches_open_linking_predicate():
    assert (
        LLMTranslator._chinese_boundary_signal(
            "所以虽然CERN可能已经成为",
            "尽管欧洲核子研究中心因LHC而成为世界上最知名的科学组织之一",
        )
        == "unfinished Chinese grammatical structure"
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("所以如果人们想要做出不同的选择", "他们应该知道这一点"),
        ("尤利乌斯·凯撒烧毁了它 黑暗时代由此开始", "而现在学界有了新的共识"),
        ("它们体现在我们的思维方式、国家和文化之中", "但也有人愿意放弃阅读"),
    ],
)
def test_chinese_boundary_signal_avoids_complete_clauses_from_full_quality_audit(
    left,
    right,
):
    assert LLMTranslator._chinese_boundary_signal(left, right) == ""


def test_chinese_boundary_signal_catches_stranded_growth_multiplier():
    assert (
        LLMTranslator._chinese_boundary_signal(
            "我猜 至少会增长到",
            "我们开始运行对撞机时的1.5到2倍",
        )
        == "numeric complement is stranded"
    )


def test_chinese_boundary_signal_allows_complete_growth_multiplier():
    assert not LLMTranslator._chinese_boundary_signal(
        "团队规模至少会增长到现在的1.5倍",
        "之后还会继续扩大",
    )


def test_chinese_boundary_signal_catches_missing_consequence_predicate():
    assert (
        LLMTranslator._chinese_boundary_signal(
            "事实上 这项研究非常成功 以至于三项诺贝尔奖",
            "它至今仍在使用",
        )
        == "consequence predicate is missing"
    )


def test_chinese_boundary_signal_accepts_completed_award_predicate():
    assert not LLMTranslator._chinese_boundary_signal(
        "事实上 这项研究非常成功 以至于后来颁发了三项诺贝尔奖",
        "这都得益于它 而且至今仍在使用",
    )


def test_chinese_boundary_signal_accepts_passive_use_before_new_subject():
    assert not LLMTranslator._chinese_boundary_signal(
        "这都得益于它 而且至今仍在使用",
        "那些被送入机器的离子来自同步加速器",
    )


@pytest.mark.parametrize(
    "left",
    [
        "而现在多伦多已提出 11 座",
        "目前已经获批三项",
        "全市已建成两栋",
    ],
)
def test_chinese_boundary_signal_catches_count_without_contextual_head_noun(left):
    assert (
        LLMTranslator._chinese_boundary_signal(left, "其中一部分已经投入使用")
        == "count classifier lacks its contextual head noun"
    )


@pytest.mark.parametrize(
    "left",
    [
        "我们现在基本上",
        "归纳起来总体上",
        "从项目性质来看本质上",
    ],
)
def test_chinese_boundary_signal_catches_stranded_sentence_adverb(left):
    assert (
        LLMTranslator._chinese_boundary_signal(left, "面对的是一个混合用途的世界")
        == "sentence adverb is separated from its predicate"
    )


def test_chinese_boundary_signal_accepts_completed_sentence_before_adverbial_reply():
    assert not LLMTranslator._chinese_boundary_signal(
        "这就是目前的情况 基本上。",
        "接下来我们看另一项计划",
    )


@pytest.mark.parametrize("left", ["基本上我们现在", "但他们目前", "所以你实际上"])
def test_chinese_boundary_signal_catches_subject_adverb_without_predicate(left):
    assert (
        LLMTranslator._chinese_boundary_signal(left, "面对的是一个混合用途的世界")
        == "subject and sentence adverb are separated from their predicate"
    )


def test_style_guidance_preserves_official_name_wordplay():
    guidance = target_language_style_rules(
        "简体中文",
        [
            "No, that is not just me calling it an epic detector.",
            "It is really what the lab named it.",
        ],
    )

    assert "Terminology must not erase the wordplay" in guidance


def test_normalizes_repeated_chinese_false_start_before_reporting_clause():
    assert (
        LLMTranslator._normalize_stacked_chinese_connectives(
            "所以我不 我在文章里写道 我不认为这是问题"
        )
        == "我在文章里写道 我不认为这是问题"
    )
