"""Tests for LLM translation response validation."""

import pytest

from subforge.core.entities import SubtitleProcessData
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


class TestValidateLLmResponse:
    """Test _validate_llm_response with standard and reflect modes."""

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
        resp = {"0": "你好", "1": "世界", "2": "美好的", "3": "天气", "4": "今天", "5": "不错", "6": "OK"}
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

    def test_cjk_target_rejects_threshold_boundary_english(self):
        """The old percentage threshold must not allow untranslated full lines."""
        t = _make_translator()
        resp = {"0": "你好", "1": "世界", "2": "好的", "3": "hello"}
        inp = {"0": "a", "1": "b", "2": "c", "3": "d"}
        ok, msg = t._validate_llm_response(resp, inp)
        assert ok is False
        assert "Untranslated keys" in msg

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

    def test_single_fallback_rejects_untranslated_cjk_result(self, monkeypatch):
        t = _make_translator()

        class _Message:
            content = "hello"

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        monkeypatch.setattr(
            "subforge.core.translate.llm_translator.call_llm",
            lambda **kwargs: _Response(),
        )

        with pytest.raises(RuntimeError, match="Single item translation failed"):
            t._translate_chunk_single([SubtitleProcessData(index=1, original_text="hello")])

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
