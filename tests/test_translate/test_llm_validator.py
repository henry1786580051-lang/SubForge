"""Tests for LLM translation response validation."""

import pytest

from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.types import TargetLanguage


def _make_translator(is_reflect=False, threshold=0.15):
    """Create a translator instance for testing."""
    t = LLMTranslator(
        thread_num=1,
        batch_num=1,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        model="test",
        custom_prompt="",
        is_reflect=is_reflect,
        update_callback=None,
    )
    t.UNTRANSLATED_THRESHOLD = threshold
    return t


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
        t = _make_translator(threshold=0.15)
        resp = {"0": "hello", "1": "world", "2": "foo", "3": "bar"}
        inp = {"0": "a", "1": "b", "2": "c", "3": "d"}
        ok, msg = t._validate_llm_response(resp, inp)
        assert ok is False
        assert "still in the source language" in msg

    def test_cjk_target_accepts_mostly_chinese(self):
        """Mostly Chinese with one English entry under threshold should pass."""
        t = _make_translator(threshold=0.15)
        # 7 entries, 1 untranslated = 14.3% < 15% threshold
        resp = {"0": "你好", "1": "世界", "2": "美好的", "3": "天气", "4": "今天", "5": "不错", "6": "OK"}
        inp = {str(i): f"text_{i}" for i in range(7)}
        ok, msg = t._validate_llm_response(resp, inp)
        assert ok is True

    def test_cjk_threshold_boundary(self):
        """Exactly at threshold should pass, over threshold should fail."""
        t = _make_translator(threshold=0.25)
        # 4 entries, 1 untranslated = 25% — at threshold, should pass
        resp = {"0": "你好", "1": "世界", "2": "好的", "3": "hello"}
        inp = {"0": "a", "1": "b", "2": "c", "3": "d"}
        ok, _ = t._validate_llm_response(resp, inp)
        assert ok is True

        # 4 entries, 2 untranslated = 50% — over threshold, should fail
        resp2 = {"0": "你好", "1": "world", "2": "好的", "3": "hello"}
        ok2, _ = t._validate_llm_response(resp2, inp)
        assert ok2 is False

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
