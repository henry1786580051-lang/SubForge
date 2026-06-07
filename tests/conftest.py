"""Root-level test configuration and shared fixtures."""

import ast
import json
import os
import re
from types import SimpleNamespace
from typing import Dict, List

import pytest

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.translate import SubtitleProcessData, TargetLanguage
from subforge.core.utils import cache

# Disable cache for testing
cache.disable_cache()


@pytest.fixture(autouse=True)
def isolate_global_caches():
    """Keep tests independent from global cache state and persisted diskcache entries."""
    cache.disable_cache()
    for cache_instance in (
        cache.get_llm_cache(),
        cache.get_tts_cache(),
        cache.get_translate_cache(),
    ):
        cache_instance.clear()
    yield
    cache.disable_cache()


@pytest.fixture
def sample_asr_data():
    """Create sample ASR data for translation testing."""
    segments = [
        ASRDataSeg(start_time=0, end_time=1000, text="I am a student"),
        ASRDataSeg(start_time=1000, end_time=2000, text="You are a teacher"),
        ASRDataSeg(start_time=2000, end_time=3000, text="SubForge is a tool for captioning videos"),
    ]
    return ASRData(segments)


@pytest.fixture
def sample_translate_data():
    """Create sample translation data for testing."""
    return [
        SubtitleProcessData(index=1, original_text="I am a student", translated_text=""),
        SubtitleProcessData(index=2, original_text="You are a teacher", translated_text=""),
        SubtitleProcessData(index=3, original_text="SubForge is a tool for captioning videos", translated_text=""),
    ]


@pytest.fixture
def target_language():
    """Default target language for translation tests."""
    return TargetLanguage.SIMPLIFIED_CHINESE


@pytest.fixture
def check_env_vars():
    """Check if required environment variables are set."""
    def _check(*var_names):
        if os.getenv("SUBFORGE_RUN_LIVE_LLM_TESTS") != "1":
            pytest.skip("Live LLM integration tests require SUBFORGE_RUN_LIVE_LLM_TESTS=1")
        missing = [var for var in var_names if not os.getenv(var)]
        if missing:
            pytest.skip(f"Required environment variables not set: {', '.join(missing)}")
    return _check


def _mock_llm_response(content: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


def _split_mock_text(text: str) -> str:
    is_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    if is_cjk:
        pieces = [part for part in re.split(r"(?<=[。！？])", text) if part]
        result: list[str] = []
        for piece in pieces or [text]:
            if len(piece) <= 15:
                result.append(piece)
            else:
                result.extend(piece[i : i + 15] for i in range(0, len(piece), 15))
        return "<br>".join(result)

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    result = []
    for sentence in sentences or [text]:
        words = sentence.split()
        if len(words) <= 12:
            result.append(sentence)
        else:
            for i in range(0, len(words), 12):
                result.append(" ".join(words[i : i + 12]))
    return "<br>".join(result)


def _translate_mock_value(text: str, target: str, reflect: bool):
    if "日本語" in target:
        translated = f"日本語訳: {text}"
    elif "繁體中文" in target:
        translated = f"繁體中文翻譯：{text}"
    elif "简体中文" in target or "中文" in target:
        translated = f"中文翻译：{text}"
    else:
        translated = f"Translated: {text}"
    if reflect:
        return {"native_translation": translated}
    return translated


@pytest.fixture
def mock_llm_client(monkeypatch):
    """Patch LLM calls used by split/translate/optimize tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("VIDEOCAPTIONER_LLM_MODEL", "gpt-4o-mini")

    def fake_call_llm(messages, model=None, **kwargs):
        system = str(messages[0].get("content", "")) if messages else ""
        user = str(messages[-1].get("content", "")) if messages else ""
        combined = f"{system}\n{user}"

        if "<input_subtitle>" in user:
            match = re.search(r"<input_subtitle>(.*?)</input_subtitle>", user, re.S)
            data = ast.literal_eval(match.group(1)) if match else {}
            return _mock_llm_response(json.dumps(data, ensure_ascii=False))

        if "Please use multiple <br> tags" in user:
            text = user.split("\n", 1)[-1]
            return _mock_llm_response(_split_mock_text(text))

        try:
            data = json.loads(user)
        except Exception:
            data = {}

        if isinstance(data, dict):
            if "transcript_excerpt" in data:
                return _mock_llm_response(
                    json.dumps(
                        {
                            "summary": "Mock video summary",
                            "terminology": [{"source": "Lexus", "target": "雷克萨斯", "note": "brand"}],
                            "style": "Natural subtitle translation",
                        },
                        ensure_ascii=False,
                    )
                )

            if isinstance(data.get("current_subtitles"), dict):
                data = data["current_subtitles"]

            reflect = "native_translation" in combined or "reflect" in combined.lower()
            target_match = re.search(r"(简体中文|繁體中文|日本語|한국어|English)", combined)
            target = target_match.group(1) if target_match else "简体中文"
            translated = {
                str(key): _translate_mock_value(str(value), target, reflect)
                for key, value in data.items()
            }
            return _mock_llm_response(json.dumps(translated, ensure_ascii=False))

        return _mock_llm_response("中文翻译：mock")

    import subforge.core.llm as llm_module
    import subforge.core.optimize.optimize as optimize_module
    import subforge.core.split.split_by_llm as split_module
    import subforge.core.translate.context as translate_context_module
    import subforge.core.translate.llm_translator as translator_module
    import subforge.ui.thread.subtitle_thread as subtitle_thread_module

    monkeypatch.setattr(llm_module, "call_llm", fake_call_llm)
    monkeypatch.setattr(optimize_module, "call_llm", fake_call_llm)
    monkeypatch.setattr(split_module, "call_llm", fake_call_llm)
    monkeypatch.setattr(translate_context_module, "call_llm", fake_call_llm)
    monkeypatch.setattr(translator_module, "call_llm", fake_call_llm)
    monkeypatch.setattr(
        subtitle_thread_module,
        "check_llm_connection",
        lambda *args, **kwargs: (True, ""),
    )
    return fake_call_llm


@pytest.fixture
def expected_translations() -> Dict[str, Dict[str, List[str]]]:
    """Expected translation keywords for quality validation."""
    return {
        "简体中文": {
            "I am a student": ["学生"],
            "You are a teacher": ["老师", "教师"],
            "SubForge is a tool for captioning videos": ["工具"],
        },
        "日本語": {
            "I am a student": ["学生"],
            "You are a teacher": ["先生", "教師"],
        },
        "English": {
            "我是学生": ["student"],
            "你是老师": ["teacher"],
        },
    }


def assert_translation_quality(original: str, translated: str, expected_keywords: List[str]) -> None:
    """Validate translation contains expected keywords."""
    assert translated, f"Translation is empty for: {original}"
    found_keywords = [kw for kw in expected_keywords if kw in translated]
    assert found_keywords, (
        f"Translation quality issue:\n"
        f"  Original: {original}\n"
        f"  Translated: {translated}\n"
        f"  Expected keywords: {expected_keywords}"
    )
