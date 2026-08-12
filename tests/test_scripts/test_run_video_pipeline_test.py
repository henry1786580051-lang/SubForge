from argparse import Namespace

import pytest

from scripts.run_video_pipeline_test import _build_config


def _args(**overrides):
    values = {
        "language": "en",
        "whisper_batch_size": 4,
        "llm_threads": 20,
        "llm_batch_size": 20,
        "llm_api_base": None,
        "llm_model": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_pipeline_runner_does_not_prefer_legacy_mimo_environment(monkeypatch):
    monkeypatch.setenv("MIMO_API_KEY", "legacy-mimo-key")
    monkeypatch.setenv("MIMO_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
    monkeypatch.setenv("MIMO_MODEL", "mimo-v2.5-pro")
    monkeypatch.setenv("OPENAI_API_KEY", "deepseek-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-v4-flash")

    config = _build_config(_args())

    assert config["llm"] == {
        "api_key": "deepseek-key",
        "api_base": "https://api.deepseek.com/v1",
        "model": "deepseek-v4-flash",
    }
    assert config["subtitle"]["max_word_count_english"] == 16


def test_pipeline_runner_accepts_explicit_subtitle_length_policy(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "deepseek-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-v4-flash")

    config = _build_config(_args(max_english=18, max_cjk=20))

    assert config["subtitle"]["max_word_count_english"] == 18
    assert config["subtitle"]["max_word_count_cjk"] == 20


def test_pipeline_runner_rejects_explicit_mixed_provider(monkeypatch):
    monkeypatch.setenv("SUBFORGE_TEST_LLM_API_KEY", "test-key")

    with pytest.raises(ValueError, match="belongs to 'deepseek'"):
        _build_config(
            _args(
                llm_api_base="https://token-plan-cn.xiaomimimo.com/v1",
                llm_model="deepseek-v4-flash",
            )
        )


def test_pipeline_runner_rejects_implicit_default_model(monkeypatch):
    for key in (
        "SUBFORGE_TEST_LLM_API_KEY",
        "SUBFORGE_TEST_LLM_BASE_URL",
        "SUBFORGE_TEST_LLM_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValueError, match="explicit LLM runtime"):
        _build_config(_args())
