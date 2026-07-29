import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

import app.api.config as config_module
import pytest
from fastapi import HTTPException


def test_get_config_value_rejects_corrupted_types(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "_settings_cache",
        {
            "threads": "ten",
            "enhance": "false",
            "model": 123,
            "ratio": 2,
        },
    )
    monkeypatch.setattr(config_module, "_cache_time", time.monotonic())

    assert config_module.get_config_value("threads", 4) == 4
    assert config_module.get_config_value("enhance", True) is True
    assert config_module.get_config_value("model", "large-v3") == "large-v3"
    assert config_module.get_config_value("ratio", 1.5) == 2.0


def test_write_settings_is_atomic_and_private(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(config_module, "_SETTINGS_CANDIDATES", [settings_path])

    config_module._write_settings({"llm_api_key": "secret"})

    assert json.loads(settings_path.read_text(encoding="utf-8")) == {"llm_api_key": "secret"}
    assert not (tmp_path / ".settings.json.tmp").exists()
    if os.name != "nt":
        assert settings_path.stat().st_mode & 0o777 == 0o600


def test_effective_config_migrates_whisperx_on_unsupported_platform(monkeypatch):
    monkeypatch.setattr(config_module, "_WHISPERX_SUPPORTED", False)

    config = config_module._effective_config({"transcribe_model": "whisperx"})

    assert config["transcribe_model"] == "whisper_cpp"


def test_effective_config_keeps_whisperx_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(config_module, "_WHISPERX_SUPPORTED", True)

    config = config_module._effective_config({"transcribe_model": "whisperx"})

    assert config["transcribe_model"] == "whisperx"


def test_effective_config_keeps_whisperx_on_windows_runtime(monkeypatch):
    monkeypatch.setattr(config_module, "_IS_APPLE_SILICON", False)
    monkeypatch.setattr(config_module, "_WHISPERX_SUPPORTED", True)

    config = config_module._effective_config({"transcribe_model": "whisperx"})

    assert config["transcribe_model"] == "whisperx"


def test_effective_config_defaults_alignment_to_automatic_language_matching():
    config = config_module._effective_config({})

    assert config["whisperx_alignment_strategy"] == "auto"


def test_effective_config_preserves_legacy_custom_alignment_as_manual():
    config = config_module._effective_config({"whisperx_align_model": "example/custom-alignment"})

    assert config["whisperx_alignment_strategy"] == "manual"


def test_effective_config_discards_corrupted_persisted_types():
    config = config_module._effective_config(
        {"thread_num": "ten", "enable_audio_enhancement": "false"}
    )

    assert config["thread_num"] == 5
    assert config["enable_audio_enhancement"] is True


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("thread_num", 0),
        ("batch_size", 101),
        ("translator", "unknown"),
        ("target_language", "hindi"),
        ("speaker_count", 1),
        ("speaker_count", 11),
        ("speaker_diarization", "five"),
    ],
)
def test_config_update_rejects_values_that_would_silently_fallback(key, value):
    with pytest.raises(HTTPException):
        config_module._validate_config_update(key, value)


def test_effective_config_migrates_legacy_llm_credentials_to_detected_provider():
    config = config_module._effective_config(
        {
            "llm_base_url": "https://token-plan-cn.xiaomimimo.com/v1",
            "llm_api_key": "mimo-secret",
            "llm_model": "mimo-v2.5-pro",
        }
    )

    assert config["llm_provider"] == "mimo"
    assert config["llm_profiles"]["mimo"] == {
        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
        "api_key": "mimo-secret",
        "model": "mimo-v2.5-pro",
    }


def test_switch_llm_provider_keeps_credentials_isolated_and_restores_them(
    tmp_path,
    monkeypatch,
):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(config_module, "_SETTINGS_CANDIDATES", [settings_path])
    config_module._write_settings(
        {
            "llm_base_url": "https://token-plan-cn.xiaomimimo.com/v1",
            "llm_api_key": "mimo-secret",
            "llm_model": "mimo-v2.5-pro",
        }
    )

    deepseek = asyncio.run(
        config_module.switch_llm_provider(
            config_module.LlmProviderSwitch(
                provider="deepseek",
                current_base_url="https://token-plan-cn.xiaomimimo.com/v1",
                current_api_key="mimo-secret",
                current_model="mimo-v2.5-pro",
            )
        )
    )

    assert deepseek == {
        "status": "ok",
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "api_key_configured": False,
        "model": "",
    }

    mimo = asyncio.run(
        config_module.switch_llm_provider(
            config_module.LlmProviderSwitch(
                provider="mimo",
                current_base_url="https://api.deepseek.com/v1",
                current_api_key="deepseek-secret",
                current_model="deepseek-chat",
            )
        )
    )

    assert mimo["base_url"] == "https://token-plan-cn.xiaomimimo.com/v1"
    assert mimo["api_key_configured"] is True
    assert mimo["model"] == "mimo-v2.5-pro"
    stored = json.loads(settings_path.read_text(encoding="utf-8"))
    assert stored["llm_profiles"]["deepseek"]["api_key"] == "deepseek-secret"
    assert stored["llm_profiles"]["mimo"]["api_key"] == "mimo-secret"


def test_public_config_reports_credentials_without_exposing_them():
    public = config_module._public_config(
        {
            "llm_api_key": "llm-secret",
            "whisper_api_key": "whisper-secret",
            "huggingface_token": "hf-secret",
            "llm_profiles": {
                "mimo": {
                    "base_url": "https://example.test",
                    "api_key": "profile-secret",
                    "model": "mimo",
                }
            },
        }
    )

    assert public["llm_api_key"] == ""
    assert public["whisper_api_key"] == ""
    assert public["huggingface_token"] == ""
    assert public["llm_api_key_configured"] is True
    assert public["llm_profiles"]["mimo"] == {
        "base_url": "https://example.test",
        "model": "mimo",
        "api_key_configured": True,
    }


def test_public_config_exposes_resolved_subtitle_length_policy():
    public = config_module._public_config(
        {
            "max_word_count_cjk": 30,
            "max_word_count_english": 16,
            "llm_profiles": {},
        }
    )

    assert public["subtitle_length_policy"] == {
        "cjk_hard_limit": 30,
        "english_soft_limit": 16,
        "english_hard_limit": 20,
    }


def test_switch_provider_blank_key_does_not_erase_current_profile(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(config_module, "_SETTINGS_CANDIDATES", [settings_path])
    config_module._write_settings(
        {
            "llm_provider": "mimo",
            "llm_base_url": "https://token-plan-cn.xiaomimimo.com/v1",
            "llm_api_key": "saved-secret",
            "llm_model": "mimo-v2.5-pro",
        }
    )

    asyncio.run(
        config_module.switch_llm_provider(
            config_module.LlmProviderSwitch(
                provider="deepseek",
                current_base_url="https://token-plan-cn.xiaomimimo.com/v1",
                current_api_key="",
                current_model="mimo-v2.5-pro",
            )
        )
    )

    stored = json.loads(settings_path.read_text(encoding="utf-8"))
    assert stored["llm_profiles"]["mimo"]["api_key"] == "saved-secret"


def test_updating_active_llm_key_updates_only_active_profile(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(config_module, "_SETTINGS_CANDIDATES", [settings_path])
    config_module._write_settings(
        {
            "llm_provider": "deepseek",
            "llm_base_url": "https://api.deepseek.com",
            "llm_api_key": "old-key",
            "llm_model": "deepseek-chat",
            "llm_profiles": {
                "deepseek": {
                    "base_url": "https://api.deepseek.com",
                    "api_key": "old-key",
                    "model": "deepseek-chat",
                },
                "mimo": {
                    "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
                    "api_key": "mimo-key",
                    "model": "mimo-v2.5-pro",
                },
            },
        }
    )

    asyncio.run(
        config_module.update_config(
            config_module.ConfigUpdate(key="llm_api_key", value="new-deepseek-key")
        )
    )

    stored = json.loads(settings_path.read_text(encoding="utf-8"))
    assert stored["llm_profiles"]["deepseek"]["api_key"] == "new-deepseek-key"
    assert stored["llm_profiles"]["mimo"]["api_key"] == "mimo-key"


def test_detects_current_and_legacy_minimax_urls():
    assert config_module._detect_llm_provider("https://api.minimaxi.com/anthropic") == "minimax"
    assert config_module._detect_llm_provider("https://api.minimaxi.com/v1") == "minimax"
    assert config_module._detect_llm_provider("https://api.minimax.chat/v1") == "minimax"


def test_detects_nvidia_provider_url():
    assert (
        config_module._detect_llm_provider("https://integrate.api.nvidia.com/v1")
        == "nvidia"
    )


def test_switch_to_nvidia_uses_default_url_and_isolated_credentials(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(config_module, "_SETTINGS_CANDIDATES", [settings_path])
    config_module._write_settings(
        {
            "llm_provider": "deepseek",
            "llm_base_url": "https://api.deepseek.com",
            "llm_api_key": "deepseek-secret",
            "llm_model": "deepseek-chat",
        }
    )

    nvidia = asyncio.run(
        config_module.switch_llm_provider(
            config_module.LlmProviderSwitch(
                provider="nvidia",
                current_base_url="https://api.deepseek.com",
                current_api_key="",
                current_model="deepseek-chat",
            )
        )
    )

    assert nvidia == {
        "status": "ok",
        "provider": "nvidia",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_configured": False,
        "model": "",
    }
    stored = json.loads(settings_path.read_text(encoding="utf-8"))
    assert stored["llm_profiles"]["deepseek"]["api_key"] == "deepseek-secret"
    assert stored["llm_profiles"]["nvidia"]["api_key"] == ""


def test_effective_config_migrates_legacy_minimax_profile_url():
    config = config_module._effective_config(
        {
            "llm_provider": "minimax",
            "llm_base_url": "https://api.minimax.chat/v1",
            "llm_api_key": "minimax-key",
            "llm_model": "MiniMax-M3",
            "llm_profiles": {
                "minimax": {
                    "base_url": "https://api.minimax.chat/v1",
                    "api_key": "minimax-key",
                    "model": "MiniMax-M3",
                }
            },
        }
    )

    assert config["llm_base_url"] == "https://api.minimaxi.com/anthropic"
    assert config["llm_profiles"]["minimax"]["base_url"] == ("https://api.minimaxi.com/anthropic")
