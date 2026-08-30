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


def test_effective_config_never_uses_unresolved_keyring_reference_as_secret():
    reference = "keyring://llm-profile:missing"

    config = config_module._effective_config(
        {
            "llm_provider": "deepseek",
            "llm_profiles": {
                "deepseek": {
                    "base_url": "https://api.deepseek.com",
                    "model": "deepseek-chat",
                    "api_key": reference,
                }
            },
            "whisper_api_key": reference,
        }
    )

    assert config["llm_api_key"] == ""
    assert config["llm_profiles"]["deepseek"]["api_key"] == ""
    assert config["whisper_api_key"] == ""


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


def test_provider_runtime_config_reads_inactive_nvidia_profile_without_switching(
    tmp_path,
    monkeypatch,
):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "llm_provider": "deepseek",
                "llm_profiles": {
                    "deepseek": {
                        "base_url": "https://api.deepseek.com",
                        "api_key": "deepseek-secret",
                        "model": "deepseek-chat",
                    },
                    "nvidia": {
                        "base_url": "https://integrate.api.nvidia.com/v1",
                        "api_key": "nvidia-secret",
                        "model": "nvidia/nemotron-3-ultra-550b-a55b",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "_SETTINGS_CANDIDATES", [settings_path])

    runtime = config_module.get_llm_provider_runtime_config("nvidia")
    status = config_module.get_llm_provider_status("nvidia")

    assert runtime.provider == "nvidia"
    assert runtime.api_key == "nvidia-secret"
    assert runtime.model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert status["api_key_configured"] is True
    assert config_module._active_llm_provider(config_module._read_settings()) == "deepseek"


def test_public_config_reports_credentials_without_exposing_them():
    public = config_module._public_config(
        {
            "llm_api_key": "llm-secret",
            "whisper_api_key": "whisper-secret",
            "huggingface_token": "hf-secret",
            "azure_translator_key": "azure-secret",
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
    assert public["azure_translator_key"] == ""
    assert public["llm_api_key_configured"] is True
    assert public["azure_translator_key_configured"] is True
    assert public["llm_profiles"]["mimo"] == {
        "base_url": "https://example.test",
        "model": "mimo",
        "api_key_configured": True,
    }


def test_get_config_does_not_unlock_keychain_credentials(monkeypatch):
    reference = "keyring://llm-profile:deepseek"
    monkeypatch.setattr(
        config_module,
        "_read_settings",
        lambda: {
            "llm_provider": "deepseek",
            "llm_profiles": {
                "deepseek": {
                    "base_url": "https://api.deepseek.com",
                    "api_key": reference,
                    "model": "deepseek-chat",
                },
                "minimax": {
                    "base_url": "https://api.minimaxi.com/anthropic",
                    "api_key": "keyring://llm-profile:minimax",
                    "model": "MiniMax-M3",
                },
            },
        },
    )
    monkeypatch.setattr(
        config_module,
        "restore_secret_value",
        lambda value: (_ for _ in ()).throw(AssertionError("must stay lazy")),
    )

    public = asyncio.run(config_module.get_config())

    assert public["llm_api_key"] == ""
    assert public["llm_api_key_configured"] is True
    assert public["llm_profiles"]["deepseek"]["api_key_configured"] is True
    assert public["llm_profiles"]["minimax"]["api_key_configured"] is True


def test_runtime_unlocks_only_active_llm_provider(monkeypatch):
    values = {
        "keyring://llm-profile:deepseek": "deepseek-secret",
        "keyring://llm-profile:minimax": "minimax-secret",
    }
    calls = []
    monkeypatch.setattr(
        config_module,
        "_read_settings",
        lambda: {
            "llm_provider": "deepseek",
            "llm_profiles": {
                "deepseek": {
                    "base_url": "https://api.deepseek.com",
                    "api_key": "keyring://llm-profile:deepseek",
                    "model": "deepseek-chat",
                },
                "minimax": {
                    "base_url": "https://api.minimaxi.com/anthropic",
                    "api_key": "keyring://llm-profile:minimax",
                    "model": "MiniMax-M3",
                },
            },
        },
    )

    def restore(value):
        calls.append(value)
        return values[value]

    monkeypatch.setattr(config_module, "restore_secret_value", restore)

    runtime = config_module.get_llm_runtime_config()

    assert runtime.api_key == "deepseek-secret"
    assert calls == ["keyring://llm-profile:deepseek"]


def test_azure_translator_endpoint_must_be_https():
    with pytest.raises(HTTPException):
        config_module._validate_config_update(
            "azure_translator_endpoint", "http://example.test"
        )

    assert config_module._validate_config_update(
        "azure_translator_endpoint",
        "https://example.cognitiveservices.azure.com/translator/text/v3.0",
    ) == "https://example.cognitiveservices.azure.com/translator/text/v3.0"


def test_azure_translator_connection_requires_key(monkeypatch):
    monkeypatch.setattr(config_module, "_read_settings", lambda: {})

    result = asyncio.run(config_module.test_azure_translator_connection())

    assert result == {
        "ok": False,
        "error": "未配置 Microsoft Azure Translator API Key",
    }


def test_llm_connection_reuses_protocol_aware_client(monkeypatch):
    calls = {}

    class Completions:
        def create(self, **kwargs):
            calls["request"] = kwargs
            return object()

    class Client:
        chat = type("Chat", (), {"completions": Completions()})()

    client = Client()
    monkeypatch.setattr(
        config_module,
        "get_llm_runtime_config",
        lambda: config_module.LlmRuntimeConfig(
            provider="minimax",
            base_url="https://api.minimaxi.com/anthropic",
            api_key="private-key",
            model="MiniMax-M3",
        ),
    )
    monkeypatch.setattr(
        "subforge.core.llm.create_client",
        lambda **kwargs: calls.setdefault("client_args", kwargs) and client,
    )
    monkeypatch.setattr(
        "subforge.core.llm.close_client",
        lambda value: calls.setdefault("closed", value),
    )

    result = asyncio.run(config_module.test_llm_connection())

    assert result == {"ok": True, "model": "MiniMax-M3"}
    assert calls["client_args"] == {
        "base_url": "https://api.minimaxi.com/anthropic",
        "api_key": "private-key",
        "timeout": 15.0,
    }
    assert calls["request"]["model"] == "MiniMax-M3"
    assert calls["closed"] is client


def test_azure_translator_connection_uses_private_persisted_key(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "_read_settings",
        lambda: {
            "azure_translator_key": "private-key",
            "azure_translator_region": "eastasia",
            "azure_translator_endpoint": "https://example.test",
        },
    )
    monkeypatch.setattr(
        "subforge.core.translate.bing_translator.BingTranslator.test_connection",
        lambda self: "你好",
    )

    result = asyncio.run(config_module.test_azure_translator_connection())

    assert result == {"ok": True, "translated": "你好"}
    assert "private-key" not in str(result)


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


def test_runtime_config_uses_active_profile_as_one_snapshot(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(config_module, "_SETTINGS_CANDIDATES", [settings_path])
    config_module._write_settings(
        {
            "llm_provider": "deepseek",
            "llm_base_url": "https://token-plan-cn.xiaomimimo.com/v1",
            "llm_api_key": "stale-flat-key",
            "llm_model": "mimo-v2.5-pro",
            "llm_profiles": {
                "deepseek": {
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key": "deepseek-key",
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

    runtime = config_module.get_llm_runtime_config()

    assert runtime.provider == "deepseek"
    assert runtime.base_url == "https://api.deepseek.com/v1"
    assert runtime.api_key == "deepseek-key"
    assert runtime.model == "deepseek-chat"


def test_update_rejects_model_from_inactive_provider_without_writing(
    tmp_path, monkeypatch
):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(config_module, "_SETTINGS_CANDIDATES", [settings_path])
    config_module._write_settings(
        {
            "llm_provider": "mimo",
            "llm_base_url": "https://token-plan-cn.xiaomimimo.com/v1",
            "llm_api_key": "mimo-key",
            "llm_model": "mimo-v2.5-pro",
        }
    )
    before = settings_path.read_text(encoding="utf-8")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            config_module.update_config(
                config_module.ConfigUpdate(key="llm_model", value="deepseek-chat")
            )
        )

    assert exc.value.status_code == 422
    assert settings_path.read_text(encoding="utf-8") == before


def test_update_rejects_endpoint_from_inactive_provider_without_writing(
    tmp_path, monkeypatch
):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(config_module, "_SETTINGS_CANDIDATES", [settings_path])
    config_module._write_settings(
        {
            "llm_provider": "deepseek",
            "llm_base_url": "https://api.deepseek.com/v1",
            "llm_api_key": "deepseek-key",
            "llm_model": "deepseek-chat",
        }
    )
    before = settings_path.read_text(encoding="utf-8")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            config_module.update_config(
                config_module.ConfigUpdate(
                    key="llm_base_url",
                    value="https://token-plan-cn.xiaomimimo.com/v1",
                )
            )
        )

    assert exc.value.status_code == 422
    assert settings_path.read_text(encoding="utf-8") == before


def test_switch_provider_repairs_corrupted_profile_and_preserves_key(
    tmp_path, monkeypatch
):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(config_module, "_SETTINGS_CANDIDATES", [settings_path])
    config_module._write_settings(
        {
            "llm_provider": "mimo",
            "llm_base_url": "https://token-plan-cn.xiaomimimo.com/v1",
            "llm_api_key": "mimo-key",
            "llm_model": "mimo-v2.5-pro",
            "llm_profiles": {
                "deepseek": {
                    "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
                    "api_key": "deepseek-key",
                    "model": "mimo-v2.5-pro",
                }
            },
        }
    )

    result = asyncio.run(
        config_module.switch_llm_provider(
            config_module.LlmProviderSwitch(
                provider="deepseek",
                current_base_url="https://token-plan-cn.xiaomimimo.com/v1",
                current_api_key="mimo-key",
                current_model="mimo-v2.5-pro",
            )
        )
    )

    assert result == {
        "status": "ok",
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "api_key_configured": True,
        "model": "",
    }


def test_model_listing_rejects_corrupted_runtime_before_network(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "get_llm_runtime_config",
        lambda: (_ for _ in ()).throw(ValueError("mixed LLM profile")),
    )

    result = asyncio.run(config_module.list_llm_models())

    assert result == {"error": "mixed LLM profile", "models": []}
