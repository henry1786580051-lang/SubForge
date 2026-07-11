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
    monkeypatch.setattr(config_module, "_IS_APPLE_SILICON", False)

    config = config_module._effective_config({"transcribe_model": "whisperx"})

    assert config["transcribe_model"] == "whisper_cpp"


def test_effective_config_keeps_whisperx_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(config_module, "_IS_APPLE_SILICON", True)

    config = config_module._effective_config({"transcribe_model": "whisperx"})

    assert config["transcribe_model"] == "whisperx"


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
    ],
)
def test_config_update_rejects_values_that_would_silently_fallback(key, value):
    with pytest.raises(HTTPException):
        config_module._validate_config_update(key, value)
