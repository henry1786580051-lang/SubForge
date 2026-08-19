"""Strongly typed canonical settings model.

Persistence remains the responsibility of interface adapters. This module owns
defaults and value types so the web API, CLI, and legacy desktop UI do not
silently drift apart.
"""

from __future__ import annotations

import platform
from copy import deepcopy
from typing import Any, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")

TranslatorName = Literal["llm", "bing", "google", "deeplx"]
TranscribeEngine = Literal["whisperx", "whisper_cpp", "faster_whisper", "whisper_api"]

SECRET_SETTING_KEYS = frozenset(
    {
        "llm_api_key",
        "whisper_api_key",
        "huggingface_token",
        "azure_translator_key",
    }
)


class LlmProfile(BaseModel):
    """Credentials and model selection isolated per LLM provider."""

    model_config = ConfigDict(extra="ignore")

    base_url: str = ""
    api_key: str = ""
    model: str = ""


class AppSettings(BaseModel):
    """Canonical runtime settings independent of any persistence format."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    transcribe_model: TranscribeEngine = "whisper_cpp"
    source_language: str = "auto"
    target_language: str = "chinese"
    translator: TranslatorName = "bing"
    work_dir: str = ""

    font_name: str = "Noto Sans SC"
    font_size: int = Field(default=40, ge=8, le=200)
    font_color: str = "#ffffff"
    outline_color: str = "#000000"
    outline_width: float = Field(default=2.0, ge=0, le=20)
    bold: bool = True
    subtitle_style: str = "classic"
    show_bilingual: bool = True

    need_optimize: bool = True
    need_translate: bool = True
    need_reflect: bool = False
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_provider: str = "custom"
    llm_profiles: dict[str, LlmProfile] = Field(default_factory=dict)
    llm_log_level: Literal["summary", "standard", "debug"] = "summary"

    azure_translator_endpoint: str = "https://api.cognitive.microsofttranslator.com"
    azure_translator_key: str = ""
    azure_translator_region: str = ""

    max_word_count_cjk: int = Field(default=25, ge=1, le=200)
    max_word_count_english: int = Field(default=18, ge=1, le=200)
    thread_num: int = Field(default=5, ge=1, le=32)
    batch_size: int = Field(default=10, ge=1, le=100)
    custom_prompt: str = ""

    whisper_model_dir: str = ""
    whisper_cpp_path: str = ""
    whisper_base_url: str = ""
    whisper_api_key: str = ""
    whisper_api_model: str = "whisper-1"
    whisper_device: str = "auto"
    whisper_n_threads: int = Field(default=4, ge=0, le=128)
    whisper_compute_type: str = "default"
    whisperx_alignment_strategy: Literal["auto", "manual"] = "auto"
    whisperx_align_model: str = "WAV2VEC2_ASR_LARGE_LV60K_960H"
    whisperx_batch_size: int = Field(default=8, ge=1, le=64)
    detect_additional_languages: bool = False
    ff_mdx_kim2: bool = False
    enable_audio_enhancement: bool = True
    speaker_diarization: Literal["off", "two", "auto", "fixed"] = "off"
    speaker_count: int = Field(default=2, ge=2, le=10)
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    huggingface_token: str = ""
    replace_chinese_punctuation: bool = True
    whisper_model_size: str = "base"

    schema_version: int = 1


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() == "arm64"


def default_app_settings(*, apple_silicon: bool | None = None) -> AppSettings:
    """Return platform-aware defaults without mutating a shared instance."""
    is_apple = _is_apple_silicon() if apple_silicon is None else apple_silicon
    return AppSettings(
        transcribe_model="whisperx" if is_apple else "whisper_cpp",
        whisper_model_size="large-v3" if is_apple else "base",
    )


def default_settings_dict(*, apple_silicon: bool | None = None) -> dict[str, Any]:
    return default_app_settings(apple_silicon=apple_silicon).model_dump()


def coerce_setting_value(value: Any, default: T) -> T:
    """Match existing persistence semantics: invalid types fall back safely."""
    if isinstance(default, bool):
        return cast(T, value if isinstance(value, bool) else default)
    if isinstance(default, int):
        return cast(T, value if isinstance(value, int) and not isinstance(value, bool) else default)
    if isinstance(default, float):
        return cast(T, float(value) if isinstance(value, (int, float)) else default)
    if isinstance(default, str):
        return cast(T, value if isinstance(value, str) else default)
    if isinstance(default, dict):
        return cast(T, deepcopy(value) if isinstance(value, dict) else deepcopy(default))
    return cast(T, value if isinstance(value, type(default)) else deepcopy(default))


def coerce_flat_settings(
    stored: dict[str, Any],
    *,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Coerce a flat persisted mapping against canonical defaults."""
    resolved_defaults = defaults or default_settings_dict()
    return {
        key: coerce_setting_value(stored.get(key, default), default)
        for key, default in resolved_defaults.items()
    }
