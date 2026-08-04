"""Adapters from legacy interface-specific configuration formats."""

from __future__ import annotations

from typing import Any

from subforge.settings.schema import (
    AppSettings,
    TranscribeEngine,
    TranslatorName,
    default_app_settings,
)


def _get(mapping: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def app_settings_from_cli(config: dict[str, Any]) -> AppSettings:
    """Map the stable nested CLI TOML format into canonical runtime settings."""
    defaults = default_app_settings()
    engine_map: dict[str, TranscribeEngine] = {
        "whisper-api": "whisper_api",
        "whisper-cpp": "whisper_cpp",
        "faster-whisper": "faster_whisper",
        "whisperx": "whisperx",
    }
    raw_engine = str(_get(config, "transcribe.asr", "whisper-api"))
    engine = engine_map.get(raw_engine, defaults.transcribe_model)
    raw_translator = str(_get(config, "translate.service", defaults.translator))
    translator_map: dict[str, TranslatorName] = {
        "llm": "llm",
        "bing": "bing",
        "google": "google",
        "deeplx": "deeplx",
    }
    translator = translator_map.get(raw_translator, defaults.translator)
    if engine == "faster_whisper":
        model_size = str(_get(config, "transcribe.faster_whisper.model", "large-v3"))
    elif engine == "whisper_cpp":
        model_size = str(_get(config, "transcribe.whisper_cpp.model", "large-v2"))
    else:
        model_size = defaults.whisper_model_size

    return AppSettings(
        transcribe_model=engine,
        source_language=str(_get(config, "transcribe.language", "auto")),
        target_language=str(_get(config, "translate.target_language", "zh-Hans")),
        translator=translator,
        need_optimize=bool(_get(config, "subtitle.optimize", True)),
        need_translate=bool(_get(config, "subtitle.translate", False)),
        need_reflect=bool(_get(config, "translate.reflect", False)),
        llm_base_url=str(_get(config, "llm.api_base", "")),
        llm_api_key=str(_get(config, "llm.api_key", "")),
        llm_model=str(_get(config, "llm.model", "gpt-4o-mini")),
        azure_translator_endpoint=str(
            _get(
                config,
                "translate.azure_endpoint",
                defaults.azure_translator_endpoint,
            )
        ),
        azure_translator_key=str(_get(config, "translate.azure_key", "")),
        azure_translator_region=str(_get(config, "translate.azure_region", "")),
        max_word_count_cjk=int(_get(config, "subtitle.max_word_count_cjk", 25)),
        max_word_count_english=int(_get(config, "subtitle.max_word_count_english", 18)),
        thread_num=int(_get(config, "subtitle.thread_num", 4)),
        batch_size=int(_get(config, "subtitle.batch_size", 20)),
        whisper_base_url=str(_get(config, "whisper_api.api_base", "")),
        whisper_api_key=str(_get(config, "whisper_api.api_key", "")),
        whisper_api_model=str(_get(config, "whisper_api.model", "whisper-1")),
        whisper_device=str(_get(config, "transcribe.faster_whisper.device", "auto")),
        enable_audio_enhancement=bool(
            _get(config, "transcribe.audio_enhancement", True)
        ),
        ff_mdx_kim2=bool(_get(config, "transcribe.faster_whisper.voice_extraction", False)),
        whisper_model_size=model_size,
    )
