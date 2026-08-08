"""Canonical application settings shared by desktop, API, and CLI adapters."""

from subforge.settings.adapters import app_settings_from_cli
from subforge.settings.llm import (
    LEGACY_MINIMAX_URLS,
    LLM_PROVIDER_URLS,
    LlmRuntimeConfig,
    detect_llm_provider,
    validate_llm_runtime_config,
)
from subforge.settings.schema import (
    SECRET_SETTING_KEYS,
    AppSettings,
    LlmProfile,
    coerce_flat_settings,
    coerce_setting_value,
    default_app_settings,
    default_settings_dict,
)

__all__ = [
    "AppSettings",
    "LlmProfile",
    "LlmRuntimeConfig",
    "LLM_PROVIDER_URLS",
    "LEGACY_MINIMAX_URLS",
    "SECRET_SETTING_KEYS",
    "coerce_flat_settings",
    "coerce_setting_value",
    "default_app_settings",
    "default_settings_dict",
    "detect_llm_provider",
    "app_settings_from_cli",
    "validate_llm_runtime_config",
]
