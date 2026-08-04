from subforge.settings import (
    SECRET_SETTING_KEYS,
    AppSettings,
    app_settings_from_cli,
    coerce_flat_settings,
    coerce_setting_value,
    default_app_settings,
    default_settings_dict,
)


def test_platform_defaults_are_explicit_and_isolated():
    apple = default_app_settings(apple_silicon=True)
    other = default_app_settings(apple_silicon=False)

    assert apple.transcribe_model == "whisperx"
    assert apple.whisper_model_size == "large-v3"
    assert other.transcribe_model == "whisper_cpp"
    assert other.whisper_model_size == "base"


def test_flat_settings_reject_corrupted_scalar_types():
    defaults = default_settings_dict(apple_silicon=False)
    settings = coerce_flat_settings(
        {
            "thread_num": "twenty",
            "enable_audio_enhancement": "false",
            "llm_profiles": [],
        },
        defaults=defaults,
    )

    assert settings["thread_num"] == 5
    assert settings["enable_audio_enhancement"] is True
    assert settings["llm_profiles"] == {}


def test_single_setting_coercion_matches_flat_settings_semantics():
    assert coerce_setting_value("twenty", 5) == 5
    assert coerce_setting_value(3, 2.5) == 3.0
    assert coerce_setting_value({"nested": True}, {}) == {"nested": True}


def test_secret_registry_covers_every_canonical_credential():
    fields = AppSettings.model_fields

    assert SECRET_SETTING_KEYS <= fields.keys()
    assert {
        "llm_api_key",
        "whisper_api_key",
        "huggingface_token",
        "azure_translator_key",
    } == SECRET_SETTING_KEYS


def test_cli_adapter_preserves_legacy_runtime_values():
    settings = app_settings_from_cli(
        {
            "llm": {"api_key": "secret", "api_base": "https://llm.test", "model": "m"},
            "transcribe": {
                "asr": "faster-whisper",
                "language": "en",
                "audio_enhancement": False,
                "faster_whisper": {"model": "small", "device": "cuda"},
            },
            "subtitle": {
                "optimize": False,
                "translate": True,
                "thread_num": 7,
                "batch_size": 9,
            },
            "translate": {
                "service": "bing",
                "target_language": "zh-Hans",
                "azure_key": "azure",
                "azure_region": "eastasia",
            },
        }
    )

    assert settings.transcribe_model == "faster_whisper"
    assert settings.whisper_model_size == "small"
    assert settings.whisper_device == "cuda"
    assert settings.enable_audio_enhancement is False
    assert settings.thread_num == 7
    assert settings.batch_size == 9
    assert settings.azure_translator_key == "azure"


def test_cli_adapter_falls_back_from_unknown_engine_and_translator():
    defaults = default_app_settings()

    settings = app_settings_from_cli(
        {
            "transcribe": {"asr": "removed-engine"},
            "translate": {"service": "removed-translator"},
        }
    )

    assert settings.transcribe_model == defaults.transcribe_model
    assert settings.translator == defaults.translator
