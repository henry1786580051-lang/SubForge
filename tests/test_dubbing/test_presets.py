import pytest

from subforge.core.dubbing.presets import (
    available_dubbing_presets,
    get_dubbing_preset,
    normalize_dubbing_voice,
    validate_dubbing_voice,
)


def test_available_presets_include_main_providers():
    presets = available_dubbing_presets()

    assert "siliconflow-cn-female" in presets
    assert "gemini-en-friendly" in presets
    assert "edge-cn-female" in presets


def test_get_dubbing_preset():
    preset = get_dubbing_preset("siliconflow-cn-female")

    assert preset.provider == "siliconflow"
    assert preset.model == "FunAudioLLM/CosyVoice2-0.5B"
    assert preset.voice.endswith(":anna")


def test_get_dubbing_preset_unknown():
    with pytest.raises(ValueError, match="Unknown dubbing preset"):
        get_dubbing_preset("missing")


def test_normalize_siliconflow_short_voice_alias():
    voice = normalize_dubbing_voice("siliconflow", "FunAudioLLM/CosyVoice2-0.5B", "anna")

    assert voice == "FunAudioLLM/CosyVoice2-0.5B:anna"


def test_validate_gemini_unknown_voice():
    error = validate_dubbing_voice("gemini", "not-a-voice")

    assert error is not None
    assert "Unknown Gemini voice" in error


def test_normalize_edge_short_voice_alias():
    voice = normalize_dubbing_voice("edge", "edge-tts", "xiaoxiao")

    assert voice == "zh-CN-XiaoxiaoNeural"


def test_validate_edge_voice_alias_and_full_id():
    assert validate_dubbing_voice("edge", "xiaoxiao") is None
    assert validate_dubbing_voice("edge", "zh-CN-XiaoxiaoNeural") is None
    assert "Edge TTS voice" in (validate_dubbing_voice("edge", "badvoice") or "")
