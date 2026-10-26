from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from subforge.core.asr import whisperx_asr as module
from subforge.core.asr.model_cache import SingleEntryModelCache


@pytest.fixture
def alignment(tmp_path, monkeypatch):
    asr = module.WhisperXASR.__new__(module.WhisperXASR)
    asr.model_dir = str(tmp_path)
    asr.align_device = "cpu"
    monkeypatch.setattr(module, "_ALIGNMENT_MODEL_CACHE", SingleEntryModelCache())
    runtime = SimpleNamespace(
        load_align_model=Mock(return_value=(object(), {"language": "en"})),
        align=Mock(side_effect=lambda segments, *args, **kwargs: {"segments": segments}),
    )
    return asr, runtime


def run_alignment(asr, runtime, language="en"):
    return asr._align_result(
        {"segments": [{"start": 0.0, "end": 1.0, "text": "Hello world"}]},
        [], language, lambda *args: None, runtime,
    )


@pytest.mark.parametrize("custom", ["example/custom-aligner", "local"])
def test_custom_model_is_loaded_without_requiring_the_language_default(alignment, tmp_path, custom):
    asr, runtime = alignment
    if custom == "local":
        directory = tmp_path / "custom-aligner"
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        custom = str(directory)
    asr.align_model = custom
    result = run_alignment(asr, runtime)
    assert result["align_model"] == custom
    runtime.load_align_model.assert_called_once_with(
        language_code="en", device="cpu", model_name=custom, model_dir=str(tmp_path)
    )
    runtime.align.assert_called_once()


@pytest.mark.parametrize("selected", ["", "WAV2VEC2_ASR_LARGE_LV60K_960H"])
def test_missing_managed_default_still_fails_before_loading(alignment, selected):
    asr, runtime = alignment
    asr.align_model = selected
    with pytest.raises(RuntimeError, match="英语.*not available locally"):
        run_alignment(asr, runtime)
    runtime.load_align_model.assert_not_called()


def test_readiness_check_uses_the_explicit_managed_model(alignment):
    asr, runtime = alignment
    asr.align_model = "VOXPOPULI_ASR_BASE_10K_FR"
    with pytest.raises(RuntimeError, match="法语.*not available locally"):
        run_alignment(asr, runtime)
    runtime.load_align_model.assert_not_called()


def test_custom_loader_failure_is_not_masked_by_a_missing_default(alignment):
    asr, runtime = alignment
    asr.align_model = "example/custom-aligner"
    runtime.load_align_model.side_effect = ValueError("Custom model is invalid")
    with pytest.raises(ValueError, match="Custom model is invalid"):
        run_alignment(asr, runtime)
    runtime.load_align_model.assert_called_once()
