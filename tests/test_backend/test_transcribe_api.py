import asyncio
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

import app.api.config as config_module
import app.api.transcribe as transcribe_api

import subforge.config as subforge_config

transcribe_module = importlib.import_module("subforge.core.asr.transcribe")


def _config_values(tmp_path: Path) -> dict:
    return {
        "transcribe_model": "whisperx",
        "whisper_model_size": "large-v3",
        "whisperx_align_model": "WAV2VEC2_ASR_LARGE_LV60K_960H",
        "whisper_model_dir": str(tmp_path / "models"),
    }


def test_model_status_reports_detected_local_mlx_model(tmp_path, monkeypatch):
    values = _config_values(tmp_path)
    local_model = tmp_path / "whisper-large-v3-fp16"
    local_model.mkdir()
    values["whisper_model_size"] = str(local_model)

    monkeypatch.setattr(
        config_module,
        "get_config_value",
        lambda key, default=None: values.get(key, default),
    )
    monkeypatch.setattr(
        transcribe_api,
        "resolve_mlx_model",
        lambda model: str(local_model) if model in {str(local_model), "large-v3"} else model,
    )
    monkeypatch.setattr(transcribe_api, "is_valid_mlx_model_dir", lambda path: Path(path) == local_model)
    monkeypatch.setattr(transcribe_api.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(transcribe_api.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(transcribe_api.platform, "machine", lambda: "arm64")

    status = transcribe_api._current_model_status()

    assert status["engine"] == "whisperx"
    assert status["model_id"] == "mlx-large-v3"
    assert status["model_value"] == "large-v3"
    assert status["model_name"] == "MLX Whisper Large V3 FP16"
    assert status["model_path"] == str(local_model)
    assert status["model_ready"] is True
    assert status["testable"] is True


def test_model_list_uses_stable_mlx_ids_and_selection(tmp_path, monkeypatch):
    values = _config_values(tmp_path)
    local_model = tmp_path / "whisper-large-v3-fp16"
    local_model.mkdir()
    values["whisper_model_size"] = str(local_model)

    monkeypatch.setattr(
        config_module,
        "get_config_value",
        lambda key, default=None: values.get(key, default),
    )
    monkeypatch.setattr(
        transcribe_api,
        "resolve_mlx_model",
        lambda model: str(local_model)
        if model in {str(local_model), "large-v3"}
        else f"mlx-community/{model}",
    )
    monkeypatch.setattr(
        transcribe_api,
        "is_valid_mlx_model_dir",
        lambda path: Path(path) == local_model,
    )

    models = asyncio.run(transcribe_api.list_whisper_models())
    large_v3 = next(model for model in models if model["id"] == "mlx-large-v3")

    assert large_v3["value"] == "large-v3"
    assert large_v3["downloaded"] is True
    assert large_v3["selected"] is True
    assert large_v3["path"] == str(local_model)


def test_model_self_test_returns_real_transcript_metadata(tmp_path, monkeypatch):
    audio = tmp_path / "en.mp3"
    audio.write_bytes(b"test")
    status = {
        "engine": "whisperx",
        "engine_name": "WhisperX",
        "model_id": "mlx-large-v3",
        "model_value": "large-v3",
        "model_name": "MLX Whisper Large V3 FP16",
        "resolved_model": str(tmp_path / "model"),
        "model_path": str(tmp_path / "model"),
        "model_ready": True,
        "model_state": "ready",
        "model_message": "本地模型已验证",
        "alignment_model": "WAV2VEC2_ASR_LARGE_LV60K_960H",
        "alignment_path": "",
        "alignment_ready": False,
        "platform_supported": True,
        "runtime_ready": True,
        "testable": True,
    }

    class Segment:
        text = "model test passed"

    class Result:
        segments = [Segment()]

    monkeypatch.setattr(transcribe_api, "_current_model_status", lambda: status)
    monkeypatch.setattr(transcribe_api, "_build_transcribe_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(subforge_config, "ASSETS_PATH", tmp_path)
    monkeypatch.setattr(transcribe_module, "transcribe", lambda *_args, **_kwargs: Result())

    result = asyncio.run(transcribe_api.test_current_model())

    assert result["ok"] is True
    assert result["transcript"] == "model test passed"
    assert result["segment_count"] == 1
