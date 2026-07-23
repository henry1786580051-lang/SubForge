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


def test_whisper_cpp_download_urls_are_pinned_to_an_immutable_revision():
    for model in transcribe_api.WHISPER_CPP_MODELS.values():
        assert "/resolve/main/" not in model["url"]
        assert transcribe_api.WHISPER_CPP_REVISION in model["url"]


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
    monkeypatch.setattr(
        transcribe_api, "is_valid_mlx_model_dir", lambda path: Path(path) == local_model
    )
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

    monkeypatch.setattr(transcribe_api.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(transcribe_api.platform, "machine", lambda: "arm64")

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


def test_windows_lists_standard_whisperx_and_downloadable_tool_models(tmp_path, monkeypatch):
    values = _config_values(tmp_path)
    monkeypatch.setattr(
        config_module,
        "get_config_value",
        lambda key, default=None: values.get(key, default),
    )
    monkeypatch.setattr(transcribe_api.platform, "system", lambda: "Windows")
    monkeypatch.setattr(transcribe_api.platform, "machine", lambda: "AMD64")

    models = asyncio.run(transcribe_api.list_whisper_models())
    whisperx_model = next(model for model in models if model["id"] == "whisperx-large-v3")
    alignment = next(model for model in models if model["type"] == "alignment")
    diarization = next(model for model in models if model["type"] == "diarization")

    assert whisperx_model["type"] == "ctranslate2"
    assert whisperx_model["selected"] is True
    assert whisperx_model["state"] == "on_demand"
    assert alignment["downloadable"] is True
    assert diarization["downloadable"] is True


def test_model_list_exposes_language_specific_alignment_models(tmp_path, monkeypatch):
    values = _config_values(tmp_path)
    values.update(
        {
            "source_language": "ko",
            "whisperx_alignment_strategy": "auto",
        }
    )
    monkeypatch.setattr(
        config_module,
        "get_config_value",
        lambda key, default=None: values.get(key, default),
    )

    models = asyncio.run(transcribe_api.list_whisper_models())
    alignment_models = [model for model in models if model["type"] == "alignment"]
    korean = next(model for model in alignment_models if model["language"] == "ko")
    chinese = next(model for model in alignment_models if model["language"] == "zh")

    assert len(alignment_models) >= 40
    assert korean["align_model"] == "kresnik/wav2vec2-large-xlsr-korean"
    assert korean["selected"] is True
    assert korean["source"] == "huggingface"
    assert chinese["selected"] is False


def test_huggingface_alignment_cache_is_reported_ready(tmp_path):
    models_dir = tmp_path / "models"
    model_id = "whisperx-align-ko"
    cache_path = transcribe_api._alignment_model_path(model_id, models_dir)
    snapshot = cache_path / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")

    assert transcribe_api._alignment_model_ready(model_id, models_dir) is True


def test_huggingface_alignment_download_uses_whisperx_cache_layout(tmp_path, monkeypatch):
    import huggingface_hub

    models_dir = tmp_path / "models"
    model_id = "whisperx-align-ko"
    completed = {}

    def fake_snapshot_download(repo_id, cache_dir):
        assert repo_id == "kresnik/wav2vec2-large-xlsr-korean"
        cache_path = Path(cache_dir) / "models--kresnik--wav2vec2-large-xlsr-korean"
        snapshot = cache_path / "snapshots" / "revision"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "pytorch_model.bin").write_bytes(b"weights")
        return str(snapshot)

    monkeypatch.setattr(transcribe_api, "_get_models_dir", lambda: models_dir)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        transcribe_api.task_manager, "update_progress", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        transcribe_api.task_manager,
        "complete_task",
        lambda task_id, result: completed.update({"task_id": task_id, **result}),
    )

    asyncio.run(transcribe_api._download_huggingface_alignment_model("task-1", model_id))

    assert completed["task_id"] == "task-1"
    assert transcribe_api._alignment_model_ready(model_id, models_dir) is True


def test_build_transcribe_config_uses_auto_or_manual_alignment(tmp_path, monkeypatch):
    values = _config_values(tmp_path)
    values["whisperx_alignment_strategy"] = "auto"
    monkeypatch.setattr(
        config_module,
        "get_config_value",
        lambda key, default=None: values.get(key, default),
    )

    automatic = transcribe_api._build_transcribe_config("whisperx", "ko")
    values["whisperx_alignment_strategy"] = "manual"
    values["whisperx_align_model"] = "example/custom-alignment"
    manual = transcribe_api._build_transcribe_config("whisperx", "ko")

    assert automatic.whisperx_align_model == ""
    assert manual.whisperx_align_model == "example/custom-alignment"


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
    monkeypatch.setattr(
        transcribe_api, "_build_transcribe_config", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(subforge_config, "ASSETS_PATH", tmp_path)
    monkeypatch.setattr(transcribe_module, "transcribe", lambda *_args, **_kwargs: Result())

    result = asyncio.run(transcribe_api.test_current_model())

    assert result["ok"] is True
    assert result["transcript"] == "model test passed"
    assert result["segment_count"] == 1
