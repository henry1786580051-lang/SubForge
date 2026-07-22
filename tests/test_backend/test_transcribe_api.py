import asyncio
import importlib
import sys
from pathlib import Path

import pytest

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


def test_faster_whisper_models_use_distinct_download_ids(tmp_path, monkeypatch):
    values = _config_values(tmp_path)
    values["transcribe_model"] = "faster_whisper"
    values["whisper_model_size"] = "small"
    monkeypatch.setattr(
        config_module,
        "get_config_value",
        lambda key, default=None: values.get(key, default),
    )

    models = asyncio.run(transcribe_api.list_whisper_models())
    model = next(item for item in models if item["id"] == "faster-whisper-small")

    assert model["category"] == "faster_whisper"
    assert model["type"] == "ctranslate2"
    assert model["value"] == "small"
    assert model["selected"] is True
    assert model["downloaded"] is False


def test_hardware_detection_requires_complete_cuda_runtime(monkeypatch):
    monkeypatch.setattr(transcribe_api.platform, "system", lambda: "Windows")
    monkeypatch.setattr(transcribe_api.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(
        transcribe_api.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "NVIDIA GeForce RTX\n",
    )
    monkeypatch.setattr(transcribe_api, "is_faster_whisper_cuda_available", lambda: False)

    hardware = transcribe_api.detect_hardware()

    assert hardware["gpu"] == "NVIDIA GeForce RTX"
    assert hardware["device"] == "cpu"
    assert hardware["compute_type"] == "int8"


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


def test_alignment_model_list_includes_pinned_english_japanese_and_korean(tmp_path, monkeypatch):
    values = _config_values(tmp_path)
    values["source_language"] = "ko"
    monkeypatch.setattr(
        config_module,
        "get_config_value",
        lambda key, default=None: values.get(key, default),
    )

    models = asyncio.run(transcribe_api.list_whisper_models())
    alignment = {model["language"]: model for model in models if model["type"] == "alignment"}

    assert set(alignment) == {"en", "ja", "ko"}
    assert alignment["en"]["align_model"] == "WAV2VEC2_ASR_LARGE_LV60K_960H"
    assert alignment["ja"]["align_model"] == ("jonatasgrosman/wav2vec2-large-xlsr-53-japanese")
    assert alignment["ko"]["align_model"] == "kresnik/wav2vec2-large-xlsr-korean"
    assert alignment["ko"]["selected"] is True
    assert all(model["deletable"] for model in alignment.values())
    for info in transcribe_api.WHISPERX_MODELS.values():
        if info.get("repo"):
            assert len(info["revision"]) == 40
            assert "pytorch_model.bin" in info["allow_patterns"] or (
                "model.safetensors" in info["allow_patterns"]
            )


def test_huggingface_alignment_download_is_pinned_and_atomically_published(tmp_path, monkeypatch):
    import huggingface_hub

    model_id = "whisperx-align-ko-large"
    info = transcribe_api.WHISPERX_MODELS[model_id]
    destination = transcribe_api.managed_hf_alignment_dir(tmp_path, info["repo"])
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        staging = Path(kwargs["local_dir"])
        staging.mkdir(parents=True)
        for name in ("config.json", "preprocessor_config.json", "vocab.json"):
            (staging / name).write_text("{}", encoding="utf-8")
        (staging / "model.safetensors").write_bytes(b"weights")
        return str(staging)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    task = transcribe_api.task_manager.create_task("download_model")

    asyncio.run(transcribe_api._download_hf_alignment_model(task.id, model_id, destination))

    completed = transcribe_api.task_manager.get_task(task.id)
    assert completed is not None
    assert completed.status.value == "completed"
    assert destination.is_dir()
    assert transcribe_api.is_hf_alignment_model_dir(destination)
    assert calls["repo_id"] == info["repo"]
    assert calls["revision"] == info["revision"]
    assert calls["allow_patterns"] == info["allow_patterns"]


def test_delete_model_removes_only_the_requested_managed_target(tmp_path, monkeypatch):
    values = _config_values(tmp_path)
    models_dir = Path(values["whisper_model_dir"])
    monkeypatch.setattr(
        config_module,
        "get_config_value",
        lambda key, default=None: values.get(key, default),
    )

    targets = {
        "tiny": models_dir / "ggml-tiny.bin",
        "faster-whisper-small": models_dir / "faster-whisper-small",
        "whisperx-align-en-large": (models_dir / "wav2vec2_fairseq_large_lv60k_asr_ls960.pth"),
        "whisperx-align-ko-large": transcribe_api.managed_hf_alignment_dir(
            models_dir, "kresnik/wav2vec2-large-xlsr-korean"
        ),
        "whisperx-diarization-community-1": (
            models_dir / "pyannote-speaker-diarization-community-1"
        ),
    }
    untouched = models_dir / "keep-me.txt"
    untouched.parent.mkdir(parents=True)
    untouched.write_text("keep", encoding="utf-8")
    for target in targets.values():
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"model")
        else:
            target.mkdir(parents=True, exist_ok=True)
            (target / "model.bin").write_bytes(b"model")

    for model_id, target in targets.items():
        result = asyncio.run(
            transcribe_api.delete_whisper_model(
                transcribe_api.DownloadModelRequest(model_id=model_id)
            )
        )
        assert result["status"] == "deleted"
        assert not target.exists()
        assert untouched.read_text(encoding="utf-8") == "keep"


def test_delete_model_path_guard_rejects_model_root(tmp_path):
    with pytest.raises(ValueError, match="unmanaged model path"):
        transcribe_api._ensure_managed_model_path(tmp_path, tmp_path)


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
