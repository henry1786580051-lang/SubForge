import asyncio
import logging
import platform
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.task_manager import task_manager
from app.security import validate_path
from subforge.core.asr.whisperx_asr import default_mlx_model

logger = logging.getLogger(__name__)

router = APIRouter()

_background_tasks: set[asyncio.Task] = set()


def _raise_if_cancelled(task_id: str) -> None:
    if task_manager.is_cancelled(task_id):
        raise asyncio.CancelledError()


def detect_hardware() -> dict:
    """Detect hardware and return optimal whisper settings."""
    result = {
        "platform": platform.system(),
        "arch": platform.machine(),
        "chip": "Unknown",
        "device": "cpu",
        "n_threads": 4,
        "compute_type": "default",
        "gpu": "None",
    }

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # Apple Silicon detection
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                timeout=5,
            ).strip()
            result["chip"] = out

            # Parse core count from sysctl
            try:
                perf_cores = int(
                    subprocess.check_output(
                        ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                        text=True,
                        timeout=5,
                    ).strip()
                )
            except Exception:
                perf_cores = 8

            result["device"] = "cpu"  # CTranslate2 uses ARM NEON
            result["n_threads"] = perf_cores  # Use performance cores
            result["compute_type"] = "int8"  # Best for CPU on Apple Silicon
            result["gpu"] = "Metal (via whisper.cpp)"
        except Exception:
            result["chip"] = "Apple Silicon"
            result["device"] = "cpu"
            result["n_threads"] = 8
            result["compute_type"] = "int8"
    else:
        # Check for NVIDIA GPU
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                timeout=5,
            ).strip()
            result["gpu"] = out.split("\n")[0]
            result["device"] = "cuda"
            result["n_threads"] = 4
            result["compute_type"] = "float16"
        except Exception:
            import os

            result["n_threads"] = os.cpu_count() or 4

    return result


@router.get("/hardware")
async def get_hardware_info():
    """Detect hardware and return optimal whisper settings."""
    return detect_hardware()


# Whisper.cpp model definitions
WHISPER_CPP_MODELS = {
    "tiny": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        "size": "75MB",
    },
    "base": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        "size": "142MB",
    },
    "small": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        "size": "466MB",
    },
    "medium": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        "size": "1.5GB",
    },
    "large-v1": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v1.bin",
        "size": "3.1GB",
    },
    "large-v2": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin",
        "size": "3.1GB",
    },
    "large-v3": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
        "size": "3.1GB",
    },
}

WHISPERX_MODELS = {
    "whisperx-align-en-large": {
        "name": "English Large LV60K Alignment",
        "category": "whisperx",
        "type": "alignment",
        "url": "https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_large_lv60k_asr_ls960.pth",
        "filename": "wav2vec2_fairseq_large_lv60k_asr_ls960.pth",
        "size": "1.18GB",
        "align_model": "WAV2VEC2_ASR_LARGE_LV60K_960H",
    },
}

_default_mlx_model = default_mlx_model()
_default_mlx_model_id = _default_mlx_model if Path(_default_mlx_model).is_dir() else "mlx-large-v3"
WHISPERX_LOCAL_MLX_MODELS = {
    _default_mlx_model_id: {
        "name": "MLX Large V3 FP16",
        "category": "whisperx",
        "type": "mlx",
        "size": "3.1GB",
        "model": _default_mlx_model,
    },
}


class TranscribeRequest(BaseModel):
    file_path: str
    model: str = "whisper_cpp"
    language: str = "auto"
    device: str = "auto"
    n_threads: int = 4
    compute_type: str = "default"


@router.post("/start")
async def start_transcription(req: TranscribeRequest):
    try:
        file_path = validate_path(req.file_path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists():
        raise HTTPException(status_code=400, detail="File not found")
    req = req.model_copy(update={"file_path": str(file_path)})

    task = task_manager.create_task("transcribe")
    # Run transcription in background
    task_obj = asyncio.create_task(_run_transcription(task.id, req))
    task_manager.register_running_task(task.id, task_obj)
    _background_tasks.add(task_obj)
    task_obj.add_done_callback(_background_tasks.discard)
    task_obj.add_done_callback(lambda _task: task_manager.unregister_running_task(task.id))
    return {"task_id": task.id, "status": "started"}


async def _run_transcription(task_id: str, req: TranscribeRequest):
    import tempfile
    from pathlib import Path as PathLib

    from subforge.core.asr.transcribe import transcribe
    from subforge.core.entities import TranscribeConfig, TranscribeModelEnum
    from subforge.core.utils.video_utils import video2audio

    temp_audio_path = None
    partial_srt_path = None
    try:
        task_manager.update_progress(task_id, 5, "Initializing transcription...")
        _raise_if_cancelled(task_id)

        from app.api.config import get_config_value

        # Map frontend engine IDs to enum values
        _model_map = {
            "whisper_cpp": "WhisperCpp",
            "whisperx": "WhisperX",
            "faster_whisper": "FasterWhisper ✨",
            "whisper_api": "Whisper [API] ✨",
        }
        model_enum = TranscribeModelEnum(_model_map.get(req.model, req.model))

        # Read performance settings from config
        device = get_config_value("whisper_device", "auto")
        n_threads = get_config_value("whisper_n_threads", 0)
        compute_type = get_config_value("whisper_compute_type", "default")

        # Auto-detect optimal settings
        hw = detect_hardware()
        if device == "auto":
            device = hw["device"]
        if n_threads == 0:
            n_threads = hw["n_threads"]
        if compute_type == "default":
            compute_type = hw["compute_type"]

        config = TranscribeConfig(
            transcribe_model=model_enum,
            transcribe_language=req.language,
            faster_whisper_device=device,
        )

        # Performance settings
        config.whisper_n_threads = n_threads
        config.faster_whisper_compute_type = compute_type
        config.whisper_cpp_path = get_config_value("whisper_cpp_path", "")
        config.enable_audio_enhancement = get_config_value("enable_audio_enhancement", True)
        config.whisperx_align_model = get_config_value(
            "whisperx_align_model",
            "WAV2VEC2_ASR_LARGE_LV60K_960H",
        )
        config.whisperx_batch_size = int(get_config_value("whisperx_batch_size", 8) or 8)

        # Vocal separation
        config.faster_whisper_ff_mdx_kim2 = get_config_value("ff_mdx_kim2", False)

        # Apply whisper-specific config
        whisper_model_dir = get_config_value("whisper_model_dir", "")
        if whisper_model_dir:
            config.faster_whisper_model_dir = whisper_model_dir

        # Apply model size selection
        whisper_model_size = get_config_value("whisper_model_size", "base")
        if req.model == "whisper_cpp":
            from subforge.core.entities import WhisperModelEnum

            try:
                config.whisper_model = WhisperModelEnum(whisper_model_size)
            except ValueError:
                config.whisper_model = WhisperModelEnum.BASE
        elif req.model == "whisperx":
            if whisper_model_size in {"", "large-v2", "mlx-large-v3"}:
                whisper_model_size = "large-v3"
            config.whisperx_model = whisper_model_size
            from subforge.core.entities import FasterWhisperModelEnum

            try:
                config.faster_whisper_model = FasterWhisperModelEnum(whisper_model_size)
            except ValueError:
                config.faster_whisper_model = None
        elif req.model == "faster_whisper":
            from subforge.core.entities import FasterWhisperModelEnum

            try:
                config.faster_whisper_model = FasterWhisperModelEnum(whisper_model_size)
            except ValueError:
                config.faster_whisper_model = FasterWhisperModelEnum.BASE

        # Pass API config for whisper_api
        if req.model == "whisper_api":
            config.whisper_api_key = get_config_value("whisper_api_key", "")
            config.whisper_api_base = get_config_value("whisper_base_url", "")
            config.whisper_api_model = get_config_value("whisper_api_model", "whisper-1")

        # Override MODEL_PATH if user configured a custom whisper model directory
        if whisper_model_dir:
            import subforge.config as vc_config
            import subforge.core.asr.whisper_cpp as wc_module

            vc_config.MODEL_PATH = PathLib(whisper_model_dir)
            wc_module.MODEL_PATH = PathLib(whisper_model_dir)

        # Extract audio from video to temp WAV file
        task_manager.update_progress(task_id, 10, "Extracting audio from video...")
        _raise_if_cancelled(task_id)
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()

        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, video2audio, req.file_path, temp_audio_path)
        if not success:
            raise RuntimeError("Failed to extract audio from video")
        _raise_if_cancelled(task_id)

        task_manager.update_progress(task_id, 30, "Running ASR engine...")

        # Partial SRT file for real-time preview
        partial_srt = tempfile.NamedTemporaryFile(suffix="_partial.srt", delete=False)
        partial_srt_path = partial_srt.name
        partial_srt.close()

        # Progress callback: map ASR progress (0-100) to overall (30-95%)
        def _on_progress(asr_progress: int, message: str):
            overall = 30 + int(asr_progress * 0.65)
            task_manager.update_progress(task_id, overall, message or "Transcribing...")

        # Save partial results as segments are transcribed
        def _on_segment(partial_data):
            try:
                partial_data.save(partial_srt_path)
                task = task_manager.get_task(task_id)
                if task:
                    task_manager.update_progress(
                        task_id, task.progress, subtitle_file=partial_srt_path
                    )
            except Exception as e:
                logger.warning(f"Failed to save partial segment: {e}")

        # Run transcription on the extracted audio
        result = await loop.run_in_executor(
            None, transcribe, temp_audio_path, config, _on_progress, _on_segment
        )
        _raise_if_cancelled(task_id)

        # Save subtitle file
        if result and len(result.segments) > 0:
            video_stem = PathLib(req.file_path).stem
            configured_work_dir = str(get_config_value("work_dir", "") or "").strip()
            if configured_work_dir:
                try:
                    work_dir = validate_path(configured_work_dir)
                except ValueError as exc:
                    raise RuntimeError("Configured output folder is outside allowed roots") from exc
                if not work_dir.is_dir():
                    raise RuntimeError("Configured output folder does not exist")
            else:
                work_dir = PathLib(req.file_path).parent
            subtitle_path = work_dir / f"{video_stem}.srt"
            _raise_if_cancelled(task_id)
            result.save(str(subtitle_path))
            task_manager.complete_task(
                task_id,
                {
                    "subtitle_file": str(subtitle_path),
                },
            )
        else:
            raise RuntimeError(
                "Transcription produced no subtitle segments. Check the selected "
                "ASR engine, whisper.cpp executable path, model file, and audio track."
            )
    except asyncio.CancelledError:
        logger.info("Transcription task %s cancelled", task_id)
        raise
    except Exception as e:
        logger.exception("Transcription task %s failed", task_id)
        task_manager.fail_task(task_id, str(e))
    finally:
        if temp_audio_path:
            PathLib(temp_audio_path).unlink(missing_ok=True)
        if partial_srt_path:
            PathLib(partial_srt_path).unlink(missing_ok=True)


def _get_models_dir() -> Path:
    """Get the whisper models directory, respecting user config."""
    from app.api.config import get_config_value

    custom_dir = get_config_value("whisper_model_dir", "")
    if custom_dir:
        models_dir = Path(custom_dir)
    else:
        from subforge.config import APPDATA_PATH

        models_dir = APPDATA_PATH / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


@router.get("/models")
async def list_whisper_models():
    """List available local ASR models and their download status."""
    models_dir = _get_models_dir()

    result = []
    for model_id, info in WHISPER_CPP_MODELS.items():
        model_path = models_dir / f"ggml-{model_id}.bin"
        result.append(
            {
                "id": model_id,
                "name": model_id,
                "category": "whisper_cpp",
                "type": "ggml",
                "size": info["size"],
                "downloaded": model_path.exists(),
                "downloadable": True,
                "path": str(model_path),
            }
        )
    for model_id, info in WHISPERX_LOCAL_MLX_MODELS.items():
        model_path = Path(info["model"])
        result.append(
            {
                "id": model_id,
                "name": info["name"],
                "category": info["category"],
                "type": info["type"],
                "size": info["size"],
                "downloaded": model_path.exists(),
                "downloadable": False,
                "path": str(model_path),
            }
        )
    for model_id, info in WHISPERX_MODELS.items():
        model_path = models_dir / info["filename"]
        result.append(
            {
                "id": model_id,
                "name": info["name"],
                "category": info["category"],
                "type": info["type"],
                "size": info["size"],
                "downloaded": model_path.exists(),
                "downloadable": True,
                "path": str(model_path),
                "align_model": info["align_model"],
            }
        )
    return result


class DownloadModelRequest(BaseModel):
    model_id: str


@router.post("/download-model")
async def download_whisper_model(req: DownloadModelRequest):
    """Start downloading a local ASR model."""
    if req.model_id in WHISPER_CPP_MODELS:
        models_dir = _get_models_dir()
        model_path = models_dir / f"ggml-{req.model_id}.bin"
    elif req.model_id in WHISPERX_MODELS:
        models_dir = _get_models_dir()
        model_path = models_dir / WHISPERX_MODELS[req.model_id]["filename"]
    elif req.model_id in WHISPERX_LOCAL_MLX_MODELS:
        model_path = Path(WHISPERX_LOCAL_MLX_MODELS[req.model_id]["model"])
        if model_path.exists():
            return {"status": "already_exists", "path": str(model_path)}
        raise HTTPException(
            status_code=400,
            detail="This MLX model is a local directory. Download it separately or set the model path.",
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model_id}")

    if model_path.exists():
        return {"status": "already_exists", "path": str(model_path)}

    task = task_manager.create_task("download_model")
    task_obj = asyncio.create_task(_download_model(task.id, req.model_id, model_path))
    task_manager.register_running_task(task.id, task_obj)
    _background_tasks.add(task_obj)
    task_obj.add_done_callback(_background_tasks.discard)
    task_obj.add_done_callback(lambda _task: task_manager.unregister_running_task(task.id))
    return {"task_id": task.id, "status": "started"}


async def _download_model(task_id: str, model_id: str, dest: Path):
    import httpx

    if model_id in WHISPER_CPP_MODELS:
        url = WHISPER_CPP_MODELS[model_id]["url"]
    else:
        url = WHISPERX_MODELS[model_id]["url"]
    tmp_dest = dest.with_name(f"{dest.name}.part")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        task_manager.update_progress(task_id, 0, f"开始下载 {model_id} 模型...")

        async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0

                with open(tmp_dest, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 64):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = int(downloaded / total * 100)
                            task_manager.update_progress(task_id, pct, f"下载中... {pct}%")

        tmp_dest.replace(dest)
        task_manager.complete_task(task_id, {"path": str(dest)})
    except asyncio.CancelledError:
        tmp_dest.unlink(missing_ok=True)
        raise
    except Exception as e:
        # Clean up partial download
        tmp_dest.unlink(missing_ok=True)
        task_manager.fail_task(task_id, str(e))
