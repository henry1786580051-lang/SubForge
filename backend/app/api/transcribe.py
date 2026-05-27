import asyncio
import os
import platform
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.task_manager import task_manager

router = APIRouter()


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
                text=True, timeout=5,
            ).strip()
            result["chip"] = out

            # Parse core count from sysctl
            try:
                perf_cores = int(subprocess.check_output(
                    ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                    text=True, timeout=5,
                ).strip())
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
                text=True, timeout=5,
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
    "tiny": {"url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin", "size": "75MB"},
    "base": {"url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin", "size": "142MB"},
    "small": {"url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin", "size": "466MB"},
    "medium": {"url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin", "size": "1.5GB"},
    "large-v1": {"url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v1.bin", "size": "3.1GB"},
    "large-v2": {"url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin", "size": "3.1GB"},
    "large-v3": {"url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin", "size": "3.1GB"},
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
    file_path = Path(req.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=400, detail="File not found")

    task = task_manager.create_task("transcribe")
    # Run transcription in background
    asyncio.create_task(_run_transcription(task.id, req))
    return {"task_id": task.id, "status": "started"}


async def _run_transcription(task_id: str, req: TranscribeRequest):
    import tempfile
    from pathlib import Path as PathLib

    from subforge.core.asr.transcribe import transcribe
    from subforge.core.entities import TranscribeConfig, TranscribeModelEnum
    from subforge.core.utils.video_utils import video2audio

    temp_audio_path = None
    try:
        task_manager.update_progress(task_id, 5, "Initializing transcription...")

        from app.api.config import get_config_value

        # Map frontend engine IDs to enum values
        _model_map = {
            "whisper_cpp": "WhisperCpp",
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
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()

        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None, video2audio, req.file_path, temp_audio_path
        )
        if not success:
            raise RuntimeError("Failed to extract audio from video")

        task_manager.update_progress(task_id, 30, "Running ASR engine...")

        # Run transcription on the extracted audio
        result = await loop.run_in_executor(None, transcribe, temp_audio_path, config)

        # Save subtitle file
        if result:
            video_stem = PathLib(req.file_path).stem
            work_dir = PathLib(get_config_value("work_dir", ""))
            if not work_dir.exists():
                work_dir = PathLib(req.file_path).parent
            subtitle_path = work_dir / f"{video_stem}.srt"
            result.save(str(subtitle_path))
            task_manager.complete_task(task_id, {
                "subtitle_file": str(subtitle_path),
            })
        else:
            task_manager.complete_task(task_id, {"subtitle_file": None})
    except Exception as e:
        task_manager.fail_task(task_id, str(e))
    finally:
        if temp_audio_path:
            PathLib(temp_audio_path).unlink(missing_ok=True)


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
    """List available whisper.cpp models and their download status."""
    models_dir = _get_models_dir()

    result = []
    for model_id, info in WHISPER_CPP_MODELS.items():
        model_path = models_dir / f"ggml-{model_id}.bin"
        result.append({
            "id": model_id,
            "size": info["size"],
            "downloaded": model_path.exists(),
            "path": str(model_path),
        })
    return result


class DownloadModelRequest(BaseModel):
    model_id: str


@router.post("/download-model")
async def download_whisper_model(req: DownloadModelRequest):
    """Start downloading a whisper.cpp model."""
    if req.model_id not in WHISPER_CPP_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model_id}")

    models_dir = _get_models_dir()
    model_path = models_dir / f"ggml-{req.model_id}.bin"

    if model_path.exists():
        return {"status": "already_exists", "path": str(model_path)}

    task = task_manager.create_task("download_model")
    asyncio.create_task(_download_model(task.id, req.model_id, model_path))
    return {"task_id": task.id, "status": "started"}


async def _download_model(task_id: str, model_id: str, dest: Path):
    import httpx

    url = WHISPER_CPP_MODELS[model_id]["url"]
    try:
        task_manager.update_progress(task_id, 0, f"开始下载 {model_id} 模型...")

        async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0

                with open(dest, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 64):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = int(downloaded / total * 100)
                            task_manager.update_progress(task_id, pct, f"下载中... {pct}%")

        task_manager.complete_task(task_id, {"path": str(dest)})
    except Exception as e:
        # Clean up partial download
        if dest.exists():
            dest.unlink()
        task_manager.fail_task(task_id, str(e))
