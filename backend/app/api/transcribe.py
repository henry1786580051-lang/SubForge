import asyncio
import importlib.util
import logging
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.blocking import run_blocking
from app.core.task_manager import TaskResourceBusyError, task_manager
from app.security import validate_path
from app.services.task_runtime import create_pipeline_context, schedule_background_task
from subforge.application import subtitle_preview_segments
from subforge.core.asr.alignment_models import (
    ALIGNMENT_MODEL_BY_ID,
    ALIGNMENT_MODELS,
    alignment_model_for_language,
    alignment_model_path,
    is_alignment_model_ready,
    normalize_alignment_language,
)
from subforge.core.asr.faster_whisper import (
    find_faster_whisper_model_dir,
    is_faster_whisper_model_dir,
)
from subforge.core.asr.speaker_embedding_models import (
    DEFAULT_SPEAKER_VERIFICATION_MODEL,
    LOCAL_SPEAKER_VERIFICATION_DIR,
    SPEAKER_VERIFICATION_MODEL_FILE,
    SPEAKER_VERIFICATION_MODEL_REVISION,
    is_speaker_verification_model_ready,
    speaker_verification_model_path,
)
from subforge.core.asr.whisperx_asr import (
    MLX_WHISPER_MODELS,
    is_valid_mlx_model_dir,
    resolve_mlx_model,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_background_tasks: set[asyncio.Task] = set()
_model_test_lock = asyncio.Lock()
_model_download_locks: dict[str, asyncio.Lock] = {}


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


# Whisper.cpp model definitions. Pin downloads to an immutable repository
# revision so an upstream branch update cannot silently replace model bytes.
WHISPER_CPP_REVISION = "5359861c739e955e79d9a303bcbc70fb988958b1"
WHISPER_CPP_MODELS = {
    "tiny": {
        "url": f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{WHISPER_CPP_REVISION}/ggml-tiny.bin",
        "size": "75MB",
    },
    "base": {
        "url": f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{WHISPER_CPP_REVISION}/ggml-base.bin",
        "size": "142MB",
    },
    "small": {
        "url": f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{WHISPER_CPP_REVISION}/ggml-small.bin",
        "size": "466MB",
    },
    "medium": {
        "url": f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{WHISPER_CPP_REVISION}/ggml-medium.bin",
        "size": "1.5GB",
    },
    "large-v1": {
        "url": f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{WHISPER_CPP_REVISION}/ggml-large-v1.bin",
        "size": "3.1GB",
    },
    "large-v2": {
        "url": f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{WHISPER_CPP_REVISION}/ggml-large-v2.bin",
        "size": "3.1GB",
    },
    "large-v3": {
        "url": f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{WHISPER_CPP_REVISION}/ggml-large-v3.bin",
        "size": "3.1GB",
    },
}

FASTER_WHISPER_MODELS = {
    f"faster-whisper-{model}": {
        "value": model,
        "size": size,
    }
    for model, size in {
        "tiny": "75MB",
        "base": "148MB",
        "small": "496MB",
        "medium": "1.5GB",
        "large-v1": "3.1GB",
        "large-v2": "3.1GB",
        "large-v3": "3.1GB",
        "large-v3-turbo": "1.7GB",
    }.items()
}

WHISPERX_MODELS = {
    model.id: {
        "name": f"{model.language_name} Alignment",
        "category": "whisperx",
        "type": "alignment",
        "url": model.url,
        "filename": model.filename,
        "size": model.size,
        "align_model": model.model_name,
        "language": model.language,
        "language_name": model.language_name,
        "source": model.source,
    }
    for model in ALIGNMENT_MODELS
}

DIARIZATION_MODELS = {
    "whisperx-diarization-community-1": {
        "name": "Pyannote Community-1",
        "category": "whisperx",
        "type": "diarization",
        "repo": "pyannote/speaker-diarization-community-1",
        "dirname": "pyannote-speaker-diarization-community-1",
        "size": "约 34MB",
    },
}

SPEAKER_VERIFICATION_MODELS = {
    "whisperx-speaker-verification-ecapa512-lm": {
        "name": "WeSpeaker ECAPA512-LM",
        "category": "whisperx",
        "type": "speaker_verification",
        "repo": DEFAULT_SPEAKER_VERIFICATION_MODEL,
        "revision": SPEAKER_VERIFICATION_MODEL_REVISION,
        "dirname": LOCAL_SPEAKER_VERIFICATION_DIR,
        "filename": SPEAKER_VERIFICATION_MODEL_FILE,
        "size": "约 25MB",
    },
}

MLX_MODEL_SIZES = {
    "tiny": "75MB",
    "base": "148MB",
    "small": "496MB",
    "medium": "1.5GB",
    "large": "3.1GB",
    "large-v1": "3.1GB",
    "large-v2": "3.1GB",
    "large-v3": "3.1GB",
    "large-v3-turbo": "1.7GB",
}


class TranscribeRequest(BaseModel):
    file_path: str = Field(max_length=4096)
    model: Literal["whisper_cpp", "whisperx", "faster_whisper", "whisper_api"] = "whisper_cpp"
    language: str = Field(default="auto", min_length=1, max_length=32)
    device: str = Field(default="auto", max_length=32)
    n_threads: int = Field(default=4, ge=1, le=128)
    compute_type: str = Field(default="default", max_length=32)


class AlignmentDecisionRequest(BaseModel):
    action: Literal["retry", "continue", "ignore"]


def _build_transcribe_config(
    model_id: str,
    language: str = "auto",
    *,
    enable_audio_enhancement: bool | None = None,
    need_word_time_stamp: bool = True,
):
    """Build the effective ASR config used by jobs and model self-tests."""
    from app.api.config import get_config_value
    from subforge.core.entities import (
        FasterWhisperModelEnum,
        TranscribeConfig,
        TranscribeModelEnum,
        WhisperModelEnum,
    )

    model_map = {
        "whisper_cpp": "WhisperCpp",
        "whisperx": "WhisperX",
        "faster_whisper": "FasterWhisper ✨",
        "whisper_api": "Whisper [API] ✨",
    }
    model_enum = TranscribeModelEnum(model_map.get(model_id, model_id))
    hardware = detect_hardware()
    device = get_config_value("whisper_device", "auto")
    n_threads = int(get_config_value("whisper_n_threads", 0) or 0)
    compute_type = get_config_value("whisper_compute_type", "default")
    if device == "auto":
        device = hardware["device"]
    if n_threads == 0:
        n_threads = hardware["n_threads"]
    if compute_type == "default":
        compute_type = hardware["compute_type"]

    config = TranscribeConfig(
        transcribe_model=model_enum,
        transcribe_language=language,
        faster_whisper_device=device,
        need_word_time_stamp=(need_word_time_stamp if model_id != "faster_whisper" else False),
    )
    config.whisper_n_threads = n_threads
    config.faster_whisper_compute_type = compute_type
    config.whisper_cpp_path = get_config_value("whisper_cpp_path", "")
    configured_enhancement = bool(get_config_value("enable_audio_enhancement", True))
    config.enable_audio_enhancement = (
        configured_enhancement if enable_audio_enhancement is None else enable_audio_enhancement
    )
    alignment_strategy = get_config_value("whisperx_alignment_strategy", "auto")
    automatic_source_language = normalize_alignment_language(language) in {"", "auto"}
    config.whisperx_align_model = (
        get_config_value(
            "whisperx_align_model",
            "WAV2VEC2_ASR_LARGE_LV60K_960H",
        )
        if alignment_strategy == "manual" and not automatic_source_language
        else ""
    )
    config.whisperx_batch_size = int(get_config_value("whisperx_batch_size", 8) or 8)
    config.speaker_diarization = get_config_value("speaker_diarization", "off")
    config.speaker_count = int(get_config_value("speaker_count", 2) or 2)
    config.diarization_model = get_config_value(
        "diarization_model", "pyannote/speaker-diarization-community-1"
    )
    config.diarization_token = get_config_value("huggingface_token", "")
    config.diarization_model_dir = str(_get_models_dir())
    config.faster_whisper_ff_mdx_kim2 = bool(get_config_value("ff_mdx_kim2", False))

    model_dir = str(get_config_value("whisper_model_dir", "") or "").strip()
    config.faster_whisper_model_dir = (
        str(Path(model_dir).expanduser()) if model_dir else str(_get_models_dir())
    )

    model_size = str(get_config_value("whisper_model_size", "large-v3") or "large-v3")
    if model_id == "whisper_cpp":
        try:
            config.whisper_model = WhisperModelEnum(model_size)
        except ValueError:
            config.whisper_model = WhisperModelEnum.BASE
    elif model_id == "whisperx":
        if model_size in {"", "mlx-large-v3"}:
            model_size = "large-v3"
        config.whisperx_model = model_size
        try:
            config.faster_whisper_model = FasterWhisperModelEnum(model_size)
        except ValueError:
            config.faster_whisper_model = None
    elif model_id == "faster_whisper":
        try:
            config.faster_whisper_model = FasterWhisperModelEnum(model_size)
        except ValueError:
            config.faster_whisper_model = FasterWhisperModelEnum.BASE
    elif model_id == "whisper_api":
        config.whisper_api_key = get_config_value("whisper_api_key", "")
        config.whisper_api_base = get_config_value("whisper_base_url", "")
        config.whisper_api_model = get_config_value("whisper_api_model", "whisper-1")

    return config


@router.post("/start")
async def start_transcription(req: TranscribeRequest):
    try:
        file_path = validate_path(req.file_path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists():
        raise HTTPException(status_code=400, detail="File not found")
    req = req.model_copy(update={"file_path": str(file_path)})

    try:
        task_id = schedule_background_task(
            task_type="transcribe",
            resource_key=f"transcribe:{file_path.resolve()}",
            runner=lambda current_task_id: _run_transcription(current_task_id, req),
            background_tasks=_background_tasks,
        )
    except TaskResourceBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"task_id": task_id, "status": "started"}


@router.post("/{task_id}/alignment-decision")
async def resolve_alignment_decision(task_id: str, req: AlignmentDecisionRequest):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.type != "transcribe":
        raise HTTPException(status_code=409, detail="Task is not a transcription task")
    attention = task.attention or {}
    if attention.get("type") != "missing_alignment_models":
        raise HTTPException(status_code=409, detail="Task is not waiting for alignment models")
    if not task_manager.resolve_attention(task_id, req.action):
        raise HTTPException(status_code=409, detail="Alignment decision is no longer active")
    return {"status": "resuming", "action": req.action}


async def _run_transcription(task_id: str, req: TranscribeRequest):
    import tempfile
    import threading
    from pathlib import Path as PathLib

    from app.api.config import get_config_value
    from subforge.core.asr.transcribe import transcribe
    from subforge.core.utils.video_utils import video2audio

    temp_audio_path = None
    partial_srt_path = None
    context = create_pipeline_context(task_id)
    try:
        context.report(5, "Initializing transcription...")
        context.checkpoint()

        config = _build_transcribe_config(req.model, req.language)
        cancel_event = threading.Event()
        config.cancel_event = cancel_event

        if req.model == "whisperx" and normalize_alignment_language(req.language) in {"", "auto"}:

            def _wait_for_alignment_models(models: list[dict]) -> str:
                attention = {
                    "type": "missing_alignment_models",
                    "source_mode": "auto",
                    "message": "检测到尚未安装对齐模型的语言，等待选择处理方式",
                    "models": models,
                }
                if not context.request_attention(attention):
                    raise RuntimeError(
                        "Transcription task ended while waiting for alignment models"
                    )
                while not cancel_event.is_set() and not context.is_cancelled():
                    resolution = context.wait_for_attention(timeout=0.5)
                    if resolution is not None:
                        return resolution
                raise RuntimeError("Transcription cancelled while waiting for alignment models")

            config.missing_alignment_model_callback = _wait_for_alignment_models

        # Extract audio from video to temp WAV file
        context.report(10, "Extracting audio from video...")
        context.checkpoint()
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()

        success = await run_blocking(
            video2audio,
            req.file_path,
            temp_audio_path,
            0,
            cancel_event,
            on_cancel=cancel_event.set,
        )
        if not success:
            raise RuntimeError("Failed to extract audio from video")
        context.checkpoint()

        context.report(30, "Running ASR engine...")

        # Partial SRT file for real-time preview
        partial_srt = tempfile.NamedTemporaryFile(suffix="_partial.srt", delete=False)
        partial_srt_path = partial_srt.name
        partial_srt.close()
        preview_lock = threading.RLock()
        last_preview_update = [0.0]
        last_preview_snapshot = [0.0]

        # Progress callback: map ASR progress (0-100) to overall (30-95%)
        def _on_progress(asr_progress: int, message: str):
            overall = 30 + int(asr_progress * 0.65)
            context.report(overall, message or "Transcribing...")

        # Save partial results as segments are transcribed
        def _on_segment(partial_data):
            try:
                with preview_lock:
                    now = time.monotonic()
                    if now - last_preview_update[0] < 0.25:
                        return
                    last_preview_update[0] = now
                    snapshot_path = None
                    if now - last_preview_snapshot[0] >= 5.0:
                        partial_data.save(partial_srt_path)
                        last_preview_snapshot[0] = now
                        snapshot_path = partial_srt_path
                    context.publish_preview(
                        subtitle_preview_segments(partial_data),
                        subtitle_file=snapshot_path,
                    )
            except Exception as e:
                logger.warning(f"Failed to save partial segment: {e}")

        # Run transcription on the extracted audio
        result = await run_blocking(
            transcribe,
            temp_audio_path,
            config,
            _on_progress,
            _on_segment,
            on_cancel=cancel_event.set,
        )
        context.checkpoint()

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
                source = PathLib(req.file_path).resolve()
                from app.api.files import UPLOAD_ROOT

                if source.is_relative_to(UPLOAD_ROOT.resolve()):
                    from subforge.config import WORK_PATH

                    work_dir = WORK_PATH.resolve()
                else:
                    work_dir = source.parent
            work_dir.mkdir(parents=True, exist_ok=True)
            subtitle_path = work_dir / f"{video_stem}.srt"
            context.checkpoint()
            result.save(str(subtitle_path))
            task_manager.complete_task(
                task_id,
                {
                    "subtitle_file": str(subtitle_path),
                    "segments": subtitle_preview_segments(result),
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


def _alignment_model_path(model_id: str, models_dir: Path) -> Path:
    spec = ALIGNMENT_MODEL_BY_ID[model_id]
    return alignment_model_path(spec, models_dir)


def _alignment_model_ready(model_id: str, models_dir: Path) -> bool:
    spec = ALIGNMENT_MODEL_BY_ID[model_id]
    return is_alignment_model_ready(spec, models_dir)


def _current_model_status() -> dict:
    """Describe the effective ASR engine and model selected in settings."""
    from app.api.config import get_config_value

    default_engine = (
        "whisperx"
        if platform.system() == "Darwin" and platform.machine() == "arm64"
        else "whisper_cpp"
    )
    default_model = "large-v3" if default_engine == "whisperx" else "base"
    engine = str(get_config_value("transcribe_model", default_engine) or default_engine)
    model_value = str(get_config_value("whisper_model_size", default_model) or default_model)
    models_dir = _get_models_dir()
    engine_names = {
        "whisperx": "WhisperX",
        "whisper_cpp": "Whisper.cpp",
        "faster_whisper": "FasterWhisper",
        "whisper_api": "Whisper API",
    }
    status = {
        "engine": engine,
        "engine_name": engine_names.get(engine, engine),
        "model_value": model_value,
        "model_id": model_value,
        "model_name": model_value,
        "resolved_model": model_value,
        "model_path": "",
        "model_ready": False,
        "model_state": "missing",
        "model_message": "模型尚未配置",
        "alignment_model": "",
        "alignment_model_id": "",
        "alignment_strategy": "auto",
        "alignment_language": "",
        "alignment_language_name": "",
        "alignment_path": "",
        "alignment_ready": False,
        "alignment_supported": True,
        "platform_supported": True,
        "runtime_ready": True,
        "testable": True,
    }

    if engine == "whisperx":
        if model_value == "mlx-large-v3":
            model_value = "large-v3"
        uses_mlx = platform.system() == "Darwin" and platform.machine().lower() == "arm64"
        if uses_mlx:
            resolved = resolve_mlx_model(model_value)
            resolved_path = Path(resolved).expanduser()
            local_ready = is_valid_mlx_model_dir(resolved_path)
            if local_ready:
                for candidate in MLX_WHISPER_MODELS:
                    candidate_resolved = Path(resolve_mlx_model(candidate)).expanduser()
                    if candidate_resolved == resolved_path:
                        model_value = candidate
                        break
            runtime_ready = (
                importlib.util.find_spec("mlx_whisper") is not None
                and importlib.util.find_spec("whisperx") is not None
            )
        else:
            local_model = find_faster_whisper_model_dir(model_value, models_dir)
            resolved = str(local_model) if local_model is not None else model_value
            resolved_path = local_model or Path()
            local_ready = local_model is not None
            runtime_ready = (
                importlib.util.find_spec("whisperx") is not None
                and importlib.util.find_spec("faster_whisper") is not None
            )
        platform_supported = uses_mlx or platform.system() in {"Windows", "Linux"}
        alignment_strategy = str(get_config_value("whisperx_alignment_strategy", "auto") or "auto")
        source_language = normalize_alignment_language(
            str(get_config_value("source_language", "auto") or "auto")
        )
        configured_align_model = str(
            get_config_value("whisperx_align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H") or ""
        )
        if alignment_strategy == "manual":
            align_spec = next(
                (spec for spec in ALIGNMENT_MODELS if spec.model_name == configured_align_model),
                None,
            )
        elif source_language and source_language != "auto":
            align_spec = alignment_model_for_language(source_language)
        else:
            align_spec = None
        align_path = _alignment_model_path(align_spec.id, models_dir) if align_spec else Path()
        status.update(
            {
                "model_value": model_value,
                "model_id": f"mlx-{model_value}" if uses_mlx else f"whisperx-{model_value}",
                "model_name": (
                    "MLX Whisper Large V3 FP16"
                    if uses_mlx and local_ready and model_value == "large-v3"
                    else f"MLX Whisper {model_value}"
                    if uses_mlx
                    else f"WhisperX {model_value} (Faster-Whisper)"
                ),
                "resolved_model": resolved,
                "model_path": str(resolved_path) if local_ready else "",
                "model_ready": local_ready,
                "model_state": "ready" if local_ready else "on_demand",
                "model_message": (
                    "本地模型已验证，转录时直接使用"
                    if local_ready
                    else "首次测试或转录时将自动下载"
                ),
                "alignment_model": (
                    align_spec.model_name
                    if align_spec
                    else configured_align_model
                    if alignment_strategy == "manual"
                    else "按语言自动选择"
                ),
                "alignment_model_id": align_spec.id if align_spec else "",
                "alignment_strategy": alignment_strategy,
                "alignment_language": align_spec.language if align_spec else source_language,
                "alignment_language_name": align_spec.language_name if align_spec else "",
                "alignment_path": str(align_path) if align_spec else "",
                "alignment_ready": bool(
                    align_spec and _alignment_model_ready(align_spec.id, models_dir)
                ),
                "alignment_supported": bool(
                    align_spec or alignment_strategy == "manual" or source_language == "auto"
                ),
                "platform_supported": platform_supported,
                "runtime_ready": runtime_ready,
                "testable": platform_supported and runtime_ready,
            }
        )
    elif engine == "whisper_cpp":
        from subforge.core.asr.whisper_cpp import detect_whisper_executable

        model_path = models_dir / f"ggml-{model_value}.bin"
        try:
            executable = detect_whisper_executable()
        except Exception:
            executable = ""
        ready = model_path.is_file() and bool(executable)
        status.update(
            {
                "model_id": model_value,
                "model_name": f"Whisper.cpp {model_value}",
                "resolved_model": str(model_path),
                "model_path": str(model_path),
                "model_ready": model_path.is_file(),
                "model_state": "ready" if ready else "missing",
                "model_message": (
                    "模型与 whisper-cli 均已就绪"
                    if ready
                    else "需要模型文件和 whisper-cli 可执行程序"
                ),
                "runtime_ready": bool(executable),
                "testable": ready,
            }
        )
    elif engine == "faster_whisper":
        model_path = models_dir / f"faster-whisper-{model_value}"
        model_ready = is_faster_whisper_model_dir(model_path)
        runtime_ready = importlib.util.find_spec("faster_whisper") is not None
        status.update(
            {
                "model_id": f"faster-whisper-{model_value}",
                "model_name": f"FasterWhisper {model_value}",
                "resolved_model": str(model_path),
                "model_path": str(model_path),
                "model_ready": model_ready,
                "model_state": "ready" if model_ready and runtime_ready else "missing",
                "model_message": (
                    "CTranslate2 模型与内置运行时已就绪"
                    if model_ready and runtime_ready
                    else "需要下载 CTranslate2 模型"
                    if runtime_ready
                    else "安装包缺少 FasterWhisper 运行时"
                ),
                "runtime_ready": runtime_ready,
                "testable": model_ready and runtime_ready,
            }
        )
    elif engine == "whisper_api":
        base_url = str(get_config_value("whisper_base_url", "") or "").strip()
        api_model = str(get_config_value("whisper_api_model", "whisper-1") or "whisper-1")
        status.update(
            {
                "model_value": api_model,
                "model_id": api_model,
                "model_name": api_model,
                "resolved_model": base_url,
                "model_ready": bool(base_url),
                "model_state": "ready" if base_url else "missing",
                "model_message": "API 配置已填写" if base_url else "尚未配置 API Base URL",
                "testable": bool(base_url),
            }
        )
    else:
        status["testable"] = False

    return status


@router.get("/model-status")
async def get_model_status():
    """Return the exact engine and model that the next transcription will use."""
    return _current_model_status()


@router.post("/test-model")
async def test_current_model():
    """Run a real short transcription with the currently selected ASR model."""
    if _model_test_lock.locked():
        raise HTTPException(status_code=409, detail="模型测试正在运行")

    status = _current_model_status()
    if not status["testable"]:
        return {**status, "ok": False, "error": status["model_message"]}

    from subforge.config import ASSETS_PATH
    from subforge.core.asr.transcribe import transcribe

    sample_audio = ASSETS_PATH / "en.mp3"
    if not sample_audio.is_file():
        return {**status, "ok": False, "error": "内置模型测试音频缺失"}

    config = _build_transcribe_config(
        status["engine"],
        "en",
        enable_audio_enhancement=False,
        need_word_time_stamp=status["engine"] == "whisperx",
    )
    started = time.monotonic()
    try:
        async with _model_test_lock:
            result = await asyncio.wait_for(
                asyncio.to_thread(transcribe, str(sample_audio), config),
                timeout=300,
            )
        transcript = " ".join(segment.text.strip() for segment in result.segments).strip()
        if not transcript:
            raise RuntimeError("模型未返回可识别文本")
        return {
            **_current_model_status(),
            "ok": True,
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "transcript": transcript[:160],
            "segment_count": len(result.segments),
        }
    except TimeoutError:
        return {**status, "ok": False, "error": "模型测试超过 5 分钟，已停止等待"}
    except Exception as exc:
        logger.exception("ASR model self-test failed")
        return {**_current_model_status(), "ok": False, "error": str(exc)[:300]}


@router.get("/models")
async def list_whisper_models():
    """List available local ASR models and their download status."""
    models_dir = _get_models_dir()

    from app.api.config import get_config_value

    default_engine = (
        "whisperx"
        if platform.system() == "Darwin" and platform.machine() == "arm64"
        else "whisper_cpp"
    )
    selected_engine = str(get_config_value("transcribe_model", default_engine) or default_engine)
    selected_model = str(get_config_value("whisper_model_size", "large-v3") or "large-v3")
    if selected_model == "mlx-large-v3":
        selected_model = "large-v3"
    uses_mlx = platform.system() == "Darwin" and platform.machine().lower() == "arm64"
    selected_resolved = (
        resolve_mlx_model(selected_model)
        if selected_engine == "whisperx" and uses_mlx
        else selected_model
    )
    selected_align = str(get_config_value("whisperx_align_model", "") or "")
    alignment_strategy = str(get_config_value("whisperx_alignment_strategy", "auto") or "auto")
    selected_language = normalize_alignment_language(
        str(get_config_value("source_language", "auto") or "auto")
    )
    selected_diarization = str(get_config_value("speaker_diarization", "off") or "off")
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
                "value": model_id,
                "selected": selected_engine == "whisper_cpp" and selected_model == model_id,
                "state": "ready" if model_path.exists() else "missing",
            }
        )
    for model_value, repo in MLX_WHISPER_MODELS.items():
        if uses_mlx:
            resolved = resolve_mlx_model(model_value)
            model_path = Path(resolved).expanduser()
            local_ready = is_valid_mlx_model_dir(model_path)
        else:
            local_model = find_faster_whisper_model_dir(model_value, models_dir)
            resolved = str(local_model) if local_model is not None else model_value
            model_path = local_model or Path()
            local_ready = local_model is not None
        result.append(
            {
                "id": f"mlx-{model_value}" if uses_mlx else f"whisperx-{model_value}",
                "value": model_value,
                "name": (
                    "MLX Large V3 FP16"
                    if uses_mlx and local_ready and model_value == "large-v3"
                    else f"MLX {model_value}"
                    if uses_mlx
                    else f"WhisperX {model_value}"
                ),
                "category": "whisperx",
                "type": "mlx" if uses_mlx else "ctranslate2",
                "size": MLX_MODEL_SIZES.get(model_value, ""),
                "downloaded": local_ready,
                "downloadable": False,
                "path": str(model_path) if local_ready else "",
                "resolved_model": resolved if local_ready else repo if uses_mlx else model_value,
                "selected": selected_engine == "whisperx"
                and (selected_model == model_value or selected_resolved == resolved),
                "state": "ready" if local_ready else "on_demand",
                "detail": ("本地模型已验证" if local_ready else "首次测试或转录时自动下载"),
            }
        )
    for model_id, info in FASTER_WHISPER_MODELS.items():
        model_value = info["value"]
        model_path = models_dir / model_id
        ready = is_faster_whisper_model_dir(model_path)
        result.append(
            {
                "id": model_id,
                "name": f"FasterWhisper {model_value}",
                "category": "faster_whisper",
                "type": "ctranslate2",
                "size": info["size"],
                "downloaded": ready,
                "downloadable": True,
                "path": str(model_path),
                "value": model_value,
                "selected": selected_engine == "faster_whisper" and selected_model == model_value,
                "state": "ready" if ready else "missing",
                "detail": "本地 CTranslate2 模型" if ready else "需要下载",
            }
        )
    for model_id, info in WHISPERX_MODELS.items():
        model_path = _alignment_model_path(model_id, models_dir)
        ready = _alignment_model_ready(model_id, models_dir)
        selected = selected_engine == "whisperx" and (
            (alignment_strategy == "manual" and selected_align == info["align_model"])
            or (
                alignment_strategy == "auto"
                and selected_language != "auto"
                and selected_language == info["language"]
            )
        )
        result.append(
            {
                "id": model_id,
                "name": info["name"],
                "category": info["category"],
                "type": info["type"],
                "size": info["size"],
                "downloaded": ready,
                "downloadable": True,
                "path": str(model_path),
                "align_model": info["align_model"],
                "value": info["align_model"],
                "selected": selected,
                "state": "ready" if ready else "missing",
                "detail": (
                    f"{info['language_name']} · "
                    f"{'TorchAudio' if info['source'] == 'torchaudio' else 'Hugging Face'}"
                ),
                "language": info["language"],
                "language_name": info["language_name"],
                "source": info["source"],
                "recommended": True,
            }
        )
    for model_id, info in DIARIZATION_MODELS.items():
        model_path = models_dir / info["dirname"]
        from subforge.core.asr.speaker_diarization import is_diarization_model_dir

        ready = is_diarization_model_dir(model_path)
        result.append(
            {
                "id": model_id,
                "name": info["name"],
                "category": info["category"],
                "type": info["type"],
                "size": info["size"],
                "downloaded": ready,
                "downloadable": True,
                "path": str(model_path),
                "value": info["repo"],
                "selected": selected_diarization != "off",
                "state": "ready" if ready else "missing",
                "detail": "双人/多人说话人标注" if ready else "需 Hugging Face 授权后下载",
            }
        )
    for model_id, info in SPEAKER_VERIFICATION_MODELS.items():
        ready = is_speaker_verification_model_ready(models_dir)
        result.append(
            {
                "id": model_id,
                "name": info["name"],
                "category": info["category"],
                "type": info["type"],
                "size": info["size"],
                "downloaded": ready,
                "downloadable": True,
                "path": str(speaker_verification_model_path(models_dir)),
                "value": info["repo"],
                "selected": selected_diarization != "off" and ready,
                "state": "ready" if ready else "missing",
                "detail": (
                    "独立声纹交叉校验已启用" if ready else "可选；缺失时保留 Community-1 保守结果"
                ),
            }
        )
    return result


class DownloadModelRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=128)


@router.post("/download-model")
async def download_whisper_model(req: DownloadModelRequest):
    """Start downloading a local ASR model."""
    if req.model_id in WHISPER_CPP_MODELS:
        models_dir = _get_models_dir()
        model_path = models_dir / f"ggml-{req.model_id}.bin"
    elif req.model_id in WHISPERX_MODELS:
        models_dir = _get_models_dir()
        model_path = _alignment_model_path(req.model_id, models_dir)
    elif req.model_id in DIARIZATION_MODELS:
        models_dir = _get_models_dir()
        model_path = models_dir / DIARIZATION_MODELS[req.model_id]["dirname"]
    elif req.model_id in SPEAKER_VERIFICATION_MODELS:
        models_dir = _get_models_dir()
        model_path = models_dir / SPEAKER_VERIFICATION_MODELS[req.model_id]["dirname"]
    elif req.model_id in FASTER_WHISPER_MODELS:
        models_dir = _get_models_dir()
        model_path = models_dir / req.model_id
    elif req.model_id.startswith("mlx-"):
        raise HTTPException(
            status_code=400,
            detail="MLX 模型会在首次测试或转录时自动下载，无需在此手动下载。",
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model_id}")

    if req.model_id in DIARIZATION_MODELS:
        from subforge.core.asr.speaker_diarization import is_diarization_model_dir

        model_ready = is_diarization_model_dir(model_path)
    elif req.model_id in SPEAKER_VERIFICATION_MODELS:
        model_ready = is_speaker_verification_model_ready(models_dir)
    elif req.model_id in FASTER_WHISPER_MODELS:
        model_ready = is_faster_whisper_model_dir(model_path)
    elif req.model_id in WHISPERX_MODELS:
        model_ready = _alignment_model_ready(req.model_id, models_dir)
    else:
        model_ready = model_path.exists()
    if model_ready:
        return {"status": "already_exists", "path": str(model_path)}

    task = task_manager.create_task("download_model")
    task_obj = asyncio.create_task(_download_model(task.id, req.model_id, model_path))
    task_manager.register_running_task(task.id, task_obj)
    _background_tasks.add(task_obj)
    task_obj.add_done_callback(_background_tasks.discard)
    task_obj.add_done_callback(lambda _task: task_manager.unregister_running_task(task.id))
    return {"task_id": task.id, "status": "started"}


async def _download_model(task_id: str, model_id: str, dest: Path):
    if model_id in DIARIZATION_MODELS:
        await _download_diarization_model(task_id, model_id, dest)
        return
    if model_id in SPEAKER_VERIFICATION_MODELS:
        await _download_speaker_verification_model(task_id, model_id, dest)
        return
    if model_id in FASTER_WHISPER_MODELS:
        await _download_faster_whisper_model(task_id, model_id, dest)
        return
    if model_id in WHISPERX_MODELS and WHISPERX_MODELS[model_id]["source"] == "huggingface":
        await _download_huggingface_alignment_model(task_id, model_id)
        return

    import httpx

    if model_id in WHISPER_CPP_MODELS:
        url = WHISPER_CPP_MODELS[model_id]["url"]
    else:
        url = WHISPERX_MODELS[model_id]["url"]
    max_bytes = {
        "tiny": 200 * 1024**2,
        "base": 300 * 1024**2,
        "small": 750 * 1024**2,
        "medium": 2 * 1024**3,
        "large-v1": 4 * 1024**3,
        "large-v2": 4 * 1024**3,
        "large-v3": 4 * 1024**3,
    }.get(model_id, 5 * 1024**3)
    tmp_dest = dest.with_name(f".{dest.name}.{task_id}.part")
    download_lock = _model_download_locks.setdefault(str(dest), asyncio.Lock())
    try:
        async with download_lock:
            if dest.exists():
                task_manager.complete_task(task_id, {"path": str(dest)})
                return
            dest.parent.mkdir(parents=True, exist_ok=True)
            task_manager.update_progress(task_id, 0, f"开始下载 {model_id} 模型...")

            async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    total = int(resp.headers.get("content-length", 0))
                    if total > max_bytes:
                        raise RuntimeError("Model download exceeds the expected size limit")
                    downloaded = 0
                    last_pct = -1

                    with open(tmp_dest, "wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                            downloaded += len(chunk)
                            if downloaded > max_bytes:
                                raise RuntimeError("Model download exceeds the expected size limit")
                            f.write(chunk)
                            if total > 0:
                                pct = min(99, int(downloaded / total * 100))
                                if pct > last_pct:
                                    last_pct = pct
                                    task_manager.update_progress(task_id, pct, f"下载中... {pct}%")

            if downloaded == 0 or (total > 0 and downloaded != total):
                raise RuntimeError("Model download was empty or incomplete")
            tmp_dest.replace(dest)
            task_manager.complete_task(task_id, {"path": str(dest)})
    except asyncio.CancelledError:
        tmp_dest.unlink(missing_ok=True)
        raise
    except Exception as e:
        # Clean up partial download
        tmp_dest.unlink(missing_ok=True)
        logger.exception("Model download failed for %s", model_id)
        task_manager.fail_task(task_id, str(e))
    finally:
        if not download_lock.locked():
            _model_download_locks.pop(str(dest), None)


async def _download_huggingface_alignment_model(task_id: str, model_id: str):
    """Download an alignment model into the cache consumed by WhisperX."""
    spec = ALIGNMENT_MODEL_BY_ID[model_id]
    models_dir = _get_models_dir()
    dest = _alignment_model_path(model_id, models_dir)
    download_lock = _model_download_locks.setdefault(str(dest), asyncio.Lock())
    try:
        async with download_lock:
            if _alignment_model_ready(model_id, models_dir):
                task_manager.complete_task(task_id, {"path": str(dest)})
                return
            task_manager.update_progress(task_id, 5, f"正在下载{spec.language_name}词级对齐模型...")

            def _snapshot_download():
                from huggingface_hub import snapshot_download

                return snapshot_download(
                    repo_id=spec.model_name,
                    cache_dir=str(models_dir),
                )

            await run_blocking(_snapshot_download)
            task_manager.update_progress(task_id, 95, "正在校验模型文件...")
            if not _alignment_model_ready(model_id, models_dir):
                raise RuntimeError("下载完成但模型配置或权重文件不完整")
            task_manager.complete_task(task_id, {"path": str(dest)})
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Alignment model download failed for %s", spec.model_name)
        task_manager.fail_task(task_id, f"{spec.language_name}对齐模型下载失败：{exc}")
    finally:
        if not download_lock.locked():
            _model_download_locks.pop(str(dest), None)


async def _download_faster_whisper_model(task_id: str, model_id: str, dest: Path):
    """Download and atomically publish a CTranslate2 FasterWhisper model."""
    model_value = FASTER_WHISPER_MODELS[model_id]["value"]
    staging = dest.with_name(f".{dest.name}.{task_id}.part")
    download_lock = _model_download_locks.setdefault(str(dest), asyncio.Lock())
    try:
        async with download_lock:
            if is_faster_whisper_model_dir(dest):
                task_manager.complete_task(task_id, {"path": str(dest)})
                return
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            staging.parent.mkdir(parents=True, exist_ok=True)
            task_manager.update_progress(
                task_id, 5, f"正在下载 FasterWhisper {model_value} 模型..."
            )

            def _download_snapshot():
                from faster_whisper.utils import download_model

                return download_model(model_value, output_dir=str(staging))

            await run_blocking(_download_snapshot)
            task_manager.update_progress(task_id, 95, "正在校验 CTranslate2 模型...")
            if not is_faster_whisper_model_dir(staging):
                raise RuntimeError(
                    "下载完成但 CTranslate2 模型不完整（缺少配置、权重或 tokenizer）"
                )
            if dest.exists():
                shutil.rmtree(dest)
            staging.replace(dest)
            task_manager.complete_task(task_id, {"path": str(dest)})
    except asyncio.CancelledError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        logger.exception("FasterWhisper model download failed for %s", model_value)
        task_manager.fail_task(task_id, f"FasterWhisper 模型下载失败：{exc}")
    finally:
        if not download_lock.locked():
            _model_download_locks.pop(str(dest), None)


async def _download_diarization_model(task_id: str, model_id: str, dest: Path):
    """Download a gated pyannote snapshot into the managed model folder."""
    from app.api.config import get_config_value

    token = str(get_config_value("huggingface_token", "") or "").strip()
    if not token:
        task_manager.fail_task(
            task_id,
            "请先在设置中填写 Hugging Face Token，并接受 Community-1 模型使用条款。",
        )
        return

    staging = dest.with_name(f".{dest.name}.{task_id}.part")
    download_lock = _model_download_locks.setdefault(str(dest), asyncio.Lock())
    try:
        async with download_lock:
            from subforge.core.asr.speaker_diarization import is_diarization_model_dir

            if is_diarization_model_dir(dest):
                task_manager.complete_task(task_id, {"path": str(dest)})
                return
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            task_manager.update_progress(task_id, 5, "正在验证 Hugging Face 授权...")

            def _snapshot_download():
                from huggingface_hub import snapshot_download

                return snapshot_download(
                    repo_id=DIARIZATION_MODELS[model_id]["repo"],
                    token=token,
                    local_dir=str(staging),
                )

            await run_blocking(_snapshot_download)
            if not is_diarization_model_dir(staging):
                raise RuntimeError("下载完成但 Community-1 核心配置或权重文件缺失")
            if dest.exists():
                shutil.rmtree(dest)
            staging.replace(dest)
            task_manager.complete_task(task_id, {"path": str(dest)})
    except asyncio.CancelledError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        logger.exception("Diarization model download failed")
        task_manager.fail_task(
            task_id,
            "说话人模型下载失败。请确认已接受 Community-1 条款且 Token 具有读取权限：" + str(exc),
        )
    finally:
        if not download_lock.locked():
            _model_download_locks.pop(str(dest), None)


async def _download_speaker_verification_model(task_id: str, model_id: str, dest: Path):
    """Download the pinned public ECAPA snapshot into the managed model folder."""
    info = SPEAKER_VERIFICATION_MODELS[model_id]
    staging = dest.with_name(f".{dest.name}.{task_id}.part")
    download_lock = _model_download_locks.setdefault(str(dest), asyncio.Lock())
    try:
        async with download_lock:
            models_dir = dest.parent
            if is_speaker_verification_model_ready(models_dir):
                task_manager.complete_task(task_id, {"path": str(dest)})
                return
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            task_manager.update_progress(task_id, 5, "正在下载独立声纹校验模型...")

            def _snapshot_download():
                from huggingface_hub import snapshot_download

                return snapshot_download(
                    repo_id=info["repo"],
                    revision=info["revision"],
                    local_dir=str(staging),
                    allow_patterns=[info["filename"], "README.md", "config.yaml"],
                )

            await run_blocking(_snapshot_download)
            staged_model = staging / info["filename"]
            if not staged_model.is_file() or staged_model.stat().st_size < 20 * 1024 * 1024:
                raise RuntimeError("下载完成但 ECAPA512-LM ONNX 权重缺失或不完整")
            if dest.exists():
                shutil.rmtree(dest)
            staging.replace(dest)
            task_manager.complete_task(task_id, {"path": str(dest)})
    except asyncio.CancelledError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        logger.exception("Speaker verification model download failed")
        task_manager.fail_task(task_id, "独立声纹校验模型下载失败：" + str(exc))
    finally:
        if not download_lock.locked():
            _model_download_locks.pop(str(dest), None)
