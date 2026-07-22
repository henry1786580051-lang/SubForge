"""Direct FasterWhisper/CTranslate2 ASR implementation."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import logging
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from .asr_data import ASRDataSeg
from .base import BaseASR
from .status import ASRStatus

logger = logging.getLogger(__name__)
_CUDA_DLL_DIRECTORY_HANDLES: list[Any] = []


def _candidate_cuda_runtime_dirs() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("SUBFORGE_CUDA_RUNTIME_DIR", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    if getattr(sys, "frozen", False):
        candidates.append(Path(getattr(sys, "_MEIPASS", "")) / "cuda")

    if platform.system() == "Windows":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        toolkit_root = program_files / "NVIDIA GPU Computing Toolkit" / "CUDA"
        if toolkit_root.is_dir():
            candidates.extend(
                path / "bin" for path in sorted(toolkit_root.glob("v*"), reverse=True)
            )
    return candidates


def _prepare_cuda_runtime() -> None:
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if platform.system() != "Windows" or not callable(add_dll_directory):
        return
    if _CUDA_DLL_DIRECTORY_HANDLES:
        return
    for directory in _candidate_cuda_runtime_dirs():
        if not all(
            (directory / name).is_file()
            for name in ("cublas64_12.dll", "cudnn64_9.dll")
        ):
            continue
        try:
            _CUDA_DLL_DIRECTORY_HANDLES.append(add_dll_directory(str(directory)))
            logger.info("Using FasterWhisper CUDA runtime from %s", directory)
            return
        except OSError:
            continue


def is_faster_whisper_cuda_available() -> bool:
    """Return whether CTranslate2 can load the required CUDA runtime."""
    if platform.system() == "Darwin":
        return False

    _prepare_cuda_runtime()
    libraries = (
        ("cublas64_12.dll", "cudnn64_9.dll")
        if platform.system() == "Windows"
        else ("libcublas.so.12", "libcudnn.so.9")
    )
    loader = getattr(ctypes, "WinDLL", ctypes.CDLL) if platform.system() == "Windows" else ctypes.CDLL
    try:
        handles = [loader(name) for name in libraries]
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except (ImportError, OSError):
        return False
    finally:
        if "handles" in locals():
            del handles


def resolve_faster_whisper_runtime(
    device: str | None, compute_type: str | None
) -> tuple[str, str]:
    """Resolve a runnable device and a compatible CTranslate2 compute type."""
    requested_device = (device or "auto").strip().lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        requested_device = "auto"

    cuda_available = is_faster_whisper_cuda_available()
    resolved_device = (
        "cuda"
        if requested_device in {"auto", "cuda"} and cuda_available
        else "cpu"
    )
    if requested_device == "cuda" and resolved_device == "cpu":
        logger.warning("FasterWhisper CUDA is unavailable; falling back to CPU")

    resolved_compute = (compute_type or "default").strip().lower()
    if resolved_compute == "default":
        resolved_compute = "float16" if resolved_device == "cuda" else "int8"
    elif resolved_device == "cpu" and resolved_compute == "float16":
        resolved_compute = "int8"
    return resolved_device, resolved_compute


def is_faster_whisper_model_dir(path: str | Path) -> bool:
    """Return whether a local CTranslate2 Whisper snapshot is complete."""
    model_dir = Path(path).expanduser()
    required = {
        "config.json": 100,
        "model.bin": 1024 * 1024,
        "tokenizer.json": 1024,
    }
    return model_dir.is_dir() and all(
        (model_dir / name).is_file()
        and (model_dir / name).stat().st_size >= minimum
        for name, minimum in required.items()
    )


class FasterWhisperASR(BaseASR):
    """Run FasterWhisper through the packaged Python runtime."""

    def __init__(
        self,
        audio_input: Union[str, bytes],
        faster_whisper_program: str = "",
        whisper_model: str = "base",
        model_dir: str = "",
        language: str = "zh",
        device: str = "auto",
        output_dir: Optional[str] = None,
        output_format: str = "srt",
        use_cache: bool = False,
        need_word_time_stamp: bool = False,
        vad_filter: bool = True,
        vad_threshold: float = 0.4,
        vad_method: str = "",
        ff_mdx_kim2: bool = False,
        one_word: int = 0,
        sentence: bool = False,
        max_line_width: int = 100,
        max_line_count: int = 1,
        max_comma: int = 20,
        max_comma_cent: int = 50,
        prompt: Optional[str] = None,
        compute_type: str = "default",
    ):
        super().__init__(audio_input, use_cache)
        self.model_path = self._resolve_model_path(whisper_model, model_dir)
        self.model_dir = model_dir
        self.language = language
        self.device, self.compute_type = resolve_faster_whisper_runtime(
            device, compute_type
        )
        self.vad_filter = vad_filter
        self.vad_threshold = max(0.0, min(1.0, float(vad_threshold)))
        self.prompt = prompt
        self.need_word_time_stamp = bool(need_word_time_stamp)
        self.ff_mdx_kim2 = ff_mdx_kim2
        self.faster_whisper_program = faster_whisper_program

    @staticmethod
    def _resolve_model_path(whisper_model: str, model_dir: str) -> Path:
        configured = Path(whisper_model).expanduser()
        if is_faster_whisper_model_dir(configured):
            return configured

        if model_dir:
            models_root = Path(model_dir).expanduser()
        else:
            from subforge.config import MODEL_PATH

            models_root = MODEL_PATH
        candidates = [
            models_root,
            models_root / f"faster-whisper-{whisper_model}",
            models_root / whisper_model,
        ]
        for candidate in candidates:
            if is_faster_whisper_model_dir(candidate):
                return candidate
        raise RuntimeError(
            f"FasterWhisper model '{whisper_model}' is not downloaded or is incomplete. "
            "Download its CTranslate2 model in ASR settings. Whisper.cpp GGML files "
            "are not compatible with FasterWhisper."
        )

    def _run(
        self, callback: Optional[Callable[[int, str], None]] = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        try:
            faster_whisper = importlib.import_module("faster_whisper")
            whisper_model_class = getattr(faster_whisper, "WhisperModel")
        except (AttributeError, ImportError) as exc:
            raise RuntimeError(
                "The FasterWhisper runtime is missing from this installation."
            ) from exc

        effective_callback = callback or (lambda _progress, _message: None)
        if self.need_word_time_stamp:
            logger.info(
                "FasterWhisper word timestamps are deferred; returning segment timestamps"
            )
        if self.ff_mdx_kim2:
            logger.info(
                "FF-MDX Kim2 is unavailable in the built-in FasterWhisper runtime; "
                "SubForge audio enhancement remains unchanged"
            )

        effective_callback(*ASRStatus.TRANSCRIBING.with_progress(5))
        with tempfile.TemporaryDirectory() as temp_path:
            if isinstance(self.audio_input, str):
                audio_path = self.audio_input
            else:
                audio_file = Path(temp_path) / "audio.wav"
                audio_file.write_bytes(self.file_binary or b"")
                audio_path = str(audio_file)

            model = whisper_model_class(
                str(self.model_path),
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=True,
            )
            segments, info = model.transcribe(
                audio_path,
                language=self.language or None,
                beam_size=5,
                vad_filter=self.vad_filter,
                vad_parameters={"threshold": self.vad_threshold},
                word_timestamps=False,
                initial_prompt=self.prompt or None,
            )
            duration = max(float(getattr(info, "duration", 0.0) or 0.0), 0.01)
            output: list[dict[str, Any]] = []
            for segment in segments:
                text = str(getattr(segment, "text", "")).strip()
                start = float(getattr(segment, "start", 0.0))
                end = float(getattr(segment, "end", start))
                if text and end > start:
                    output.append({"text": text, "start": start, "end": end})
                progress = min(99, 5 + int(end / duration * 90))
                effective_callback(progress, f"{progress}%")

        if not output:
            raise RuntimeError("FasterWhisper returned no transcription segments")
        effective_callback(*ASRStatus.COMPLETED.callback_tuple())
        return output

    def _make_segments(self, resp_data: list[dict[str, Any]]) -> List[ASRDataSeg]:
        return [
            ASRDataSeg(
                text=str(item["text"]).strip(),
                start_time=max(0, round(float(item["start"]) * 1000)),
                end_time=max(1, round(float(item["end"]) * 1000)),
                timestamp_granularity="sentence",
                timing_source="native",
            )
            for item in resp_data
            if str(item.get("text", "")).strip()
            and float(item.get("end", 0)) > float(item.get("start", 0))
        ]

    def _get_key(self) -> str:
        settings = (
            str(self.model_path.resolve()),
            self.language,
            self.device,
            self.compute_type,
            self.vad_filter,
            self.vad_threshold,
            self.prompt,
        )
        digest = hashlib.sha256(repr(settings).encode()).hexdigest()
        return f"{self.crc32_hex}-{digest}"
