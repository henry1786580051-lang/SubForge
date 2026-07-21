"""Direct FasterWhisper/CTranslate2 ASR implementation."""

from __future__ import annotations

import ctypes
import hashlib
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
                path / "bin"
                for path in sorted(toolkit_root.glob("v*"), reverse=True)
            )
        home = Path.home()
        for distribution in ("anaconda3", "miniconda3", "miniforge3"):
            envs = home / distribution / "envs"
            if envs.is_dir():
                candidates.extend(
                    env / "Lib" / "site-packages" / "torch" / "lib"
                    for env in envs.iterdir()
                    if env.is_dir()
                )
    return candidates


def _prepare_cuda_runtime() -> None:
    if platform.system() != "Windows" or not hasattr(os, "add_dll_directory"):
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
            _CUDA_DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
            logger.info("Using private FasterWhisper CUDA runtime from %s", directory)
            return
        except OSError:
            continue


def is_faster_whisper_cuda_available() -> bool:
    """Return whether CTranslate2 can actually load its CUDA dependencies."""
    if platform.system() == "Darwin":
        return False

    _prepare_cuda_runtime()
    libraries = (
        ("cublas64_12.dll", "cudnn64_9.dll")
        if platform.system() == "Windows"
        else ("libcublas.so.12", "libcudnn.so.9")
    )
    loader = ctypes.WinDLL if platform.system() == "Windows" else ctypes.CDLL
    try:
        handles = [loader(name) for name in libraries]
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except (ImportError, OSError):
        return False
    finally:
        # Keep explicit references until after CTranslate2 probes the device.
        if "handles" in locals():
            del handles


def resolve_faster_whisper_runtime(
    device: str | None, compute_type: str | None
) -> tuple[str, str]:
    """Choose a runnable CTranslate2 device and compatible compute type."""
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
        logger.warning(
            "FasterWhisper CUDA runtime is incomplete; falling back to CPU"
        )

    resolved_compute = (compute_type or "default").strip().lower()
    if resolved_compute == "default":
        resolved_compute = "float16" if resolved_device == "cuda" else "int8"
    elif resolved_device == "cpu" and resolved_compute == "float16":
        resolved_compute = "int8"
    return resolved_device, resolved_compute


def is_faster_whisper_model_dir(path: str | Path) -> bool:
    """Return whether a CTranslate2 Whisper model directory is complete."""
    model_dir = Path(path).expanduser()
    required = {
        "config.json": 100,
        "model.bin": 1024 * 1024,
        "tokenizer.json": 1024,
    }
    return model_dir.is_dir() and all(
        (model_dir / name).is_file() and (model_dir / name).stat().st_size >= minimum
        for name, minimum in required.items()
    )


class FasterWhisperASR(BaseASR):
    """Run FasterWhisper directly through its packaged Python runtime."""

    def __init__(
        self,
        audio_input: Union[str, bytes],
        faster_whisper_program: str = "",
        whisper_model: str = "base",
        model_dir: str = "",
        language: str = "zh",
        device: str = "cpu",
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
        self.need_word_time_stamp = need_word_time_stamp
        self.vad_filter = vad_filter
        self.vad_threshold = vad_threshold
        self.prompt = prompt
        self.ff_mdx_kim2 = ff_mdx_kim2
        # Retained for backwards-compatible construction; the packaged app no
        # longer depends on an external faster-whisper-xxl executable.
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
        managed = models_root / f"faster-whisper-{whisper_model}"
        if is_faster_whisper_model_dir(managed):
            return managed
        raise RuntimeError(
            f"FasterWhisper model '{whisper_model}' is not downloaded or is incomplete. "
            "Download the CTranslate2 model in ASR settings; Whisper.cpp GGML .bin "
            "files are not compatible with FasterWhisper."
        )

    def _run(
        self, callback: Optional[Callable[[int, str], None]] = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "The FasterWhisper runtime is missing from this installation."
            ) from exc

        if callback is None:
            def callback(_progress: int, _message: str) -> None:
                return None
        if self.ff_mdx_kim2:
            logger.warning(
                "FF-MDX Kim2 is an external faster-whisper-xxl feature and is ignored "
                "by the built-in FasterWhisper runtime"
            )

        callback(*ASRStatus.TRANSCRIBING.with_progress(5))
        with tempfile.TemporaryDirectory() as temp_path:
            if isinstance(self.audio_input, str):
                audio_path = self.audio_input
            else:
                audio_file = Path(temp_path) / "audio.wav"
                audio_file.write_bytes(self.file_binary or b"")
                audio_path = str(audio_file)

            model = WhisperModel(
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
                word_timestamps=self.need_word_time_stamp,
                initial_prompt=self.prompt or None,
            )
            duration = max(float(getattr(info, "duration", 0.0) or 0.0), 0.01)
            output: list[dict[str, Any]] = []
            for segment in segments:
                words = list(getattr(segment, "words", None) or [])
                if self.need_word_time_stamp and words:
                    for word in words:
                        text = str(getattr(word, "word", "")).strip()
                        if text:
                            output.append(
                                {
                                    "text": text,
                                    "start": float(getattr(word, "start", segment.start)),
                                    "end": float(getattr(word, "end", segment.end)),
                                }
                            )
                else:
                    text = str(getattr(segment, "text", "")).strip()
                    if text:
                        output.append(
                            {
                                "text": text,
                                "start": float(segment.start),
                                "end": float(segment.end),
                            }
                        )
                progress = min(99, 5 + int(float(segment.end) / duration * 90))
                callback(progress, f"{progress}%")

        if not output:
            raise RuntimeError("FasterWhisper returned no transcription segments")
        callback(*ASRStatus.COMPLETED.callback_tuple())
        return output

    def _make_segments(self, resp_data: list[dict[str, Any]]) -> List[ASRDataSeg]:
        return [
            ASRDataSeg(
                text=str(item["text"]).strip(),
                start_time=max(0, round(float(item["start"]) * 1000)),
                end_time=max(1, round(float(item["end"]) * 1000)),
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
            self.need_word_time_stamp,
            self.prompt,
        )
        return f"{self.crc32_hex}-{hashlib.sha256(repr(settings).encode()).hexdigest()}"
