"""Direct FasterWhisper/CTranslate2 ASR implementation."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from .asr_data import ASRDataSeg
from .base import BaseASR
from .status import ASRStatus

logger = logging.getLogger(__name__)
_CUDA_DLL_DIRECTORY_HANDLES: list[Any] = []
_FASTER_WORKER_FLAG = "SUBFORGE_FASTER_WHISPER_WORKER"
_FASTER_WORKER_REQUEST = "SUBFORGE_FASTER_WHISPER_REQUEST"
_FASTER_WORKER_OUTPUT = "SUBFORGE_FASTER_WHISPER_OUTPUT"
_FASTER_WORKER_PROGRESS = "SUBFORGE_FASTER_WHISPER_PROGRESS"


def _atomic_json_write(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _candidate_cuda_runtime_dirs() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("SUBFORGE_CUDA_RUNTIME_DIR", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    if getattr(sys, "frozen", False):
        # Prefer the CUDA libraries shipped with the CUDA-enabled PyTorch
        # wheel. Loading a second cuBLAS/cuDNN set in the same WhisperX process
        # can terminate Windows before Python can report an exception.
        candidates.append(Path(getattr(sys, "_MEIPASS", "")) / "torch" / "lib")
        candidates.append(Path(getattr(sys, "_MEIPASS", "")) / "cuda")

    if platform.system() == "Windows":
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec and torch_spec.submodule_search_locations:
            torch_package = Path(next(iter(torch_spec.submodule_search_locations)))
            candidates.append(torch_package / "lib")
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
        if (
            platform.system() == "Windows"
            and getattr(sys, "frozen", False)
            and self.device == "cuda"
            and os.environ.get(_FASTER_WORKER_FLAG) != "1"
        ):
            return self._run_in_packaged_worker(callback)
        return self._run_direct(callback)

    def _run_in_packaged_worker(
        self, callback: Optional[Callable[[int, str], None]] = None
    ) -> list[dict[str, Any]]:
        """Run CUDA CTranslate2 in a disposable process on packaged Windows.

        Some Windows CTranslate2 builds can fast-fail while their CUDA allocator
        tears down worker threads. The child writes its result while the model is
        still alive and then exits with os._exit, so native cleanup cannot take
        down the desktop/backend process.
        """
        effective_callback = callback or (lambda _progress, _message: None)
        with tempfile.TemporaryDirectory(prefix="subforge-faster-worker-") as temp_path:
            temp_dir = Path(temp_path)
            if isinstance(self.audio_input, str):
                audio_path = self.audio_input
            else:
                worker_audio = temp_dir / "audio.wav"
                worker_audio.write_bytes(self.file_binary or b"")
                audio_path = str(worker_audio)

            request_path = temp_dir / "request.json"
            output_path = temp_dir / "output.json"
            progress_path = temp_dir / "progress.json"
            _atomic_json_write(
                request_path,
                {
                    "audio_input": audio_path,
                    "whisper_model": str(self.model_path),
                    "language": self.language,
                    "device": self.device,
                    "compute_type": self.compute_type,
                    "vad_filter": self.vad_filter,
                    "vad_threshold": self.vad_threshold,
                    "need_word_time_stamp": self.need_word_time_stamp,
                    "prompt": self.prompt,
                },
            )
            env = os.environ.copy()
            env.update(
                {
                    _FASTER_WORKER_FLAG: "1",
                    _FASTER_WORKER_REQUEST: str(request_path),
                    _FASTER_WORKER_OUTPUT: str(output_path),
                    _FASTER_WORKER_PROGRESS: str(progress_path),
                }
            )
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            process = subprocess.Popen(
                [sys.executable],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            last_progress: tuple[int, str] | None = None
            try:
                while process.poll() is None:
                    if progress_path.is_file():
                        try:
                            progress_data = json.loads(progress_path.read_text(encoding="utf-8"))
                            current = (
                                int(progress_data["progress"]),
                                str(progress_data["message"]),
                            )
                            if current != last_progress:
                                effective_callback(*current)
                                last_progress = current
                        except (OSError, ValueError, KeyError, TypeError):
                            pass
                    time.sleep(0.2)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)

            payload: dict[str, Any] | None = None
            if output_path.is_file():
                try:
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    payload = None
            if process.returncode != 0 or not payload or not payload.get("ok"):
                detail = str((payload or {}).get("error") or "").strip()
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(
                    f"FasterWhisper CUDA worker exited with code {process.returncode}{suffix}"
                )
            data = payload.get("data")
            if not isinstance(data, list):
                raise RuntimeError("FasterWhisper CUDA worker returned invalid data")
            return data

    def _run_direct(
        self, callback: Optional[Callable[[int, str], None]] = None
    ) -> list[dict[str, Any]]:
        try:
            faster_whisper = importlib.import_module("faster_whisper")
            whisper_model_class = getattr(faster_whisper, "WhisperModel")
        except (AttributeError, ImportError) as exc:
            raise RuntimeError(
                "The FasterWhisper runtime is missing from this installation."
            ) from exc

        effective_callback = callback or (lambda _progress, _message: None)
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
            # Keep the native model alive until the packaged worker has written
            # its result and terminates with os._exit. Releasing it at function
            # return is the Windows CUDA crash this worker boundary prevents.
            self._active_model = model
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
                        start = float(
                            getattr(word, "start", getattr(segment, "start", 0.0))
                        )
                        end = float(
                            getattr(word, "end", getattr(segment, "end", start))
                        )
                        if text and end > start:
                            output.append({"text": text, "start": start, "end": end})
                else:
                    text = str(getattr(segment, "text", "")).strip()
                    start = float(getattr(segment, "start", 0.0))
                    end = float(getattr(segment, "end", start))
                    if text and end > start:
                        output.append({"text": text, "start": start, "end": end})
                end = float(getattr(segment, "end", 0.0))
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
                timestamp_granularity=(
                    "word" if self.need_word_time_stamp else "sentence"
                ),
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
            self.need_word_time_stamp,
            self.prompt,
        )
        digest = hashlib.sha256(repr(settings).encode()).hexdigest()
        return f"{self.crc32_hex}-{digest}"


def run_packaged_faster_whisper_worker() -> None:
    """Execute one packaged FasterWhisper request and bypass native teardown."""
    request_path = Path(os.environ[_FASTER_WORKER_REQUEST])
    output_path = Path(os.environ[_FASTER_WORKER_OUTPUT])
    progress_path = Path(os.environ[_FASTER_WORKER_PROGRESS])
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        asr = FasterWhisperASR(**request)

        def report(progress: int, message: str) -> None:
            try:
                _atomic_json_write(
                    progress_path,
                    {"progress": int(progress), "message": str(message)},
                )
            except OSError:
                # Progress is advisory. Antivirus/indexing or the parent reader
                # can briefly hold this file on Windows; never abort CUDA ASR
                # because one UI progress update could not be published.
                pass

        data = asr._run_direct(report)
        _atomic_json_write(output_path, {"ok": True, "data": data})
        exit_code = 0
    except BaseException:
        _atomic_json_write(output_path, {"ok": False, "error": traceback.format_exc()})
        exit_code = 1
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream:
                stream.flush()
        except Exception:
            pass
    os._exit(exit_code)
