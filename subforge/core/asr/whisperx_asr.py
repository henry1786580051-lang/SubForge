import json
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from ...config import MODEL_PATH
from ..utils.logger import setup_logger
from .asr_data import ASRDataSeg
from .base import BaseASR
from .status import ASRStatus

logger = setup_logger("whisperx_asr")


DEFAULT_EN_ALIGN_MODEL = "WAV2VEC2_ASR_LARGE_LV60K_960H"
DEFAULT_LOCAL_MLX_MODEL = "/Users/guwenhan/Desktop/YouTube/model/whisper-large-v3-fp16"
LOCAL_MLX_MODEL_NAMES = (
    "whisper-large-v3-fp16",
    "mlx-whisper-large-v3-fp16",
    "large-v3-fp16",
)
MLX_WHISPER_MODELS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-mlx",
    "large-v1": "mlx-community/whisper-large-v1-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def _candidate_local_model_dirs() -> list[Path]:
    candidates = [Path(DEFAULT_LOCAL_MLX_MODEL)]
    env_path = os.environ.get("SUBFORGE_MLX_WHISPER_MODEL", "").strip()
    if env_path:
        candidates.insert(0, Path(env_path).expanduser())

    roots = [
        Path.home() / "Desktop" / "YouTube" / "model",
        Path.home() / "SubForge" / "models",
        Path.home() / "Library" / "Application Support" / "SubForge" / "models",
        MODEL_PATH,
    ]
    for root in roots:
        for name in LOCAL_MLX_MODEL_NAMES:
            candidates.append(root / name)

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _is_valid_mlx_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "config.json").is_file()
    has_weights = any(
        (path / name).is_file()
        for name in ("weights.safetensors", "weights.npz", "model.safetensors")
    )
    return has_config and has_weights


def _find_local_mlx_model() -> Path | None:
    for candidate in _candidate_local_model_dirs():
        if _is_valid_mlx_model_dir(candidate):
            return candidate
    return None


def default_mlx_model() -> str:
    path = _find_local_mlx_model()
    if path:
        return str(path)
    return "large-v3"


def _normalize_language(language: str | None) -> str | None:
    value = (language or "").strip().lower()
    if not value or value == "auto":
        return None
    return value


def _normalize_align_device(device: str | None) -> str:
    value = (device or "cpu").strip().lower()
    if value == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    if value == "mps":
        # MLX handles Apple Silicon acceleration; WhisperX alignment is more stable on CPU.
        return "cpu"
    if value in {"cuda", "cpu"}:
        return value
    return "cpu"


def _normalize_compute_type(device: str, compute_type: str | None) -> str:
    value = (compute_type or "default").strip().lower()
    if value and value != "default":
        return value
    return "float16" if device == "cuda" else "int8"


def _mlx_model_repo(model: str) -> str:
    value = (model or default_mlx_model()).strip()
    if value in {"", "auto", "default"}:
        return default_mlx_model()
    if "/" in value or Path(value).exists():
        return value
    # Prefer a local MLX Large V3 FP16 directory for common short aliases. This
    # keeps the packaged app from unexpectedly falling back to an online
    # HuggingFace repo when the user has already downloaded the local model.
    if value in {"large-v3", "large-v3-fp16", "whisper-large-v3-fp16"}:
        local = _find_local_mlx_model()
        if local:
            return str(local)
    return MLX_WHISPER_MODELS.get(value, value)


def _prepare_mlx_model_path(model: str, tmp_dir: Path) -> str:
    model_path = Path(model)
    if not model_path.exists() or not model_path.is_dir():
        return model
    if not _is_valid_mlx_model_dir(model_path):
        raise RuntimeError(
            "Invalid MLX Whisper model directory. Expected config.json and "
            f"weights/model safetensors in: {model_path}"
        )

    weights = model_path / "weights.safetensors"
    weights_npz = model_path / "weights.npz"
    model_weights = model_path / "model.safetensors"
    if weights.exists() or weights_npz.exists() or not model_weights.exists():
        return str(model_path)

    alias_dir = tmp_dir / "mlx_model"
    alias_dir.mkdir(parents=True, exist_ok=True)
    for item in model_path.iterdir():
        if not item.is_file() or item.name == ".DS_Store":
            continue
        target_name = "weights.safetensors" if item.name == "model.safetensors" else item.name
        target = alias_dir / target_name
        if target.exists():
            continue
        try:
            os.symlink(item, target)
        except OSError:
            os.link(item, target)
    return str(alias_dir)


def _segments_for_alignment(result: dict) -> list[dict]:
    segments: list[dict] = []
    for item in result.get("segments") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        start = item.get("start")
        end = item.get("end")
        if not text or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        segments.append({"text": text, "start": float(start), "end": float(end)})
    return segments


def install_whisperx_runtime_stubs() -> None:
    """Avoid importing WhisperX features that SubForge does not execute.

    Python executes whisperx.__init__ before loading whisperx.alignment. That
    __init__ imports transcribe/diarize, which then pull faster-whisper VAD,
    pyannote, sklearn, matplotlib and other unused packages. SubForge uses
    MLX Whisper for transcription and only needs WhisperX audio/alignment.
    """

    def _unsupported(*_args, **_kwargs):
        raise RuntimeError("This WhisperX feature is not bundled in SubForge")

    if "whisperx.transcribe" not in sys.modules:
        transcribe = types.ModuleType("whisperx.transcribe")
        transcribe.load_model = _unsupported
        sys.modules["whisperx.transcribe"] = transcribe

    if "whisperx.diarize" not in sys.modules:
        diarize = types.ModuleType("whisperx.diarize")
        diarize.assign_word_speakers = _unsupported
        diarize.DiarizationPipeline = _unsupported
        sys.modules["whisperx.diarize"] = diarize


def _float_seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    return None


def _word_text(word: dict) -> str:
    return str(word.get("word") or word.get("text") or "").strip()


def _word_has_timing(word: dict) -> bool:
    return _float_seconds(word.get("start")) is not None and _float_seconds(word.get("end")) is not None


def _word_duration_weight(text: str) -> int:
    alnum = sum(1 for ch in text if ch.isalnum())
    return max(1, alnum)


def _append_word_run(
    output: list[ASRDataSeg],
    words: list[dict],
    left_bound: float | None,
    right_bound: float | None,
) -> None:
    """Add unaligned WhisperX words by distributing the adjacent aligned gap.

    WhisperX keeps unalignable words such as numerals in the segment word list,
    but without start/end. Dropping them makes the exported transcript lose real
    content. The safest recovery is local interpolation between the surrounding
    aligned words inside the same ASR segment.
    """
    texts = [_word_text(word) for word in words]
    valid = [(word, text) for word, text in zip(words, texts) if text]
    if not valid:
        return

    if left_bound is None and right_bound is None:
        return
    if left_bound is None:
        left_bound = max(0.0, right_bound - 0.25 * len(valid))  # type: ignore[operator]
    if right_bound is None:
        right_bound = left_bound + 0.25 * len(valid)
    if right_bound <= left_bound:
        right_bound = left_bound + 0.02 * len(valid)

    weights = [_word_duration_weight(text) for _, text in valid]
    total_weight = max(1, sum(weights))
    current = left_bound
    for index, ((_, text), weight) in enumerate(zip(valid, weights)):
        if index == len(valid) - 1:
            end = right_bound
        else:
            end = current + (right_bound - left_bound) * weight / total_weight
        start_ms = max(0, int(round(current * 1000)))
        end_ms = max(start_ms, int(round(end * 1000)))
        output.append(ASRDataSeg(text, start_ms, end_ms))
        current = end


def _words_to_segments(
    words: list[dict],
    segment_start: float | None = None,
    segment_end: float | None = None,
) -> list[ASRDataSeg]:
    output: list[ASRDataSeg] = []
    pending: list[dict] = []
    last_known_end = segment_start

    for word in words:
        text = _word_text(word)
        if not text:
            continue

        if _word_has_timing(word):
            start = _float_seconds(word.get("start"))
            end = _float_seconds(word.get("end"))
            if start is None or end is None:
                continue
            if pending:
                if last_known_end is not None or segment_start is not None:
                    _append_word_run(output, pending, last_known_end, start)
                pending = []
            start_ms = max(0, int(round(start * 1000)))
            end_ms = max(start_ms, int(round(end * 1000)))
            output.append(ASRDataSeg(text, start_ms, end_ms))
            last_known_end = end
        else:
            pending.append(word)

    if pending:
        if segment_end is not None:
            _append_word_run(output, pending, last_known_end, segment_end)

    return output


class WhisperXASR(BaseASR):
    """MLX Whisper ASR on Apple Silicon followed by WhisperX forced alignment."""

    def __init__(
        self,
        audio_input: Union[str, bytes],
        whisper_model: str = "",
        model_dir: str = "",
        language: str = "en",
        device: str = "auto",
        compute_type: str = "default",
        align_model: str = "",
        batch_size: int = 4,
        use_cache: bool = False,
        need_word_time_stamp: bool = True,
    ):
        super().__init__(audio_input, use_cache, need_word_time_stamp)
        self.whisper_model = whisper_model or default_mlx_model()
        self.mlx_model = _mlx_model_repo(self.whisper_model)
        self.model_dir = model_dir or str(MODEL_PATH)
        self.language = _normalize_language(language)
        self.align_device = _normalize_align_device(device)
        self.compute_type = _normalize_compute_type(self.align_device, compute_type)
        self.align_model = (align_model or "").strip()
        self.batch_size = max(1, int(batch_size or 4))
        self.need_word_time_stamp = need_word_time_stamp

    def _write_audio_to_temp(self, tmp_dir: Path) -> str:
        if isinstance(self.audio_input, str):
            return self.audio_input
        if not self.file_binary:
            raise ValueError("No audio data available")
        audio_path = tmp_dir / "whisperx_audio.wav"
        audio_path.write_bytes(self.file_binary)
        return str(audio_path)

    def _resolve_align_model_name(self, language_code: str) -> str | None:
        if self.align_model:
            return self.align_model
        if language_code == "en":
            return DEFAULT_EN_ALIGN_MODEL
        return None

    def _run(
        self,
        callback: Optional[Callable[[int, str], None]] = None,
        **kwargs: Any,
    ) -> dict:
        def _default_callback(_progress: int, _message: str) -> None:
            pass

        if callback is None:
            callback = _default_callback

        try:
            import mlx_whisper
            install_whisperx_runtime_stubs()
            from whisperx.alignment import align as whisperx_align
            from whisperx.alignment import load_align_model
            from whisperx.audio import load_audio
        except ImportError as exc:
            raise RuntimeError(
                "WhisperX/MLX Whisper is not installed. Install the optional WhisperX stack "
                "or rebuild SubForge with the whisperx extra."
            ) from exc
        except RuntimeError as exc:
            message = str(exc)
            if "No Metal device available" in message:
                raise RuntimeError(
                    "MLX Whisper could not access an Apple Metal device. Launch SubForge from "
                    "the macOS app bundle/Finder on Apple Silicon, and avoid running it in a "
                    "headless or sandboxed terminal environment."
                ) from exc
            if "metallib" in message.lower():
                raise RuntimeError(
                    "MLX Whisper could not load mlx.metallib. Rebuild or reinstall SubForge "
                    "with the bundled MLX Metal resources."
                ) from exc
            raise

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            audio_path = self._write_audio_to_temp(tmp_dir)
            mlx_model_path = _prepare_mlx_model_path(self.mlx_model, tmp_dir)
            callback(20, "Transcribing with MLX Whisper...")
            logger.info(
                "Transcribing with MLX Whisper model=%s align_device=%s",
                mlx_model_path,
                self.align_device,
            )
            if not Path(mlx_model_path).exists() and "/" not in mlx_model_path:
                logger.warning(
                    "MLX Whisper model %s is not a local path. mlx-whisper may try to download it.",
                    mlx_model_path,
                )
            transcribe_kwargs: dict[str, Any] = {
                "path_or_hf_repo": mlx_model_path,
                "word_timestamps": False,
                "condition_on_previous_text": False,
                "verbose": False,
            }
            if self.language:
                transcribe_kwargs["language"] = self.language
            try:
                result = mlx_whisper.transcribe(audio_path, **transcribe_kwargs)
            except TypeError:
                transcribe_kwargs.pop("condition_on_previous_text", None)
                result = mlx_whisper.transcribe(audio_path, **transcribe_kwargs)

            callback(35, "Loading audio...")
            audio = load_audio(audio_path)

            align_segments = _segments_for_alignment(result)
            if not align_segments:
                raise RuntimeError("MLX Whisper did not return alignable transcript segments")

            language_code = (result.get("language") or self.language or "en").lower()
            callback(65, "Loading forced alignment model...")
            align_model_name = self._resolve_align_model_name(language_code)
            align_kwargs: dict[str, Any] = {
                "language_code": language_code,
                "device": self.align_device,
            }
            if align_model_name:
                align_kwargs["model_name"] = align_model_name
            if self.model_dir:
                align_kwargs["model_dir"] = self.model_dir
            try:
                model_a, metadata = load_align_model(**align_kwargs)
            except TypeError:
                align_kwargs.pop("model_dir", None)
                model_a, metadata = load_align_model(**align_kwargs)

            callback(78, "Running forced alignment...")
            aligned = whisperx_align(
                align_segments,
                model_a,
                metadata,
                audio,
                self.align_device,
                return_char_alignments=False,
            )
            aligned["language"] = language_code
            aligned["align_model"] = align_model_name or ""
            aligned["asr_backend"] = "mlx-whisper"
            aligned["mlx_model"] = self.mlx_model

            try:
                del model_a
                if self.align_device == "cuda":
                    import torch

                    torch.cuda.empty_cache()
            except Exception:
                pass

            callback(*ASRStatus.COMPLETED.callback_tuple())
            return aligned

    def _make_segments(self, resp_data: dict) -> List[ASRDataSeg]:
        segments: list[ASRDataSeg] = []

        if self.need_word_time_stamp:
            for item in resp_data.get("segments") or []:
                if not isinstance(item, dict):
                    continue
                words = item.get("words")
                if not isinstance(words, list):
                    continue
                word_dicts = [word for word in words if isinstance(word, dict)]
                segment_start = _float_seconds(item.get("start"))
                segment_end = _float_seconds(item.get("end"))
                segments.extend(_words_to_segments(word_dicts, segment_start, segment_end))

            if segments:
                return segments

            words = resp_data.get("word_segments") or []
            if isinstance(words, list):
                word_dicts = [word for word in words if isinstance(word, dict)]
                segments.extend(_words_to_segments(word_dicts))

            if segments:
                return segments

        for item in resp_data.get("segments") or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            start = item.get("start")
            end = item.get("end")
            if not text or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            start_ms = max(0, int(round(float(start) * 1000)))
            end_ms = max(start_ms, int(round(float(end) * 1000)))
            segments.append(ASRDataSeg(text, start_ms, end_ms))

        return segments

    def _get_key(self) -> str:
        key = {
            "crc32": self.crc32_hex,
            "model": self.mlx_model,
            "language": self.language or "auto",
            "align_device": self.align_device,
            "compute_type": self.compute_type,
            "align_model": self.align_model or "auto",
            "batch_size": self.batch_size,
            "word": self.need_word_time_stamp,
        }
        return json.dumps(key, sort_keys=True)
