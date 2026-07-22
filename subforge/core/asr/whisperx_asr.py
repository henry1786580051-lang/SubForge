import json
import os
import platform
import re
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from ...config import MODEL_PATH
from ..utils.logger import setup_logger
from .asr_data import ASRData, ASRDataSeg, ASRWord, TimestampSource
from .base import BaseASR
from .faster_whisper import is_faster_whisper_model_dir
from .model_cache import SingleEntryModelCache
from .status import ASRStatus

logger = setup_logger("whisperx_asr")
_ALIGNMENT_MODEL_CACHE = SingleEntryModelCache()


DEFAULT_EN_ALIGN_MODEL = "WAV2VEC2_ASR_LARGE_LV60K_960H"
DEFAULT_EN_ALIGN_FILENAME = "wav2vec2_fairseq_large_lv60k_asr_ls960.pth"
DEFAULT_HF_ALIGN_MODELS = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
}


def clear_alignment_model_cache() -> None:
    """Release the process-local forced-alignment model, if any."""
    _ALIGNMENT_MODEL_CACHE.clear()


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
    candidates: list[Path] = []
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


def resolve_mlx_model(model: str = "") -> str:
    """Resolve a configured MLX model name to its local path or HF repo."""
    return _mlx_model_repo(model)


def is_valid_mlx_model_dir(path: str | Path) -> bool:
    """Return whether a directory contains a usable MLX Whisper model."""
    return _is_valid_mlx_model_dir(Path(path).expanduser())


def managed_hf_alignment_dir(model_dir: str | Path, repo_id: str) -> Path:
    """Return the managed local snapshot path for a Hugging Face aligner."""
    owner, separator, name = repo_id.partition("/")
    if not separator or not owner or not name:
        raise ValueError(f"Invalid Hugging Face alignment model id: {repo_id}")
    safe_owner = re.sub(r"[^A-Za-z0-9._-]+", "-", owner)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    return Path(model_dir).expanduser() / f"alignment--{safe_owner}--{safe_name}"


def is_hf_alignment_model_dir(path: str | Path) -> bool:
    """Return whether a local Hugging Face CTC aligner snapshot is usable."""
    model_path = Path(path).expanduser()
    has_weights = any(
        (model_path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
    )
    return (
        model_path.is_dir()
        and (model_path / "config.json").is_file()
        and (model_path / "preprocessor_config.json").is_file()
        and (model_path / "vocab.json").is_file()
        and has_weights
    )


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


def _normalize_align_model(model: str | None) -> str:
    """Map the downloaded torchaudio weight file to its pipeline identifier."""
    value = (model or "").strip()
    if value and Path(value).name.lower() == DEFAULT_EN_ALIGN_FILENAME:
        return DEFAULT_EN_ALIGN_MODEL
    return value


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


_SMALL_NUMBERS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_UNIT_NAMES = {
    "hp": "horsepower",
    "mph": "miles per hour",
    "kph": "kilometers per hour",
    "kmh": "kilometers per hour",
    "kg": "kilograms",
    "lb": "pounds",
    "lbs": "pounds",
    "ft": "feet",
    "nm": "newton meters",
    "rpm": "R P M",
}


@dataclass(frozen=True)
class _AlignmentToken:
    display_text: str
    spoken_word_count: int


@dataclass(frozen=True)
class _AlignmentSegmentPlan:
    text: str
    start: float
    end: float
    tokens: tuple[_AlignmentToken, ...]


def _integer_to_english(value: int) -> str:
    if value < 0:
        return f"minus {_integer_to_english(-value)}"
    if value < 20:
        return _SMALL_NUMBERS[value]
    if value < 100:
        tens, remainder = divmod(value, 10)
        return _TENS[tens] if not remainder else f"{_TENS[tens]} {_SMALL_NUMBERS[remainder]}"
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        result = f"{_SMALL_NUMBERS[hundreds]} hundred"
        return result if not remainder else f"{result} {_integer_to_english(remainder)}"
    for scale, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")):
        if value >= scale:
            leading, remainder = divmod(value, scale)
            result = f"{_integer_to_english(leading)} {name}"
            return result if not remainder else f"{result} {_integer_to_english(remainder)}"
    return " ".join(_SMALL_NUMBERS[int(digit)] for digit in str(value))


def _number_to_english(value: str, *, allow_year: bool = True) -> str | None:
    compact = value.replace(",", "")
    if not re.fullmatch(r"\d+(?:\.\d+)?", compact):
        return None
    if "." in compact:
        whole, fraction = compact.split(".", 1)
        return f"{_integer_to_english(int(whole))} point {' '.join(_SMALL_NUMBERS[int(d)] for d in fraction)}"
    number = int(compact)
    if allow_year and len(compact) == 4 and 1900 <= number <= 2099:
        first, second = divmod(number, 100)
        if second == 0:
            return f"{_integer_to_english(first)} hundred"
        if number < 2000:
            return f"{_integer_to_english(first)} {_integer_to_english(second)}"
        if number < 2010:
            return f"two thousand {_integer_to_english(second)}"
        return f"twenty {_integer_to_english(second)}"
    return _integer_to_english(number)


def _spoken_token(token: str) -> str:
    """Return deterministic English speech text for one display token."""
    leading = re.match(r"^[\(\[\{\"']*", token).group(0)  # type: ignore[union-attr]
    trailing_match = re.search(r"[\)\]\}\"',.!?;:]*$", token)
    trailing = trailing_match.group(0) if trailing_match else ""
    core_end = len(token) - len(trailing) if trailing else len(token)
    core = token[len(leading) : core_end]
    if not core:
        return token

    punctuation = next((ch for ch in reversed(trailing) if ch in ".!?"), "")

    def finish(text: str) -> str:
        return f"{text}{punctuation}" if punctuation else text

    if core == "&":
        return finish("and")
    if core == "+":
        return finish("plus")

    currency = re.fullmatch(r"([\$£€])(\d[\d,]*(?:\.\d+)?)", core)
    if currency:
        amount = _number_to_english(currency.group(2), allow_year=False)
        names = {"$": "dollars", "£": "pounds", "€": "euros"}
        return finish(f"{amount} {names[currency.group(1)]}")

    percent = re.fullmatch(r"(\d[\d,]*(?:\.\d+)?)%", core)
    if percent:
        return finish(f"{_number_to_english(percent.group(1), allow_year=False)} percent")

    number_range = re.fullmatch(r"(\d+(?:\.\d+)?)[-–](\d+(?:\.\d+)?)", core)
    if number_range:
        left = _number_to_english(number_range.group(1), allow_year=False)
        right = _number_to_english(number_range.group(2), allow_year=False)
        return finish(f"{left} to {right}")

    number_unit = re.fullmatch(r"(\d[\d,]*(?:\.\d+)?)([A-Za-z]+)", core)
    if number_unit:
        number = _number_to_english(number_unit.group(1), allow_year=False)
        unit = number_unit.group(2)
        spoken_unit = _UNIT_NAMES.get(unit.lower())
        if spoken_unit:
            return finish(f"{number} {spoken_unit}")
        if len(unit) <= 3:
            return finish(f"{number} {' '.join(unit.upper())}")

    model_name = re.fullmatch(r"([A-Za-z]{1,3})(\d+)", core)
    if model_name:
        letters = " ".join(model_name.group(1).upper())
        number = _number_to_english(model_name.group(2), allow_year=False)
        return finish(f"{letters} {number}")

    number = _number_to_english(core)
    if number:
        return finish(number)

    if core.isupper() and core.isalpha() and 3 <= len(core) <= 6:
        return finish(" ".join(core))
    return token


def _prepare_spoken_alignment(
    segments: list[dict], language_code: str
) -> tuple[list[dict], list[_AlignmentSegmentPlan] | None]:
    if language_code != "en":
        return segments, None

    normalized: list[dict] = []
    plans: list[_AlignmentSegmentPlan] = []
    changed = False
    for segment in segments:
        display_tokens = re.findall(r"\S+", segment["text"])
        spoken_tokens = [_spoken_token(token) for token in display_tokens]
        changed |= spoken_tokens != display_tokens
        plan_tokens = tuple(
            _AlignmentToken(display, max(1, len(spoken.split())))
            for display, spoken in zip(display_tokens, spoken_tokens)
        )
        normalized.append({**segment, "text": " ".join(spoken_tokens)})
        plans.append(
            _AlignmentSegmentPlan(
                text=segment["text"],
                start=float(segment["start"]),
                end=float(segment["end"]),
                tokens=plan_tokens,
            )
        )
    return (normalized, plans) if changed else (segments, None)


def _restore_display_alignment(aligned: dict, plans: list[_AlignmentSegmentPlan]) -> dict | None:
    spoken_words = [
        word
        for segment in aligned.get("segments") or []
        if isinstance(segment, dict)
        for word in segment.get("words") or []
        if isinstance(word, dict)
    ]
    expected = sum(token.spoken_word_count for plan in plans for token in plan.tokens)
    if len(spoken_words) != expected:
        return None

    word_index = 0
    restored_segments: list[dict] = []
    restored_words: list[dict] = []
    for plan in plans:
        words: list[dict] = []
        for token in plan.tokens:
            group = spoken_words[word_index : word_index + token.spoken_word_count]
            word_index += token.spoken_word_count
            restored: dict[str, Any] = {"word": token.display_text}
            starts = [_float_seconds(word.get("start")) for word in group]
            ends = [_float_seconds(word.get("end")) for word in group]
            valid_starts = [value for value in starts if value is not None]
            valid_ends = [value for value in ends if value is not None]
            if len(valid_starts) == len(group) and len(valid_ends) == len(group):
                restored["start"] = min(valid_starts)
                restored["end"] = max(valid_ends)
            scores = [
                float(score)
                for word in group
                if isinstance((score := word.get("score")), (int, float))
            ]
            if scores:
                restored["score"] = round(sum(scores) / len(scores), 3)
            words.append(restored)
            restored_words.append(restored)

        timed_starts = [_float_seconds(word.get("start")) for word in words]
        timed_ends = [_float_seconds(word.get("end")) for word in words]
        starts = [value for value in timed_starts if value is not None]
        ends = [value for value in timed_ends if value is not None]
        restored_segments.append(
            {
                "text": plan.text,
                "start": min(starts) if starts else plan.start,
                "end": max(ends) if ends else plan.end,
                "words": words,
            }
        )

    result = dict(aligned)
    result["segments"] = restored_segments
    result["word_segments"] = restored_words
    return result


def install_whisperx_runtime_stubs() -> None:
    """Avoid importing WhisperX features that SubForge does not execute.

    Older WhisperX releases eagerly import transcribe/diarize, while frozen
    builds can still probe those modules during collection. SubForge uses MLX
    Whisper for transcription and only needs WhisperX audio/alignment.
    """

    def _unsupported(*_args, **_kwargs):
        raise RuntimeError("This WhisperX feature is not bundled in SubForge")

    if "whisperx.transcribe" not in sys.modules:
        transcribe = types.ModuleType("whisperx.transcribe")
        setattr(transcribe, "load_model", _unsupported)
        sys.modules["whisperx.transcribe"] = transcribe

    if "whisperx.diarize" not in sys.modules:
        diarize = types.ModuleType("whisperx.diarize")
        setattr(diarize, "assign_word_speakers", _unsupported)
        setattr(diarize, "DiarizationPipeline", _unsupported)
        sys.modules["whisperx.diarize"] = diarize


class _OfflineSentenceTokenizer:
    """Small Punkt fallback used when the packaged app has no NLTK data."""

    def span_tokenize(self, text: str):
        start = 0
        for match in re.finditer(r"[.!?。！？]+(?=\s+|$)", text):
            end = match.end()
            if end > start:
                yield start, end
            start = end
            while start < len(text) and text[start].isspace():
                start += 1
        if start < len(text):
            yield start, len(text)


def _install_offline_sentence_tokenizer(alignment_module: Any) -> None:
    """Prevent WhisperX 3.8+ from downloading Punkt data at runtime."""
    if getattr(alignment_module, "_subforge_offline_tokenizer_installed", False):
        return
    nltk_load = getattr(alignment_module, "nltk_load", None)
    if not callable(nltk_load):
        return

    fallback_tokenizer: _OfflineSentenceTokenizer | None = None
    fallback_logged = False
    unavailable_resources: set[str] = set()

    def _load_or_fallback(resource: str):
        nonlocal fallback_tokenizer, fallback_logged
        if resource in unavailable_resources:
            if fallback_tokenizer is None:
                fallback_tokenizer = _OfflineSentenceTokenizer()
            return fallback_tokenizer
        try:
            return nltk_load(resource)
        except LookupError:
            unavailable_resources.add(resource)
            if not fallback_logged:
                logger.warning("NLTK Punkt data is unavailable; using offline sentence boundaries")
                fallback_logged = True
            if fallback_tokenizer is None:
                fallback_tokenizer = _OfflineSentenceTokenizer()
            return fallback_tokenizer

    alignment_module.nltk_load = _load_or_fallback
    alignment_module._subforge_offline_tokenizer_installed = True


def _float_seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    return None


def _word_text(word: dict) -> str:
    return str(word.get("word") or word.get("text") or "").strip()


def _word_has_timing(word: dict) -> bool:
    return (
        _float_seconds(word.get("start")) is not None
        and _float_seconds(word.get("end")) is not None
    )


def _word_duration_weight(text: str) -> int:
    alnum = sum(1 for ch in text if ch.isalnum())
    return max(1, alnum)


def _refine_words_with_char_alignments(words: list[dict], chars: list[dict] | None) -> list[dict]:
    """Use WhisperX character timings to tighten aligned word boundaries.

    WhisperX already derives words from characters, but older releases can
    leave a word partially timed when punctuation or an unsupported character
    is involved. Character data is treated as supporting evidence only: text
    order and existing timings are preserved when an exact sequential match is
    unavailable.
    """
    if not chars:
        return words

    timed_chars = [char for char in chars if isinstance(char, dict) and str(char.get("char") or "")]
    uses_space_delimiters = any(str(char.get("char") or "").isspace() for char in timed_chars)
    char_index = 0
    refined: list[dict] = []

    for original in words:
        word = dict(original)
        target = "".join(ch.lower() for ch in _word_text(word) if not ch.isspace())
        if not target:
            refined.append(word)
            continue

        matched: list[dict] = []
        target_index = 0
        scan_index = char_index
        while scan_index < len(timed_chars):
            value = str(timed_chars[scan_index].get("char") or "")
            if not value.isspace():
                break
            scan_index += 1

        if uses_space_delimiters:
            while scan_index < len(timed_chars):
                item = timed_chars[scan_index]
                value = str(item.get("char") or "")
                if value.isspace():
                    break
                matched.append(item)
                scan_index += 1
            candidate = "".join(str(item.get("char") or "").lower() for item in matched)
            target_index = len(target) if candidate == target else 0
            char_index = scan_index
        else:
            while scan_index < len(timed_chars) and target_index < len(target):
                item = timed_chars[scan_index]
                value = str(item.get("char") or "")
                scan_index += 1
                if value.lower() != target[target_index]:
                    matched = []
                    break
                matched.append(item)
                target_index += 1

        if target_index == len(target) and matched:
            starts = [_float_seconds(item.get("start")) for item in matched]
            ends = [_float_seconds(item.get("end")) for item in matched]
            valid_starts = [value for value in starts if value is not None]
            valid_ends = [value for value in ends if value is not None]
            if valid_starts:
                word["start"] = min(valid_starts)
            if valid_ends:
                word["end"] = max(valid_ends)
            char_index = scan_index

        refined.append(word)

    return refined


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
        output.append(
            _make_word_segment(
                text,
                start_ms,
                end_ms,
                timing_source="estimated",
            )
        )
        current = end


def _make_word_segment(
    text: str,
    start_ms: int,
    end_ms: int,
    *,
    timing_source: TimestampSource,
    confidence: float | None = None,
) -> ASRDataSeg:
    word = ASRWord(
        text=text,
        start_time=start_ms,
        end_time=end_ms,
        confidence=confidence,
        alignment_score=confidence if timing_source == "forced_alignment" else None,
        timing_source=timing_source,
    )
    return ASRDataSeg(
        text,
        start_ms,
        end_ms,
        words=[word],
        timestamp_granularity="word",
        timing_source=timing_source,
    )


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
            score = word.get("score")
            confidence = float(score) if isinstance(score, (int, float)) else None
            output.append(
                _make_word_segment(
                    text,
                    start_ms,
                    end_ms,
                    timing_source="forced_alignment",
                    confidence=confidence,
                )
            )
            last_known_end = end
        else:
            pending.append(word)

    if pending:
        if segment_end is not None:
            _append_word_run(output, pending, last_known_end, segment_end)

    return output


class WhisperXASR(BaseASR):
    """Platform-native Whisper transcription followed by WhisperX alignment.

    Apple Silicon uses MLX Whisper. Other platforms use WhisperX's
    Faster-Whisper backend so Windows can use the same forced-alignment model.
    """

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
        segment_callback: Optional[Callable[[ASRData], None]] = None,
        use_cache: bool = False,
        need_word_time_stamp: bool = True,
    ):
        super().__init__(audio_input, use_cache, need_word_time_stamp)
        self.uses_mlx = platform.system() == "Darwin" and platform.machine() == "arm64"
        self.model_dir = model_dir or str(MODEL_PATH)
        requested_model = whisper_model or (default_mlx_model() if self.uses_mlx else "large-v3")
        if not self.uses_mlx:
            managed_model = Path(self.model_dir) / f"faster-whisper-{requested_model}"
            if is_faster_whisper_model_dir(requested_model):
                requested_model = str(Path(requested_model).expanduser())
            elif is_faster_whisper_model_dir(managed_model):
                requested_model = str(managed_model)
        self.whisper_model = requested_model
        self.mlx_model = _mlx_model_repo(self.whisper_model) if self.uses_mlx else ""
        self.language = _normalize_language(language)
        self.align_device = _normalize_align_device(device)
        self.compute_type = _normalize_compute_type(self.align_device, compute_type)
        self.align_model = _normalize_align_model(align_model)
        self.batch_size = max(1, int(batch_size or 4))
        self.segment_callback = segment_callback
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
        normalized_language = _normalize_language(language_code)
        requested_model = self.align_model
        if requested_model == DEFAULT_EN_ALIGN_MODEL and normalized_language != "en":
            requested_model = DEFAULT_HF_ALIGN_MODELS.get(normalized_language or "", "")
        elif not requested_model:
            requested_model = DEFAULT_HF_ALIGN_MODELS.get(normalized_language or "", "")

        if requested_model:
            model_dir = getattr(self, "model_dir", "")
            if model_dir and "/" in requested_model:
                local_model = managed_hf_alignment_dir(model_dir, requested_model)
                if is_hf_alignment_model_dir(local_model):
                    return str(local_model)
            return requested_model
        if normalized_language == "en":
            return DEFAULT_EN_ALIGN_MODEL
        if self.align_model == DEFAULT_EN_ALIGN_MODEL:
            logger.info(
                "Ignoring English-only forced alignment model for language=%s; "
                "using WhisperX language default",
                normalized_language,
            )
        return None

    def _run(
        self,
        callback: Optional[Callable[[int, str], None]] = None,
        **kwargs: Any,
    ) -> dict:
        if self.uses_mlx:
            return self._run_mlx(callback=callback, **kwargs)
        return self._run_standard(callback=callback, **kwargs)

    def _run_standard(
        self,
        callback: Optional[Callable[[int, str], None]] = None,
        **_kwargs: Any,
    ) -> dict:
        def _default_callback(_progress: int, _message: str) -> None:
            pass

        callback = callback or _default_callback
        try:
            import whisperx.alignment as whisperx_alignment
            from whisperx.asr import load_model
            from whisperx.audio import load_audio
        except ImportError as exc:
            raise RuntimeError(
                f"WhisperX runtime import failed: {exc}. Reinstall SubForge with "
                "the WhisperX desktop runtime included."
            ) from exc

        with tempfile.TemporaryDirectory() as tmp:
            audio_path = self._write_audio_to_temp(Path(tmp))
            callback(15, "Loading WhisperX transcription model...")
            logger.info(
                "Transcribing with standard WhisperX model=%s device=%s compute=%s",
                self.whisper_model,
                self.align_device,
                self.compute_type,
            )
            model = load_model(
                self.whisper_model,
                self.align_device,
                compute_type=self.compute_type,
                language=self.language,
                vad_method="silero",
                download_root=self.model_dir,
            )
            callback(30, "Loading audio...")
            audio = load_audio(audio_path)
            callback(40, "Transcribing with WhisperX...")
            result = dict(
                model.transcribe(
                    audio,
                    batch_size=self.batch_size,
                    language=self.language,
                )
            )

            if self.segment_callback:
                raw_segments = self._make_segments(result)
                if raw_segments:
                    self.segment_callback(ASRData(raw_segments))

            language_code = str(result.get("language") or self.language or "en").lower()
            aligned = self._align_result(
                result,
                audio,
                language_code,
                callback,
                whisperx_alignment,
            )
            aligned["asr_backend"] = "faster-whisper"
            aligned["whisper_model"] = self.whisper_model

            if self.segment_callback:
                aligned_segments = self._make_segments(aligned)
                if aligned_segments:
                    self.segment_callback(ASRData(aligned_segments))

            try:
                del model
                if self.align_device == "cuda":
                    import torch

                    torch.cuda.empty_cache()
            except Exception:
                pass

            callback(*ASRStatus.COMPLETED.callback_tuple())
            return aligned

    def _align_result(
        self,
        result: dict,
        audio: Any,
        language_code: str,
        callback: Callable[[int, str], None],
        whisperx_alignment: Any,
    ) -> dict:
        align_segments = _segments_for_alignment(result)
        if not align_segments:
            raise RuntimeError("WhisperX did not return alignable transcript segments")

        spoken_align_segments, alignment_plans = _prepare_spoken_alignment(
            align_segments, language_code
        )
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

        def _load_alignment_model():
            try:
                return whisperx_alignment.load_align_model(**align_kwargs)
            except TypeError:
                fallback_kwargs = dict(align_kwargs)
                fallback_kwargs.pop("model_dir", None)
                return whisperx_alignment.load_align_model(**fallback_kwargs)

        cache_key = (
            language_code,
            align_model_name or "",
            self.align_device,
            str(self.model_dir or ""),
            id(whisperx_alignment.load_align_model),
        )
        with _ALIGNMENT_MODEL_CACHE.acquire(cache_key, _load_alignment_model) as loaded:
            model_a, metadata = loaded
            callback(78, "Running forced alignment...")

            def _align(segments_to_align: list[dict]) -> dict:
                try:
                    return dict(
                        whisperx_alignment.align(
                            segments_to_align,
                            model_a,
                            metadata,
                            audio,
                            self.align_device,
                            return_char_alignments=True,
                        )
                    )
                except TypeError:
                    return dict(
                        whisperx_alignment.align(
                            segments_to_align,
                            model_a,
                            metadata,
                            audio,
                            self.align_device,
                        )
                    )

            aligned = _align(spoken_align_segments)
            if alignment_plans:
                restored = _restore_display_alignment(aligned, alignment_plans)
                if restored is None:
                    logger.warning(
                        "Spoken alignment mapping was incomplete; retrying original text"
                    )
                    aligned = _align(align_segments)
                else:
                    aligned = restored
        aligned["language"] = language_code
        aligned["align_model"] = align_model_name or ""

        return aligned

    def _run_mlx(
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
            import whisperx.alignment as whisperx_alignment
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

        _install_offline_sentence_tokenizer(whisperx_alignment)
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

            if self.segment_callback:
                raw_segments = self._make_segments(result)
                if raw_segments:
                    self.segment_callback(ASRData(raw_segments))

            callback(35, "Loading audio...")
            audio = load_audio(audio_path)

            language_code = str(result.get("language") or self.language or "en").lower()
            aligned = self._align_result(
                result,
                audio,
                language_code,
                callback,
                whisperx_alignment,
            )
            aligned["asr_backend"] = "mlx-whisper"
            aligned["mlx_model"] = self.mlx_model

            if self.segment_callback:
                aligned_segments = self._make_segments(aligned)
                if aligned_segments:
                    self.segment_callback(ASRData(aligned_segments))

            try:
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
                chars = item.get("chars")
                word_dicts = _refine_words_with_char_alignments(
                    word_dicts,
                    chars if isinstance(chars, list) else None,
                )
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
            segments.append(
                ASRDataSeg(
                    text,
                    start_ms,
                    end_ms,
                    timestamp_granularity="sentence",
                    timing_source="native",
                )
            )

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
