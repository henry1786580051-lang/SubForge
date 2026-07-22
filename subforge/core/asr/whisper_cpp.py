import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from ...config import BIN_PATH, BUNDLED_BIN_PATH, MODEL_PATH
from ..utils.logger import setup_logger
from ..utils.subprocess_helper import StreamReader
from .asr_data import ASRData, ASRDataSeg, reasonable_word_duration_ms
from .base import BaseASR
from .status import ASRStatus

logger = setup_logger("whisper_asr")


_TERMINAL_PUNCTUATION = ".!?"
_SOFT_PUNCTUATION = ",;:"
_PUNCTUATION = set(_TERMINAL_PUNCTUATION + _SOFT_PUNCTUATION + "。，！？；：")


def _is_special_token(text: str) -> bool:
    stripped = text.strip()
    return not stripped or (stripped.startswith("[_") and stripped.endswith("]"))


def _tokens_to_word_segments(transcription: list[dict]) -> list[ASRDataSeg]:
    """Convert whisper.cpp JSON-full tokens to word-like timed segments."""
    words: list[ASRDataSeg] = []
    current_text = ""
    current_start: int | None = None
    current_end: int | None = None

    def flush() -> None:
        nonlocal current_text, current_start, current_end
        text = current_text.strip()
        if text and current_start is not None and current_end is not None:
            words.append(
                ASRDataSeg(
                    text,
                    current_start,
                    max(current_start, current_end),
                    timestamp_granularity="word",
                    timing_source="native",
                )
            )
        current_text = ""
        current_start = None
        current_end = None

    for item in transcription:
        for token in item.get("tokens", []):
            token_text = token.get("text", "")
            if _is_special_token(token_text):
                continue

            offsets = token.get("offsets") or {}
            start = offsets.get("from")
            end = offsets.get("to")
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if end < start:
                continue

            stripped = token_text.strip()
            if not stripped:
                continue

            starts_new_word = bool(token_text[:1].isspace())
            is_punctuation = stripped in _PUNCTUATION or all(ch in _PUNCTUATION for ch in stripped)

            if current_text and starts_new_word and not is_punctuation:
                flush()

            if current_start is None:
                current_start = start
            current_text += stripped if is_punctuation else token_text
            current_end = end

    flush()
    return _cap_unreasonable_word_durations(words)


def _cap_unreasonable_word_durations(words: list[ASRDataSeg]) -> list[ASRDataSeg]:
    """Cap implausibly long whisper.cpp token spans for word-level output.

    whisper.cpp JSON-full can assign a short token the duration of a whole
    decoder span, especially near silence or music. Keep the token end, which is
    normally closest to the next token boundary, and move only the start forward.
    """
    for word in words:
        duration = word.end_time - word.start_time
        max_duration = reasonable_word_duration_ms(word.text)
        if duration <= max_duration * 1.5:
            continue
        word.start_time = max(0, word.end_time - max_duration)
        if word.words:
            word.words[0].start_time = word.start_time
    return words


def _segments_from_words(
    words: list[ASRDataSeg],
    *,
    max_words: int = 16,
    max_duration_ms: int = 6500,
) -> list[ASRDataSeg]:
    """Group timed words into readable sentence-like subtitle segments."""
    if not words:
        return []

    segments: list[ASRDataSeg] = []
    current: list[ASRDataSeg] = []

    def text_of(items: list[ASRDataSeg]) -> str:
        text = ""
        for item in items:
            value = item.text.strip()
            if not value:
                continue
            if not text:
                text = value
            elif value in _PUNCTUATION or all(ch in _PUNCTUATION for ch in value):
                text += value
            else:
                text += f" {value}"
        return re.sub(r"\s+([,.;:!?])", r"\1", text).strip()

    def flush() -> None:
        if not current:
            return
        text = text_of(current)
        if text:
            segments.append(ASRDataSeg.from_segments(current, text=text))
        current.clear()

    for word in words:
        current.append(word)
        text = text_of(current)
        duration = current[-1].end_time - current[0].start_time
        word_count = len(current)
        ended_sentence = bool(re.search(r"[.!?。！？]\s*$", text))
        ended_soft = bool(re.search(r"[,;:，；：]\s*$", text))

        if ended_sentence and (word_count >= 3 or duration >= 1200):
            flush()
        elif word_count >= max_words and (ended_soft or duration >= 3000):
            flush()
        elif duration >= max_duration_ms and word_count >= 8:
            flush()

    flush()
    return segments


def _segments_from_whisper_json(resp_data: str, need_word_time_stamp: bool) -> list[ASRDataSeg]:
    data = json.loads(resp_data)
    transcription = data.get("transcription")
    if not isinstance(transcription, list):
        return []

    words = _tokens_to_word_segments(transcription)
    if need_word_time_stamp:
        return words
    return _segments_from_words(words)


class WhisperCppASR(BaseASR):
    """Whisper.cpp local ASR implementation.

    Runs whisper.cpp binary for local ASR processing.
    """

    def __init__(
        self,
        audio_input: Union[str, bytes],
        language="en",
        whisper_cpp_path=None,
        whisper_model=None,
        model_dir: str = "",
        use_cache: bool = False,
        need_word_time_stamp: bool = False,
        n_threads: int = 4,
        use_vad: bool = True,
        segment_callback: Optional[Callable[[ASRData], None]] = None,
    ):
        super().__init__(audio_input, use_cache)

        if isinstance(audio_input, str):
            assert os.path.exists(audio_input), f"Audio file not found: {audio_input}"
            assert audio_input.endswith(
                ".wav"
            ), f"Audio must be WAV format: {audio_input}"

        # Auto-detect whisper executable if not provided
        if whisper_cpp_path is None:
            whisper_cpp_path = detect_whisper_executable()

        # Find model file in models directory
        if whisper_model:
            models_dir = Path(model_dir).expanduser() if model_dir else Path(MODEL_PATH)
            model_files = list(models_dir.glob(f"*ggml*{whisper_model}*.bin"))
            if not model_files:
                raise ValueError(
                    f"Model file not found in {models_dir} for: {whisper_model}"
                )
            model_path = str(model_files[0])
            logger.debug(f"Model found: {model_path}")
        else:
            raise ValueError("whisper_model cannot be empty")

        self.model_path = model_path
        self.whisper_cpp_path = Path(whisper_cpp_path)
        self.need_word_time_stamp = need_word_time_stamp
        self.language = language
        self.n_threads = n_threads
        self.use_vad = use_vad
        self.segment_callback = segment_callback
        self._live_segments: list[ASRDataSeg] = []

        # Find VAD model in same directory as whisper model
        self.vad_model_path = None
        if use_vad:
            vad_candidates = list(Path(model_path).parent.glob("*silero*vad*"))
            if vad_candidates:
                self.vad_model_path = str(vad_candidates[0])

        self.process = None

    @staticmethod
    def _parse_cli_segment_line(line: str) -> Optional[ASRDataSeg]:
        match = re.match(
            r"^\s*\[(\d{2}:\d{2}:\d{2}[,.]\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}[,.]\d{3})\]\s*(.*)$",
            line,
        )
        if not match:
            return None

        def to_ms(value: str) -> int:
            hh, mm, rest = value.replace(",", ".").split(":")
            ss, ms = rest.split(".")
            return (
                int(hh) * 3600 * 1000
                + int(mm) * 60 * 1000
                + int(ss) * 1000
                + int(ms)
            )

        text = match.group(3).strip()
        if not text:
            return None
        return ASRDataSeg(
            text,
            to_ms(match.group(1)),
            to_ms(match.group(2)),
            timestamp_granularity="sentence",
            timing_source="native",
        )

    def _emit_live_segment(self, line: str) -> None:
        if not self.segment_callback:
            return
        segment = self._parse_cli_segment_line(line)
        if not segment:
            return
        self._live_segments.append(segment)
        try:
            self.segment_callback(ASRData(list(self._live_segments)))
        except Exception as e:
            logger.debug("whisper.cpp live segment callback failed: %s", e, exc_info=True)

    def _make_segments(self, resp_data: str) -> List[ASRDataSeg]:
        if resp_data.lstrip().startswith("{"):
            try:
                json_segments = _segments_from_whisper_json(
                    resp_data,
                    need_word_time_stamp=self.need_word_time_stamp,
                )
                if json_segments:
                    return json_segments
            except Exception as e:
                logger.warning("Failed to parse whisper.cpp JSON output, falling back to SRT: %s", e)

        asr_data = ASRData.from_srt(resp_data)
        # 过滤掉纯音乐标记
        filtered_segments = []
        for seg in asr_data.segments:
            text = seg.text.strip()
            # 保留不以【、[、(、（开头的文本
            if not (
                text.startswith("【")
                or text.startswith("[")
                or text.startswith("(")
                or text.startswith("（")
            ):
                filtered_segments.append(seg)
                seg.timestamp_granularity = "sentence"
                seg.timing_source = "native"
        return filtered_segments

    def _build_command(
        self, wav_path, output_path, is_const_me_version: bool
    ) -> list[str]:
        """Build whisper-cpp command line arguments."""
        whisper_params = [
            str(self.whisper_cpp_path),
            "-m",
            str(self.model_path),
            "-f",
            str(wav_path),
            "-l",
            self.language or "auto",
            "--output-srt",
            "--output-json",
            "--output-json-full",
        ]

        if not is_const_me_version:
            if sys.platform != "darwin":
                whisper_params.append("--no-gpu")

            whisper_params.extend(
                ["--output-file", str(output_path.with_suffix(""))]
            )

        if self.n_threads and self.n_threads > 0:
            whisper_params.extend(["-t", str(self.n_threads)])

        if self.language == "zh":
            whisper_params.extend(
                ["--prompt", "你好，我们需要使用简体中文，以下是普通话的句子。"]
            )

        # Intentionally do not pass whisper.cpp internal VAD flags. Its
        # compacted-audio time mapping can drop or shift quiet intros; full
        # audio JSON token timestamps are more reliable for SubForge.

        return whisper_params

    def _run(
        self, callback: Optional[Callable[[int, str], None]] = None, **kwargs: Any
    ) -> str:
        def _default_callback(_progress: int, _message: str) -> None:
            pass

        if callback is None:
            callback = _default_callback

        # SubForge bundles the official whisper.cpp CLI on Windows. The old
        # Const-me port used different output arguments and is not supported.
        is_const_me_version = False

        with tempfile.TemporaryDirectory() as temp_path:
            temp_dir = Path(temp_path)
            wav_path = temp_dir / "whisper_cpp_audio.wav"
            output_path = wav_path.with_suffix(".srt")

            try:
                # 复制音频文件
                if isinstance(self.audio_input, str):
                    shutil.copy2(self.audio_input, wav_path)
                else:
                    if self.file_binary:
                        wav_path.write_bytes(self.file_binary)
                    else:
                        raise ValueError("No audio data available")

                # Build command
                whisper_params = self._build_command(
                    wav_path, output_path, is_const_me_version
                )
                logger.debug("Whisper.cpp command: %s", " ".join(whisper_params))

                # Get audio duration
                total_duration = self.audio_duration
                logger.debug("Audio duration: %d seconds", total_duration)

                # Start process
                self.process = subprocess.Popen(
                    whisper_params,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                    creationflags=(
                        getattr(subprocess, "CREATE_NO_WINDOW", 0)
                        if os.name == "nt"
                        else 0
                    ),
                )

                logger.debug(f"Whisper.cpp process started, PID: {self.process.pid}")

                # Process output with StreamReader
                reader = StreamReader(self.process)
                reader.start_reading()

                last_progress = 0

                while True:
                    # Check process status
                    if self.process.poll() is not None:
                        time.sleep(0.2)
                        for stream_name, line in reader.get_remaining_output():
                            if stream_name == "stderr":
                                logger.debug(f"[stderr] {line.strip()}")
                        break

                    # Non-blocking output reading
                    output = reader.get_output(timeout=0.1)
                    if output:
                        stream_name, line = output

                        if stream_name == "stdout":
                            logger.debug(f"[stdout] {line.strip()}")
                            self._emit_live_segment(line)

                            # Parse progress
                            if " --> " in line and "[" in line:
                                try:
                                    time_str = (
                                        line.split("[")[1].split(" -->")[0].strip()
                                    )
                                    parts = time_str.split(":")
                                    current_time = sum(
                                        float(x) * y
                                        for x, y in zip(reversed(parts), [1, 60, 3600])
                                    )
                                    progress = int(
                                        min(current_time / total_duration * 100, 98)
                                    )

                                    if progress > last_progress:
                                        last_progress = progress
                                        callback(progress, f"{progress}%")
                                except (ValueError, IndexError) as e:
                                    logger.debug(f"Progress parse failed: {e}")
                        else:
                            logger.debug(f"[stderr] {line.strip()}")

                # Check return code
                if self.process.returncode != 0:
                    raise RuntimeError(
                        f"Whisper.cpp failed with code: {self.process.returncode}"
                    )

                callback(*ASRStatus.COMPLETED.callback_tuple())
                logger.debug("Whisper.cpp ASR completed")

                # Read result file
                json_path = output_path.with_suffix(".json")
                if json_path.exists():
                    return json_path.read_text(encoding="utf-8")

                srt_path = output_path
                if not srt_path.exists():
                    time.sleep(5)
                    if json_path.exists():
                        return json_path.read_text(encoding="utf-8")
                    if not srt_path.exists():
                        raise RuntimeError(f"Output file not generated: {srt_path}")

                return srt_path.read_text(encoding="utf-8")

            except Exception as e:
                logger.exception("ASR processing failed")
                if self.process and self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
                raise RuntimeError(f"SRT generation failed: {str(e)}")

    def _get_key(self):
        effective_vad = False
        return f"{self.crc32_hex}-{self.need_word_time_stamp}-{self.model_path}-{self.language}-vad{effective_vad}"

    def get_audio_duration(self, filepath: str) -> int:
        """Get audio file duration in seconds using ffmpeg."""
        try:
            cmd = ["ffmpeg", "-i", filepath]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            info = result.stderr
            if duration_match := re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", info):
                hours, minutes, seconds = map(float, duration_match.groups())
                duration_seconds = hours * 3600 + minutes * 60 + seconds
                return int(duration_seconds)
            return 600
        except Exception as e:
            logger.exception("Failed to get audio duration: %s", str(e))
            return 600


def _whisper_executable_search_dirs() -> list[Path]:
    return [
        Path(BIN_PATH),
        Path(BUNDLED_BIN_PATH),
        Path(MODEL_PATH),
        Path.home() / "SubForge" / "bin",
        Path.home() / ".local" / "bin",
        Path("/opt/homebrew/bin"),
        Path("/usr/local/bin"),
    ]


def detect_whisper_executable() -> str:
    """Detect available whisper.cpp executable path."""
    executable_names = (
        "whisper-cli.exe",
        "whisper-cpp.exe",
        "main.exe",
        "whisper-cli",
        "whisper-cpp",
        "main",
    )

    for name in executable_names:
        if found := shutil.which(name):
            return found

    for directory in _whisper_executable_search_dirs():
        for name in executable_names:
            candidate = directory / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)

    raise RuntimeError(
        "whisper.cpp executable not found. Install or build whisper.cpp and set "
        "the executable path to 'whisper-cli' in Settings. The GGML model file "
        "alone is not enough to run local transcription."
    )


if __name__ == "__main__":
    # 简短示例
    asr = WhisperCppASR(
        audio_input="audio.mp3",
        whisper_model="tiny",
        whisper_cpp_path="bin/whisper-cpp.exe",
        language="en",
        need_word_time_stamp=True,
    )
    asr_data = asr._run(callback=print)
