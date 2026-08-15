import base64
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from openai import OpenAI

from subforge.core.llm.client import normalize_base_url

from ..utils.logger import setup_logger
from .asr_data import ASRDataSeg, ASRWord
from .base import BaseASR

logger = setup_logger("whisper_api")

# Models that use chat completions + audio_url instead of /audio/transcriptions
_AUDIO_CHAT_MODELS = {"mimo-v2-omni", "mimo-omni"}
# Max audio size for base64 encoding (200MB)
_MAX_AUDIO_BYTES = 200 * 1024 * 1024


class WhisperAPI(BaseASR):
    """OpenAI-compatible Whisper API implementation.

    Supports two modes:
    - Standard Whisper API: /v1/audio/transcriptions (OpenAI, etc.)
    - Audio chat models: /v1/chat/completions with audio_url (mimo-omni, etc.)
    """

    def __init__(
        self,
        audio_input: Union[str, bytes],
        whisper_model: str,
        need_word_time_stamp: bool = False,
        language: str = "zh",
        prompt: str = "",
        base_url: str = "",
        api_key: str = "",
        use_cache: bool = False,
    ):
        """Initialize Whisper API.

        Args:
            audio_input: Path to audio file or raw audio bytes
            whisper_model: Model name (e.g. whisper-1, mimo-v2-omni)
            need_word_time_stamp: Return word-level timestamps
            language: Language code (default: zh)
            prompt: Initial prompt for model
            base_url: API base URL
            api_key: API key
            use_cache: Enable caching
        """
        super().__init__(audio_input, use_cache)

        self.base_url = normalize_base_url(base_url)
        self.api_key = api_key.strip()

        if not self.base_url or not self.api_key:
            raise ValueError("Whisper BASE_URL and API_KEY must be set")

        self.model = whisper_model
        self.language = language
        self.prompt = prompt
        self.need_word_time_stamp = need_word_time_stamp

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self._use_audio_chat = whisper_model.lower() in _AUDIO_CHAT_MODELS

    def _run(
        self, callback: Optional[Callable[[int, str], None]] = None, **kwargs: Any
    ) -> dict:
        """Execute ASR via API."""
        if self._use_audio_chat:
            return self._submit_audio_chat()
        return self._submit_whisper()

    def _make_segments(self, resp_data: dict) -> List[ASRDataSeg]:
        """Convert API response to segments."""
        if self._use_audio_chat:
            # mimo-omni returns plain text, no timestamps
            text = resp_data.get("text", "").strip()
            if not text:
                return []
            # Use audio duration as end_time (model doesn't provide timestamps)
            duration_ms = int(self._get_audio_duration() * 1000)
            return [
                ASRDataSeg(
                    text=text,
                    start_time=0,
                    end_time=max(duration_ms, 1),
                    timestamp_granularity="sentence",
                    timing_source="estimated",
                )
            ]

        if self.need_word_time_stamp and "words" in resp_data:
            return [
                ASRDataSeg(
                    text=word["word"],
                    start_time=int(float(word["start"]) * 1000),
                    end_time=int(float(word["end"]) * 1000),
                    words=[
                        ASRWord(
                            text=word["word"],
                            start_time=int(float(word["start"]) * 1000),
                            end_time=int(float(word["end"]) * 1000),
                            confidence=(
                                float(word["confidence"])
                                if isinstance(word.get("confidence"), (int, float))
                                else None
                            ),
                            timing_source="native",
                        )
                    ],
                    timestamp_granularity="word",
                    timing_source="native",
                )
                for word in resp_data["words"]
            ]
        else:
            return [
                ASRDataSeg(
                    text=seg["text"].strip(),
                    start_time=int(float(seg["start"]) * 1000),
                    end_time=int(float(seg["end"]) * 1000),
                    timestamp_granularity="sentence",
                    timing_source="native",
                )
                for seg in resp_data["segments"]
            ]

    def _get_key(self) -> str:
        """Get cache key including model and language."""
        return f"{self.crc32_hex}-{self.model}-{self.language}-{self.prompt}"

    def _submit_whisper(self) -> dict:
        """Submit audio via standard Whisper /audio/transcriptions endpoint."""
        try:
            if self.language == "zh" and not self.prompt:
                self.prompt = "你好，我们需要使用简体中文，以下是普通话的句子"

            if not self.base_url:
                raise ValueError("Whisper BASE_URL must be set")

            api_kwargs: dict[str, Any] = {
                "model": self.model,
                "response_format": "verbose_json",
                "file": ("audio.mp3", self.file_binary or b"", "audio/mp3"),
                "prompt": self.prompt,
                "timestamp_granularities": ["word", "segment"],
            }
            if self.language:
                api_kwargs["language"] = self.language

            completion = self.client.audio.transcriptions.create(**api_kwargs)
            if isinstance(completion, str):
                raise ValueError(
                    "WhisperAPI returned type error, please check your base URL."
                )
            return completion.to_dict()
        except Exception:
            logger.exception("WhisperAPI failed")
            raise

    def _submit_audio_chat(self) -> dict:
        """Submit audio via chat completions with audio_url (for mimo-omni etc.)."""
        try:
            # Read audio bytes
            if isinstance(self.audio_input, bytes):
                audio_bytes = self.audio_input
            elif isinstance(self.audio_input, str):
                audio_bytes = Path(self.audio_input).read_bytes()
            else:
                raise ValueError(f"Invalid audio input type: {type(self.audio_input)}")

            if len(audio_bytes) > _MAX_AUDIO_BYTES:
                raise ValueError(
                    f"Audio file too large ({len(audio_bytes) / 1024 / 1024:.0f}MB). "
                    f"Max for base64 encoding: {_MAX_AUDIO_BYTES // 1024 // 1024}MB"
                )
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            # Detect format
            if isinstance(self.audio_input, str):
                ext = Path(self.audio_input).suffix.lower().lstrip(".")
                fmt = {"wav": "wav", "mp3": "mp3", "m4a": "m4a", "flac": "flac", "ogg": "ogg"}.get(ext, "wav")
            else:
                fmt = "wav"

            # Build language-aware prompt
            lang_hint = {
                "zh": "请逐字转录这段音频，使用简体中文。",
                "en": "Please transcribe this audio verbatim in English.",
                "ja": "この音声を逐字的に書き起こしてください。",
                "auto": "Please transcribe this audio verbatim in the original language.",
            }.get(self.language, "Please transcribe this audio verbatim.")

            system_prompt = "You are a professional audio transcription assistant. Output ONLY the transcription text, nothing else."
            user_prompt = lang_hint
            if self.prompt:
                user_prompt += f"\n\nContext: {self.prompt}"

            messages: Any = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/{fmt};base64,{audio_b64}"},
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
            )

            # Check content first, then reasoning_content (mimo-omni puts transcription in reasoning)
            text = response.choices[0].message.content or ""
            if not text:
                text = getattr(response.choices[0].message, "reasoning_content", "") or ""
            text = text.strip()

            if not text:
                raise ValueError("Audio chat model returned empty transcription")

            logger.info(f"Audio chat transcription ({self.model}): {text[:100]}...")
            return {"text": text}

        except Exception:
            logger.exception("Audio chat ASR failed")
            raise
