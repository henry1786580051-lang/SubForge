"""LLM-based speaker diarization using multimodal audio understanding."""

import base64
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import json_repair

from ..asr.asr_data import ASRData, ASRDataSeg
from ..llm import call_llm
from ..llm.client import get_llm_client
from ..prompts import get_prompt
from ..utils.logger import setup_logger

logger = setup_logger("speaker_diarizer")

MAX_STEPS = 3


class SpeakerDiarizer:
    """Speaker diarization using multimodal LLM with audio understanding.

    Sends audio chunks + transcription text to a multimodal LLM (e.g. mimo-v2-omni)
    to identify different speakers based on voice characteristics.
    """

    def __init__(
        self,
        model: str = "mimo-v2-omni",
        batch_num: int = 25,
        thread_num: int = 2,
        audio_path: Optional[str] = None,
        update_callback: Optional[Callable] = None,
    ):
        self.model = model
        self.batch_num = batch_num
        self.thread_num = thread_num
        self.audio_path = audio_path
        self.update_callback = update_callback
        self._audio = None

    def _load_audio(self):
        """Lazy load the audio file."""
        if self._audio is None and self.audio_path:
            from pydub import AudioSegment
            self._audio = AudioSegment.from_file(self.audio_path)
        return self._audio

    def _extract_audio_chunk(self, start_ms: int, end_ms: int) -> Optional[bytes]:
        """Extract a chunk of audio as WAV bytes with padding."""
        audio = self._load_audio()
        if audio is None:
            return None

        # Add 500ms padding on each side for context
        padded_start = max(0, start_ms - 500)
        padded_end = min(len(audio), end_ms + 500)
        chunk = audio[padded_start:padded_end]

        # Export as WAV mono 16kHz for smaller size
        chunk = chunk.set_channels(1).set_frame_rate(16000)
        buf = io.BytesIO()
        chunk.export(buf, format="wav")
        return buf.getvalue()

    def diarize(self, asr_data: ASRData) -> ASRData:
        """Run speaker diarization on transcribed ASRData.

        Args:
            asr_data: Transcribed ASRData with segment timestamps

        Returns:
            ASRData with speaker_id set on each segment
        """
        if not asr_data.segments:
            return asr_data

        # Build segment info list
        seg_infos = []
        for i, seg in enumerate(asr_data.segments, 1):
            seg_infos.append({
                "index": i,
                "start_ms": seg.start_time,
                "end_ms": seg.end_time,
                "text": seg.text,
            })

        # Split into batches
        batches = self._make_batches(seg_infos)
        logger.info(f"Diarizing {len(seg_infos)} segments in {len(batches)} batches")

        # Process batches with thread pool
        speaker_map: Dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=self.thread_num) as executor:
            future_to_batch = {}
            for batch_idx, batch in enumerate(batches):
                future = executor.submit(self._diarize_batch, batch, batch_idx)
                future_to_batch[future] = batch_idx

            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    result = future.result()
                    # Only keep core segment assignments (exclude overlap zones)
                    core_start, core_end = batches[batch_idx]["core_range"]
                    for idx, speaker in result.items():
                        if core_start <= idx <= core_end:
                            speaker_map[idx] = speaker
                except Exception as e:
                    logger.error(f"Batch {batch_idx} diarization failed: {e}")

                if self.update_callback:
                    self.update_callback(batch_idx + 1, len(batches))

        # Apply speaker_id to segments
        for i, seg in enumerate(asr_data.segments, 1):
            seg.speaker_id = speaker_map.get(i, "")

        # Normalize speaker labels to sequential numbers
        self._normalize_speakers(asr_data.segments)

        logger.info(f"Diarization complete: {len(set(speaker_map.values()))} speakers identified")
        return asr_data

    def _make_batches(self, seg_infos: List[dict]) -> List[dict]:
        """Split segments into overlapping batches.

        Each batch has a core zone and overlap zones on both sides.
        Overlap ensures speaker transitions at boundaries are captured.
        """
        overlap = 5
        batches = []
        n = len(seg_infos)

        for start in range(0, n, self.batch_num):
            end = min(start + self.batch_num, n)

            # Extend with overlap on both sides
            batch_start = max(0, start - overlap)
            batch_end = min(n, end + overlap)

            # Core zone (what we actually keep from this batch)
            core_start_idx = start
            core_end_idx = end - 1

            batch_segs = seg_infos[batch_start:batch_end]
            batches.append({
                "segments": batch_segs,
                "core_range": (core_start_idx + 1, core_end_idx + 1),  # 1-indexed
                "audio_start_ms": seg_infos[batch_start]["start_ms"],
                "audio_end_ms": seg_infos[batch_end - 1]["end_ms"],
            })

        return batches

    def _diarize_batch(self, batch: dict, batch_idx: int) -> Dict[int, str]:
        """Diarize a single batch using multimodal LLM."""
        prompt = get_prompt("diarization/speaker")

        # Build segment text for the prompt
        seg_text_parts = []
        for seg in batch["segments"]:
            seg_text_parts.append(f'{seg["index"]}: [{seg["start_ms"]}ms - {seg["end_ms"]}ms] {seg["text"]}')
        seg_text = "\n".join(seg_text_parts)

        # Try audio-based diarization first
        audio_chunk = self._extract_audio_chunk(batch["audio_start_ms"], batch["audio_end_ms"])

        if audio_chunk:
            return self._call_with_audio(prompt, seg_text, audio_chunk, batch["segments"])
        else:
            # Fallback to text-only if audio extraction fails
            return self._call_text_only(prompt, seg_text, batch["segments"])

    def _call_with_audio(
        self, system_prompt: str, seg_text: str, audio_bytes: bytes, segments: List[dict]
    ) -> Dict[int, str]:
        """Call multimodal LLM with audio input for speaker identification."""
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav",
                        },
                    },
                    {
                        "type": "text",
                        "text": f"以下是这段音频对应的转录文本（共 {len(segments)} 段），请识别每段的说话人：\n\n{seg_text}",
                    },
                ],
            },
        ]

        for step in range(MAX_STEPS):
            try:
                client = get_llm_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # pyright: ignore[reportArgumentType]
                    temperature=0.3,
                )
                content = response.choices[0].message.content.strip()
                result = json_repair.loads(content)

                if isinstance(result, dict):
                    # Validate: all segment indices must be present
                    expected = {str(seg["index"]) for seg in segments}
                    actual = set(result.keys())
                    if expected.issubset(actual):
                        return {int(k): str(v) for k, v in result.items() if k.isdigit()}
                    else:
                        missing = expected - actual
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": f"缺少以下段落的说话人: {sorted(missing, key=int)}。请补充完整，输出包含所有 {len(segments)} 个段落的 JSON。",
                        })
                else:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": f"输出必须是 JSON 字典格式，如 {{\"1\": \"说话人1\", \"2\": \"说话人2\"}}。请重新输出。",
                    })
            except Exception as e:
                logger.warning(f"Audio diarization step {step + 1} failed: {e}")
                if step == MAX_STEPS - 1:
                    raise

        return {}

    def _call_text_only(
        self, system_prompt: str, seg_text: str, segments: List[dict]
    ) -> Dict[int, str]:
        """Fallback: call LLM with text only (no audio)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"以下是转录文本（共 {len(segments)} 段），请根据对话模式推断每段的说话人：\n\n{seg_text}"},
        ]

        for step in range(MAX_STEPS):
            try:
                response = call_llm(messages=messages, model=self.model, temperature=0.3)
                content = response.choices[0].message.content.strip()
                result = json_repair.loads(content)

                if isinstance(result, dict):
                    expected = {str(seg["index"]) for seg in segments}
                    actual = set(result.keys())
                    if expected.issubset(actual):
                        return {int(k): str(v) for k, v in result.items() if k.isdigit()}
                    else:
                        missing = expected - actual
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": f"缺少段落: {sorted(missing, key=int)}。请补充完整。",
                        })
                else:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": "输出必须是 JSON 字典格式。请重新输出。",
                    })
            except Exception as e:
                logger.warning(f"Text diarization step {step + 1} failed: {e}")
                if step == MAX_STEPS - 1:
                    raise

        return {}

    @staticmethod
    def _normalize_speakers(segments: List[ASRDataSeg]) -> None:
        """Normalize speaker labels to sequential numbers (说话人1, 说话人2, ...)."""
        # Collect unique speakers in order of appearance
        seen = {}
        counter = 1
        for seg in segments:
            if seg.speaker_id and seg.speaker_id not in seen:
                seen[seg.speaker_id] = f"说话人{counter}"
                counter += 1

        # Apply normalized labels
        for seg in segments:
            if seg.speaker_id in seen:
                seg.speaker_id = seen[seg.speaker_id]
