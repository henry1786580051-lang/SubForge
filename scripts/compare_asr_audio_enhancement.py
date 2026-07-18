#!/usr/bin/env python3
"""Compare WhisperX transcription coverage with and without audio enhancement."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from subforge.core.asr.asr_data import ASRData
from subforge.core.asr.transcribe import transcribe
from subforge.core.entities import TranscribeConfig, TranscribeModelEnum
from subforge.core.utils.video_utils import video2audio


def _config(model: str, model_dir: str, enhancement: bool) -> TranscribeConfig:
    return TranscribeConfig(
        transcribe_model=TranscribeModelEnum.WHISPERX,
        transcribe_language="en",
        need_word_time_stamp=True,
        whisperx_model=model,
        whisperx_align_model="WAV2VEC2_ASR_LARGE_LV60K_960H",
        whisperx_batch_size=8,
        faster_whisper_model_dir=model_dir,
        faster_whisper_device="cpu",
        enable_audio_enhancement=enhancement,
        speaker_diarization="off",
    )


def _summary(data: ASRData) -> dict[str, int | float]:
    segments = data.segments
    normalized_words = [
        token.lower()
        for segment in segments
        for token in re.findall(r"[A-Za-z0-9']+", segment.text)
    ]
    return {
        "segments": len(segments),
        "words": len(normalized_words),
        "unique_words": len(set(normalized_words)),
        "first_start_ms": segments[0].start_time if segments else 0,
        "last_end_ms": segments[-1].end_time if segments else 0,
        "timed_speech_seconds": round(
            sum(max(0, segment.end_time - segment.start_time) for segment in segments) / 1000,
            3,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_path = Path(audio_file.name)
    try:
        if not video2audio(str(args.video), str(audio_path)):
            raise RuntimeError("Failed to extract test audio")
        summaries = {}
        for name, enhancement in (("original_audio", False), ("deepfilternet_audio", True)):
            print(f"\n=== {name} ===", flush=True)
            data = transcribe(
                str(audio_path),
                _config(args.model, args.model_dir, enhancement),
                lambda progress, message: print(f"[{progress:3d}%] {message}", flush=True),
            )
            output_path = args.output_dir / f"{name}.srt"
            data.save(str(output_path))
            summaries[name] = _summary(data)
            summaries[name]["path"] = str(output_path)
        report_path = args.output_dir / "comparison.json"
        report_path.write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)
    finally:
        audio_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
