#!/usr/bin/env python3
"""Run the production multi-speaker ASR path and write a compact test report."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subforge.core.asr.asr_data import ASRData  # noqa: E402
from subforge.core.asr.transcribe import transcribe  # noqa: E402
from subforge.core.entities import TranscribeConfig, TranscribeModelEnum  # noqa: E402
from subforge.core.utils.video_utils import video2audio  # noqa: E402


def _srt_summary(path: Path) -> dict[str, int | str]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    data = ASRData.from_srt(text)
    words = [
        token
        for segment in data.segments
        for token in re.findall(r"[A-Za-z0-9']+", segment.text)
    ]
    return {
        "entries": len(data.segments),
        "words": len(words),
        "first_start": data.segments[0]._ms_to_srt_time(data.segments[0].start_time)
        if data.segments
        else "",
        "last_end": data.segments[-1]._ms_to_srt_time(data.segments[-1].end_time)
        if data.segments
        else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--speakers",
        type=int,
        default=0,
        help="Known speaker count (2-10); omit or use 0 for automatic detection",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable ASR and diarization caches for performance diagnostics",
    )
    args = parser.parse_args()

    if args.no_cache:
        from subforge.core.utils.cache import disable_cache

        disable_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        audio_path = Path(temp_file.name)
    try:
        if not video2audio(str(args.video), str(audio_path)):
            raise RuntimeError("Failed to extract video audio")
        config = TranscribeConfig(
            transcribe_model=TranscribeModelEnum.WHISPERX,
            transcribe_language="en",
            need_word_time_stamp=True,
            whisperx_model=str(args.model_dir / "whisper-large-v3-fp16"),
            whisperx_align_model="WAV2VEC2_ASR_LARGE_LV60K_960H",
            whisperx_batch_size=10,
            faster_whisper_model_dir=str(args.model_dir),
            faster_whisper_device="auto",
            enable_audio_enhancement=True,
            speaker_diarization="fixed" if args.speakers else "auto",
            speaker_count=args.speakers or 2,
            diarization_model="pyannote/speaker-diarization-community-1",
            diarization_model_dir=str(args.model_dir),
        )

        result = transcribe(
            str(audio_path),
            config,
            callback=lambda progress, message: print(
                f"[{progress:3d}%] {message}", flush=True
            ),
        )
        result.save(str(args.output))
        speaker_words = Counter()
        for segment in result.segments:
            speaker_words[segment.speaker_id or "unassigned"] += max(
                1, len(re.findall(r"[A-Za-z0-9']+", segment.text))
            )
        report: dict[str, object] = {
            "output": str(args.output),
            "result": _srt_summary(args.output),
            "speaker_words": dict(speaker_words),
        }
        if args.baseline and args.baseline.is_file():
            report["baseline"] = _srt_summary(args.baseline)
        report_path = args.output.with_suffix(".report.json")
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        audio_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
