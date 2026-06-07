#!/usr/bin/env python3
"""Run SubForge transcription + subtitle processing for one local video."""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

from subforge.cli import exit_codes as EXIT
from subforge.cli.commands import subtitle as subtitle_cmd
from subforge.cli.commands import transcribe as transcribe_cmd
from subforge.cli.config import DEFAULTS


def _build_config(args: argparse.Namespace) -> dict:
    config = deepcopy(DEFAULTS)
    config.setdefault("output", {})["format"] = "srt"
    config["transcribe"]["asr"] = "whisperx"
    config["transcribe"]["language"] = args.language
    config["transcribe"]["audio_enhancement"] = True
    config["transcribe"].setdefault("whisperx", {})
    config["transcribe"]["whisperx"]["align_model"] = "WAV2VEC2_ASR_LARGE_LV60K_960H"
    config["transcribe"]["whisperx"]["batch_size"] = args.whisper_batch_size

    config["subtitle"]["optimize"] = True
    config["subtitle"]["translate"] = True
    config["subtitle"]["split"] = True
    config["subtitle"]["thread_num"] = args.llm_threads
    config["subtitle"]["batch_size"] = args.llm_batch_size
    config["subtitle"]["max_word_count_cjk"] = 18
    config["subtitle"]["max_word_count_english"] = 12
    config["subtitle"]["max_chars_en"] = 42
    config["subtitle"]["max_chars_cjk"] = 16

    config["translate"]["service"] = "llm"
    config["translate"]["target_language"] = "zh-Hans"
    config["translate"]["reflect"] = False
    config["synthesize"]["layout"] = "target-above"

    api_key = os.environ.get("MIMO_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("MIMO_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("MIMO_MODEL") or os.environ.get("OPENAI_MODEL")
    if api_key:
        config["llm"]["api_key"] = api_key
    if api_base:
        config["llm"]["api_base"] = api_base
    if model:
        config["llm"]["model"] = model
    return config


def _assert_output(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Expected output was not created: {path}")


def _summarize_srt(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = [block for block in text.replace("\r\n", "\n").split("\n\n") if "-->" in block]
    bilingual = 0
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        subtitle_lines = [line for line in lines[2:] if not line.isdigit()]
        if len(subtitle_lines) >= 2:
            bilingual += 1
    return len(blocks), bilingual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--language", default="en")
    parser.add_argument("--whisper-batch-size", type=int, default=4)
    parser.add_argument("--llm-threads", type=int, default=2)
    parser.add_argument("--llm-batch-size", type=int, default=12)
    parser.add_argument("--skip-transcribe", action="store_true")
    args = parser.parse_args()

    video = args.video.expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(video)

    raw_srt = video.with_suffix(".srt")
    processed_srt = video.with_name(f"{video.stem}_processed.srt")
    config = _build_config(args)

    if not args.skip_transcribe:
        transcribe_args = Namespace(
            input=str(video),
            output=str(raw_srt),
            format="srt",
            verbose=True,
            quiet=False,
            word_timestamps=True,
        )
        ret = transcribe_cmd.run(transcribe_args, config)
        if ret != EXIT.SUCCESS:
            return ret
    _assert_output(raw_srt)

    subtitle_args = Namespace(
        input=str(raw_srt),
        output=str(processed_srt),
        format="srt",
        verbose=True,
        quiet=False,
        translator="llm",
        target_language="zh-Hans",
        no_translate=False,
        layout="target-above",
        prompt=None,
        prompt_file=None,
    )
    ret = subtitle_cmd.run(subtitle_args, config)
    if ret != EXIT.SUCCESS:
        return ret
    _assert_output(processed_srt)

    raw_count, _ = _summarize_srt(raw_srt)
    processed_count, bilingual_count = _summarize_srt(processed_srt)
    print(f"RAW_SRT={raw_srt}")
    print(f"PROCESSED_SRT={processed_srt}")
    print(f"RAW_SEGMENTS={raw_count}")
    print(f"PROCESSED_SEGMENTS={processed_count}")
    print(f"BILINGUAL_SEGMENTS={bilingual_count}")
    return EXIT.SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
