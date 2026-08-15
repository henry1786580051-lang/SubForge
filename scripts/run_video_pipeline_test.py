#!/usr/bin/env python3
# ruff: noqa: E402
"""Run SubForge transcription + subtitle processing for one local video."""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    config["subtitle"]["max_word_count_cjk"] = int(
        getattr(args, "max_cjk", None) or 16
    )
    config["subtitle"]["max_word_count_english"] = int(
        getattr(args, "max_english", None) or 18
    )
    config["subtitle"]["max_chars_en"] = 42
    config["subtitle"]["max_chars_cjk"] = 16

    config["translate"]["service"] = "llm"
    config["translate"]["target_language"] = "zh-Hans"
    config["translate"]["reflect"] = bool(getattr(args, "reflect", False))
    config["subtitle"]["layout"] = "target-above"

    # Keep this test runner provider-neutral. Provider-specific environment
    # variables previously made MiMo silently override an explicitly requested
    # DeepSeek run.
    api_key = os.environ.get("SUBFORGE_TEST_LLM_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    api_base = args.llm_api_base or os.environ.get(
        "SUBFORGE_TEST_LLM_BASE_URL"
    ) or os.environ.get("OPENAI_BASE_URL")
    model = args.llm_model or os.environ.get("SUBFORGE_TEST_LLM_MODEL") or os.environ.get(
        "OPENAI_MODEL"
    )
    missing = [
        name
        for name, value in (
            ("API key", api_key),
            ("Base URL", api_base),
            ("model", model),
        )
        if not str(value or "").strip()
    ]
    if missing:
        raise ValueError(
            "Pipeline tests require an explicit LLM runtime; missing " + ", ".join(missing)
        )
    config["llm"]["api_key"] = api_key
    config["llm"]["api_base"] = api_base
    config["llm"]["model"] = model

    from subforge.settings import (
        LlmRuntimeConfig,
        detect_llm_provider,
        validate_llm_runtime_config,
    )

    validate_llm_runtime_config(
        LlmRuntimeConfig(
            provider=detect_llm_provider(config["llm"]["api_base"]),
            base_url=config["llm"]["api_base"],
            api_key=config["llm"]["api_key"],
            model=config["llm"]["model"],
        )
    )
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
    parser.add_argument(
        "--max-english",
        type=int,
        default=18,
        help="English soft subtitle limit; the hard limit is four words higher",
    )
    parser.add_argument("--max-cjk", type=int, default=16)
    parser.add_argument("--llm-api-base", help="Explicit LLM service URL for this test")
    parser.add_argument("--llm-model", help="Explicit LLM model for this test")
    parser.add_argument("--reflect", action="store_true", help="Enable translation review")
    parser.add_argument("--skip-transcribe", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Processed SRT path; defaults to the media directory",
    )
    args = parser.parse_args()

    # Keep symlink paths intact so an isolated test fixture cannot silently write
    # its result beside the original media file.
    video = Path(os.path.abspath(os.path.expanduser(str(args.video))))
    if not video.exists():
        raise FileNotFoundError(video)

    raw_srt = video.with_suffix(".srt")
    processed_srt = (
        Path(os.path.abspath(os.path.expanduser(str(args.output))))
        if args.output
        else video.with_name(f"{video.stem}_processed.srt")
    )
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
