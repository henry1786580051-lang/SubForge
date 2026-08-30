#!/usr/bin/env python3
# ruff: noqa: E402
"""Run the production subtitle pipeline with NVIDIA Kimi K3 only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subforge.cli import exit_codes as EXIT
from subforge.cli.commands import subtitle as subtitle_cmd
from subforge.cli.config import DEFAULTS
from subforge.config import SETTINGS_PATH
from subforge.core.llm.request_logger import LLM_LOG_FILE
from subforge.settings.credentials import restore_secret_value, usable_secret_value

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
KIMI_K3_MODEL = "moonshotai/kimi-k3"
MAX_FREE_ACCOUNT_CONCURRENCY = 5


def _log_offset() -> int:
    try:
        return LLM_LOG_FILE.stat().st_size
    except FileNotFoundError:
        return 0


def _usage_since(offset: int) -> dict[str, object]:
    try:
        current_size = LLM_LOG_FILE.stat().st_size
        with LLM_LOG_FILE.open("r", encoding="utf-8") as log_file:
            if current_size >= offset:
                log_file.seek(offset)
            entries = []
            for line in log_file:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("model") == KIMI_K3_MODEL:
                    entries.append(entry)
    except FileNotFoundError:
        entries = []

    successful = [entry for entry in entries if entry.get("status") == 200]
    prompt_tokens = sum(int(entry.get("prompt_tokens") or 0) for entry in successful)
    cached_tokens = sum(int(entry.get("cached_tokens") or 0) for entry in successful)
    return {
        "successful_requests": len(successful),
        "rate_limit_responses": sum(entry.get("status") == 429 for entry in entries),
        "timeout_responses": sum(
            entry.get("status") == 0 and "timeout" in str(entry.get("error", "")).lower()
            for entry in entries
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": sum(
            int(entry.get("completion_tokens") or 0) for entry in successful
        ),
        "reasoning_tokens": sum(
            int(entry.get("reasoning_tokens") or 0) for entry in successful
        ),
        "cached_tokens": cached_tokens,
        "cache_hit_rate": round(cached_tokens / prompt_tokens, 4) if prompt_tokens else 0.0,
    }


def _nvidia_api_key() -> str:
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    profile = settings.get("llm_profiles", {}).get("nvidia", {})
    api_key = usable_secret_value(restore_secret_value(profile.get("api_key", "")))
    if not api_key:
        raise RuntimeError("The saved NVIDIA API key is unavailable")
    return api_key


def _build_config(*, threads: int, batch_size: int, optimize: bool) -> dict:
    if not 1 <= threads <= MAX_FREE_ACCOUNT_CONCURRENCY:
        raise ValueError(
            f"Kimi K3 benchmark concurrency must be between 1 and "
            f"{MAX_FREE_ACCOUNT_CONCURRENCY}"
        )
    if batch_size < 1:
        raise ValueError("batch size must be positive")

    config = deepcopy(DEFAULTS)
    config["subtitle"]["split"] = True
    config["subtitle"]["optimize"] = optimize
    config["subtitle"]["translate"] = True
    config["subtitle"]["thread_num"] = threads
    config["subtitle"]["batch_size"] = batch_size
    config["subtitle"]["max_word_count_cjk"] = 16
    config["subtitle"]["max_word_count_english"] = 18
    config["subtitle"]["layout"] = "target-above"
    config["translate"]["service"] = "llm"
    config["translate"]["target_language"] = "zh-Hans"
    config["translate"]["reflect"] = True
    config["llm"]["api_key"] = _nvidia_api_key()
    config["llm"]["api_base"] = NVIDIA_BASE_URL
    config["llm"]["model"] = KIMI_K3_MODEL
    config["output"]["format"] = "srt"
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=MAX_FREE_ACCOUNT_CONCURRENCY)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--translate-only",
        action="store_true",
        help="Keep existing sentence boundaries and skip the optimize stage",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = _build_config(
        threads=args.threads,
        batch_size=args.batch_size,
        optimize=not args.translate_only,
    )
    started = time.monotonic()
    log_offset = _log_offset()
    result = subtitle_cmd.run(
        Namespace(
            input=str(input_path),
            output=str(output_path),
            format="srt",
            verbose=True,
            quiet=False,
            translator="llm",
            target_language="zh-Hans",
            no_translate=False,
            layout="target-above",
            prompt=None,
            prompt_file=None,
        ),
        config,
    )
    elapsed = time.monotonic() - started
    print(
        json.dumps(
            {
                "model": KIMI_K3_MODEL,
                "threads": args.threads,
                "batch_size": args.batch_size,
                "elapsed_seconds": round(elapsed, 2),
                "usage": _usage_since(log_offset),
                "output": str(output_path),
                "exit_code": result,
            },
            ensure_ascii=False,
        )
    )
    return result if result != EXIT.SUCCESS else 0


if __name__ == "__main__":
    raise SystemExit(main())
