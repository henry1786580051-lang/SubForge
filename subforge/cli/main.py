"""SubForge command-line interface."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from subforge.cli import exit_codes as EXIT


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", metavar="FILE", help="Path to config file")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="Only print results")


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output", metavar="PATH", help="Output file or directory")
    parser.add_argument(
        "--format",
        choices=["srt", "ass", "txt", "json"],
        help="Output subtitle format (default: srt)",
    )


def _add_llm_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("LLM options")
    group.add_argument("--api-key", metavar="KEY", help="LLM API key")
    group.add_argument("--api-base", metavar="URL", help="LLM API base URL")
    group.add_argument("--model", metavar="NAME", help="LLM model name")


def _build_transcribe_parser(subparsers) -> None:
    parser = subparsers.add_parser("transcribe", help="Transcribe audio/video")
    parser.add_argument("input", help="Audio or video file path")
    _add_common_options(parser)
    _add_output_options(parser)
    group = parser.add_argument_group("ASR options")
    group.add_argument(
        "--asr",
        choices=["whisper-api", "whisper-cpp", "faster-whisper", "whisperx"],
        help="ASR engine",
    )
    group.add_argument("--language", metavar="CODE", help="Source language or auto")
    group.add_argument("--word-timestamps", action="store_true")
    group.add_argument("--whisper-api-key", metavar="KEY")
    group.add_argument("--whisper-api-base", metavar="URL")
    group.add_argument("--whisper-model", metavar="NAME")
    for name in ["--fw-model", "--fw-device", "--fw-vad-method", "--fw-prompt", "--whisper-prompt"]:
        parser.add_argument(name, help=argparse.SUPPRESS)
    parser.add_argument("--fw-vad-threshold", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--fw-voice-extraction", action="store_true", help=argparse.SUPPRESS)
    parser.set_defaults(func=_run_transcribe)


def _add_subtitle_options(parser: argparse.ArgumentParser) -> None:
    _add_llm_options(parser)
    processing = parser.add_argument_group("Processing options")
    processing.add_argument("--no-optimize", action="store_true")
    processing.add_argument("--no-translate", action="store_true")
    processing.add_argument("--no-split", action="store_true")
    translation = parser.add_argument_group("Translation options")
    translation.add_argument("--translator", choices=["llm", "bing", "google"])
    translation.add_argument("--target-language", "--to", dest="target_language", metavar="CODE")
    translation.add_argument("--reflect", action="store_true")
    subtitle = parser.add_argument_group("Subtitle options")
    subtitle.add_argument("--max-cjk", type=int, metavar="N")
    subtitle.add_argument("--max-english", type=int, metavar="N")
    subtitle.add_argument("--max-chars-en", type=int, metavar="N")
    subtitle.add_argument("--max-chars-cjk", type=int, metavar="N")
    subtitle.add_argument("--prompt", metavar="TEXT")
    subtitle.add_argument("--thread-num", type=int, metavar="N")
    subtitle.add_argument("--batch-size", type=int, metavar="N")
    subtitle.add_argument(
        "--layout",
        choices=["target-above", "source-above", "target-only", "source-only"],
    )
    parser.add_argument("--prompt-file", metavar="FILE", help=argparse.SUPPRESS)


def _build_subtitle_parser(subparsers) -> None:
    parser = subparsers.add_parser("subtitle", help="Optimize and translate subtitles")
    parser.add_argument("input", help="Subtitle file path")
    _add_common_options(parser)
    _add_output_options(parser)
    _add_subtitle_options(parser)
    parser.set_defaults(func=_run_subtitle)


def _build_process_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "process",
        help="Transcribe, optimize, and translate media",
        description="Run transcription and subtitle processing without modifying the source video.",
    )
    parser.add_argument("input", help="Audio or video file path")
    _add_common_options(parser)
    _add_output_options(parser)
    _add_subtitle_options(parser)
    parser.add_argument(
        "--asr",
        choices=["whisper-api", "whisper-cpp", "faster-whisper", "whisperx"],
    )
    parser.add_argument("--language", metavar="CODE")
    parser.add_argument("--whisper-api-key", metavar="KEY")
    parser.add_argument("--whisper-api-base", metavar="URL", help=argparse.SUPPRESS)
    parser.add_argument("--whisper-model", metavar="NAME", help=argparse.SUPPRESS)
    parser.set_defaults(func=_run_process)


def _build_download_parser(subparsers) -> None:
    parser = subparsers.add_parser("download", help="Download online media")
    parser.add_argument("url", help="Media URL")
    _add_common_options(parser)
    parser.add_argument("-o", "--output", metavar="DIR")
    parser.set_defaults(func=_run_download)


def _build_config_parser(subparsers) -> None:
    parser = subparsers.add_parser("config", help="Manage CLI configuration")
    actions = parser.add_subparsers(dest="config_action", metavar="action")
    actions.add_parser("show")
    actions.add_parser("path")
    actions.add_parser("edit")
    init_parser = actions.add_parser("init")
    init_parser.add_argument("--non-interactive", action="store_true")
    init_parser.add_argument("--force", action="store_true")
    init_parser.add_argument("--print-template", action="store_true")
    init_parser.add_argument("--llm-api-key", metavar="KEY")
    init_parser.add_argument("--llm-api-base", metavar="URL")
    init_parser.add_argument("--llm-model", metavar="NAME")
    init_parser.add_argument("--asr", choices=["whisper-api", "whisper-cpp", "faster-whisper", "whisperx"])
    init_parser.add_argument("--translator", choices=["llm", "bing", "google"])
    init_parser.add_argument("--target-language", "--to", dest="target_language")
    init_parser.add_argument("--no-optimize", action="store_true")
    init_parser.add_argument("--no-split", action="store_true")
    set_parser = actions.add_parser("set")
    set_parser.add_argument("key")
    set_parser.add_argument("value")
    get_parser = actions.add_parser("get")
    get_parser.add_argument("key")
    parser.set_defaults(func=_run_config)


def _build_doctor_parser(subparsers) -> None:
    parser = subparsers.add_parser("doctor", help="Diagnose dependencies and configuration")
    _add_common_options(parser)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check-api", action="store_true")
    parser.set_defaults(func=_run_doctor)


def _get_version() -> str:
    try:
        import importlib.metadata

        return f"subforge {importlib.metadata.version('subforge')}"
    except Exception:
        return "subforge (version unknown)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subforge",
        description="Transcribe speech, optimize subtitles, and translate them.",
    )
    parser.add_argument("--version", action="version", version=_get_version())
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    _build_transcribe_parser(subparsers)
    _build_subtitle_parser(subparsers)
    _build_process_parser(subparsers)
    _build_download_parser(subparsers)
    _build_config_parser(subparsers)
    _build_doctor_parser(subparsers)
    return parser


def _build_cli_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}

    def set_value(key: str, value) -> None:
        if value is not None:
            from subforge.cli.config import _set_nested

            _set_nested(overrides, key, value)

    for attr, key in {
        "api_key": "llm.api_key",
        "api_base": "llm.api_base",
        "model": "llm.model",
        "whisper_api_key": "whisper_api.api_key",
        "whisper_api_base": "whisper_api.api_base",
        "whisper_model": "whisper_api.model",
        "asr": "transcribe.asr",
        "language": "transcribe.language",
        "fw_model": "transcribe.faster_whisper.model",
        "fw_device": "transcribe.faster_whisper.device",
        "fw_vad_method": "transcribe.faster_whisper.vad_method",
        "fw_vad_threshold": "transcribe.faster_whisper.vad_threshold",
        "fw_prompt": "transcribe.faster_whisper.prompt",
        "whisper_prompt": "whisper_api.prompt",
        "max_cjk": "subtitle.max_word_count_cjk",
        "max_english": "subtitle.max_word_count_english",
        "max_chars_en": "subtitle.max_chars_en",
        "max_chars_cjk": "subtitle.max_chars_cjk",
        "thread_num": "subtitle.thread_num",
        "batch_size": "subtitle.batch_size",
        "translator": "translate.service",
        "target_language": "translate.target_language",
        "layout": "subtitle.layout",
        "format": "output.format",
    }.items():
        set_value(key, getattr(args, attr, None))
    if getattr(args, "fw_voice_extraction", False):
        set_value("transcribe.faster_whisper.voice_extraction", True)
    for attr, key in {
        "no_optimize": "subtitle.optimize",
        "no_translate": "subtitle.translate",
        "no_split": "subtitle.split",
    }.items():
        if getattr(args, attr, False):
            set_value(key, False)
    if getattr(args, "reflect", False):
        set_value("translate.reflect", True)
    return overrides


def _load_config(args: argparse.Namespace) -> dict:
    from subforge.cli.config import build_config

    config_path = Path(args.config) if getattr(args, "config", None) else None
    if config_path is not None and not config_path.exists():
        from subforge.cli import output

        output.warn(f"Config file not found: {config_path}, using defaults")
        config_path = None
    return build_config(
        cli_overrides=_build_cli_overrides(args), config_path=config_path
    )


def _run_transcribe(args: argparse.Namespace) -> int:
    from subforge.cli.commands.transcribe import run

    return run(args, _load_config(args))


def _run_subtitle(args: argparse.Namespace) -> int:
    from subforge.cli.commands.subtitle import run

    return run(args, _load_config(args))


def _run_process(args: argparse.Namespace) -> int:
    from subforge.cli.commands.process import run

    return run(args, _load_config(args))


def _run_download(args: argparse.Namespace) -> int:
    from subforge.cli.commands.download import run

    return run(args, _load_config(args))


def _run_config(args: argparse.Namespace) -> int:
    from subforge.cli.commands.config_cmd import run

    return run(args, _load_config(args))


def _run_doctor(args: argparse.Namespace) -> int:
    from subforge.cli.commands.doctor import run

    return run(args, _load_config(args))


def main(argv: Optional[List[str]] = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return EXIT.USAGE_ERROR
    if not hasattr(args, "func"):
        parser.print_help()
        return EXIT.USAGE_ERROR

    logging.getLogger().setLevel(
        logging.CRITICAL
        if getattr(args, "quiet", False)
        else logging.DEBUG
        if getattr(args, "verbose", False)
        else logging.WARNING
    )
    try:
        return args.func(args) or EXIT.SUCCESS
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        from subforge.cli.output import error

        error(str(exc))
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        return EXIT.GENERAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
