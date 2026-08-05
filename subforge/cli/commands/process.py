"""Run transcription followed by optional subtitle processing."""

from argparse import Namespace
from pathlib import Path

from subforge.cli import exit_codes as EXIT
from subforge.cli import output
from subforge.cli.config import get


def run(args: Namespace, config: dict) -> int:
    input_path = Path(args.input)
    if not input_path.is_file():
        output.error(f"Input file not found: {input_path}")
        return EXIT.FILE_NOT_FOUND

    from subforge.cli.validators import validate_process

    if not validate_process(config):
        return EXIT.USAGE_ERROR

    output_arg = getattr(args, "output", None)
    if output_arg:
        requested = Path(output_arg)
        output_dir = requested.parent if requested.suffix else requested
    else:
        requested = None
        output_dir = input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    verbose = bool(getattr(args, "verbose", False))
    quiet = bool(getattr(args, "quiet", False))
    optimize = bool(get(config, "subtitle.optimize", True))
    translate = bool(get(config, "subtitle.translate", False))
    split = bool(get(config, "subtitle.split", True))
    if getattr(args, "translator", None) or getattr(args, "target_language", None):
        translate = True

    raw_path = output_dir / f"{input_path.stem}.srt"
    if not quiet:
        output.info("Step 1/2: Transcribing...")
    transcribe_args = Namespace(
        input=str(input_path),
        output=str(raw_path),
        format="srt",
        word_timestamps=optimize or split,
        verbose=verbose,
        quiet=quiet,
        config=getattr(args, "config", None),
        asr=getattr(args, "asr", None),
        language=getattr(args, "language", None),
        fw_model=None,
        fw_device=None,
        fw_vad_method=None,
        fw_vad_threshold=None,
        fw_voice_extraction=False,
        fw_prompt=None,
        whisper_api_key=getattr(args, "whisper_api_key", None),
        whisper_api_base=getattr(args, "whisper_api_base", None),
        whisper_model=getattr(args, "whisper_model", None),
        whisper_prompt=None,
    )
    from subforge.cli.commands.transcribe import run as transcribe

    result = transcribe(transcribe_args, config)
    if result != EXIT.SUCCESS:
        return result

    if not (optimize or translate or split):
        if requested and requested.suffix and requested != raw_path:
            raw_path.replace(requested)
        if not quiet:
            output.success("Pipeline complete")
        return EXIT.SUCCESS

    processed_path = (
        requested
        if requested is not None and requested.suffix
        else output_dir / f"{input_path.stem}_processed.srt"
    )
    if not quiet:
        output.info("Step 2/2: Processing subtitles...")
    subtitle_args = Namespace(
        input=str(raw_path),
        output=str(processed_path),
        format=get(config, "output.format", "srt"),
        no_optimize=not optimize,
        no_translate=not translate,
        no_split=not split,
        verbose=verbose,
        quiet=quiet,
        config=getattr(args, "config", None),
        api_key=getattr(args, "api_key", None),
        api_base=getattr(args, "api_base", None),
        model=getattr(args, "model", None),
        translator=getattr(args, "translator", None),
        target_language=getattr(args, "target_language", None),
        reflect=bool(getattr(args, "reflect", False)),
        max_cjk=getattr(args, "max_cjk", None),
        max_english=getattr(args, "max_english", None),
        max_chars_en=getattr(args, "max_chars_en", None),
        max_chars_cjk=getattr(args, "max_chars_cjk", None),
        prompt=getattr(args, "prompt", None),
        prompt_file=getattr(args, "prompt_file", None),
        thread_num=getattr(args, "thread_num", None),
        batch_size=getattr(args, "batch_size", None),
        layout=getattr(args, "layout", None),
    )
    from subforge.cli.commands.subtitle import run as process_subtitle

    result = process_subtitle(subtitle_args, config)
    if result == EXIT.SUCCESS and not quiet:
        output.success("Pipeline complete")
    return result
