"""download command — download online video via yt-dlp."""

from argparse import Namespace
from pathlib import Path
from typing import Any

from subforge.cli import exit_codes as EXIT
from subforge.cli import output


def run(args: Namespace, config: dict) -> int:
    url = args.url
    out_dir = getattr(args, "output", None) or "."
    quiet = getattr(args, "quiet", False)

    try:
        import yt_dlp
    except ImportError:
        output.error("yt-dlp is not available")
        output.hint("Install the official package with: pip install subforge")
        return EXIT.DEPENDENCY_MISSING

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    progress = None if quiet else output.ProgressLine(f"Downloading {url}").start()

    try:
        ydl_opts: dict[str, Any] = {
            "format": "bestvideo+bestaudio/best",
            "outtmpl": f"{out_dir}/%(title)s.%(ext)s",
            "noplaylist": True,
            "quiet": quiet,
            "no_warnings": quiet,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[arg-type]
            ydl.download([url])

        if progress:
            progress.finish(f"Downloaded to {out_dir}/")
        return EXIT.SUCCESS

    except Exception as e:
        if progress:
            progress.fail(str(e))
        else:
            output.error(str(e))
        return EXIT.RUNTIME_ERROR
