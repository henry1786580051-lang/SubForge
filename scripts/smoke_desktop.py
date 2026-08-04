#!/usr/bin/env python3
"""Run real packaged-app smoke tests against a desktop bundle."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path


def _run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess:
    print("+ " + " ".join(str(part) for part in cmd))
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True, text=True)


def _find_executable(bundle: Path) -> Path:
    if bundle.is_file():
        return bundle
    candidates = []
    if platform.system() == "Windows":
        candidates.append(bundle / "SubForge.exe")
    else:
        if bundle.suffix == ".app":
            candidates.append(bundle / "Contents" / "MacOS" / "SubForge")
        candidates.append(bundle / "SubForge")
        candidates.append(bundle / "SubForge.app" / "Contents" / "MacOS" / "SubForge")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"SubForge executable not found under {bundle}")


def _find_bundled_tool(bundle: Path, name: str) -> Path:
    exe_name = f"{name}.exe" if platform.system() == "Windows" else name
    candidates = [
        bundle / "_internal" / "resource" / "bin" / exe_name,
        bundle / "resource" / "bin" / exe_name,
        bundle / "Contents" / "Frameworks" / "resource" / "bin" / exe_name,
        bundle / "Contents" / "Resources" / "resource" / "bin" / exe_name,
        bundle / "SubForge.app" / "Contents" / "Frameworks" / "resource" / "bin" / exe_name,
        bundle / "SubForge.app" / "Contents" / "Resources" / "resource" / "bin" / exe_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    system_tool = shutil.which(name)
    if system_tool:
        return Path(system_tool)
    raise FileNotFoundError(f"{name} not found in bundle or PATH")


def _write_sample_srt(path: Path) -> None:
    path.write_text(
        "1\n"
        "00:00:00,100 --> 00:00:01,400\n"
        "Hello from SubForge.\n\n"
        "2\n"
        "00:00:01,500 --> 00:00:02,600\n"
        "这是一条真实合成测试字幕。\n",
        encoding="utf-8",
    )


def _create_sample_video(ffmpeg: Path, output: Path) -> None:
    _run([
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=640x360:rate=24",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=880:sample_rate=44100",
        "-t",
        "3",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        "-y",
        str(output),
    ])


def _duration(ffprobe: Path, media: Path) -> float:
    result = subprocess.run([
        str(ffprobe),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(media),
    ], check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", help="Path to dist/SubForge or an executable")
    args = parser.parse_args()

    bundle = Path(args.bundle).resolve()
    exe = _find_executable(bundle)
    ffmpeg = _find_bundled_tool(bundle, "ffmpeg")
    ffprobe = _find_bundled_tool(bundle, "ffprobe")

    with tempfile.TemporaryDirectory(prefix="subforge-smoke-") as tmp:
        tmp_path = Path(tmp)

        video = tmp_path / "sample.mp4"
        subtitle = tmp_path / "sample.srt"
        _write_sample_srt(subtitle)
        _create_sample_video(ffmpeg, video)

        # Verify ffmpeg/ffprobe work
        _duration(ffprobe, video)
        print(f"Verified ffmpeg/ffprobe: {video.stat().st_size} bytes")

        # Verify the executable exists
        if not exe.exists():
            raise RuntimeError(f"Executable not found: {exe}")
        print(f"Verified executable: {exe}")

        env = os.environ.copy()
        env["SUBFORGE_CHECK_BACKEND"] = "1"
        _run([str(exe)], env=env)
        print("Verified packaged FastAPI routes and HTTP runtime")

        if platform.system() == "Windows":
            whisper_cli = _find_bundled_tool(bundle, "whisper-cli")
            _run([str(whisper_cli), "--help"])
            print("Verified bundled whisper.cpp CLI")

            env = os.environ.copy()
            env["SUBFORGE_CHECK_DENOISE"] = "1"
            env["SUBFORGE_DENOISE_AUDIO_PATH"] = str(video)
            _run([str(exe)], env=env)
            print("Verified packaged DeepFilterNet3 inference")

            env = os.environ.copy()
            env["SUBFORGE_CHECK_FASTER_WHISPER"] = "1"
            _run([str(exe)], env=env)
            print("Verified packaged FasterWhisper/CTranslate2/PyAV imports")

            env = os.environ.copy()
            env["SUBFORGE_CHECK_WHISPERX"] = "1"
            _run([str(exe)], env=env)
            print("Verified packaged WhisperX and forced-alignment imports")

            env = os.environ.copy()
            env["SUBFORGE_CHECK_DIARIZATION"] = "1"
            _run([str(exe)], env=env)
            print("Verified packaged pyannote speaker diarization imports")

        if platform.system() == "Darwin" and platform.machine() == "arm64":
            env = os.environ.copy()
            env["SUBFORGE_CHECK_ASR"] = "1"
            env.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
            _run([str(exe)], env=env)
            print("Verified packaged MLX Whisper and WhisperX imports")

            env = os.environ.copy()
            env["SUBFORGE_CHECK_DIARIZATION"] = "1"
            env.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
            _run([str(exe)], env=env)
            print("Verified packaged pyannote speaker diarization imports")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
