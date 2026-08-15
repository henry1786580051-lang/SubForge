#!/usr/bin/env python3
"""Run real packaged-app smoke tests against a desktop bundle."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def _run(
    cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None
) -> subprocess.CompletedProcess:
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


def _create_sample_video(ffmpeg: Path, output: Path) -> None:
    _run(
        [
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
        ]
    )


def _duration(ffprobe: Path, media: Path) -> float:
    result = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(media),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def _verify_ffmpeg_runtime(ffmpeg: Path, ffprobe: Path) -> None:
    for executable in (ffmpeg, ffprobe):
        result = subprocess.run(
            [str(executable), "-version"],
            check=True,
            capture_output=True,
            text=True,
        )
        first_line = result.stdout.splitlines()[0]
        if not re.search(r"\bversion 8\.1(?:\.\d+)?\b", first_line):
            raise RuntimeError(f"Expected FFmpeg 8.1 runtime, got: {first_line}")

    encoders = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if " pcm_s16le " not in encoders:
        raise RuntimeError("Bundled FFmpeg is missing the PCM encoder required by ASR")


def _run_macos_aqua_check(
    bundle: Path,
    env: dict[str, str],
    required_markers: tuple[str, ...],
    *,
    timeout_seconds: float = 120.0,
) -> None:
    """Run a packaged check through LaunchServices so Metal sees the Aqua session."""
    if bundle.suffix != ".app":
        raise RuntimeError("Aqua runtime checks require a macOS .app bundle")

    with tempfile.TemporaryDirectory(prefix="subforge-aqua-smoke-") as tmp:
        stdout_path = Path(tmp) / "stdout.log"
        stderr_path = Path(tmp) / "stderr.log"
        command = ["open", "-W", "-n", "-g"]
        for name, value in sorted(env.items()):
            command.extend(["--env", f"{name}={value}"])
        command.extend(
            [
                "-o",
                str(stdout_path),
                "--stderr",
                str(stderr_path),
                str(bundle),
            ]
        )
        print("+ " + " ".join(command))
        try:
            result = subprocess.run(command, check=False, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Timed out waiting for the macOS Aqua runtime check") from exc

        stdout = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
        stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if stderr:
            print(stderr, end="" if stderr.endswith("\n") else "\n")

        missing = [marker for marker in required_markers if marker not in stdout]
        if result.returncode != 0 or missing:
            details = f"; missing output: {', '.join(missing)}" if missing else ""
            raise RuntimeError(
                f"macOS Aqua runtime check failed with exit code {result.returncode}{details}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", help="Path to dist/SubForge or an executable")
    args = parser.parse_args()

    bundle = Path(args.bundle).resolve()
    exe = _find_executable(bundle)
    ffmpeg = _find_bundled_tool(bundle, "ffmpeg")
    ffprobe = _find_bundled_tool(bundle, "ffprobe")
    _verify_ffmpeg_runtime(ffmpeg, ffprobe)

    with tempfile.TemporaryDirectory(prefix="subforge-smoke-") as tmp:
        tmp_path = Path(tmp)

        video = tmp_path / "sample.mp4"
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
            _run_macos_aqua_check(
                bundle,
                {
                    "SUBFORGE_CHECK_ASR": "1",
                    "TORCH_USE_RTLD_GLOBAL": "1",
                },
                (
                    "MLX Whisper import: ok",
                    "MLX Metal inference: ok",
                    "PyTorch MPS inference: ok",
                    "WhisperX alignment import: ok",
                ),
            )
            print("Verified packaged MLX Metal, PyTorch MPS, and WhisperX imports")

            env = os.environ.copy()
            env["SUBFORGE_CHECK_DIARIZATION"] = "1"
            env.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
            _run([str(exe)], env=env)
            print("Verified packaged pyannote speaker diarization imports")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
