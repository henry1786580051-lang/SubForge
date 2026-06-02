#!/usr/bin/env python3
"""Build a desktop bundle for the current platform with PyInstaller."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC_FILE = ROOT / "SubForge.spec"
BUILD_DIR = ROOT / "build"
DIST_DIR = ROOT / "dist"
ARTIFACT_DIR = ROOT / "artifacts"
RUNTIME_DIR = BUILD_DIR / "desktop-runtime"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(ROOT), check=True, **kwargs)


def _version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("subforge").lstrip("v")
    except Exception:
        pass
    try:
        result = subprocess.run(
            [sys.executable, "-m", "hatchling", "version"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().lstrip("v")
    except Exception:
        result = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().lstrip("v")
    return "0.0.0-dev"


def ensure_version_file(version: str) -> None:
    version_file = ROOT / "subforge" / "_version.py"
    if version_file.exists():
        return
    version_file.write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    print(f"Generated {version_file.relative_to(ROOT)} ({version})")


def clean() -> None:
    for path in [BUILD_DIR, DIST_DIR, ARTIFACT_DIR]:
        if path.exists():
            print(f"Removing {path.relative_to(ROOT)}")
            shutil.rmtree(path)


def prepare_ffmpeg() -> None:
    """Download the current platform's static ffmpeg/ffprobe into runtime resources."""
    try:
        from static_ffmpeg.run import (
            get_or_fetch_platform_executables_else_raise,
            get_platform_key,
        )
    except ImportError as exc:
        raise RuntimeError(
            "static-ffmpeg is required for desktop builds. "
            "Run with: uv run --with pyinstaller --with static-ffmpeg python scripts/build_desktop.py"
        ) from exc

    runtime_bin = RUNTIME_DIR / "resource" / "bin"
    runtime_bin.mkdir(parents=True, exist_ok=True)
    cache_dir = BUILD_DIR / "static-ffmpeg" / get_platform_key()
    ffmpeg, ffprobe = get_or_fetch_platform_executables_else_raise(download_dir=str(cache_dir))
    for src in [Path(ffmpeg), Path(ffprobe)]:
        dst = runtime_bin / src.name
        if dst.exists():
            dst.chmod(dst.stat().st_mode | stat.S_IWUSR)
        shutil.copy2(src, dst)
        if platform.system() != "Windows":
            mode = dst.stat().st_mode
            dst.chmod(mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"Bundled {dst.relative_to(ROOT)}")


def build_pyinstaller() -> None:
    env = os.environ.copy()
    env["VIDEOCAPTIONER_DESKTOP_RUNTIME_DIR"] = str(RUNTIME_DIR)
    _run([
        sys.executable,
        "-m",
        "PyInstaller",
        str(SPEC_FILE),
        "--noconfirm",
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR / "pyinstaller"),
    ], env=env)


def patch_packaged_torch() -> None:
    """Patch PyInstaller-collected torch for macOS frozen import semantics.

    Some torch wheels successfully execute ``from torch._C import *`` inside a
    frozen app but do not leave the ``_C`` module binding in torch globals. The
    upstream torch module then crashes when it calls ``dir(_C)``. Importing the
    extension module explicitly at that point preserves normal Python behavior.
    """
    candidates = [
        DIST_DIR / "SubForge" / "_internal" / "torch" / "__init__.py",
        DIST_DIR / "SubForge.app" / "Contents" / "Resources" / "torch" / "__init__.py",
        DIST_DIR / "SubForge.app" / "Contents" / "Frameworks" / "torch" / "__init__.py",
    ]
    needle = "for name in dir(_C):\n"
    replacement = (
        "import torch._C as _C\n"
        "\n"
        "for name in dir(_C):\n"
    )
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if replacement in text:
            continue
        if needle not in text:
            continue
        path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
        print(f"Patched packaged torch import: {path.relative_to(ROOT)}")


def _platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower().replace("amd64", "x64").replace("x86_64", "x64")
    if system == "darwin":
        system = "macos"
    return f"{system}-{machine}"


def _archive_dir(source: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file in sorted(source.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(source.parent))
    print(f"Created {archive.relative_to(ROOT)}")


def verify_bundle() -> None:
    bundle = DIST_DIR / "SubForge"
    if platform.system() == "Windows":
        exe = bundle / "SubForge.exe"
    else:
        exe = bundle / "SubForge"
    if not exe.exists():
        raise RuntimeError(f"Executable not found: {exe}")

    data_root = bundle / "_internal"
    required = [
        data_root / "resource" / "assets" / "logo.png",
        data_root / "resource" / "fonts" / "NotoSansSC-Regular.ttf",
        data_root / "resource" / "subtitle_style" / "ass-default.json",
        data_root / "resource" / "bin" / ("ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"),
        data_root / "resource" / "bin" / ("ffprobe.exe" if platform.system() == "Windows" else "ffprobe"),
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Missing bundled resources:\n  - " + "\n  - ".join(missing))
    print(f"Verified desktop bundle: {bundle.relative_to(ROOT)}")


def archive(version: str) -> None:
    bundle = DIST_DIR / "SubForge"
    tag = _platform_tag()
    _archive_dir(bundle, ARTIFACT_DIR / f"SubForge-{version}-{tag}.zip")
    app = DIST_DIR / "SubForge.app"
    if app.exists():
        _archive_dir(app, ARTIFACT_DIR / f"SubForge-{version}-{tag}-app.zip")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="Remove build/dist/artifacts first")
    parser.add_argument("--no-archive", action="store_true", help="Build and verify without creating zip archives")
    args = parser.parse_args()

    version = _version()
    if args.clean:
        clean()
    ensure_version_file(version)
    prepare_ffmpeg()
    build_pyinstaller()
    patch_packaged_torch()
    verify_bundle()
    if not args.no_archive:
        archive(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
