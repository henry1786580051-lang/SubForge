#!/usr/bin/env python3
"""Build Subtitle Web App macOS DMG.

Usage:
    python build_macos.py

Output:
    ~/Desktop/Subtitle.dmg
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
APP_NAME = "SubForge"
DMG_OUTPUT = Path.home() / "Desktop" / f"{APP_NAME}.dmg"


def build_app_bundle() -> Path:
    """Build and verify the macOS app using the desktop packaging pipeline."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts import build_desktop

    version = build_desktop._version()
    build_desktop.clean()
    build_desktop.ensure_version_file(version)
    build_desktop.build_frontend()
    build_desktop.prepare_ffmpeg()
    build_desktop.build_pyinstaller()
    build_desktop.inject_packaged_mlx_runtime()
    build_desktop.patch_packaged_torch()
    build_desktop.dedupe_packaged_torch_libs()
    build_desktop.patch_packaged_mlx_metallib()
    build_desktop.resign_macos_app()
    build_desktop.verify_bundle()
    app_path = PROJECT_ROOT / "dist" / f"{APP_NAME}.app"
    if not app_path.exists():
        raise RuntimeError(f"{app_path} not found after desktop build")

    print(f"  -> {app_path}")
    return app_path


def create_dmg(app_path: Path):
    """Create DMG with Applications symlink and nice layout."""
    import dmgbuild

    print(f"Creating DMG -> {DMG_OUTPUT} ...")
    DMG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if DMG_OUTPUT.exists():
        DMG_OUTPUT.unlink()

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "staging"
        staging.mkdir()
        # PyInstaller's macOS bundle relies on symlinks between Resources and
        # Frameworks. Dereferencing them changes the sealed bundle contents and
        # invalidates the app signature inside the DMG.
        shutil.copytree(app_path, staging / f"{APP_NAME}.app", symlinks=True)
        os.symlink("/Applications", staging / "Applications")
        app_size_mb = sum(
            path.stat().st_size for path in (staging / f"{APP_NAME}.app").rglob("*") if path.is_file()
        ) // (1024 * 1024)
        dmg_size_mb = max(1024, int(app_size_mb * 1.35) + 256)

        settings = {
            "files": [str(staging / f"{APP_NAME}.app")],
            "symlinks": {"Applications": "/Applications"},
            "icon_locations": {
                f"{APP_NAME}.app": (160, 190),
                "Applications": (500, 190),
            },
            "window_rect": ((200, 120), (660, 400)),
            "background": "builtin-arrow",
            "icon_size": 96,
            "size": f"{dmg_size_mb}M",
            "format": "UDZO",
        }

        dmgbuild.build_dmg(
            filename=str(DMG_OUTPUT),
            volume_name=APP_NAME,
            settings=settings,
        )

    size_mb = DMG_OUTPUT.stat().st_size / (1024 * 1024)
    print(f"  -> {DMG_OUTPUT} ({size_mb:.1f} MB)")


def main():
    print(f"=== Building {APP_NAME} DMG ===\n")

    app_path = build_app_bundle()
    create_dmg(app_path)
    print(f"\nDone! DMG is at: {DMG_OUTPUT}")


if __name__ == "__main__":
    main()
