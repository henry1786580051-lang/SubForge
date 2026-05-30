#!/usr/bin/env python3
"""Build Subtitle Web App macOS DMG.

Usage:
    python build_macos.py

Output:
    ~/Desktop/Subtitle.dmg
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
APP_NAME = "SubForge"
DMG_OUTPUT = Path.home() / "Desktop" / f"{APP_NAME}.dmg"
SPEC_FILE = PROJECT_ROOT / "SubForge.spec"


def run_pyinstaller():
    """Run PyInstaller with the spec file."""
    print("Running PyInstaller ...")

    for d in ["build", "dist"]:
        p = PROJECT_ROOT / d
        if p.exists():
            shutil.rmtree(p)

    subprocess.run([
        sys.executable, "-m", "PyInstaller",
        str(SPEC_FILE),
        "--noconfirm",
        "--clean",
        "--log-level", "WARN",
    ], check=True, cwd=str(PROJECT_ROOT))

    app_path = PROJECT_ROOT / "dist" / f"{APP_NAME}.app"
    if not app_path.exists():
        print(f"ERROR: {app_path} not found after build")
        sys.exit(1)

    print(f"  -> {app_path}")
    return app_path


def create_dmg(app_path: Path):
    """Create DMG with Applications symlink and nice layout."""
    import dmgbuild

    print(f"Creating DMG -> {DMG_OUTPUT} ...")

    if DMG_OUTPUT.exists():
        DMG_OUTPUT.unlink()

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "staging"
        staging.mkdir()
        shutil.copytree(app_path, staging / f"{APP_NAME}.app")
        os.symlink("/Applications", staging / "Applications")

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

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("ERROR: PyInstaller not found. Install with: pip install pyinstaller")
        sys.exit(1)

    app_path = run_pyinstaller()
    create_dmg(app_path)
    print(f"\nDone! DMG is at: {DMG_OUTPUT}")


if __name__ == "__main__":
    main()
