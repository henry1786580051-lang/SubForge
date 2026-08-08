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


def build_app_bundle() -> Path:
    """Build and verify the macOS app using the desktop packaging pipeline."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts import build_desktop

    version = build_desktop._version()
    build_desktop.clean()
    original_version_file = build_desktop.ensure_version_file(version)
    try:
        build_desktop.build_frontend(version)
        build_desktop.prepare_ffmpeg()
        build_desktop.prepare_whisper_cpp()
        build_desktop.build_pyinstaller(version)
        build_desktop.inject_packaged_mlx_runtime()
        build_desktop.patch_packaged_torch()
        build_desktop.dedupe_packaged_torch_libs()
        build_desktop.patch_packaged_mlx_metallib()
        build_desktop.resign_macos_app()
        build_desktop.verify_bundle(version)
    finally:
        build_desktop.restore_version_file(original_version_file)
    app_path = PROJECT_ROOT / "dist" / f"{APP_NAME}.app"
    if not app_path.exists():
        raise RuntimeError(f"{app_path} not found after desktop build")

    print(f"  -> {app_path}")
    return app_path


def seal_app_inside_dmg(dmg_path: Path) -> None:
    """Clear copy-time metadata and sign the exact app stored in the image."""
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        writable_dmg = work_dir / "SubForge-writable.dmg"
        rebuilt_dmg = work_dir / "SubForge-sealed.dmg"
        mount_point = work_dir / "mount"
        mount_point.mkdir()

        subprocess.run(
            [
                "hdiutil",
                "convert",
                str(dmg_path),
                "-format",
                "UDRW",
                "-o",
                str(writable_dmg),
                "-ov",
            ],
            check=True,
        )
        subprocess.run(
            [
                "hdiutil",
                "attach",
                str(writable_dmg),
                "-readwrite",
                "-nobrowse",
                "-mountpoint",
                str(mount_point),
            ],
            check=True,
        )
        try:
            image_app = mount_point / f"{APP_NAME}.app"
            subprocess.run(["xattr", "-cr", str(image_app)], check=True)
            # Copying into HFS does not change signed file contents; it only
            # adds Finder/provenance xattrs. Re-signing thousands of nested
            # binaries in-place can fail inside the macOS signing subsystem,
            # while clearing those xattrs restores the already-valid seal.
            subprocess.run(
                ["codesign", "--verify", "--deep", "--strict", str(image_app)],
                check=True,
            )
        finally:
            subprocess.run(["hdiutil", "detach", str(mount_point)], check=True)

        subprocess.run(
            [
                "hdiutil",
                "convert",
                str(writable_dmg),
                "-format",
                "UDZO",
                "-imagekey",
                "zlib-level=9",
                "-o",
                str(rebuilt_dmg),
                "-ov",
            ],
            check=True,
        )
        shutil.move(rebuilt_dmg, dmg_path)


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
        staged_app = staging / f"{APP_NAME}.app"
        shutil.copytree(app_path, staged_app, symlinks=True)
        # Finder may reattach metadata after the verified app is moved back
        # under Desktop. Sign the exact copy that will enter the DMG so those
        # attributes cannot invalidate the distributed bundle.
        subprocess.run(["xattr", "-cr", str(staged_app)], check=True)
        subprocess.run(
            ["codesign", "--force", "--deep", "--sign", "-", str(staged_app)],
            check=True,
        )
        subprocess.run(
            ["codesign", "--verify", "--deep", "--strict", str(staged_app)],
            check=True,
        )
        os.symlink("/Applications", staging / "Applications")
        app_size_mb = sum(
            path.stat().st_size for path in staged_app.rglob("*") if path.is_file()
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

    seal_app_inside_dmg(DMG_OUTPUT)

    size_mb = DMG_OUTPUT.stat().st_size / (1024 * 1024)
    print(f"  -> {DMG_OUTPUT} ({size_mb:.1f} MB)")


def main():
    print(f"=== Building {APP_NAME} DMG ===\n")

    app_path = build_app_bundle()
    create_dmg(app_path)
    print(f"\nDone! DMG is at: {DMG_OUTPUT}")


if __name__ == "__main__":
    main()
