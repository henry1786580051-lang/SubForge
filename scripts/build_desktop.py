#!/usr/bin/env python3
"""Build a desktop bundle for the current platform with PyInstaller."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import platform
import plistlib
import re
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC_FILE = ROOT / "SubForge.spec"
BUILD_DIR = ROOT / "build"
DIST_DIR = ROOT / "dist"
ARTIFACT_DIR = ROOT / "artifacts"
RUNTIME_DIR = BUILD_DIR / "desktop-runtime"
FRONTEND_DIR = ROOT / "frontend"
FRONTEND_OUT_DIR = FRONTEND_DIR / "out"
WHISPER_CPP_WINDOWS_URL = (
    "https://github.com/ggml-org/whisper.cpp/releases/download/"
    "v1.9.1/whisper-bin-x64.zip"
)
WHISPER_CPP_WINDOWS_SHA256 = "7d8be46ecd31828e1eb7a2ecdd0d6b314feafd82163038ab6092594b0a063539"
WINDOWS_CUDA_RUNTIME_DLLS = (
    "cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll",
    "cudnn64_9.dll", "cudnn_adv64_9.dll", "cudnn_cnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll", "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll", "cudnn_ops64_9.dll",
    "nvrtc64_120_0.dll", "nvrtc-builtins64_121.dll",
)


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd))
    kwargs.setdefault("cwd", str(ROOT))
    return subprocess.run(cmd, check=True, **kwargs)


def _npm_command() -> str:
    npm = shutil.which("npm.cmd") if platform.system() == "Windows" else shutil.which("npm")
    if npm:
        return npm
    return "npm.cmd" if platform.system() == "Windows" else "npm"


def _requires_mlx_metallib() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _version() -> str:
    explicit_version = os.environ.get("SUBFORGE_BUILD_VERSION", "").strip()
    if explicit_version:
        return explicit_version.lstrip("v")

    version_file = ROOT / "subforge" / "_version.py"
    if version_file.is_file():
        match = re.search(
            r"^__version__\s*=\s*(?:version\s*=\s*)?['\"]([^'\"]+)['\"]",
            version_file.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        if match:
            return match.group(1).lstrip("v")
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
    try:
        import importlib.metadata

        return importlib.metadata.version("subforge").lstrip("v")
    except Exception:
        pass
    return "0.0.0-dev"


def ensure_version_file(version: str) -> bytes | None:
    """Inject the build version and return the original file for restoration."""
    version_file = ROOT / "subforge" / "_version.py"
    original = version_file.read_bytes() if version_file.exists() else None
    version_file.write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    print(f"Injected {version_file.relative_to(ROOT)} ({version})")
    return original


def restore_version_file(original: bytes | None) -> None:
    version_file = ROOT / "subforge" / "_version.py"
    if original is None:
        version_file.unlink(missing_ok=True)
    else:
        version_file.write_bytes(original)


def clean() -> None:
    for path in [BUILD_DIR, DIST_DIR, ARTIFACT_DIR]:
        if path.exists():
            print(f"Removing {path.relative_to(ROOT)}")
            shutil.rmtree(path)


def build_frontend(version: str, skip: bool = False) -> None:
    """Refresh the static frontend used by packaged desktop builds."""
    if skip:
        print("Skipping frontend build")
    elif (FRONTEND_DIR / "package.json").exists() and (FRONTEND_DIR / "node_modules").is_dir():
        _run(
            [_npm_command(), "run", "build"],
            cwd=str(FRONTEND_DIR),
            env={**os.environ, "NEXT_PUBLIC_APP_VERSION": version},
        )
    elif not skip:
        raise RuntimeError(
            "frontend/node_modules is not installed. Run `cd frontend && npm install`, "
            "or pass --skip-frontend-build to explicitly reuse frontend/out."
        )

    required = [
        FRONTEND_OUT_DIR / "index.html",
        FRONTEND_OUT_DIR / "_next",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Missing frontend static export:\n  - " + "\n  - ".join(missing))


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
            "Run with: uv run --extra denoise --extra whisperx "
            "--with pyinstaller --with static-ffmpeg python scripts/build_desktop.py"
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


def prepare_whisper_cpp() -> None:
    """Bundle the official whisper.cpp CLI and DLLs in Windows builds."""
    if platform.system() != "Windows":
        return

    archive = BUILD_DIR / "downloads" / "whisper-bin-x64-v1.9.1.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f"Downloading {WHISPER_CPP_WINDOWS_URL}")
        urllib.request.urlretrieve(WHISPER_CPP_WINDOWS_URL, archive)

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != WHISPER_CPP_WINDOWS_SHA256:
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            "whisper.cpp archive checksum mismatch: "
            f"expected {WHISPER_CPP_WINDOWS_SHA256}, got {digest}"
        )

    extract_dir = BUILD_DIR / "whisper-cpp-windows"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_dir)

    runtime_bin = RUNTIME_DIR / "resource" / "bin"
    runtime_bin.mkdir(parents=True, exist_ok=True)
    runtime_files = [
        path for path in extract_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".exe", ".dll"}
    ]
    cli = next((path for path in runtime_files if path.name.lower() == "whisper-cli.exe"), None)
    if cli is None:
        raise RuntimeError("Official whisper.cpp archive does not contain whisper-cli.exe")
    for source in runtime_files:
        shutil.copy2(source, runtime_bin / source.name)
    print(f"Bundled whisper.cpp runtime: {cli.name} and required DLLs")


def build_pyinstaller(version: str) -> None:
    env = os.environ.copy()
    env["VIDEOCAPTIONER_DESKTOP_RUNTIME_DIR"] = str(RUNTIME_DIR)
    env["SUBFORGE_BUILD_VERSION"] = version
    if platform.system() == "Darwin":
        env.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
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


def refresh_windows_vc_runtime() -> None:
    """Replace VC++ DLLs discovered through PATH with the system runtime.

    PyInstaller's dependency scan can otherwise pick up an older copy from a
    Conda installation on the build machine.  Native ML libraries such as
    CTranslate2 then load against that incompatible copy and terminate with an
    access violation before Python can report a useful exception.
    """
    if platform.system() != "Windows":
        return

    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    system32 = system_root / "System32"
    bundle_internal = DIST_DIR / "SubForge" / "_internal"
    for name in ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        source = system32 / name
        destination = bundle_internal / name
        if source.is_file() and destination.is_file():
            shutil.copy2(source, destination)
            print(f"Refreshed Windows VC++ runtime: {name}")


def inject_windows_cuda_runtime() -> None:
    """Optionally create a self-contained NVIDIA FasterWhisper bundle."""
    configured = os.environ.get("SUBFORGE_CUDA_RUNTIME_DIR", "").strip()
    if platform.system() != "Windows" or not configured:
        return

    source_dir = Path(configured).expanduser().resolve()
    missing = [name for name in WINDOWS_CUDA_RUNTIME_DLLS if not (source_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            "Incomplete SUBFORGE_CUDA_RUNTIME_DIR; missing: " + ", ".join(missing)
        )
    destination = DIST_DIR / "SubForge" / "_internal" / "cuda"
    destination.mkdir(parents=True, exist_ok=True)
    for name in WINDOWS_CUDA_RUNTIME_DLLS:
        shutil.copy2(source_dir / name, destination / name)
    print(f"Bundled private CUDA runtime: {len(WINDOWS_CUDA_RUNTIME_DLLS)} DLLs")


def inject_packaged_mlx_runtime() -> None:
    """Copy MLX packages without importing them during PyInstaller analysis."""
    if not _requires_mlx_metallib():
        return

    package_sources: dict[str, Path] = {}
    for package_name in ("mlx", "mlx_whisper"):
        spec = importlib.util.find_spec(package_name)
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError(f"Missing {package_name}; install the whisperx extra before packaging")
        package_sources[package_name] = Path(next(iter(spec.submodule_search_locations)))

    bundle_roots = [
        DIST_DIR / "SubForge" / "_internal",
        DIST_DIR / "SubForge.app" / "Contents" / "Resources",
    ]
    for root in bundle_roots:
        if not root.exists():
            continue
        for package_name, source in package_sources.items():
            destination = root / package_name
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(
                source,
                destination,
                symlinks=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
            if package_name == "mlx_whisper":
                _make_packaged_mlx_numba_optional(destination)
            print(f"Injected {package_name} runtime: {destination.relative_to(ROOT)}")

    app_contents = DIST_DIR / "SubForge.app" / "Contents"
    resources = app_contents / "Resources"
    frameworks = app_contents / "Frameworks"
    if resources.is_dir() and frameworks.is_dir():
        for package_name in package_sources:
            link = frameworks / package_name
            if link.exists() or link.is_symlink():
                if link.is_dir() and not link.is_symlink():
                    shutil.rmtree(link)
                else:
                    link.unlink()
            link.symlink_to(Path("..") / "Resources" / package_name)
            print(f"Linked injected runtime: {link.relative_to(ROOT)}")


def _make_packaged_mlx_numba_optional(package_dir: Path) -> None:
    """Keep injected MLX Whisper importable without the 100+ MB LLVM runtime.

    SubForge requests segment timestamps from MLX Whisper and performs word
    alignment with WhisperX. MLX Whisper imports its optional DTW accelerator
    eagerly, so the frozen copy needs a correct Python fallback even though the
    accelerated function is not used by the production path.
    """
    timing_path = package_dir / "timing.py"
    if not timing_path.is_file():
        raise RuntimeError(f"Missing MLX Whisper timing module: {timing_path}")
    needle = "import mlx.core as mx\nimport numba\n"
    replacement = """import mlx.core as mx
try:
    import numba
except ImportError:
    class _NumbaFallback:
        @staticmethod
        def jit(*_args, **_kwargs):
            return lambda function: function

    numba = _NumbaFallback()
"""
    source = timing_path.read_text(encoding="utf-8")
    if replacement in source:
        return
    if needle not in source:
        raise RuntimeError("Unsupported MLX Whisper timing module; numba import not found")
    timing_path.write_text(source.replace(needle, replacement, 1), encoding="utf-8")


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


def dedupe_packaged_torch_libs() -> None:
    """Replace duplicate top-level torch dylibs with symlinks to torch/lib."""
    bundle_roots = [
        DIST_DIR / "SubForge" / "_internal",
        DIST_DIR / "SubForge.app" / "Contents" / "Frameworks",
    ]
    for root in bundle_roots:
        target = root / "torch" / "lib" / "libtorch_cpu.dylib"
        duplicate = root / "libtorch_cpu.dylib"
        if not target.exists() or not duplicate.exists() or duplicate.is_symlink():
            continue
        if target.stat().st_size != duplicate.stat().st_size:
            continue
        duplicate.unlink()
        duplicate.symlink_to(Path("torch") / "lib" / "libtorch_cpu.dylib")
        print(f"Deduped torch library: {duplicate.relative_to(ROOT)}")


def patch_packaged_mlx_metallib() -> None:
    """Expose MLX's default Metal shader library where the loaded dylib expects it.

    PyInstaller collects ``mlx.metallib`` under ``mlx/lib`` but ``mlx.core`` loads
    the top-level ``libmlx.dylib`` via ``@loader_path/..``. At runtime MLX then
    looks next to that loaded dylib, so the ASR engine fails during import unless
    the metallib is also visible at the bundle root.
    """
    bundle_roots = [
        DIST_DIR / "SubForge" / "_internal",
        DIST_DIR / "SubForge.app" / "Contents" / "Resources",
        DIST_DIR / "SubForge.app" / "Contents" / "Frameworks",
    ]
    for root in bundle_roots:
        source = root / "mlx" / "lib" / "mlx.metallib"
        alias = root / "mlx.metallib"
        if not source.exists():
            continue
        if alias.exists() or alias.is_symlink():
            if alias.is_symlink() and alias.readlink() == Path("mlx") / "lib" / "mlx.metallib":
                continue
            alias.unlink()
        alias.symlink_to(Path("mlx") / "lib" / "mlx.metallib")
        print(f"Linked MLX metallib: {alias.relative_to(ROOT)}")


def resign_macos_app() -> None:
    """Refresh the ad-hoc signature after post-build bundle modifications."""
    if platform.system() != "Darwin":
        return
    app = DIST_DIR / "SubForge.app"
    if not app.exists():
        return
    _run(["codesign", "--force", "--deep", "--sign", "-", str(app)])
    _run(["codesign", "--verify", "--deep", "--strict", str(app)])


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


def _verify_data_root(data_root: Path, label: str) -> None:
    required = [
        data_root / "frontend" / "out" / "index.html",
        data_root / "frontend" / "out" / "_next",
        data_root / "resource" / "assets" / "logo.png",
        data_root / "resource" / "fonts" / "NotoSansSC-Regular.ttf",
        data_root / "resource" / "subtitle_style" / "ass-default.json",
        data_root / "resource" / "bin" / ("ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"),
        data_root / "resource" / "bin" / ("ffprobe.exe" if platform.system() == "Windows" else "ffprobe"),
    ]
    if platform.system() == "Windows":
        required.append(data_root / "resource" / "bin" / "whisper-cli.exe")
    if _requires_mlx_metallib():
        required.extend(
            [
                data_root / "mlx.metallib",
                data_root / "mlx" / "lib" / "mlx.metallib",
                data_root / "mlx_whisper" / "__init__.py",
            ]
        )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing bundled resources in {label}:\n  - " + "\n  - ".join(missing))


def _verify_macos_app(app: Path, expected_version: str) -> None:
    exe = app / "Contents" / "MacOS" / "SubForge"
    if not exe.exists():
        raise RuntimeError(f"Executable not found: {exe}")

    candidate_roots = [
        app / "Contents" / "Resources",
        app / "Contents" / "Frameworks",
    ]
    data_root = next(
        (
            root
            for root in candidate_roots
            if (root / "frontend" / "out" / "index.html").exists()
        ),
        None,
    )
    if data_root is None:
        roots = "\n  - ".join(str(root.relative_to(ROOT)) for root in candidate_roots)
        raise RuntimeError(f"Cannot locate packaged frontend in macOS app. Checked:\n  - {roots}")

    _verify_data_root(data_root, str(app.relative_to(ROOT)))
    info_plist = app / "Contents" / "Info.plist"
    with info_plist.open("rb") as file:
        bundle_info = plistlib.load(file)
    actual_version = bundle_info.get("CFBundleShortVersionString")
    if actual_version != expected_version:
        raise RuntimeError(
            f"macOS bundle version mismatch: expected {expected_version}, got {actual_version}"
        )
    print(f"Verified macOS app bundle: {app.relative_to(ROOT)}")


def verify_bundle(version: str) -> None:
    bundle = DIST_DIR / "SubForge"
    if platform.system() == "Windows":
        exe = bundle / "SubForge.exe"
    else:
        exe = bundle / "SubForge"
    if not exe.exists():
        raise RuntimeError(f"Executable not found: {exe}")

    _verify_data_root(bundle / "_internal", str(bundle.relative_to(ROOT)))
    print(f"Verified desktop bundle: {bundle.relative_to(ROOT)}")
    app = DIST_DIR / "SubForge.app"
    if platform.system() == "Darwin" and app.exists():
        _verify_macos_app(app, version)


def verify_windows_torch_cuda_build() -> None:
    """Reject CPU-only PyTorch before producing a Windows desktop bundle."""
    if platform.system() != "Windows":
        return
    import torch

    if torch.version.cuda is None:
        raise RuntimeError(
            "Windows desktop packaging requires a CUDA-enabled PyTorch wheel. "
            "Sync the project from the configured pytorch-cu128 index first."
        )
    print(f"Verified CUDA-enabled PyTorch build: {torch.__version__} (CUDA {torch.version.cuda})")


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
    parser.add_argument("--skip-frontend-build", action="store_true", help="Use the existing frontend/out static export")
    args = parser.parse_args()

    version = _version()
    if args.clean:
        clean()
    original_version_file = ensure_version_file(version)
    try:
        verify_windows_torch_cuda_build()
        build_frontend(version, skip=args.skip_frontend_build)
        prepare_ffmpeg()
        prepare_whisper_cpp()
        build_pyinstaller(version)
        refresh_windows_vc_runtime()
        inject_windows_cuda_runtime()
        inject_packaged_mlx_runtime()
        patch_packaged_torch()
        dedupe_packaged_torch_libs()
        patch_packaged_mlx_metallib()
        resign_macos_app()
        verify_bundle(version)
        if not args.no_archive:
            archive(version)
    finally:
        restore_version_file(original_version_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
