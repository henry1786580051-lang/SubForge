import hashlib
import zipfile

import pytest

from scripts import build_desktop


def test_windows_installer_removes_previous_pyinstaller_runtime():
    installer = (build_desktop.ROOT / "scripts" / "windows_installer.iss").read_text(
        encoding="utf-8"
    )

    assert "[InstallDelete]" in installer
    assert 'Name: "{app}\\_internal"' in installer


def test_build_version_prefers_explicit_release_environment(monkeypatch):
    monkeypatch.setenv("SUBFORGE_BUILD_VERSION", "v9.8.7")

    assert build_desktop._version() == "9.8.7"


def test_build_version_uses_repository_version_file_without_vcs(monkeypatch, tmp_path):
    package_dir = tmp_path / "subforge"
    package_dir.mkdir()
    (package_dir / "_version.py").write_text(
        "__version__ = version = '9.8.7'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(build_desktop, "ROOT", tmp_path)

    assert build_desktop._version() == "9.8.7"


def test_build_version_prefers_git_tag_over_stale_version_file(monkeypatch, tmp_path):
    package_dir = tmp_path / "subforge"
    package_dir.mkdir()
    (package_dir / "_version.py").write_text(
        "__version__ = version = '1.1.7'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SUBFORGE_BUILD_VERSION", raising=False)
    monkeypatch.setattr(build_desktop, "ROOT", tmp_path)

    class Result:
        returncode = 0
        stdout = "v1.1.13\n"

    monkeypatch.setattr(build_desktop.subprocess, "run", lambda *_args, **_kwargs: Result())

    assert build_desktop._version() == "1.1.13"


def test_build_version_removes_vcs_development_metadata(monkeypatch, tmp_path):
    package_dir = tmp_path / "subforge"
    package_dir.mkdir()
    (package_dir / "_version.py").write_text(
        "__version__ = version = '1.1.6.dev4+g1234567'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SUBFORGE_BUILD_VERSION", raising=False)
    monkeypatch.setattr(build_desktop, "ROOT", tmp_path)

    assert build_desktop._version() == "1.1.6"


def test_pyinstaller_receives_release_version(monkeypatch, tmp_path):
    captured = {}
    dist_dir = tmp_path / "dist"
    stale_bundle = dist_dir / "SubForge"
    stale_bundle.mkdir(parents=True)

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]

    monkeypatch.setattr(build_desktop, "DIST_DIR", dist_dir)
    monkeypatch.setattr(build_desktop, "_run", fake_run)

    build_desktop.build_pyinstaller("9.8.7")

    assert captured["env"]["SUBFORGE_BUILD_VERSION"] == "9.8.7"
    assert not stale_bundle.exists()


def test_managed_dist_cleanup_removes_numbered_duplicates_only(monkeypatch, tmp_path):
    dist_dir = tmp_path / "dist"
    managed_paths = [
        dist_dir / "SubForge",
        dist_dir / "SubForge.app",
        dist_dir / "SubForge 2",
        dist_dir / "SubForge 3.app",
    ]
    for path in managed_paths:
        path.mkdir(parents=True)
    unrelated = dist_dir / "SubForge-1.1.13-macos-arm64.zip"
    unrelated.write_bytes(b"archive")
    notes = dist_dir / "notes"
    notes.mkdir()
    monkeypatch.setattr(build_desktop, "DIST_DIR", dist_dir)

    removed = build_desktop.clean_managed_dist_outputs()

    assert set(removed) == set(managed_paths)
    assert all(not path.exists() for path in managed_paths)
    assert unrelated.read_bytes() == b"archive"
    assert notes.is_dir()


def test_clean_preserves_evaluation_artifacts(monkeypatch, tmp_path):
    build_dir = tmp_path / "build"
    dist_dir = tmp_path / "dist"
    artifact_dir = tmp_path / "artifacts"
    frontend_dir = tmp_path / "frontend"
    for path in (build_dir, dist_dir, frontend_dir / ".next"):
        path.mkdir(parents=True)
    manifest = artifact_dir / "translation-quality" / "corpus.local.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(build_desktop, "ROOT", tmp_path)
    monkeypatch.setattr(build_desktop, "BUILD_DIR", build_dir)
    monkeypatch.setattr(build_desktop, "DIST_DIR", dist_dir)
    monkeypatch.setattr(build_desktop, "ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(build_desktop, "FRONTEND_DIR", frontend_dir)

    build_desktop.clean()

    assert not build_dir.exists()
    assert not dist_dir.exists()
    assert not (frontend_dir / ".next").exists()
    assert manifest.read_text(encoding="utf-8") == "{}"


def test_macos_build_discards_intermediate_bundle_after_verification(monkeypatch, tmp_path):
    dist_dir = tmp_path / "dist"
    standalone = dist_dir / "SubForge"
    app = dist_dir / "SubForge.app"
    standalone.mkdir(parents=True)
    app.mkdir()
    monkeypatch.setattr(build_desktop, "DIST_DIR", dist_dir)
    monkeypatch.setattr(build_desktop.platform, "system", lambda: "Darwin")

    assert build_desktop.discard_standalone_macos_bundle() is True
    assert not standalone.exists()
    assert app.is_dir()


def test_windows_build_keeps_standalone_bundle(monkeypatch, tmp_path):
    dist_dir = tmp_path / "dist"
    standalone = dist_dir / "SubForge"
    standalone.mkdir(parents=True)
    monkeypatch.setattr(build_desktop, "DIST_DIR", dist_dir)
    monkeypatch.setattr(build_desktop.platform, "system", lambda: "Windows")

    assert build_desktop.discard_standalone_macos_bundle() is False
    assert standalone.is_dir()


def test_macos_resign_clears_bundle_metadata_before_codesign(tmp_path, monkeypatch):
    app = tmp_path / "dist" / "SubForge.app"
    app.mkdir(parents=True)
    commands = []
    monkeypatch.setattr(build_desktop, "DIST_DIR", tmp_path / "dist")
    monkeypatch.setattr(build_desktop.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(build_desktop, "_run", lambda command, **_kwargs: commands.append(command))

    build_desktop.resign_macos_app()

    staged_app = commands[0][-1]
    assert commands[0][:2] == ["xattr", "-cr"]
    assert commands[1][:2] == ["codesign", "--force"]
    assert commands[2][:2] == ["codesign", "--verify"]
    assert commands[1][-1] == staged_app
    assert commands[2][-1] == staged_app
    assert app.exists()


def test_packaged_mlx_whisper_gets_numba_fallback(tmp_path):
    package_dir = tmp_path / "mlx_whisper"
    package_dir.mkdir()
    timing_path = package_dir / "timing.py"
    timing_path.write_text(
        "import mlx.core as mx\nimport numba\n\n@numba.jit(nopython=True)\ndef dtw():\n    pass\n",
        encoding="utf-8",
    )

    build_desktop._make_packaged_mlx_numba_optional(package_dir)
    patched = timing_path.read_text(encoding="utf-8")

    assert "except ImportError:" in patched
    assert "class _NumbaFallback:" in patched
    assert "import numba\n\n@numba.jit" not in patched


def test_frontend_build_failure_is_not_hidden_by_stale_export(tmp_path, monkeypatch):
    frontend = tmp_path / "frontend"
    output = frontend / "out"
    (frontend / "node_modules").mkdir(parents=True)
    output.mkdir()
    (frontend / "package.json").write_text("{}", encoding="utf-8")
    (output / "index.html").write_text("stale", encoding="utf-8")
    (output / "_next").mkdir()
    monkeypatch.setattr(build_desktop, "FRONTEND_DIR", frontend)
    monkeypatch.setattr(build_desktop, "FRONTEND_OUT_DIR", output)
    monkeypatch.setattr(
        build_desktop,
        "_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("npm failed")),
    )

    with pytest.raises(RuntimeError, match="npm failed"):
        build_desktop.build_frontend("1.2.3")


def test_frontend_build_requires_explicit_skip_when_dependencies_are_missing(tmp_path, monkeypatch):
    frontend = tmp_path / "frontend"
    output = frontend / "out"
    frontend.mkdir()
    output.mkdir()
    (frontend / "package.json").write_text("{}", encoding="utf-8")
    (output / "index.html").write_text("existing", encoding="utf-8")
    (output / "_next").mkdir()
    monkeypatch.setattr(build_desktop, "FRONTEND_DIR", frontend)
    monkeypatch.setattr(build_desktop, "FRONTEND_OUT_DIR", output)

    with pytest.raises(RuntimeError, match="node_modules"):
        build_desktop.build_frontend("1.2.3")

    build_desktop.build_frontend("1.2.3", skip=True)


def test_ffmpeg_download_retries_transient_failures(tmp_path, monkeypatch):
    attempts = []
    delays = []
    payload = b"ffmpeg archive"

    def fake_download(url, destination):
        attempts.append((url, destination))
        if len(attempts) < 3:
            raise RuntimeError("temporary gateway timeout")
        destination.write_bytes(payload)

    monkeypatch.setattr(build_desktop, "_download_file", fake_download)
    monkeypatch.setattr(build_desktop.time, "sleep", delays.append)
    destination = tmp_path / "ffmpeg.zip"

    result = build_desktop._download_verified_archive(
        "https://example.test/ffmpeg.zip",
        destination,
        hashlib.sha256(payload).hexdigest(),
        attempts=4,
        initial_delay=0.5,
    )

    assert result == destination
    assert attempts == [("https://example.test/ffmpeg.zip", destination)] * 3
    assert delays == [0.5, 1.0]


def test_ffmpeg_download_raises_after_retry_limit(tmp_path, monkeypatch):
    attempts = []

    def fake_download(url, destination):
        attempts.append((url, destination))
        raise RuntimeError("persistent gateway timeout")

    monkeypatch.setattr(build_desktop, "_download_file", fake_download)
    monkeypatch.setattr(build_desktop.time, "sleep", lambda _delay: None)
    destination = tmp_path / "ffmpeg.zip"

    with pytest.raises(RuntimeError, match="persistent gateway timeout"):
        build_desktop._download_verified_archive(
            "https://example.test/ffmpeg.zip",
            destination,
            "0" * 64,
            attempts=3,
            initial_delay=0,
        )

    assert attempts == [("https://example.test/ffmpeg.zip", destination)] * 3


def test_ffmpeg_download_rejects_checksum_mismatch(tmp_path, monkeypatch):
    destination = tmp_path / "ffmpeg.zip"

    def fake_download(_url, path):
        path.write_bytes(b"tampered")

    monkeypatch.setattr(build_desktop, "_download_file", fake_download)
    monkeypatch.setattr(build_desktop.time, "sleep", lambda _delay: None)

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        build_desktop._download_verified_archive(
            "https://example.test/ffmpeg.zip",
            destination,
            "0" * 64,
            attempts=1,
        )

    assert not destination.exists()


def test_ffmpeg_zip_rejects_path_traversal(tmp_path):
    archive = tmp_path / "ffmpeg.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../ffmpeg", b"unsafe")

    with pytest.raises(RuntimeError, match="Unsafe path"):
        build_desktop._safe_extract_zip(archive, tmp_path / "runtime")


def test_windows_ffmpeg_runtime_includes_optional_dlls(tmp_path, monkeypatch):
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffprobe = tmp_path / "ffprobe.exe"
    codec = tmp_path / "avcodec-59.dll"
    for path in (ffmpeg, ffprobe, codec):
        path.write_bytes(b"runtime")
    monkeypatch.setattr(build_desktop.platform, "system", lambda: "Windows")

    assert build_desktop._ffmpeg_runtime_files(ffmpeg, ffprobe) == [
        ffmpeg,
        ffprobe,
        codec,
    ]


def test_windows_static_ffmpeg_runtime_does_not_require_dlls(tmp_path, monkeypatch):
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffprobe = tmp_path / "ffprobe.exe"
    ffmpeg.write_bytes(b"runtime")
    ffprobe.write_bytes(b"runtime")
    monkeypatch.setattr(build_desktop.platform, "system", lambda: "Windows")

    assert build_desktop._ffmpeg_runtime_files(ffmpeg, ffprobe) == [ffmpeg, ffprobe]


def test_windows_bundle_requires_faster_whisper_vad_asset(tmp_path, monkeypatch):
    data_root = tmp_path / "bundle" / "_internal"
    required = [
        data_root / "frontend" / "out" / "index.html",
        data_root / "frontend" / "out" / "_next" / "build.js",
        data_root / "resource" / "assets" / "logo.png",
        data_root / "resource" / "bin" / "ffmpeg.exe",
        data_root / "resource" / "bin" / "ffprobe.exe",
        data_root / "resource" / "bin" / "whisper-cli.exe",
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
    monkeypatch.setattr(build_desktop, "ROOT", tmp_path)
    monkeypatch.setattr(build_desktop.platform, "system", lambda: "Windows")

    with pytest.raises(RuntimeError, match="silero_vad_v6.onnx"):
        build_desktop._verify_data_root(data_root, "test bundle")

    vad_asset = data_root / "faster_whisper" / "assets" / "silero_vad_v6.onnx"
    vad_asset.parent.mkdir(parents=True)
    vad_asset.write_bytes(b"onnx")
    build_desktop._verify_data_root(data_root, "test bundle")


@pytest.mark.parametrize("has_faster_whisper", [False, True])
def test_macos_intel_vad_requirement_matches_optional_runtime(
    tmp_path, monkeypatch, has_faster_whisper
):
    data_root = tmp_path / "bundle" / "Contents" / "Resources"
    required = [
        data_root / "frontend" / "out" / "index.html",
        data_root / "frontend" / "out" / "_next" / "build.js",
        data_root / "resource" / "assets" / "logo.png",
        data_root / "resource" / "bin" / "ffmpeg",
        data_root / "resource" / "bin" / "ffprobe",
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
    monkeypatch.setattr(build_desktop, "ROOT", tmp_path)
    monkeypatch.setattr(build_desktop.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(build_desktop, "_requires_mlx_metallib", lambda: False)
    monkeypatch.setattr(
        build_desktop.importlib.util,
        "find_spec",
        lambda name: object() if has_faster_whisper and name == "faster_whisper" else None,
    )

    if not has_faster_whisper:
        build_desktop._verify_data_root(data_root, "test bundle")
        return

    with pytest.raises(RuntimeError, match="silero_vad_v6.onnx"):
        build_desktop._verify_data_root(data_root, "test bundle")

    vad_asset = data_root / "faster_whisper" / "assets" / "silero_vad_v6.onnx"
    vad_asset.parent.mkdir(parents=True)
    vad_asset.write_bytes(b"onnx")
    build_desktop._verify_data_root(data_root, "test bundle")
