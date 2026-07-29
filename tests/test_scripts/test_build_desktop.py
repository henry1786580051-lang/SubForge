import pytest

from scripts import build_desktop


def test_build_version_prefers_explicit_release_environment(monkeypatch):
    monkeypatch.setenv("SUBFORGE_BUILD_VERSION", "v9.8.7")

    assert build_desktop._version() == "9.8.7"


def test_build_version_prefers_repository_version_file(monkeypatch, tmp_path):
    package_dir = tmp_path / "subforge"
    package_dir.mkdir()
    (package_dir / "_version.py").write_text(
        "__version__ = version = '9.8.7'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(build_desktop, "ROOT", tmp_path)

    assert build_desktop._version() == "9.8.7"


def test_pyinstaller_receives_release_version(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]

    monkeypatch.setattr(build_desktop, "_run", fake_run)

    build_desktop.build_pyinstaller("9.8.7")

    assert captured["env"]["SUBFORGE_BUILD_VERSION"] == "9.8.7"


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


def test_static_ffmpeg_download_retries_transient_failures(tmp_path, monkeypatch):
    attempts = []
    delays = []

    def fake_fetch(*, download_dir):
        attempts.append(download_dir)
        if len(attempts) < 3:
            raise RuntimeError("temporary gateway timeout")
        return "ffmpeg", "ffprobe"

    monkeypatch.setattr(build_desktop.time, "sleep", delays.append)

    result = build_desktop._fetch_static_ffmpeg(
        fake_fetch,
        tmp_path,
        attempts=4,
        initial_delay=0.5,
    )

    assert result == ("ffmpeg", "ffprobe")
    assert attempts == [str(tmp_path)] * 3
    assert delays == [0.5, 1.0]


def test_static_ffmpeg_download_raises_after_retry_limit(tmp_path, monkeypatch):
    attempts = []

    def fake_fetch(*, download_dir):
        attempts.append(download_dir)
        raise RuntimeError("persistent gateway timeout")

    monkeypatch.setattr(build_desktop.time, "sleep", lambda _delay: None)

    with pytest.raises(RuntimeError, match="persistent gateway timeout"):
        build_desktop._fetch_static_ffmpeg(
            fake_fetch,
            tmp_path,
            attempts=3,
            initial_delay=0,
        )

    assert attempts == [str(tmp_path)] * 3


def test_windows_bundle_requires_faster_whisper_vad_asset(tmp_path, monkeypatch):
    data_root = tmp_path / "bundle" / "_internal"
    required = [
        data_root / "frontend" / "out" / "index.html",
        data_root / "frontend" / "out" / "_next" / "build.js",
        data_root / "resource" / "assets" / "logo.png",
        data_root / "resource" / "fonts" / "NotoSansSC-Regular.ttf",
        data_root / "resource" / "subtitle_style" / "ass-default.json",
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
