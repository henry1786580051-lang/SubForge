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
