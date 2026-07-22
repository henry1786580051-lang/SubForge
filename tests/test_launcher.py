import launcher


def test_configure_frozen_standard_streams_replaces_missing_streams(monkeypatch, tmp_path):
    devnull = tmp_path / "null"
    monkeypatch.setattr(launcher.sys, "frozen", True, raising=False)
    monkeypatch.setattr(launcher.sys, "stdout", None)
    monkeypatch.setattr(launcher.sys, "stderr", None)
    monkeypatch.setattr(launcher.os, "devnull", str(devnull))

    launcher._configure_frozen_standard_streams()

    assert launcher.sys.stdout is not None
    assert launcher.sys.stderr is not None
    launcher.sys.stdout.write("stdout")
    launcher.sys.stderr.write("stderr")


def test_configure_runtime_paths_adds_source_backend(monkeypatch, tmp_path):
    launcher_file = tmp_path / "launcher.py"
    backend = tmp_path / "backend"
    backend.mkdir()
    launcher_file.touch()
    monkeypatch.setattr(launcher.sys, "frozen", False, raising=False)
    monkeypatch.setattr(launcher, "__file__", str(launcher_file))
    monkeypatch.setattr(launcher.sys, "path", [])

    launcher._configure_frozen_runtime_paths()

    assert launcher.sys.path == [str(backend)]


def test_configure_frozen_runtime_paths_adds_macos_bundle_roots(monkeypatch, tmp_path):
    contents = tmp_path / "SubForge.app" / "Contents"
    executable = contents / "MacOS" / "SubForge"
    resources = contents / "Resources"
    frameworks = contents / "Frameworks"
    runtime = frameworks
    for path in (executable.parent, resources, frameworks):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(launcher.sys, "frozen", True, raising=False)
    monkeypatch.setattr(launcher.sys, "_MEIPASS", str(runtime), raising=False)
    monkeypatch.setattr(launcher.sys, "executable", str(executable))
    original_path = list(launcher.sys.path)
    monkeypatch.setattr(launcher.sys, "path", original_path.copy())

    launcher._configure_frozen_runtime_paths()

    assert str(resources) in launcher.sys.path
    assert str(frameworks) in launcher.sys.path


def test_cleanup_desktop_session_requests_shutdown_and_removes_uploads(monkeypatch):
    calls = []

    class FakeServer:
        should_exit = False

    server = FakeServer()
    monkeypatch.setattr("app.api.files.cleanup_session_uploads", lambda: calls.append("cleanup"))

    launcher._cleanup_desktop_session(server)

    assert server.should_exit is True
    assert calls == ["cleanup"]


def test_cleanup_desktop_session_can_leave_active_uploads_in_place(monkeypatch):
    calls = []

    class FakeServer:
        should_exit = False

    server = FakeServer()
    monkeypatch.setattr("app.api.files.cleanup_session_uploads", lambda: calls.append("cleanup"))

    launcher._cleanup_desktop_session(server, cleanup_uploads=False)

    assert server.should_exit is True
    assert calls == []


def test_start_server_disables_console_logging(monkeypatch):
    captured = {}

    class FakeConfig:
        def __init__(self, app, **kwargs):
            captured["app"] = app
            captured.update(kwargs)

    class FakeServer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            captured["ran"] = True

    fake_uvicorn = type("FakeUvicorn", (), {"Config": FakeConfig, "Server": FakeServer})
    monkeypatch.setitem(launcher.sys.modules, "uvicorn", fake_uvicorn)

    errors = []
    launcher.start_server(8765, errors)

    assert errors == []
    assert captured["app"] == "app.main:app"
    assert captured["log_config"] is None
    assert captured["access_log"] is False
    assert captured["ran"] is True


def test_start_server_preserves_exception_type(monkeypatch, tmp_path):
    class FailingConfig:
        def __init__(self, *args, **kwargs):
            raise ValueError("bad configuration")

    fake_uvicorn = type("FakeUvicorn", (), {"Config": FailingConfig, "Server": object})
    monkeypatch.setitem(launcher.sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr(launcher.tempfile, "gettempdir", lambda: str(tmp_path))

    errors = []
    launcher.start_server(8765, errors)

    assert errors == ["ValueError: bad configuration"]
    startup_log = tmp_path / "SubForge" / "startup-error.log"
    assert "ValueError: bad configuration" in startup_log.read_text(encoding="utf-8")
