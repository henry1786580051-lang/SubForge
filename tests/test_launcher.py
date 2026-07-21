import sys
from types import SimpleNamespace

import launcher


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


def test_start_server_disables_uvicorn_default_log_config(monkeypatch):
    captured = {}

    class FakeConfig:
        def __init__(self, app, **kwargs):
            captured["app"] = app
            captured.update(kwargs)

    class FakeServer:
        def __init__(self, config):
            captured["config"] = config
            captured["server"] = self

        def run(self):
            captured["ran"] = True

    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(Config=FakeConfig, Server=FakeServer),
    )

    errors = []
    holder = []
    launcher.start_server(8765, errors, holder)

    assert errors == []
    assert captured["app"] == "app.main:app"
    assert captured["host"] == launcher.HOST
    assert captured["port"] == 8765
    assert captured["log_config"] is None
    assert captured["ran"] is True
    assert holder == [captured["server"]]


def test_backend_startup_timeout_is_longer_for_frozen_app(monkeypatch):
    monkeypatch.delenv("SUBFORGE_BACKEND_STARTUP_TIMEOUT", raising=False)
    monkeypatch.setattr(launcher.sys, "frozen", True, raising=False)

    assert launcher.backend_startup_timeout_seconds() == 120.0


def test_backend_startup_timeout_uses_positive_environment_override(monkeypatch):
    monkeypatch.setenv("SUBFORGE_BACKEND_STARTUP_TIMEOUT", "45")
    monkeypatch.setattr(launcher.sys, "frozen", True, raising=False)

    assert launcher.backend_startup_timeout_seconds() == 45.0


def test_backend_startup_timeout_ignores_invalid_override(monkeypatch):
    monkeypatch.setenv("SUBFORGE_BACKEND_STARTUP_TIMEOUT", "not-a-number")
    monkeypatch.setattr(launcher.sys, "frozen", False, raising=False)

    assert launcher.backend_startup_timeout_seconds() == 30.0


def test_wait_for_server_allows_cold_start_longer_than_old_timeout(monkeypatch):
    clock = [0.0]

    class HealthyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def delayed_health_check(*args, **kwargs):
        if clock[0] < 16.0:
            raise OSError("backend is still starting")
        return HealthyResponse()

    monkeypatch.setattr(launcher.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(launcher.time, "sleep", lambda _seconds: clock.__setitem__(0, clock[0] + 1.0))
    monkeypatch.setattr(launcher.urllib.request, "urlopen", delayed_health_check)

    assert launcher.wait_for_server("http://127.0.0.1:8000", [], timeout_seconds=120.0)
    assert clock[0] == 16.0
