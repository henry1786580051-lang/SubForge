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
