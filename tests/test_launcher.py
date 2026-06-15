import launcher


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
