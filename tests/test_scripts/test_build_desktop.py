from scripts import build_desktop


def test_build_version_prefers_repository_version_file(monkeypatch, tmp_path):
    package_dir = tmp_path / "subforge"
    package_dir.mkdir()
    (package_dir / "_version.py").write_text(
        "__version__ = version = '9.8.7'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(build_desktop, "ROOT", tmp_path)

    assert build_desktop._version() == "9.8.7"
