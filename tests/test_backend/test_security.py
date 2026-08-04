import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app import security


def test_native_file_grant_allows_exact_external_file(tmp_path, monkeypatch):
    external = tmp_path / "external-drive" / "video.mp4"
    external.parent.mkdir()
    external.touch()
    monkeypatch.setattr(security, "_get_allowed_roots", lambda: [])
    security.clear_granted_paths()

    security.grant_path(external)

    assert security.validate_path(str(external)) == external.resolve()


def test_native_file_grant_does_not_allow_sibling_paths(tmp_path, monkeypatch):
    selected = tmp_path / "external-drive" / "selected.mp4"
    sibling = selected.with_name("private.txt")
    selected.parent.mkdir()
    selected.touch()
    sibling.touch()
    monkeypatch.setattr(security, "_get_allowed_roots", lambda: [])
    security.clear_granted_paths()
    security.grant_path(selected)

    with pytest.raises(ValueError, match="Path not allowed"):
        security.validate_path(str(sibling))


def test_windows_videos_directory_is_an_allowed_root(tmp_path, monkeypatch):
    videos = tmp_path / "Videos"
    video = videos / "clip.mp4"
    videos.mkdir()
    video.touch()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("subforge.config.APPDATA_PATH", tmp_path / "AppData")
    monkeypatch.setattr("subforge.config.RESOURCE_PATH", tmp_path / "resources")
    monkeypatch.setattr("subforge.config.WORK_PATH", tmp_path / "work")
    security.clear_granted_paths()

    assert security.validate_path(str(video)) == video.resolve()
