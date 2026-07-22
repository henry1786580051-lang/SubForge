import os

import pytest

from subforge.core.utils import atomic_write


def test_atomic_write_replaces_complete_file_and_preserves_mode(tmp_path):
    destination = tmp_path / "subtitle.srt"
    destination.write_text("old", encoding="utf-8")
    if os.name != "nt":
        destination.chmod(0o640)

    atomic_write.atomic_write_text(destination, "new subtitle")

    assert destination.read_text(encoding="utf-8") == "new subtitle"
    if os.name != "nt":
        assert destination.stat().st_mode & 0o777 == 0o640


def test_atomic_write_failure_keeps_previous_file(tmp_path, monkeypatch):
    destination = tmp_path / "subtitle.srt"
    destination.write_text("previous subtitle", encoding="utf-8")
    monkeypatch.setattr(
        atomic_write.os,
        "fsync",
        lambda _fd: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    with pytest.raises(OSError, match="disk unavailable"):
        atomic_write.atomic_write_text(destination, "partial replacement")

    assert destination.read_text(encoding="utf-8") == "previous subtitle"
    assert list(tmp_path.glob(".subtitle.srt.*.tmp")) == []
