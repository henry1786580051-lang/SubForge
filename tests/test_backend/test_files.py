import asyncio
import io
import sys
from pathlib import Path

import pytest
from starlette.datastructures import UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

import app.api.files as files_module
from app.api.files import _parse_range_header


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("bytes=0-99", (0, 99)),
        ("bytes=100-", (100, 999)),
        ("bytes=-100", (900, 999)),
        ("bytes=900-1200", (900, 999)),
    ],
)
def test_parse_range_header(header, expected):
    assert _parse_range_header(header, 1000) == expected


@pytest.mark.parametrize(
    "header",
    ["items=0-10", "bytes=", "bytes=100-50", "bytes=1000-", "bytes=-0", "bytes=0-1,4-5"],
)
def test_parse_range_header_rejects_invalid_ranges(header):
    with pytest.raises((ValueError, IndexError)):
        _parse_range_header(header, 1000)


def test_uploads_with_the_same_name_do_not_overwrite_each_other(tmp_path, monkeypatch):
    monkeypatch.setattr(files_module, "UPLOAD_DIR", tmp_path)

    async def upload(content: bytes):
        return await files_module.upload_file(
            UploadFile(file=io.BytesIO(content), filename="video.mp4")
        )

    first = asyncio.run(upload(b"first"))
    second = asyncio.run(upload(b"second"))

    first_path = Path(first["file_path"])
    second_path = Path(second["file_path"])
    assert first_path != second_path
    assert first_path.read_bytes() == b"first"
    assert second_path.read_bytes() == b"second"


def test_cleanup_session_uploads_removes_only_current_session(tmp_path, monkeypatch):
    session_dir = tmp_path / "session-current"
    other_dir = tmp_path / "session-other"
    session_dir.mkdir()
    other_dir.mkdir()
    (session_dir / "video.mp4").write_bytes(b"current")
    (other_dir / "video.mp4").write_bytes(b"other")
    monkeypatch.setattr(files_module, "UPLOAD_DIR", session_dir)

    files_module.cleanup_session_uploads()

    assert not session_dir.exists()
    assert other_dir.exists()
