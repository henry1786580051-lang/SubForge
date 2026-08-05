import asyncio
import io
import sys
from pathlib import Path

from starlette.datastructures import UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

import app.api.files as files_module


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
