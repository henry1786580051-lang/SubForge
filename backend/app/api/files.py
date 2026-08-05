import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile

from app.security import validate_path

router = APIRouter()

MAX_UPLOAD_SIZE = 10 * 1024 * 1024 * 1024  # 10 GB

UPLOAD_ROOT = Path(tempfile.gettempdir()) / "SubForge" / "uploads"
UPLOAD_DIR = UPLOAD_ROOT / f"session-{uuid.uuid4().hex}"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_stale_uploads(max_age_seconds: int = 7 * 24 * 60 * 60) -> None:
    """Remove abandoned uploads from earlier application sessions."""
    cutoff = time.time() - max_age_seconds
    if not UPLOAD_ROOT.exists():
        return
    for path in UPLOAD_ROOT.iterdir():
        if path == UPLOAD_DIR:
            continue
        try:
            if path.stat().st_mtime >= cutoff:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        except OSError:
            continue


def cleanup_session_uploads() -> None:
    """Remove uploads owned by this application process."""
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)


@router.post("/upload")
async def upload_file(file: UploadFile):
    """Upload a media file for processing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    from os.path import basename

    safe_name = basename(file.filename)
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "._- ")
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    upload_dir = UPLOAD_DIR / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=False)
    dest = upload_dir / safe_name

    size = 0
    try:
        with open(dest, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE:
                    raise HTTPException(status_code=413, detail="File too large")
                f.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        try:
            upload_dir.rmdir()
        except OSError:
            pass
        raise

    return {"file_path": str(dest), "filename": safe_name}


@router.get("/info")
def get_file_info(path: str = Query(..., description="File path")):
    """Get media file information using ffprobe."""
    import json
    import subprocess

    try:
        file_path = validate_path(path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            ),
        )

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="ffprobe failed")

        probe = json.loads(result.stdout)

        video_stream = next(
            (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
            None,
        )
        audio_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
        fmt = probe.get("format", {})

        return {
            "file_path": str(file_path),
            "filename": file_path.name,
            "duration": float(fmt.get("duration", 0)),
            "size": int(fmt.get("size", 0)),
            "bit_rate": int(fmt.get("bit_rate", 0)),
            "video": {
                "width": video_stream.get("width"),
                "height": video_stream.get("height"),
                "codec": video_stream.get("codec_name"),
                "fps": video_stream.get("r_frame_rate"),
            }
            if video_stream
            else None,
            "audio_tracks": [
                {
                    "index": i,
                    "codec": s.get("codec_name"),
                    "channels": s.get("channels"),
                    "sample_rate": s.get("sample_rate"),
                    "language": s.get("tags", {}).get("language", "und"),
                }
                for i, s in enumerate(audio_streams)
            ],
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="ffprobe timed out")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffprobe not installed")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
