import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.security import validate_path

router = APIRouter()

MAX_UPLOAD_SIZE = 10 * 1024 * 1024 * 1024  # 10 GB

UPLOAD_DIR = Path("/tmp/subforge/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _parse_range_header(range_header: str, file_size: int) -> tuple[int, int]:
    """Parse one HTTP byte range and reject malformed or multipart ranges."""
    if file_size <= 0 or not range_header.startswith("bytes="):
        raise ValueError("Invalid Range header")
    value = range_header[6:].strip()
    if not value or "," in value or "-" not in value:
        raise ValueError("Invalid Range header")

    start_text, end_text = value.split("-", 1)
    if not start_text:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise ValueError("Invalid suffix range")
        return max(0, file_size - suffix_length), file_size - 1

    start = int(start_text)
    end = int(end_text) if end_text else file_size - 1
    if start < 0 or start >= file_size or end < start:
        raise ValueError("Range out of bounds")
    return start, min(end, file_size - 1)


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


@router.get("/thumbnail")
def get_thumbnail(path: str = Query(...)):
    """Generate a video thumbnail."""
    import subprocess

    try:
        file_path = validate_path(path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    path_hash = hashlib.sha256(str(file_path).encode("utf-8")).hexdigest()[:16]
    thumb_path = UPLOAD_DIR / f"{file_path.stem}_{path_hash}_thumb.jpg"

    if not thumb_path.exists() or thumb_path.stat().st_mtime_ns < file_path.stat().st_mtime_ns:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(file_path),
                "-ss",
                "00:00:01",
                "-vframes",
                "1",
                "-vf",
                "scale=320:-1",
                str(thumb_path),
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="Thumbnail generation failed")

    return FileResponse(thumb_path, media_type="image/jpeg")


@router.get("/stream")
def stream_video(request: Request, path: str = Query(...)):
    """Stream a video file for HTML5 video player with range support."""
    import mimetypes
    import os

    try:
        file_path = validate_path(path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    media_type = mimetypes.guess_type(str(file_path))[0] or "video/mp4"
    file_size = os.path.getsize(file_path)

    # Handle range requests for video seeking
    range_header = request.headers.get("range")
    if range_header:
        try:
            start, end = _parse_range_header(range_header, file_size)
        except (ValueError, IndexError):
            raise HTTPException(status_code=416, detail="Invalid Range header")

        def iter_range():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                chunk_size = 64 * 1024
                while remaining > 0:
                    read_size = min(chunk_size, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
            "Content-Type": media_type,
        }
        return StreamingResponse(iter_range(), status_code=206, headers=headers)

    return FileResponse(file_path, media_type=media_type)
