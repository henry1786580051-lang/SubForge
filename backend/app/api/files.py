import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.core.task_manager import task_manager
from app.security import validate_path

router = APIRouter()

MAX_UPLOAD_SIZE = 10 * 1024 * 1024 * 1024  # 10 GB

UPLOAD_DIR = Path("/tmp/subforge/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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
    dest = UPLOAD_DIR / safe_name

    size = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_SIZE:
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large")
            f.write(chunk)

    return {"file_path": str(dest), "filename": safe_name}


@router.get("/info")
async def get_file_info(path: str = Query(..., description="File path")):
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
                "-v", "quiet",
                "-print_format", "json",
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
        audio_streams = [
            s for s in probe.get("streams", []) if s.get("codec_type") == "audio"
        ]
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
            } if video_stream else None,
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/thumbnail")
async def get_thumbnail(path: str = Query(...)):
    """Generate a video thumbnail."""
    import subprocess

    try:
        file_path = validate_path(path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    thumb_path = UPLOAD_DIR / f"{file_path.stem}_thumb.jpg"

    if not thumb_path.exists():
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(file_path),
                "-ss", "00:00:01",
                "-vframes", "1",
                "-vf", "scale=320:-1",
                str(thumb_path),
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="Thumbnail generation failed")

    return FileResponse(thumb_path, media_type="image/jpeg")


@router.get("/stream")
async def stream_video(path: str = Query(...), request: Request = None):
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
    range_header = request.headers.get("range") if request else None
    if range_header:
        try:
            range_match = range_header.replace("bytes=", "").split("-")
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if range_match[1] else file_size - 1
        except (ValueError, IndexError):
            raise HTTPException(status_code=416, detail="Invalid Range header")
        if start >= file_size or start < 0:
            raise HTTPException(status_code=416, detail="Range out of bounds")
        end = min(end, file_size - 1)

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
