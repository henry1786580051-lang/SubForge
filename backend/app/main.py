import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure subforge package is importable
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from app.api import tasks, transcribe, subtitle, config, websocket, files, subtitles, llm_logs  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create necessary directories
    from subforge.config import APPDATA_PATH
    APPDATA_PATH.mkdir(parents=True, exist_ok=True)
    yield
    # Shutdown: cleanup resources


app = FastAPI(
    title="SubForge API",
    description="AI-powered video captioning backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(transcribe.router, prefix="/api/transcribe", tags=["transcribe"])
app.include_router(subtitle.router, prefix="/api/subtitle", tags=["subtitle"])
app.include_router(config.router, prefix="/api/config", tags=["config"])
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(subtitles.router, prefix="/api/subtitles", tags=["subtitles"])
app.include_router(llm_logs.router, prefix="/api/llm-logs", tags=["llm-logs"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])


@app.get("/api/health")
async def health_check():
    import shutil
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ffprobe_ok = shutil.which("ffprobe") is not None
    return {"status": "ok", "ffmpeg": ffmpeg_ok, "ffprobe": ffprobe_ok}


def _find_static_dir() -> Path | None:
    """Find the frontend static export directory."""
    import sys
    if getattr(sys, 'frozen', False):
        base = Path(sys._MEIPASS)
    else:
        base = _project_root
    static = base / "frontend" / "out"
    if static.is_dir():
        return static
    return None


_static_dir = _find_static_dir()
if _static_dir:
    from fastapi.responses import FileResponse

    app.mount("/_next", StaticFiles(directory=str(_static_dir / "_next")), name="next_static")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = _static_dir / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_static_dir / "index.html"))
