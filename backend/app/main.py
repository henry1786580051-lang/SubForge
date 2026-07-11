import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure subforge package is importable
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from app.api import (  # noqa: E402
    config,
    files,
    llm_logs,
    subtitle,
    subtitles,
    tasks,
    transcribe,
    websocket,
)
from subforge.config import VERSION  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create necessary directories and set event loop
    import asyncio

    from app.api.websocket import set_event_loop
    set_event_loop(asyncio.get_running_loop())
    files.cleanup_stale_uploads()
    from subforge.config import APPDATA_PATH
    APPDATA_PATH.mkdir(parents=True, exist_ok=True)
    yield
    # Shutdown: cleanup resources
    files.cleanup_session_uploads()


app = FastAPI(
    title="SubForge API",
    description="AI-powered video captioning backend",
    version=VERSION,
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

    import subforge

    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ffprobe_ok = shutil.which("ffprobe") is not None
    return {
        "status": "ok",
        "version": VERSION,
        "ffmpeg": ffmpeg_ok,
        "ffprobe": ffprobe_ok,
        "pid": os.getpid(),
        "subforge_module": str(Path(subforge.__file__).resolve()),
        "timeline_overlap_fix": True,
    }


def _find_static_dir() -> Path | None:
    """Find the frontend static export directory."""
    import sys
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS"))
    else:
        base = _project_root
    static = base / "frontend" / "out"
    if static.is_dir():
        return static
    return None


_static_dir = _find_static_dir()
if _static_dir:
    from fastapi.responses import FileResponse

    static_root = _static_dir
    app.mount("/_next", StaticFiles(directory=str(static_root / "_next")), name="next_static")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = (static_root / full_path).resolve()
        if not file_path.is_relative_to(static_root.resolve()):
            return FileResponse(str(static_root / "index.html"))
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(static_root / "index.html"))
