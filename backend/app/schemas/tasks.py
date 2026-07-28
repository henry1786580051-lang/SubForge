from pydantic import BaseModel


class TaskResponse(BaseModel):
    id: str
    type: str
    status: str
    progress: int = 0
    message: str = ""
    result: dict | None = None
    error: str | None = None
    subtitle_file: str | None = None
    preview_segments: list[dict] | None = None
    preview_revision: int = 0
    attention: dict | None = None
