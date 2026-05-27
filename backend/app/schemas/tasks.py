from pydantic import BaseModel


class TaskResponse(BaseModel):
    id: str
    type: str
    status: str
    progress: int = 0
    message: str = ""
    result: dict | None = None
    error: str | None = None
