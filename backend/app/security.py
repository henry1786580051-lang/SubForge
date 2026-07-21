import tempfile
from pathlib import Path


def _get_allowed_roots() -> list[Path]:
    """Get allowed root directories for file access."""
    from subforge.config import APPDATA_PATH, RESOURCE_PATH, WORK_PATH
    roots = [APPDATA_PATH.resolve(), RESOURCE_PATH.resolve(), WORK_PATH.resolve()]
    # Allow specific home subdirectories (not entire home — prevents ~/.ssh etc.)
    home = Path.home()
    for subdir in ["Desktop", "Downloads", "Documents", "Movies"]:
        p = home / subdir
        if p.exists():
            roots.append(p.resolve())
    roots.append(Path(tempfile.gettempdir()).resolve())
    return roots


def validate_path(user_path: str) -> Path:
    """Resolve and validate that the path is under an allowed root."""
    resolved = Path(user_path).resolve()
    for root in _get_allowed_roots():
        if resolved.is_relative_to(root):
            return resolved
    raise ValueError(f"Path not allowed: {user_path}")
