import tempfile
import threading
from pathlib import Path

_granted_paths: set[Path] = set()
_grant_lock = threading.RLock()


def grant_path(user_path: str | Path) -> Path:
    """Allow one native-dialog selection for the lifetime of this process."""
    resolved = Path(user_path).expanduser().resolve()
    with _grant_lock:
        _granted_paths.add(resolved)
    return resolved


def clear_granted_paths() -> None:
    """Clear native-dialog grants during desktop-session cleanup and tests."""
    with _grant_lock:
        _granted_paths.clear()


def _is_granted(resolved: Path) -> bool:
    with _grant_lock:
        return resolved in _granted_paths


def _get_allowed_roots() -> list[Path]:
    """Get allowed root directories for file access."""
    from subforge.config import APPDATA_PATH, RESOURCE_PATH, WORK_PATH
    roots = [APPDATA_PATH.resolve(), RESOURCE_PATH.resolve(), WORK_PATH.resolve()]
    # Allow specific home subdirectories (not entire home — prevents ~/.ssh etc.)
    home = Path.home()
    for subdir in ["Desktop", "Downloads", "Documents", "Movies", "Videos"]:
        p = home / subdir
        if p.exists():
            roots.append(p.resolve())
    roots.append(Path(tempfile.gettempdir()).resolve())
    return roots


def validate_path(user_path: str) -> Path:
    """Resolve and validate that the path is under an allowed root."""
    resolved = Path(user_path).expanduser().resolve()
    if _is_granted(resolved):
        return resolved
    for root in _get_allowed_roots():
        if resolved.is_relative_to(root):
            return resolved
    raise ValueError(f"Path not allowed: {user_path}")
