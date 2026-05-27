import logging
import os
import shutil
import sys
from pathlib import Path

try:
    from subforge._version import __version__ as _raw_version
    # Strip dev suffix (e.g. "1.5.0.dev103+g38544177c" → "1.5.0")
    VERSION = _raw_version.split(".dev")[0]
except Exception:
    VERSION = "0.0.0-dev"
YEAR = 2026
APP_NAME = "SubForge"
AUTHOR = "Weifeng"

HELP_URL = "https://github.com/WEIFENG2333/SubForge"
GITHUB_REPO_URL = "https://github.com/WEIFENG2333/SubForge"
RELEASE_URL = "https://github.com/WEIFENG2333/SubForge/releases/latest"
FEEDBACK_URL = "https://github.com/WEIFENG2333/SubForge/issues"

# Detect whether running from source tree or pip-installed
_PACKAGE_DIR = Path(__file__).parent
_PROJECT_ROOT = _PACKAGE_DIR.parent
_IS_FROZEN = getattr(sys, "frozen", False)
_PACKAGE_RESOURCE_PATH = _PACKAGE_DIR / "resources"

# Development mode: resource/ exists next to the package
_IS_DEV = (_PROJECT_ROOT / "resource").is_dir() and not _IS_FROZEN

if _IS_FROZEN:
    from platformdirs import user_data_path

    ROOT_PATH = Path(sys.executable).resolve().parent
    RESOURCE_PATH = Path(getattr(sys, "_MEIPASS")) / "resource"
    APPDATA_PATH = user_data_path(APP_NAME)
    WORK_PATH = Path.home() / APP_NAME
elif _IS_DEV:
    ROOT_PATH = _PROJECT_ROOT
    RESOURCE_PATH = ROOT_PATH / "resource"
    APPDATA_PATH = ROOT_PATH / "AppData"
    WORK_PATH = ROOT_PATH / "work-dir"
else:
    # Installed via pip — use platform-appropriate directories
    from platformdirs import user_data_path

    ROOT_PATH = user_data_path(APP_NAME)
    RESOURCE_PATH = _PACKAGE_RESOURCE_PATH if _PACKAGE_RESOURCE_PATH.exists() else ROOT_PATH / "resource"
    APPDATA_PATH = ROOT_PATH
    WORK_PATH = Path.home() / APP_NAME

ASSETS_PATH = RESOURCE_PATH / "assets"
TRANSLATIONS_PATH = RESOURCE_PATH / "translations"

# Writable user data. Keep generated/downloaded files out of frozen bundles and
# package directories so app upgrades are just replacing the program files.
if _IS_DEV:
    BIN_PATH = RESOURCE_PATH / "bin"
    SUBTITLE_STYLE_PATH = RESOURCE_PATH / "subtitle_style"
    FONTS_PATH = RESOURCE_PATH / "fonts"
else:
    BIN_PATH = APPDATA_PATH / "bin"
    SUBTITLE_STYLE_PATH = APPDATA_PATH / "resource" / "subtitle_style"
    FONTS_PATH = APPDATA_PATH / "resource" / "fonts"

BUNDLED_BIN_PATH = RESOURCE_PATH / "bin"

LOG_PATH = APPDATA_PATH / "logs"
LLM_LOG_FILE = LOG_PATH / "llm_requests.jsonl"
SETTINGS_PATH = APPDATA_PATH / "settings.json"
CACHE_PATH = APPDATA_PATH / "cache"
MODEL_PATH = APPDATA_PATH / "models"

FASTER_WHISPER_PATH = BIN_PATH / "Faster-Whisper-XXL"

# Logging
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def _copy_missing_tree(src: Path, dst: Path) -> None:
    """Copy bundled default files into the writable user directory."""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            _copy_missing_tree(item, target)
        elif not target.exists():
            shutil.copy2(item, target)


# Create data directories
for p in [APPDATA_PATH, CACHE_PATH, LOG_PATH, WORK_PATH, MODEL_PATH, BIN_PATH]:
    p.mkdir(parents=True, exist_ok=True)

if not _IS_DEV:
    _copy_missing_tree(RESOURCE_PATH / "subtitle_style", SUBTITLE_STYLE_PATH)
    _copy_missing_tree(RESOURCE_PATH / "fonts", FONTS_PATH)

# Add bin paths to PATH. User-downloaded binaries take precedence over bundled
# tools, while packaged ffmpeg/ffprobe still work out of the box.
for _path in [FASTER_WHISPER_PATH, BIN_PATH, BUNDLED_BIN_PATH]:
    if _path.exists():
        os.environ["PATH"] = str(_path) + os.pathsep + os.environ["PATH"]

if (BIN_PATH / "vlc").exists():
    os.environ["PYTHON_VLC_MODULE_PATH"] = str(BIN_PATH / "vlc")
