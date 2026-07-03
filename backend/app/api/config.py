import json
import os
import platform
import threading
import time
from pathlib import Path
from typing import TypeVar, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

_settings_lock = threading.RLock()
T = TypeVar("T")

# Path to subforge settings.json
try:
    from subforge.config import SETTINGS_PATH
except Exception:
    SETTINGS_PATH = None

_SETTINGS_CANDIDATES = [
    SETTINGS_PATH,
    Path.home() / "SubForge" / "settings.json",
    Path.home() / "Desktop" / "Project" / "SubForge" / "AppData" / "settings.json",
]


def _find_settings_path() -> Path | None:
    for p in _SETTINGS_CANDIDATES:
        if p and p.exists():
            return p
    return None


def _read_settings() -> dict:
    with _settings_lock:
        path = _find_settings_path()
        if path:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}


_settings_cache: dict | None = None
_cache_time: float = 0
_CACHE_TTL = 5.0


def _coerce_config_value(value, default: T) -> T:
    """Return a persisted value only when it matches the default's type."""
    if isinstance(default, bool):
        return cast(T, value if isinstance(value, bool) else default)
    if isinstance(default, int):
        return cast(T, value if isinstance(value, int) and not isinstance(value, bool) else default)
    if isinstance(default, float):
        return cast(T, float(value) if isinstance(value, (int, float)) else default)
    if isinstance(default, str):
        return cast(T, value if isinstance(value, str) else default)
    return cast(T, value if isinstance(value, type(default)) else default)


def get_config_value(key: str, default: T) -> T:
    """Read a single config value with TTL cache."""
    global _settings_cache, _cache_time
    with _settings_lock:
        now = time.monotonic()
        if _settings_cache is None or (now - _cache_time) > _CACHE_TTL:
            _settings_cache = _effective_config(_read_settings())
            _cache_time = now
        return _coerce_config_value(_settings_cache.get(key, default), default)


def invalidate_config_cache():
    """Invalidate the config cache so next read fetches fresh values."""
    global _settings_cache
    with _settings_lock:
        _settings_cache = None


def _write_settings(data: dict):
    with _settings_lock:
        path = _find_settings_path()
        if not path:
            # Create in the first candidate location
            path = _SETTINGS_CANDIDATES[0]
            if path is None:
                path = Path.home() / "SubForge" / "settings.json"
            path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.tmp")
        try:
            temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                os.chmod(temp_path, 0o600)
            except OSError:
                pass
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)


# Default config values
_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine().lower() == "arm64"

_DEFAULTS = {
    "transcribe_model": "whisperx" if _IS_APPLE_SILICON else "whisper_cpp",
    "source_language": "auto",
    "target_language": "chinese",
    "translator": "bing",
    "work_dir": "",
    "font_name": "Noto Sans SC",
    "font_size": 40,
    "font_color": "#ffffff",
    "outline_color": "#000000",
    "outline_width": 2.0,
    "bold": True,
    "subtitle_style": "classic",
    "show_bilingual": True,
    "need_optimize": True,
    "need_translate": True,
    "need_reflect": False,
    "llm_base_url": "",
    "llm_api_key": "",
    "llm_model": "gpt-4o-mini",
    "max_word_count_cjk": 25,
    "max_word_count_english": 18,
    "thread_num": 5,
    "batch_size": 10,
    "custom_prompt": "",
    "whisper_model_dir": "",
    "whisper_cpp_path": "",
    "whisper_base_url": "",
    "whisper_api_key": "",
    "whisper_api_model": "whisper-1",
    "whisper_device": "auto",
    "whisper_n_threads": 4,
    "whisper_compute_type": "default",
    "whisperx_align_model": "WAV2VEC2_ASR_LARGE_LV60K_960H",
    "whisperx_batch_size": 8,
    "ff_mdx_kim2": False,
    "enable_audio_enhancement": True,
    "whisper_model_size": "large-v3" if _IS_APPLE_SILICON else "base",
}


def _effective_config(stored: dict) -> dict:
    """Apply platform constraints to persisted settings without rewriting them."""
    config = {**_DEFAULTS, **stored}
    if not _IS_APPLE_SILICON and config.get("transcribe_model") == "whisperx":
        config["transcribe_model"] = "whisper_cpp"
    return config


class ConfigUpdate(BaseModel):
    key: str
    value: str | int | float | bool


@router.get("/")
async def get_config():
    """Get current application configuration."""
    stored = _read_settings()
    config = _effective_config(stored)
    return {
        **config,
        "runtime_platform": platform.system().lower(),
        "whisperx_supported": _IS_APPLE_SILICON,
    }


@router.post("/")
async def update_config(update: ConfigUpdate):
    """Update a configuration value."""
    if update.key not in _DEFAULTS:
        raise HTTPException(status_code=400, detail=f"Unknown config key: {update.key}")
    if update.key == "transcribe_model" and update.value == "whisperx" and not _IS_APPLE_SILICON:
        raise HTTPException(
            status_code=400,
            detail="WhisperX MLX 转录仅支持 Apple Silicon；Windows 请使用 Whisper.cpp。",
        )
    stored = _read_settings()
    stored[update.key] = update.value
    _write_settings(stored)
    invalidate_config_cache()
    return {"status": "ok", "key": update.key, "value": update.value}


@router.get("/test-llm")
async def test_llm_connection():
    """Test LLM connection with current config."""
    stored = _read_settings()
    config = {**_DEFAULTS, **stored}

    from subforge.core.llm.client import normalize_base_url

    raw_base_url = (config.get("llm_base_url") or "").strip()
    base_url = normalize_base_url(raw_base_url).rstrip("/") if raw_base_url else ""
    api_key = config.get("llm_api_key") or ""
    model = config.get("llm_model") or ""

    if not base_url or not api_key:
        return {"ok": False, "error": "未配置 Base URL 或 API Key"}

    try:
        import httpx

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5,
                },
            )
            resp.raise_for_status()
            return {"ok": True, "model": model}
    except httpx.HTTPStatusError as e:
        return {"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


@router.get("/test-whisper")
async def test_whisper_connection():
    """Test Whisper API connection."""
    stored = _read_settings()
    config = {**_DEFAULTS, **stored}

    base_url = (config.get("whisper_base_url") or "").rstrip("/")
    api_key = config.get("whisper_api_key") or ""

    if not base_url:
        return {"ok": False, "error": "未配置 Whisper API Base URL"}

    try:
        import httpx

        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            )
            resp.raise_for_status()
            return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


@router.get("/whisper-models")
async def list_whisper_models():
    """Fetch available models from the configured Whisper API provider."""
    import httpx

    stored = _read_settings()
    config = {**_DEFAULTS, **stored}

    base_url = (config.get("whisper_base_url") or "").rstrip("/")
    api_key = config.get("whisper_api_key") or ""

    if not base_url:
        return {"error": "未配置 Whisper API Base URL", "models": []}

    url = f"{base_url}/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        models = []
        if isinstance(data, dict) and "data" in data:
            for m in data["data"]:
                mid = m.get("id", "")
                if mid:
                    models.append(mid)
        elif isinstance(data, list):
            for m in data:
                mid = m.get("id", "") if isinstance(m, dict) else str(m)
                if mid:
                    models.append(mid)

        models.sort()
        return {"models": models}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}", "models": []}
    except Exception as e:
        return {"error": str(e)[:200], "models": []}


@router.get("/models")
async def list_llm_models():
    """Fetch available models from the configured LLM provider."""
    import httpx

    stored = _read_settings()
    config = {**_DEFAULTS, **stored}

    from subforge.core.llm.client import normalize_base_url

    raw_base_url = (config.get("llm_base_url") or "").strip()
    base_url = normalize_base_url(raw_base_url).rstrip("/") if raw_base_url else ""
    api_key = config.get("llm_api_key") or ""

    if not base_url:
        return {"error": "未配置 Base URL", "models": []}

    # Most OpenAI-compatible APIs support GET /models
    url = f"{base_url}/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        # Parse response — OpenAI format: { data: [{ id: "model-name", ... }] }
        models = []
        if isinstance(data, dict) and "data" in data:
            for m in data["data"]:
                mid = m.get("id", "")
                if mid:
                    models.append(mid)
        elif isinstance(data, list):
            for m in data:
                mid = m.get("id", "") if isinstance(m, dict) else str(m)
                if mid:
                    models.append(mid)

        models.sort()
        return {"models": models}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}", "models": []}
    except Exception as e:
        return {"error": str(e)[:200], "models": []}
