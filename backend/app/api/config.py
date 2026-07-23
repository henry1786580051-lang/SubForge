import json
import os
import platform
import threading
import time
from pathlib import Path
from typing import TypeVar, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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
_WHISPERX_SUPPORTED = _IS_APPLE_SILICON or platform.system() in {"Windows", "Linux"}

_LLM_PROVIDER_URLS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
    "mimo": "https://token-plan-cn.xiaomimimo.com/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "moonshot": "https://api.moonshot.cn/v1",
    "baichuan": "https://api.baichuan-ai.com/v1",
    "yi": "https://api.lingyiwanwu.com/v1",
    "minimax": "https://api.minimaxi.com/anthropic",
    "siliconflow": "https://api.siliconflow.cn/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "custom": "",
}
_LEGACY_MINIMAX_URLS = {
    "https://api.minimax.chat/v1",
    "https://api.minimaxi.com/v1",
}

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
    "llm_provider": "custom",
    "llm_profiles": {},
    "llm_log_level": "summary",
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
    "whisperx_alignment_strategy": "auto",
    "whisperx_align_model": "WAV2VEC2_ASR_LARGE_LV60K_960H",
    "whisperx_batch_size": 8,
    "ff_mdx_kim2": False,
    "enable_audio_enhancement": True,
    "speaker_diarization": "off",
    "speaker_count": 2,
    "diarization_model": "pyannote/speaker-diarization-community-1",
    "huggingface_token": "",
    "replace_chinese_punctuation": True,
    "whisper_model_size": "large-v3" if _IS_APPLE_SILICON else "base",
}


def _detect_llm_provider(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if normalized.startswith(("https://api.minimax.chat", "https://api.minimaxi.com")):
        return "minimax"
    for provider, default_url in _LLM_PROVIDER_URLS.items():
        if default_url and normalized.startswith(default_url.rstrip("/")):
            return provider
    return "custom"


def _sanitize_llm_profiles(value) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        return {}
    profiles = {}
    for provider, profile in value.items():
        if provider not in _LLM_PROVIDER_URLS or not isinstance(profile, dict):
            continue
        base_url = str(profile.get("base_url") or "")[:8192].rstrip("/")
        if provider == "minimax" and base_url in _LEGACY_MINIMAX_URLS:
            base_url = _LLM_PROVIDER_URLS["minimax"]
        profiles[provider] = {
            "base_url": base_url,
            "api_key": str(profile.get("api_key") or "")[:8192],
            "model": str(profile.get("model") or "")[:256],
        }
    return profiles


def _active_llm_provider(stored: dict) -> str:
    provider = stored.get("llm_provider")
    if provider in _LLM_PROVIDER_URLS:
        return provider
    return _detect_llm_provider(str(stored.get("llm_base_url") or ""))


def _effective_config(stored: dict) -> dict:
    """Apply platform constraints to persisted settings without rewriting them."""
    config = {
        key: _coerce_config_value(stored.get(key, default), default)
        for key, default in _DEFAULTS.items()
    }
    if not _WHISPERX_SUPPORTED and config.get("transcribe_model") == "whisperx":
        config["transcribe_model"] = "whisper_cpp"
    if "whisperx_alignment_strategy" not in stored:
        legacy_align_model = str(stored.get("whisperx_align_model") or "")
        if legacy_align_model and legacy_align_model != "WAV2VEC2_ASR_LARGE_LV60K_960H":
            config["whisperx_alignment_strategy"] = "manual"
    provider = _active_llm_provider(stored)
    profiles = _sanitize_llm_profiles(stored.get("llm_profiles"))
    if provider not in profiles and any(
        str(stored.get(key) or "").strip() for key in ("llm_base_url", "llm_api_key", "llm_model")
    ):
        profiles[provider] = {
            "base_url": str(stored.get("llm_base_url") or ""),
            "api_key": str(stored.get("llm_api_key") or ""),
            "model": str(stored.get("llm_model") or ""),
        }
    config["llm_provider"] = provider
    config["llm_profiles"] = profiles
    active_profile = profiles.get(provider)
    if active_profile:
        config["llm_base_url"] = active_profile["base_url"]
        config["llm_api_key"] = active_profile["api_key"]
        config["llm_model"] = active_profile["model"]
    return config


def _public_config(config: dict) -> dict:
    """Return configuration metadata without exposing persisted credentials."""
    public = dict(config)
    public["llm_api_key_configured"] = bool(config.get("llm_api_key"))
    public["whisper_api_key_configured"] = bool(config.get("whisper_api_key"))
    public["huggingface_token_configured"] = bool(config.get("huggingface_token"))
    public["llm_api_key"] = ""
    public["whisper_api_key"] = ""
    public["huggingface_token"] = ""
    public["llm_profiles"] = {
        provider: {
            "base_url": profile.get("base_url", ""),
            "model": profile.get("model", ""),
            "api_key_configured": bool(profile.get("api_key")),
        }
        for provider, profile in config.get("llm_profiles", {}).items()
        if isinstance(profile, dict)
    }
    return public


class ConfigUpdate(BaseModel):
    key: str
    value: str | int | float | bool


class LlmProviderSwitch(BaseModel):
    provider: str = Field(max_length=64)
    current_base_url: str = Field(default="", max_length=8192)
    current_api_key: str = Field(default="", max_length=8192)
    current_model: str = Field(default="", max_length=256)


_INTEGER_RANGES = {
    "font_size": (8, 200),
    "max_word_count_cjk": (1, 200),
    "max_word_count_english": (1, 200),
    "thread_num": (1, 32),
    "batch_size": (1, 100),
    "whisper_n_threads": (0, 128),
    "whisperx_batch_size": (1, 64),
    "speaker_count": (2, 10),
}

_CHOICES = {
    "transcribe_model": {"whisperx", "whisper_cpp", "faster_whisper", "whisper_api"},
    "translator": {"llm", "bing", "google", "deeplx"},
    "target_language": {
        "chinese",
        "english",
        "japanese",
        "korean",
        "french",
        "german",
        "spanish",
        "portuguese",
        "russian",
        "cantonese",
        "thai",
        "vietnamese",
        "indonesian",
        "malay",
        "tagalog",
        "italian",
        "dutch",
        "polish",
        "turkish",
        "swedish",
        "ukrainian",
        "arabic",
    },
    "llm_provider": set(_LLM_PROVIDER_URLS),
    "llm_log_level": {"summary", "standard", "debug"},
    "speaker_diarization": {"off", "two", "auto", "fixed"},
    "whisperx_alignment_strategy": {"auto", "manual"},
}


def _validate_config_update(key: str, value: str | int | float | bool):
    default = _DEFAULTS[key]
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise HTTPException(status_code=422, detail=f"{key} must be a boolean")
    elif isinstance(default, int):
        if not isinstance(value, int) or isinstance(value, bool):
            raise HTTPException(status_code=422, detail=f"{key} must be an integer")
    elif isinstance(default, float):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise HTTPException(status_code=422, detail=f"{key} must be a number")
        value = float(value)
    elif isinstance(default, str):
        if not isinstance(value, str):
            raise HTTPException(status_code=422, detail=f"{key} must be a string")
        max_length = 100_000 if key == "custom_prompt" else 8_192
        if len(value) > max_length:
            raise HTTPException(status_code=422, detail=f"{key} is too long")

    if key in _INTEGER_RANGES:
        if not isinstance(value, int) or isinstance(value, bool):
            raise HTTPException(status_code=422, detail=f"{key} must be an integer")
        minimum, maximum = _INTEGER_RANGES[key]
        if not minimum <= value <= maximum:
            raise HTTPException(
                status_code=422,
                detail=f"{key} must be between {minimum} and {maximum}",
            )
    if key == "outline_width":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise HTTPException(status_code=422, detail="outline_width must be a number")
        if not 0 <= value <= 20:
            raise HTTPException(status_code=422, detail="outline_width must be between 0 and 20")
    if key in _CHOICES and value not in _CHOICES[key]:
        raise HTTPException(status_code=422, detail=f"Unsupported {key}: {value}")
    return value


@router.get("/")
async def get_config():
    """Get current application configuration."""
    stored = _read_settings()
    config = _public_config(_effective_config(stored))
    return {
        **config,
        "runtime_platform": platform.system().lower(),
        "whisperx_supported": _WHISPERX_SUPPORTED,
        "whisperx_backend": "mlx" if _IS_APPLE_SILICON else "faster-whisper",
    }


@router.post("/")
async def update_config(update: ConfigUpdate):
    """Update a configuration value."""
    if update.key not in _DEFAULTS:
        raise HTTPException(status_code=400, detail=f"Unknown config key: {update.key}")
    if update.key in {"llm_provider", "llm_profiles"}:
        raise HTTPException(
            status_code=400,
            detail="LLM providers must be changed through the provider switch endpoint",
        )
    value = _validate_config_update(update.key, update.value)
    if update.key == "transcribe_model" and value == "whisperx" and not _WHISPERX_SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail="当前平台不支持此 WhisperX 桌面运行时。",
        )
    stored = _read_settings()
    stored[update.key] = value
    if update.key in {"llm_base_url", "llm_api_key", "llm_model"}:
        provider = _active_llm_provider(stored)
        profiles = _sanitize_llm_profiles(stored.get("llm_profiles"))
        profile = profiles.setdefault(
            provider,
            {
                "base_url": str(stored.get("llm_base_url") or ""),
                "api_key": str(stored.get("llm_api_key") or ""),
                "model": str(stored.get("llm_model") or ""),
            },
        )
        profile_key = {
            "llm_base_url": "base_url",
            "llm_api_key": "api_key",
            "llm_model": "model",
        }[update.key]
        profile[profile_key] = str(value)
        stored["llm_profiles"] = profiles
    _write_settings(stored)
    invalidate_config_cache()
    response_value = (
        "" if update.key in {"llm_api_key", "whisper_api_key", "huggingface_token"} else value
    )
    return {"status": "ok", "key": update.key, "value": response_value}


@router.post("/llm-provider")
async def switch_llm_provider(update: LlmProviderSwitch):
    """Atomically save the current LLM profile and activate another one."""
    if update.provider not in _LLM_PROVIDER_URLS:
        raise HTTPException(status_code=422, detail="Unsupported LLM provider")

    stored = _read_settings()
    current_provider = _active_llm_provider(stored)
    profiles = _sanitize_llm_profiles(stored.get("llm_profiles"))
    current_profile = profiles.get(current_provider, {})
    profiles[current_provider] = {
        "base_url": update.current_base_url.strip(),
        "api_key": update.current_api_key
        or str(current_profile.get("api_key") or stored.get("llm_api_key") or ""),
        "model": update.current_model.strip(),
    }
    target = profiles.get(update.provider)
    if target is None:
        target = {
            "base_url": _LLM_PROVIDER_URLS[update.provider],
            "api_key": "",
            "model": "",
        }
        profiles[update.provider] = target

    stored.update(
        {
            "llm_provider": update.provider,
            "llm_base_url": target["base_url"],
            "llm_api_key": target["api_key"],
            "llm_model": target["model"],
            "llm_profiles": profiles,
        }
    )
    _write_settings(stored)
    invalidate_config_cache()
    return {
        "status": "ok",
        "provider": update.provider,
        "base_url": target["base_url"],
        "api_key_configured": bool(target["api_key"]),
        "model": target["model"],
    }


@router.get("/test-llm")
async def test_llm_connection():
    """Test LLM connection with current config."""
    stored = _read_settings()
    config = {**_DEFAULTS, **stored}

    from subforge.core.llm.client import is_anthropic_base_url, normalize_base_url

    raw_base_url = (config.get("llm_base_url") or "").strip()
    base_url = normalize_base_url(raw_base_url).rstrip("/") if raw_base_url else ""
    api_key = config.get("llm_api_key") or ""
    model = config.get("llm_model") or ""

    if not base_url or not api_key:
        return {"ok": False, "error": "未配置 Base URL 或 API Key"}

    try:
        import httpx

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            anthropic_api = is_anthropic_base_url(base_url)
            resp = await client.post(
                f"{base_url}/v1/messages" if anthropic_api else f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 16 if anthropic_api else 5,
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

    from subforge.core.llm.client import is_anthropic_base_url, normalize_base_url

    raw_base_url = (config.get("llm_base_url") or "").strip()
    base_url = normalize_base_url(raw_base_url).rstrip("/") if raw_base_url else ""
    api_key = config.get("llm_api_key") or ""

    if not base_url:
        return {"error": "未配置 Base URL", "models": []}

    # Most OpenAI-compatible APIs support GET /models
    url = f"{base_url}/v1/models" if is_anthropic_base_url(base_url) else f"{base_url}/models"
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
