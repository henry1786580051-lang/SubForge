import asyncio
import json
import os
import platform
import threading
import time
from pathlib import Path
from typing import TypeVar
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from subforge.core.split.length_policy import resolve_length_policy
from subforge.settings import (
    LEGACY_MINIMAX_URLS,
    LLM_PROVIDER_URLS,
    SECRET_SETTING_KEYS,
    LlmRuntimeConfig,
    coerce_flat_settings,
    coerce_setting_value,
    default_settings_dict,
    detect_llm_provider,
    validate_llm_runtime_config,
)
from subforge.settings.credentials import (
    is_secret_configured,
    protect_settings_credentials,
    restore_secret_value,
    usable_secret_value,
)

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
    """Read persisted settings without unlocking OS credential entries."""
    with _settings_lock:
        path = _find_settings_path()
        if path:
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
                return stored if isinstance(stored, dict) else {}
            except (json.JSONDecodeError, OSError):
                pass
        return {}


_settings_cache: dict | None = None
_cache_time: float = 0
_CACHE_TTL = 5.0


def get_config_value(key: str, default: T) -> T:
    """Read a single config value with TTL cache."""
    global _settings_cache, _cache_time
    with _settings_lock:
        if key in SECRET_SETTING_KEYS:
            stored = _read_settings()
            value = usable_secret_value(
                restore_secret_value(_stored_secret_value(stored, key))
            )
            return coerce_setting_value(value, default)
        now = time.monotonic()
        if _settings_cache is None or (now - _cache_time) > _CACHE_TTL:
            _settings_cache = _effective_config(_read_settings())
            _cache_time = now
        return coerce_setting_value(_settings_cache.get(key, default), default)


def invalidate_config_cache():
    """Invalidate the config cache so next read fetches fresh values."""
    global _settings_cache
    with _settings_lock:
        _settings_cache = None


def _write_settings(data: dict):
    with _settings_lock:
        persisted = protect_settings_credentials(data)
        path = _find_settings_path()
        if not path:
            # Create in the first candidate location
            path = _SETTINGS_CANDIDATES[0]
            if path is None:
                path = Path.home() / "SubForge" / "settings.json"
            path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.tmp")
        try:
            temp_path.write_text(
                json.dumps(persisted, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
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

_LLM_PROVIDER_URLS = LLM_PROVIDER_URLS
_LEGACY_MINIMAX_URLS = LEGACY_MINIMAX_URLS

_DEFAULTS = default_settings_dict(apple_silicon=_IS_APPLE_SILICON)


def _detect_llm_provider(base_url: str) -> str:
    return detect_llm_provider(base_url)


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


def _repair_llm_profile(provider: str, profile: dict[str, str]) -> dict[str, str]:
    """Repair legacy mixed-provider fields without discarding credentials."""
    repaired = {
        "base_url": str(profile.get("base_url") or "").strip(),
        "api_key": str(profile.get("api_key") or ""),
        "model": str(profile.get("model") or "").strip(),
    }
    if provider != "custom" and _detect_llm_provider(repaired["base_url"]) != provider:
        repaired["base_url"] = _LLM_PROVIDER_URLS[provider]
    try:
        validate_llm_runtime_config(
            LlmRuntimeConfig(provider=provider, **repaired)
        )
    except ValueError:
        repaired["model"] = ""
    return repaired


def _active_llm_provider(stored: dict) -> str:
    provider = stored.get("llm_provider")
    if provider in _LLM_PROVIDER_URLS:
        return provider
    return _detect_llm_provider(str(stored.get("llm_base_url") or ""))


def _stored_secret_value(stored: dict, key: str) -> str:
    """Select one persisted secret without resolving unrelated credentials."""
    if key == "llm_api_key":
        provider = _active_llm_provider(stored)
        profiles = _sanitize_llm_profiles(stored.get("llm_profiles"))
        profile = profiles.get(provider)
        if profile and profile.get("api_key"):
            return str(profile["api_key"])
    return str(stored.get(key) or "")


def _effective_config(stored: dict, *, preserve_secret_references: bool = False) -> dict:
    """Apply platform constraints to persisted settings without rewriting them."""
    config = coerce_flat_settings(stored, defaults=_DEFAULTS)
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
    if not preserve_secret_references:
        for key in SECRET_SETTING_KEYS:
            config[key] = usable_secret_value(config.get(key))
        for profile in profiles.values():
            profile["api_key"] = usable_secret_value(profile.get("api_key"))
    return config


def get_llm_runtime_config() -> LlmRuntimeConfig:
    """Read and validate one atomic active-provider snapshot for a task."""
    with _settings_lock:
        stored = _read_settings()
        config = _effective_config(stored)
        runtime = LlmRuntimeConfig(
            provider=str(config.get("llm_provider") or "custom"),
            base_url=str(config.get("llm_base_url") or "").strip(),
            api_key=usable_secret_value(
                restore_secret_value(_stored_secret_value(stored, "llm_api_key"))
            ),
            model=str(config.get("llm_model") or "").strip(),
        )
    validate_llm_runtime_config(runtime)
    return runtime


def get_llm_provider_runtime_config(provider: str) -> LlmRuntimeConfig:
    """Resolve one saved provider profile without changing the active provider."""
    if provider not in _LLM_PROVIDER_URLS:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    with _settings_lock:
        stored = _read_settings()
        profiles = _sanitize_llm_profiles(stored.get("llm_profiles"))
        profile = profiles.get(provider)
        if profile is None and _active_llm_provider(stored) == provider:
            profile = {
                "base_url": str(stored.get("llm_base_url") or ""),
                "api_key": str(stored.get("llm_api_key") or ""),
                "model": str(stored.get("llm_model") or ""),
            }
        profile = profile or {
            "base_url": _LLM_PROVIDER_URLS[provider],
            "api_key": "",
            "model": "",
        }
        runtime = LlmRuntimeConfig(
            provider=provider,
            base_url=str(profile.get("base_url") or _LLM_PROVIDER_URLS[provider]).strip(),
            api_key=usable_secret_value(
                restore_secret_value(profile.get("api_key"))
            ),
            model=str(profile.get("model") or "").strip(),
        )
    validate_llm_runtime_config(runtime)
    return runtime


def get_llm_provider_status(provider: str) -> dict[str, str | bool]:
    """Return non-secret saved profile metadata without opening the keychain."""
    if provider not in _LLM_PROVIDER_URLS:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    with _settings_lock:
        stored = _read_settings()
        profiles = _sanitize_llm_profiles(stored.get("llm_profiles"))
        profile = profiles.get(provider)
        if profile is None and _active_llm_provider(stored) == provider:
            profile = {
                "base_url": str(stored.get("llm_base_url") or ""),
                "api_key": str(stored.get("llm_api_key") or ""),
                "model": str(stored.get("llm_model") or ""),
            }
        profile = profile or {}
    return {
        "provider": provider,
        "base_url": str(profile.get("base_url") or _LLM_PROVIDER_URLS[provider]).strip(),
        "model": str(profile.get("model") or "").strip(),
        "api_key_configured": is_secret_configured(profile.get("api_key")),
    }


def _public_config(config: dict) -> dict:
    """Return configuration metadata without exposing persisted credentials."""
    public = dict(config)
    for key in SECRET_SETTING_KEYS:
        public[f"{key}_configured"] = is_secret_configured(config.get(key))
        public[key] = ""
    public["llm_profiles"] = {
        provider: {
            "base_url": profile.get("base_url", ""),
            "model": profile.get("model", ""),
            "api_key_configured": is_secret_configured(profile.get("api_key")),
        }
        for provider, profile in config.get("llm_profiles", {}).items()
        if isinstance(profile, dict)
    }
    length_policy = resolve_length_policy(
        config.get("max_word_count_cjk", _DEFAULTS["max_word_count_cjk"]),
        config.get("max_word_count_english", _DEFAULTS["max_word_count_english"]),
    )
    public["subtitle_length_policy"] = {
        "cjk_hard_limit": length_policy.cjk_hard_limit,
        "english_soft_limit": length_policy.english_soft_limit,
        "english_hard_limit": length_policy.english_hard_limit,
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
    if key == "azure_translator_endpoint":
        parsed = urlparse(str(value).strip())
        if parsed.scheme != "https" or not parsed.hostname or parsed.query or parsed.fragment:
            raise HTTPException(
                status_code=422,
                detail="azure_translator_endpoint must be an HTTPS service endpoint",
            )
    if key in _CHOICES and value not in _CHOICES[key]:
        raise HTTPException(status_code=422, detail=f"Unsupported {key}: {value}")
    return value


@router.get("/")
async def get_config():
    """Get current application configuration."""
    stored = _read_settings()
    config = _public_config(
        _effective_config(stored, preserve_secret_references=True)
    )
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
    if update.key in {"llm_provider", "llm_profiles", "schema_version"}:
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
    if update.key in {"llm_base_url", "llm_model"}:
        active = _effective_config(stored)
        prospective = LlmRuntimeConfig(
            provider=str(active.get("llm_provider") or "custom"),
            base_url=(
                str(value).strip()
                if update.key == "llm_base_url"
                else str(active.get("llm_base_url") or "").strip()
            ),
            api_key=str(active.get("llm_api_key") or ""),
            model=(
                str(value).strip()
                if update.key == "llm_model"
                else str(active.get("llm_model") or "").strip()
            ),
        )
        try:
            validate_llm_runtime_config(prospective)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
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
    response_value = "" if update.key in SECRET_SETTING_KEYS else value
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
    profiles[current_provider] = _repair_llm_profile(
        current_provider,
        {
            "base_url": update.current_base_url,
            "api_key": update.current_api_key
            or str(current_profile.get("api_key") or stored.get("llm_api_key") or ""),
            "model": update.current_model,
        },
    )
    target = profiles.get(update.provider)
    if target is None:
        target = {
            "base_url": _LLM_PROVIDER_URLS[update.provider],
            "api_key": "",
            "model": "",
        }
        profiles[update.provider] = target
    target = _repair_llm_profile(update.provider, target)
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
    from subforge.core.llm import close_client, create_client
    from subforge.core.llm.client import normalize_base_url

    try:
        runtime = get_llm_runtime_config()
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    raw_base_url = runtime.base_url
    base_url = normalize_base_url(raw_base_url).rstrip("/") if raw_base_url else ""
    api_key = runtime.api_key
    model = runtime.model

    if not base_url or not api_key or not model:
        return {"ok": False, "error": "未配置 Base URL、API Key 或模型"}

    client = None
    try:
        client = create_client(base_url=base_url, api_key=api_key, timeout=15.0)
        await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            temperature=0,
            max_tokens=16,
        )
        return {"ok": True, "model": model}
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        prefix = f"HTTP {status_code}: " if status_code else ""
        return {"ok": False, "error": f"{prefix}{str(exc)[:200]}"}
    finally:
        if client is not None:
            close_client(client)


@router.get("/test-whisper")
async def test_whisper_connection():
    """Test Whisper API connection."""
    config = _effective_config(_read_settings())

    base_url = (config.get("whisper_base_url") or "").rstrip("/")
    api_key = get_config_value("whisper_api_key", "")

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


@router.get("/test-azure-translator")
async def test_azure_translator_connection():
    """Test the official Azure Translator credentials with a minimal request."""
    config = _effective_config(_read_settings())
    api_key = str(get_config_value("azure_translator_key", "") or "").strip()
    if not api_key:
        return {"ok": False, "error": "未配置 Microsoft Azure Translator API Key"}

    from subforge.core.translate.bing_translator import BingTranslator
    from subforge.core.translate.types import TargetLanguage

    translator = None
    try:
        translator = BingTranslator(
            thread_num=1,
            batch_num=1,
            target_language=TargetLanguage.SIMPLIFIED_CHINESE,
            update_callback=None,
            use_cache=False,
            api_key=api_key,
            region=str(config.get("azure_translator_region") or ""),
            endpoint=str(config.get("azure_translator_endpoint") or ""),
            timeout=15,
            max_retries=1,
        )
        translated = await asyncio.to_thread(translator.test_connection)
        return {"ok": True, "translated": translated}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:300]}
    finally:
        if translator is not None:
            translator.stop()


@router.get("/whisper-models")
async def list_whisper_models():
    """Fetch available models from the configured Whisper API provider."""
    import httpx

    config = _effective_config(_read_settings())

    base_url = (config.get("whisper_base_url") or "").rstrip("/")
    api_key = get_config_value("whisper_api_key", "")

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

    from subforge.core.llm.client import is_anthropic_base_url, normalize_base_url

    try:
        runtime = get_llm_runtime_config()
    except ValueError as exc:
        return {"error": str(exc), "models": []}

    raw_base_url = runtime.base_url
    base_url = normalize_base_url(raw_base_url).rstrip("/") if raw_base_url else ""
    api_key = runtime.api_key

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
