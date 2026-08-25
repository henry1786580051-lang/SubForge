"""System credential-store integration for persisted application secrets."""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import threading
from typing import Any

from subforge.settings.schema import SECRET_SETTING_KEYS

logger = logging.getLogger(__name__)

SERVICE_NAME = "SubForge"
REFERENCE_PREFIX = "keyring://"

_secret_cache: dict[str, str] = {}
_secret_cache_lock = threading.RLock()


def _account(kind: str, name: str) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:20]
    return f"{kind}:{digest}"


def _get_keyring():
    # Unit tests must never write to the developer's real login keychain.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    try:
        import keyring

        backend = keyring.get_keyring()
        if float(getattr(backend, "priority", 0)) <= 0:
            return None
        return backend
    except Exception:
        logger.debug("System credential store is unavailable", exc_info=True)
        return None


def _store_secret(backend: Any, account: str, value: str) -> str:
    backend.set_password(SERVICE_NAME, account, value)
    with _secret_cache_lock:
        _secret_cache[account] = value
    return REFERENCE_PREFIX + account


def is_credential_reference(value: Any) -> bool:
    return str(value or "").startswith(REFERENCE_PREFIX)


def is_secret_configured(value: Any) -> bool:
    """Return whether a plaintext secret or credential reference is present."""
    return bool(str(value or ""))


def usable_secret_value(value: Any) -> str:
    """Return plaintext credentials only; opaque references are never API keys."""
    text = str(value or "")
    return "" if is_credential_reference(text) else text


def restore_secret_value(value: Any) -> str:
    """Resolve one credential reference on demand and cache it for this run."""
    text = str(value or "")
    if not is_credential_reference(text):
        return text
    account = text[len(REFERENCE_PREFIX) :]
    with _secret_cache_lock:
        cached = _secret_cache.get(account)
    if cached is not None:
        return cached

    backend = _get_keyring()
    if backend is None:
        return text
    try:
        secret = backend.get_password(SERVICE_NAME, account)
    except Exception:
        logger.warning("Unable to read a credential from the system store", exc_info=True)
        return text
    # Keep the opaque reference when the keychain entry is temporarily
    # inaccessible. This prevents an unrelated settings save from erasing it.
    if secret is None:
        return text
    resolved = str(secret)
    with _secret_cache_lock:
        _secret_cache[account] = resolved
    return resolved


def protect_settings_credentials(data: dict[str, Any]) -> dict[str, Any]:
    """Move plaintext secrets into the OS credential store when available."""
    protected = copy.deepcopy(data)
    backend = _get_keyring()
    if backend is None:
        return protected

    try:
        for key in SECRET_SETTING_KEYS:
            value = str(protected.get(key) or "")
            account = _account("setting", key)
            if value.startswith(REFERENCE_PREFIX):
                continue
            if value:
                protected[key] = _store_secret(backend, account, value)
            else:
                try:
                    backend.delete_password(SERVICE_NAME, account)
                except Exception:
                    pass

        profiles = protected.get("llm_profiles")
        if isinstance(profiles, dict):
            for provider, profile in profiles.items():
                if not isinstance(profile, dict):
                    continue
                value = str(profile.get("api_key") or "")
                account = _account("llm-profile", str(provider))
                if value.startswith(REFERENCE_PREFIX):
                    continue
                if value:
                    profile["api_key"] = _store_secret(backend, account, value)
                else:
                    try:
                        backend.delete_password(SERVICE_NAME, account)
                    except Exception:
                        pass
    except Exception:
        logger.warning(
            "Unable to persist credentials in the system store; retaining private settings file",
            exc_info=True,
        )
        return copy.deepcopy(data)
    return protected


def restore_settings_credentials(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve all references for explicit migrations and compatibility tools.

    Normal application reads should call :func:`restore_secret_value` for only
    the credential required by the active operation. Resolving every provider
    at startup causes one macOS Keychain authorization dialog per item.
    """
    restored = copy.deepcopy(data)
    try:
        for key in SECRET_SETTING_KEYS:
            if key in restored:
                restored[key] = restore_secret_value(restored.get(key))
        profiles = restored.get("llm_profiles")
        if isinstance(profiles, dict):
            for profile in profiles.values():
                if isinstance(profile, dict):
                    profile["api_key"] = restore_secret_value(profile.get("api_key"))
    except Exception:
        logger.warning("Unable to read credentials from the system store", exc_info=True)
    return restored
