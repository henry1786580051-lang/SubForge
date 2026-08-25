from subforge.settings import credentials


class _MemoryKeyring:
    priority = 1

    def __init__(self):
        self.values = {}
        self.get_calls = []

    def set_password(self, service, account, value):
        self.values[(service, account)] = value

    def get_password(self, service, account):
        self.get_calls.append((service, account))
        return self.values.get((service, account))

    def delete_password(self, service, account):
        self.values.pop((service, account), None)


def test_settings_secrets_round_trip_through_system_store(monkeypatch):
    backend = _MemoryKeyring()
    monkeypatch.setattr(credentials, "_get_keyring", lambda: backend)
    source = {
        "llm_api_key": "active-secret",
        "huggingface_token": "hf-secret",
        "llm_profiles": {
            "deepseek": {
                "base_url": "https://example.test/v1",
                "model": "model",
                "api_key": "profile-secret",
            }
        },
    }

    protected = credentials.protect_settings_credentials(source)

    assert protected["llm_api_key"].startswith(credentials.REFERENCE_PREFIX)
    assert protected["huggingface_token"].startswith(credentials.REFERENCE_PREFIX)
    assert protected["llm_profiles"]["deepseek"]["api_key"].startswith(
        credentials.REFERENCE_PREFIX
    )
    assert "active-secret" not in str(protected)
    assert credentials.restore_settings_credentials(protected) == source


def test_credentials_fall_back_to_private_settings_when_store_is_unavailable(monkeypatch):
    monkeypatch.setattr(credentials, "_get_keyring", lambda: None)
    source = {"llm_api_key": "legacy-secret"}

    assert credentials.protect_settings_credentials(source) == source
    assert credentials.restore_settings_credentials(source) == source
    reference = credentials.REFERENCE_PREFIX + "missing"
    assert credentials.restore_settings_credentials({"llm_api_key": reference}) == {
        "llm_api_key": reference
    }
    assert credentials.usable_secret_value(reference) == ""


def test_single_secret_is_cached_after_first_keychain_authorization(monkeypatch):
    backend = _MemoryKeyring()
    account = "llm-profile:cached"
    backend.values[(credentials.SERVICE_NAME, account)] = "secret"
    monkeypatch.setattr(credentials, "_get_keyring", lambda: backend)
    monkeypatch.setattr(credentials, "_secret_cache", {})
    reference = credentials.REFERENCE_PREFIX + account

    assert credentials.restore_secret_value(reference) == "secret"
    assert credentials.restore_secret_value(reference) == "secret"
    assert backend.get_calls == [(credentials.SERVICE_NAME, account)]
