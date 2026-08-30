from subforge.core.llm import client as client_module


def test_global_client_uses_protocol_aware_factory(monkeypatch):
    sentinel = object()
    calls = []
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.minimaxi.com/anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(client_module, "_global_client", None)

    def fake_create_client(base_url, api_key):
        calls.append((base_url, api_key))
        return sentinel

    monkeypatch.setattr(client_module, "create_client", fake_create_client)

    assert client_module.get_llm_client() is sentinel
    assert calls == [("https://api.minimaxi.com/anthropic", "test-key")]


def test_global_client_is_recreated_when_provider_profile_changes(monkeypatch):
    clients = []
    monkeypatch.setattr(client_module, "_global_client", None)
    monkeypatch.setattr(client_module, "_global_client_identity", None)

    def fake_create_client(base_url, api_key):
        client = object()
        clients.append((base_url, api_key, client))
        return client

    monkeypatch.setattr(client_module, "create_client", fake_create_client)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.minimaxi.com/anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "minimax-key")
    first = client_module.get_llm_client()

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("OPENAI_API_KEY", "deepseek-key")
    second = client_module.get_llm_client()

    assert first is not second
    assert [(base, key) for base, key, _ in clients] == [
        ("https://api.minimaxi.com/anthropic", "minimax-key"),
        ("https://api.deepseek.com/v1", "deepseek-key"),
    ]


def test_global_call_cache_is_scoped_to_provider_base_url(monkeypatch):
    calls = []

    def fake_cached(messages, model, temperature, **kwargs):
        calls.append((messages, model, temperature, kwargs))
        return object()

    monkeypatch.setattr(client_module, "_call_llm_cached", fake_cached)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")

    client_module.call_llm([{"role": "user", "content": "hello"}], "shared-model")

    assert calls[0][3]["_subforge_cache_namespace"] == "https://api.deepseek.com/v1"


def test_global_call_cache_appends_pipeline_namespace_without_changing_default(monkeypatch):
    calls = []

    def fake_cached(messages, model, temperature, **kwargs):
        calls.append(kwargs["_subforge_cache_namespace"])
        return object()

    monkeypatch.setattr(client_module, "_call_llm_cached", fake_cached)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")

    client_module.call_llm([{"role": "user", "content": "legacy"}], "shared-model")
    client_module.call_llm(
        [{"role": "user", "content": "candidate"}],
        "shared-model",
        cache_namespace="translation-quality:candidate:phase8-r1",
    )

    assert calls == [
        "https://api.deepseek.com/v1",
        "https://api.deepseek.com/v1|translation-quality:candidate:phase8-r1",
    ]
