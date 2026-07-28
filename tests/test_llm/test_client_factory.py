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
