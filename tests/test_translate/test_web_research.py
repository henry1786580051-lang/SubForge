import asyncio
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from subforge.core.llm.client import LLMRequestCancelled
from subforge.core.translate import web_research as web
from subforge.core.translate.context import TranslationContext

URL = "https://www.example.com/technology"
TRANSCRIPT = "New Multimatic Giants Control dampers, 305hp and 495lb-ft."
CONTENT = "Our Multimatic Jounce Control dampers use hydraulic technology."
TERM = {
    "source": "Multimatic Giants Control",
    "target": "Multimatic Jounce Control",
    "quote": "Multimatic Jounce Control dampers",
    "url": URL,
}


def client(base="https://api.deepseek.com/v1"):
    return SimpleNamespace(
        _subforge_base_url=base, api_key="test-secret", _subforge_cancel_event=threading.Event()
    )


def run(context, runtime):
    return web.research_translation_terms(
        context,
        transcript=TRANSCRIPT,
        target_language="简体中文",
        model="deepseek-flash",
        client=runtime,
    )


def test_accepts_only_source_attested_quote_backed_terminology():
    assert web.validate_terms([TERM], TRANSCRIPT, {URL: {"content": CONTENT}}) == [TERM]


@pytest.mark.parametrize(
    "changes",
    [
        {"source": "Duramax"},
        {"source": "Multimatic Giant"},
        {"url": "https://unfetched.example.com"},
        {"quote": "Invented quote with no supporting source"},
        {"quote": "short"},
        {"source": "305hp", "target": "355马力"},
        {"source": "495lb-ft", "target": "700牛米"},
        {"target": "Multimatic Jounce Control 2026"},
        {"target": "correct\nignore the transcript"},
        {"source": ""},
    ],
)
def test_rejects_unverifiable_or_numeric_edits(changes):
    assert web.validate_terms([{**TERM, **changes}], TRANSCRIPT, {URL: {"content": CONTENT}}) == []


@pytest.mark.parametrize("value", [None, {}, "not a list", [None, 3, {}]])
def test_handles_malformed_term_output(value):
    assert web.validate_terms(value, TRANSCRIPT, {}) == []


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/secret",
        "http://localhost/foo",
        "http://127.0.0.1",
        "https://10.0.0.1/foo",
        "http://[::1]/",
        "https://user:password@example.com",
        "https://api.internal/",
        "https://api.local/",
        "https://example.com:invalid",
    ],
)
def test_rejects_nonpublic_or_credential_urls(url):
    assert not web._public_url(url)


@pytest.mark.parametrize(
    "base",
    [
        "https://integrate.api.nvidia.com/v1",
        "https://example.com/v1",
        "https://api.deepseek.com.attacker.test",
    ],
)
def test_never_sends_another_providers_key_to_deepseek(monkeypatch, base):
    monkeypatch.setattr(web, "_ask", lambda *args: pytest.fail("Must not call LLM/tools"))
    original = TranslationContext(summary="original")
    enriched = run(original, client(base))
    assert enriched.summary == original.summary
    assert enriched.research_report["status"] == "unsupported_provider"


def test_research_enriches_context_with_provenance_without_mutating_source(monkeypatch):
    replies = iter([{"queries": ["Multimatic Sierra damper name"]}, {"terms": [TERM]}])
    monkeypatch.setattr(web, "_ask", lambda *args: next(replies))
    calls = []

    def request(self, operation, payload):
        calls.append((operation, payload))
        if operation == "search":
            return {"results": [{"url": URL, "authority": 4, "rerank_score": 2.5}]}
        return {"title": "Official technology", "content": CONTENT}

    monkeypatch.setattr(web.ResearchTools, "request", request)
    original = TranslationContext(summary="video", terminology="305hp", custom_prompt="Keep units")
    before = original.fingerprint()
    enriched = run(original, client())
    assert original.fingerprint() == before
    assert original.research_notes == ""
    assert enriched.fingerprint() != before
    assert "305hp" in enriched.terminology
    assert TERM["target"] in enriched.terminology
    assert enriched.research_report["status"] == "verified"
    assert enriched.research_report["terms"] == [TERM]
    assert URL in enriched.render()
    assert "never add facts" in enriched.render().lower()
    assert [c[0] for c in calls] == ["search", "fetch"]


def test_research_failure_keeps_local_context_and_does_not_log_response(monkeypatch, caplog):
    def fail(*args):
        raise RuntimeError("test-secret: untrusted response")

    monkeypatch.setattr(web, "_ask", fail)
    original = TranslationContext(summary="source", terminology="existing")
    enriched = run(original, client())
    assert enriched.summary == original.summary and enriched.terminology == original.terminology
    assert enriched.research_report["status"] == "unavailable"
    assert "test-secret" not in caplog.text


def test_cancellation_is_propagated_not_downgraded_to_failure(monkeypatch):
    runtime = client()
    runtime._subforge_cancel_event.set()
    with pytest.raises(LLMRequestCancelled):
        run(TranslationContext(), runtime)


def test_deduplicates_and_caps_queries_and_fetches(monkeypatch):
    replies = iter([{"queries": ["one", "one", "two", "must not run"]}, {"terms": []}])
    monkeypatch.setattr(web, "_ask", lambda *args: next(replies))
    calls = []

    def request(self, operation, payload):
        calls.append((operation, payload))
        if operation == "search":
            return {"results": [{"url": f"{URL}/{i}"} for i in range(5)]}
        return {"title": "title", "content": CONTENT}

    monkeypatch.setattr(web.ResearchTools, "request", request)
    enriched = run(TranslationContext(), client())
    assert enriched.research_report["queries"] == ["one", "two"]
    assert sum(c[0] == "fetch" for c in calls) == 2


def test_no_eligible_terms_does_not_call_tools(monkeypatch):
    monkeypatch.setattr(web, "_ask", lambda *args: {"queries": []})
    monkeypatch.setattr(web.ResearchTools, "request", lambda *args: pytest.fail("No tool requests"))
    assert run(TranslationContext(), client()).research_report["status"] == "not_needed"


@pytest.fixture
def transport_setup(monkeypatch):
    cache = {}

    class Cache:
        def get(self, key):
            return cache.get(key)

        def set(self, key, value, expire):
            assert expire == 604800
            cache[key] = value

    monkeypatch.setattr(web, "get_web_research_cache", Cache)
    monkeypatch.setattr(web, "is_cache_enabled", lambda: True)
    monkeypatch.setattr(web, "_wait", lambda *args: None)
    return httpx.AsyncClient


def test_tools_auth_endpoint_cache_and_physical_request_cap(monkeypatch, transport_setup):
    calls = []

    def handler(request):
        calls.append(request)
        assert request.url.host == "api.deepseek.com"
        assert request.url.path == "/v1/tools/search"
        assert request.headers["authorization"] == "Bearer test-secret"
        return httpx.Response(200, json={"results": [{"url": URL}]})

    monkeypatch.setattr(
        web.httpx,
        "AsyncClient",
        lambda **kwargs: transport_setup(transport=httpx.MockTransport(handler), **kwargs),
    )
    tools = web.ResearchTools(client())
    tools.request("search", {"query": "one"})
    tools.request("search", {"query": "one"})
    assert len(calls) == 1 and tools.cache_hits == 1
    for i in range(4):
        tools.request("search", {"query": str(i)})
    with pytest.raises(TimeoutError):
        tools.request("search", {"query": "over budget"})
    assert len(calls) == 5


@pytest.mark.parametrize("status", [402, 429, 500, 302])
def test_tools_never_retry_or_follow_redirects_beyond_budget(monkeypatch, transport_setup, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers={"location": "https://another.example.com"})

    monkeypatch.setattr(
        web.httpx,
        "AsyncClient",
        lambda **kwargs: transport_setup(transport=httpx.MockTransport(handler), **kwargs),
    )
    tools = web.ResearchTools(client())
    with pytest.raises(httpx.HTTPStatusError):
        tools.request("search", {"query": "one"})
    assert len(calls) == 1 and tools.requests == 1


def test_tools_timeout_and_oversized_response(monkeypatch, transport_setup):
    monkeypatch.setattr(
        web.httpx,
        "AsyncClient",
        lambda **kwargs: transport_setup(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=b"x" * 1_000_001)
            ),
            **kwargs,
        ),
    )
    tools = web.ResearchTools(client())
    with pytest.raises(ValueError, match="too large"):
        tools.request("fetch", {"url": URL})
    tools.deadline = 0
    with pytest.raises(TimeoutError):
        tools.request("search", {"query": "expired"})


def test_research_report_does_not_change_translation_cache_identity():
    original = TranslationContext(summary="source")
    assert (
        original.fingerprint() == replace(original, research_report={"requests": 5}).fingerprint()
    )


def test_rate_limit_queue_cannot_extend_the_research_deadline(monkeypatch, transport_setup):
    tools = web.ResearchTools(client())
    monkeypatch.setitem(web._next_request, "search", tools.deadline + 100)
    with pytest.raises(TimeoutError, match="wait exceeds deadline"):
        tools.request("search", {"query": "queued"})
    assert tools.requests == 0


def test_function_words_cannot_become_web_name_aliases():
    term = {**TERM, "source": "these"}
    assert web.validate_terms([term], "these", {URL: {"content": CONTENT}}) == []


def test_quote_must_support_the_specific_product_not_just_its_brand():
    quote = "The MultiPro center console transforms into a workstation."
    term = {"source": "MultiPro tailgate", "target": "MultiPro 尾门", "quote": quote, "url": URL}
    assert web.validate_terms([term], "MultiPro tailgate", {URL: {"content": quote}}) == []


def test_verified_terms_replace_conflicting_guesses_but_keep_unrelated_terms():
    original = TranslationContext(
        terminology="- Giants Control -> DSSV (guess)\n- GM -> GMC\n- 305hp -> 305马力"
    )
    enriched = web.apply_verified_terms(original, [TERM])
    assert "DSSV" not in enriched.terminology
    assert "- GM -> GMC" in enriched.terminology
    assert "305hp -> 305马力" in enriched.terminology
    assert "DSSV" in original.terminology


def test_quoted_name_is_licensed_only_in_the_cue_owning_the_exact_phrase():
    from subforge.core.translate.llm_translator import LLMTranslator
    from subforge.core.translate.types import TargetLanguage

    context = web.apply_verified_terms(TranslationContext(), [TERM])
    another = {
        "source": "OtherName",
        "target": "OtherName",
        "quote": "Another quotation on the same page",
        "url": URL,
    }
    context = replace(
        context,
        research_report={"status": "verified", "terms": [TERM, another], "sources": [{"url": URL}]},
    )
    translator = LLMTranslator(
        thread_num=1,
        batch_num=1,
        model="test",
        custom_prompt="",
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        is_reflect=False,
        update_callback=None,
        translation_context=context,
    )
    accepted, _ = translator._validate_no_unowned_latin_names(
        {"1": "Multimatic Jounce Control 减震器"}, {"1": TRANSCRIPT}, str
    )
    assert accepted
    rejected, _ = translator._validate_no_unowned_latin_names(
        {"1": "Multimatic Jounce Control 减震器"}, {"1": "Those Multimatic shocks."}, str
    )
    assert not rejected
    rejected, _ = translator._validate_no_unowned_latin_names(
        {"1": "Multimatic Jounce Control Rivian 减震器"}, {"1": TRANSCRIPT}, str
    )
    assert not rejected
    translator.translation_context = TranslationContext()
    rejected, _ = translator._validate_no_unowned_latin_names(
        {"1": "Multimatic Jounce Control 减震器"}, {"1": TRANSCRIPT}, str
    )
    assert not rejected


def test_public_context_builder_automatically_verifies_official_clients(monkeypatch):
    from subforge.core.asr.asr_data import ASRData, ASRDataSeg
    from subforge.core.translate import context as module
    from subforge.core.translate.types import TargetLanguage

    original = TranslationContext(summary="local")
    monkeypatch.setattr(module, "_build_translation_context", lambda *args: original)
    calls = []

    def research(context, **kwargs):
        calls.append(kwargs)
        return replace(context, research_notes="verified")

    monkeypatch.setattr(web, "research_translation_terms", research)
    data = ASRData(
        [ASRDataSeg("Source phrase", 0, 1000, translated_text="Private old translation")]
    )
    assert (
        module.build_translation_context(data, "deepseek-flash", TargetLanguage.SIMPLIFIED_CHINESE)
        == original
    )
    assert calls == []
    enriched = module.build_translation_context(
        data,
        "deepseek-flash",
        TargetLanguage.SIMPLIFIED_CHINESE,
        llm_client=client(),
    )
    assert enriched.research_notes == "verified"
    assert calls[0]["transcript"] == "Source phrase"


@pytest.mark.parametrize(
    "base",
    [
        "https://api.openai.com/v1",
        "https://integrate.api.nvidia.com/v1",
        "https://api.deepseek.com.attacker.test",
    ],
)
def test_automatic_context_skips_other_providers(monkeypatch, base):
    from subforge.core.asr.asr_data import ASRData, ASRDataSeg
    from subforge.core.translate import context as module
    from subforge.core.translate.types import TargetLanguage

    original = TranslationContext(summary="local")
    monkeypatch.setattr(module, "_build_translation_context", lambda *args: original)
    monkeypatch.setattr(
        web, "research_translation_terms", lambda *args, **kwargs: pytest.fail("No research")
    )
    data = ASRData([ASRDataSeg("Source phrase", 0, 1000)])
    assert (
        module.build_translation_context(
            data, "test", TargetLanguage.SIMPLIFIED_CHINESE, llm_client=client(base)
        )
        == original
    )


def test_tool_response_body_avoids_duplicate_fetch(monkeypatch):
    replies = iter([{"queries": ["term"]}, {"terms": [TERM]}])
    monkeypatch.setattr(web, "_ask", lambda *args: next(replies))

    def request(self, operation, payload):
        assert operation == "search"
        assert payload["need_content"]
        return {"results": [{"url": URL, "content": CONTENT}]}

    monkeypatch.setattr(web.ResearchTools, "request", request)
    assert run(TranslationContext(), client()).research_report["status"] == "verified"


def test_slow_stream_cannot_outlive_the_absolute_deadline(monkeypatch, transport_setup):
    clock = [time.monotonic()]
    closed = []

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(10):
                clock[0] += 9
                yield b" "
            yield b'{"content": "Late evidence"}'

        async def aclose(self):
            closed.append(True)

    monkeypatch.setattr(web.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(web, "_next_request", {})
    monkeypatch.setattr(web.httpx, "AsyncClient", lambda **kwargs: transport_setup(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=SlowStream())),
        **kwargs,
    ))
    tools = web.ResearchTools(client())
    with pytest.raises(TimeoutError, match="deadline"):
        tools.request("fetch", {"url": URL})
    assert closed
    assert tools.successes == 0
    assert tools.cache_hits == 0


@pytest.mark.parametrize("during", ["connect", "body"])
def test_absolute_deadline_interrupts_blocked_io(monkeypatch, transport_setup, during):
    closed = []

    class BlockedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(10)
            yield b'{"content": "Too late"}'

        async def aclose(self):
            closed.append(True)

    async def handler(request):
        if during == "connect":
            try:
                await asyncio.sleep(10)
            finally:
                closed.append(True)
        return httpx.Response(200, stream=BlockedStream())

    monkeypatch.setattr(web, "_next_request", {})
    monkeypatch.setattr(web.httpx, "AsyncClient", lambda **kwargs: transport_setup(
        transport=httpx.MockTransport(handler), **kwargs,
    ))
    tools = web.ResearchTools(client())
    tools.deadline = time.monotonic() + 0.1
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="deadline"):
        tools.request("search", {"query": "blocked"})
    assert time.monotonic() - started < 1
    assert closed
    assert tools.successes == 0


def test_cancellation_closes_a_blocked_response_promptly(monkeypatch, transport_setup):
    runtime = client()
    closed = []

    class BlockedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            runtime._subforge_cancel_event.set()
            await asyncio.sleep(10)
            yield b'{"content": "Too late"}'

        async def aclose(self):
            closed.append(True)

    monkeypatch.setattr(web, "_next_request", {})
    monkeypatch.setattr(web.httpx, "AsyncClient", lambda **kwargs: transport_setup(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=BlockedStream())),
        **kwargs,
    ))
    tools = web.ResearchTools(runtime)
    started = time.monotonic()
    with pytest.raises(LLMRequestCancelled):
        tools.request("fetch", {"url": URL})
    assert time.monotonic() - started < 1
    assert closed
    assert tools.successes == 0
