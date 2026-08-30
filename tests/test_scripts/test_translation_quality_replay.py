import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.translation_quality.replay import (
    capture_agent_replays,
    fingerprint,
    replay_agent,
    stable_json,
)
from subforge.core.translate import llm_translator as engine
from subforge.core.translate.types import TargetLanguage


def _translator(*, reflect=False):
    return engine.LLMTranslator(
        thread_num=1, batch_num=20, target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        model="glm-5.3-flash", custom_prompt="", is_reflect=reflect,
        update_callback=None, use_cache=False,
    )


def _response(payload):
    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=30, total_tokens=130),
        api_key="must-not-be-recorded",
    )


def _capture(tmp_path, monkeypatch, outputs, *, reflect=False):
    responses = iter(outputs)
    monkeypatch.setattr(engine, "call_llm", lambda **kwargs: _response(next(responses)))
    translator = _translator(reflect=reflect)
    translator._all_source_by_index = {1: "Hello.", 2: "Thanks."}
    try:
        with capture_agent_replays(tmp_path, provider="zhipu", revision="test") as errors:
            result = translator._agent_loop("Translate.", {"1": "Hello.", "2": "Thanks."})
        assert errors == []
    finally:
        translator.stop()
    path = next(tmp_path.glob("*.json"))
    return json.loads(path.read_text()), result


def test_replay_matches_real_agent_parser_normalizer_validator_and_retries(tmp_path, monkeypatch):
    fixture, result = _capture(tmp_path, monkeypatch, [
        {"1": "你好"}, {"1": "你好", "2": "谢谢"},
    ])
    assert result == {"1": "你好", "2": "谢谢"}

    def forbid_network(**kwargs):
        pytest.fail("Replay attempted an API call")

    monkeypatch.setattr(engine, "call_llm", forbid_network)
    first = replay_agent(fixture)
    second = replay_agent(fixture)
    assert stable_json(first) == stable_json(second)
    assert first["matches_capture"]
    assert first["requests_replayed"] == 2
    assert first["validations"][0]["diagnostics"][0]["rule_id"] == "schema.missing_key"
    assert "must-not-be-recorded" not in stable_json(fixture)
    assert "你好" not in stable_json(first)
    assert "Hello" not in stable_json(first)


@pytest.mark.parametrize("reflect", [False, True])
def test_replay_supports_plain_and_compact_reflect_response(tmp_path, monkeypatch, reflect):
    values = {"1": "你好", "2": "谢谢"}
    if reflect:
        values = {key: {"native_translation": value} for key, value in values.items()}
    fixture, _ = _capture(tmp_path, monkeypatch, [values], reflect=reflect)
    assert replay_agent(fixture)["matches_capture"]


def test_corrupted_fixture_rejected_before_execution(tmp_path, monkeypatch):
    fixture, _ = _capture(tmp_path, monkeypatch, [{"1": "你好", "2": "谢谢"}])
    fixture["source_by_key"]["1"] = "Changed"
    with pytest.raises(ValueError, match="hash"):
        replay_agent(fixture)


def test_exhausted_responses_do_not_fall_back_to_network(tmp_path, monkeypatch):
    fixture, _ = _capture(tmp_path, monkeypatch, [{"1": "你好", "2": "谢谢"}])
    fixture["responses"] = []
    report = replay_agent(fixture)
    assert not report["matches_capture"]
    assert report["outcome"]["error_type"] == "RuntimeError"


def test_concurrent_capture_does_not_mix_batches(tmp_path, monkeypatch):
    def respond(**kwargs):
        payload = json.loads(kwargs["messages"][-1]["content"])
        return _response({key: "你好" for key in payload["current_subtitles"]})

    monkeypatch.setattr(engine, "call_llm", respond)
    translator = _translator()
    try:
        with capture_agent_replays(tmp_path, provider="zhipu", revision="test") as errors:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(translator._agent_loop, "Translate.", {str(i): "Hello."})
                           for i in (1, 2)]
                assert [future.result() for future in futures] == [{"1": "你好"}, {"2": "你好"}]
        assert errors == []
    finally:
        translator.stop()
    fixtures = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert len(fixtures) == 2
    assert {tuple(f["source_by_key"]) for f in fixtures} == {("1",), ("2",)}
    assert all(replay_agent(f)["matches_capture"] for f in fixtures)


def test_capture_write_failure_does_not_discard_translation(tmp_path, monkeypatch):
    def fail_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(engine, "call_llm", lambda **kwargs: _response({"1": "你好"}))
    translator = _translator()
    try:
        with capture_agent_replays(tmp_path, provider="zhipu", revision="test") as errors:
            monkeypatch.setattr(Path, "open", fail_open)
            assert translator._agent_loop("Translate.", {"1": "Hello."}) == {"1": "你好"}
        assert errors == ["OSError"]
    finally:
        translator.stop()


def test_fingerprint_is_mapping_order_independent():
    assert fingerprint({"1": "a", "2": "b"}) == fingerprint({"2": "b", "1": "a"})


def test_replay_preserves_numeric_key_order_across_json_round_trip(tmp_path, monkeypatch):
    translator = _translator()
    source = {str(i): "Hello." for i in range(1, 21)}
    values = {key: "你好" for key in source}
    monkeypatch.setattr(engine, "call_llm", lambda **kwargs: _response(values))
    try:
        with capture_agent_replays(tmp_path, provider="zhipu", revision="test", origin="synthetic"):
            translator._agent_loop("Translate.", source)
    finally:
        translator.stop()
    fixture = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert fixture["responses"][0]["wall_seconds"] is None
    assert replay_agent(fixture)["matches_capture"]
