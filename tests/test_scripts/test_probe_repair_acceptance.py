import json

from scripts.probe_translation_repair_acceptance import build_verdict_request


def test_probe_uses_existing_validator_without_sending_reference_judgment():
    request = build_verdict_request({
        "expected": "do not disclose this reference",
        "items": {"1": {"source": "I love it.", "current": "我喜欢", "proposed": "我很喜欢", "speaker": "S1"}},
    })
    payload = json.loads(request["messages"][1]["content"])
    assert payload["1"]["source"] == "I love it."
    assert payload["1"]["translation"] == "我很喜欢"
    assert payload["1"]["speaker"] == "S1"
    assert "expected" not in json.dumps(request["messages"])
    assert request["reasoning_mode"] == "disabled"
