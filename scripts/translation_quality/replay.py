"""Private, offline replay of the legacy translation agent loop, not the ASR pipeline.

Capture is opt-in and process-local. It changes neither prompts nor responses.
Fixtures contain source text; public replay reports contain only hashes/counts/IDs.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

from subforge.core.llm.response import get_response_history_message
from subforge.core.translate import llm_translator as engine
from subforge.core.translate.context import TranslationContext
from subforge.core.translate.types import TargetLanguage

_MAP_FIELDS = (
    "_all_source_by_index", "_all_speaker_by_index",
    "_all_language_by_index", "_gap_after_index",
)
_USAGE_FIELDS = (
    "prompt_tokens", "completion_tokens", "total_tokens",
    "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def fingerprint(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _snapshot(translator: engine.LLMTranslator) -> dict[str, Any]:
    return {
        "model": translator.model,
        "target_language": translator.target_language.value,
        "reflect": translator.is_reflect,
        "custom_prompt": translator.custom_prompt,
        "context": asdict(translator.translation_context),
        "maps": {name: dict(getattr(translator, name)) for name in _MAP_FIELDS},
    }


@contextmanager
def _observe_validation(local: threading.local) -> Iterator[None]:
    validate = engine.LLMTranslator._validate_llm_response
    record = engine.LLMTranslator._record_shadow_repair_plan

    def record_diagnostics(self, diagnostics):
        frame = getattr(local, "frame", None)
        if frame is not None:
            frame["diagnostics"].extend({
                "rule_id": item.rule_id,
                "category": item.category.value,
                "severity": item.severity.value,
                "cue_keys": list(item.cue_keys),
            } for item in diagnostics)
        return record(self, diagnostics)

    def validate_response(self, response, source, **kwargs):
        frame = getattr(local, "frame", None)
        if frame is None:
            return validate(self, response, source, **kwargs)
        frame["diagnostics"] = []
        result = validate(self, response, source, **kwargs)
        frame.setdefault("parsed_responses", []).append(json.loads(stable_json(response)))
        frame["validations"].append({
            "parsed_sha256": fingerprint(response),
            "valid": result[0],
            "feedback_sha256": fingerprint(result[1]),
            "diagnostics": frame["diagnostics"],
        })
        return result

    with (
        patch.object(engine.LLMTranslator, "_record_shadow_repair_plan", record_diagnostics),
        patch.object(engine.LLMTranslator, "_validate_llm_response", validate_response),
    ):
        yield


def _outcome(result: Any, error: Exception | None) -> dict[str, Any]:
    return {
        "action": "return" if error is None else "raise",
        "result_sha256": fingerprint(result),
        "error_type": type(error).__name__ if error else None,
        "error_sha256": fingerprint(str(error)) if error else None,
    }


@contextmanager
def capture_agent_replays(
    directory: Path, *, provider: str, revision: str, origin: str = "api_capture",
) -> Iterator[list[str]]:
    """Capture main translation batches only; never persist client/config secrets.

    Use in a dedicated harness process, not a shared app server. Network exceptions
    are recorded as non-replayable; this harness does not simulate provider retry.
    """
    if origin not in {"api_capture", "srt_reconstruction", "synthetic"}:
        raise ValueError("Unsupported replay origin")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    local = threading.local()
    lock = threading.Lock()
    errors: list[str] = []
    agent = engine.LLMTranslator._agent_loop
    call = engine.call_llm

    def recorded_call(*args, **kwargs):
        frame = getattr(local, "frame", None)
        if frame is None:
            return call(*args, **kwargs)
        start = time.monotonic()
        request = {
            "messages_sha256": fingerprint(kwargs.get("messages")),
            "reasoning_mode": kwargs.get("reasoning_mode"),
            "max_output_tokens": kwargs.get("max_output_tokens"),
        }
        try:
            response = call(*args, **kwargs)
        except Exception as exc:
            frame["network_error"] = type(exc).__name__
            raise
        try:
            message = get_response_history_message(response)
            # Explicit allowlist excludes provider headers, endpoint and credential data.
            request["message"] = {
                key: message[key] for key in ("content", "reasoning_content", "reasoning")
                if key in message
            }
            usage = getattr(response, "usage", None)
            request["usage"] = {key: getattr(usage, key, None) for key in _USAGE_FIELDS}
            request["wall_seconds"] = time.monotonic() - start if origin == "api_capture" else None
            frame["responses"].append(request)
        except Exception as exc:
            frame["capture_error"] = type(exc).__name__
            with lock:
                errors.append(type(exc).__name__)
        return response

    def recorded_agent(self, system_prompt, subtitle_dict):
        previous = getattr(local, "frame", None)
        frame = {
            "schema_version": 1,
            "origin": origin,
            "stage": "translation_agent_loop",
            "provider": provider,
            "revision": revision,
            "snapshot": _snapshot(self),
            "system_prompt": system_prompt,
            "source_by_key": dict(subtitle_dict),
            "source_key_order": list(subtitle_dict),
            "input_sha256": fingerprint(subtitle_dict),
            "responses": [], "validations": [], "diagnostics": [],
        }
        local.frame = frame
        result, error = None, None
        try:
            result = agent(self, system_prompt, subtitle_dict)
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            frame["expected"] = _outcome(result, error)
            frame.pop("diagnostics", None)
            local.frame = previous
            try:
                path = directory / f"{fingerprint(frame)}.json"
                with path.open("x", encoding="utf-8") as stream:
                    path.chmod(0o600)
                    stream.write(stable_json(frame))
            except Exception as exc:
                # Capture failure must not throw away a successful paid response.
                with lock:
                    errors.append(type(exc).__name__)

    with (
        _observe_validation(local),
        patch.object(engine, "call_llm", recorded_call),
        patch.object(engine.LLMTranslator, "_agent_loop", recorded_agent),
    ):
        yield errors


def replay_agent(fixture: dict[str, Any]) -> dict[str, Any]:
    """Run the real legacy parser/normalizer/validator/retry loop with no API calls."""
    if fixture.get("schema_version") != 1 or fixture.get("stage") != "translation_agent_loop":
        raise ValueError("Unsupported replay schema or stage")
    if fixture.get("network_error") or fixture.get("capture_error"):
        raise ValueError("Provider exceptions or incomplete captures cannot be replayed")
    source = fixture["source_by_key"]
    if not isinstance(source, dict) or not source or fingerprint(source) != fixture["input_sha256"]:
        raise ValueError("Replay input is empty or its hash does not match")
    key_order = fixture["source_key_order"]
    if len(key_order) != len(source) or set(key_order) != set(source):
        raise ValueError("Replay source key order does not match input")
    source = {key: source[key] for key in key_order}
    snapshot = fixture["snapshot"]
    translator = engine.LLMTranslator(
        thread_num=1, batch_num=max(1, len(source)),
        target_language=TargetLanguage(snapshot["target_language"]),
        model=snapshot["model"], custom_prompt=snapshot["custom_prompt"],
        is_reflect=snapshot["reflect"], update_callback=None, use_cache=False,
        translation_context=TranslationContext(**snapshot["context"]),
    )
    for name in _MAP_FIELDS:
        setattr(translator, name, {int(key): value for key, value in snapshot["maps"][name].items()})
    local = threading.local()
    local.frame = {"validations": [], "diagnostics": []}
    responses = fixture["responses"]
    consumed: list[dict[str, Any]] = []

    def replay_call(*args, **kwargs):
        index = len(consumed)
        if index >= len(responses):
            raise RuntimeError("Replay response sequence exhausted; API access is disabled")
        item = responses[index]
        consumed.append({
            "messages_sha256": fingerprint(kwargs.get("messages")),
            "reasoning_mode": kwargs.get("reasoning_mode"),
            "max_output_tokens": kwargs.get("max_output_tokens"),
        })
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(**item["message"]))])

    result, error = None, None
    try:
        with _observe_validation(local), patch.object(engine, "call_llm", replay_call):
            result = translator._agent_loop(fixture["system_prompt"], dict(source))
    except Exception as exc:
        error = exc
    finally:
        translator.stop()
    outcome = _outcome(result, error)
    request_matches = len(consumed) == len(responses) and all(
        all(actual[key] == saved.get(key) for key in actual)
        for actual, saved in zip(consumed, responses)
    )
    return {
        "schema_version": 1,
        "scope": "translation_agent_loop_only",
        "fixture_sha256": fingerprint(fixture),
        "input_sha256": fixture["input_sha256"],
        "requests_replayed": len(consumed),
        "request_sequence_matches": request_matches,
        "validations": local.frame["validations"],
        "outcome": outcome,
        "matches_capture": (
            request_matches and outcome == fixture.get("expected")
            and local.frame["validations"] == fixture.get("validations")
        ),
    }
