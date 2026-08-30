#!/usr/bin/env python3
"""Compare legacy and paired fidelity verdicts on private, frozen local windows.

This is an evaluation tool, not a production repair selector. Expected judgments
are never sent to the model, and no subtitle files are modified.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_translation_quality_shadow import isolated_settings_source  # noqa: E402
from scripts.translation_quality.replay import stable_json  # noqa: E402
from subforge.core.entities import SubtitleProcessData  # noqa: E402
from subforge.core.llm import (  # noqa: E402
    call_llm,
    create_client,
    get_response_text,
    parse_json_object,
)
from subforge.core.translate import llm_translator as engine  # noqa: E402
from subforge.core.translate.types import TargetLanguage  # noqa: E402

PAIR_GUIDANCE = (
    " Also compare the proposed translation with current_translation. Judge both against "
    "the source, not against each other as a reference. For the proposal, valid retains the "
    "fidelity/readability meaning specified above. Return current_valid as a boolean and "
    "preference as current, proposed, or neither. Prefer current when both are accurate and "
    "readable unless the proposal fixes a concrete omission, mistranslation, broken boundary "
    "or unmistakably unnatural expression without damaging another key. Mere rewording, "
    "literalizing idioms, more ornate language, or extra detail is not an improvement. "
    "If current is defective and proposed is a genuine improvement, prefer proposed. "
    "If neither is acceptable, choose neither. Give one brief source-grounded explanation "
    "in issues. Return only the specified JSON, not a rewritten subtitle."
)


def build_verdict_request(case: dict) -> dict:
    translator = engine.LLMTranslator(
        thread_num=1, batch_num=20, target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        model="glm-5.3-flash", custom_prompt="", is_reflect=False,
        update_callback=None, use_cache=False,
    )
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"valid": true, "issues": []}'
        ))])

    source = [SubtitleProcessData(index=int(key), original_text=item["source"])
              for key, item in case["items"].items()]
    proposed = [SubtitleProcessData(index=int(key), original_text=item["source"],
                                   translated_text=item["proposed"])
                for key, item in case["items"].items()]
    translator._all_speaker_by_index = {int(k): v.get("speaker", "") for k, v in case["items"].items()}
    translator._gap_after_index = {int(k): v.get("gap_after_ms", 0) for k, v in case["items"].items()}
    try:
        with patch.object(engine, "call_llm", capture):
            translator._validate_chinese_window_fidelity(source, proposed)
    finally:
        translator.stop()
    return captured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings-file", type=Path, required=True)
    parser.add_argument("--with-context", action="store_true")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())["cases"]
    if args.output.exists():
        raise FileExistsError(args.output)
    requests = []
    for case in cases:
        original = build_verdict_request(case)
        for variant in ("legacy", "paired"):
            messages = json.loads(json.dumps(original["messages"]))
            if variant == "paired":
                messages[0]["content"] += PAIR_GUIDANCE
                payload = json.loads(messages[1]["content"])
                for key, value in payload.items():
                    value["current_translation"] = case["items"][key]["current"]
                messages[1]["content"] = json.dumps(payload, ensure_ascii=False)
            if args.with_context and case.get("readonly_context"):
                messages[0]["content"] += (
                    " The request contains items and readonly_context. Evaluate only items. "
                    "Use readonly_context solely to disambiguate referents, ellipsis and word "
                    "sense, never as extra content to translate. Do not copy surrounding facts."
                )
                messages[1]["content"] = json.dumps({
                    "items": json.loads(messages[1]["content"]),
                    "readonly_context": case["readonly_context"],
                }, ensure_ascii=False)
            requests.append((case["id"], variant, messages))
    with isolated_settings_source(args.settings_file):
        from app.api.config import get_llm_runtime_config

        runtime = get_llm_runtime_config()
        if (runtime.provider, runtime.model) != ("zhipu", "glm-5.3-flash") or not runtime.api_key:
            raise RuntimeError("Expected an authorized zhipu/glm-5.3-flash profile")
        client = create_client(base_url=runtime.base_url, api_key=runtime.api_key)
    try:
        def evaluate(request):
            case_id, variant, messages = request
            start = time.monotonic()
            response = call_llm(
                messages=messages, model="glm-5.3-flash", client=client,
                temperature=0, use_cache=False, reasoning_mode="disabled", max_output_tokens=2048,
            )
            usage = response.usage
            return {
                "id": case_id, "variant": variant,
                "verdict": parse_json_object(get_response_text(response)),
                "tokens": usage.total_tokens,
                "wall_seconds": time.monotonic() - start,
            }

        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(evaluate, requests))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            args.output.chmod(0o600)
            stream.write(stable_json({"results": results, "model": "glm-5.3-flash"}))
        print(f"verdicts={len(results)} tokens={sum(r['tokens'] for r in results)}")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
