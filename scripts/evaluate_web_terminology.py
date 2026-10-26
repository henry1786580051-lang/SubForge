#!/usr/bin/env python3
"""Opt-in live A/B probe against an existing bilingual subtitle, without editing it."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def cue_indices(value: str, total: int) -> list[int]:
    indices: set[int] = set()
    for part in value.split(","):
        bounds = part.strip().split("-")
        start = int(bounds[0])
        end = int(bounds[1]) if len(bounds) == 2 else start
        if len(bounds) > 2 or not 1 <= start <= end <= total:
            raise ValueError("Cue selection is outside the source subtitle")
        indices.update(range(start - 1, end))
    if not indices or len(indices) > 64:
        raise ValueError("Choose between 1 and 64 cues for this bounded live probe")
    return sorted(indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cues", default="1-24", help="One-based ranges, e.g. 1-8,209-216")
    parser.add_argument("--baseline-context", type=Path)
    parser.add_argument(
        "--live", action="store_true", help="Authorize real DeepSeek API calls and charges"
    )
    args = parser.parse_args()
    if not args.live:
        parser.error("This probe makes billable API requests; explicitly pass --live")

    from backend.app.api.config import get_llm_provider_runtime_config
    from subforge.core.asr.asr_data import ASRData
    from subforge.core.entities import SubtitleLayoutEnum
    from subforge.core.llm import close_client, create_client, request_logger
    from subforge.core.subtitle.validation import lock_source_segments, validate_bilingual_result
    from subforge.core.translate.context import (
        TranslationContext,
        _build_translation_context,
        _compact_transcript,
    )
    from subforge.core.translate.llm_translator import LLMTranslator
    from subforge.core.translate.types import TargetLanguage
    from subforge.core.translate.web_research import research_translation_terms

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source.is_relative_to(output):
        raise ValueError("Output folder must be separate from the source files")
    output.mkdir(parents=True, exist_ok=True)
    # Keep source text and responses in this local evaluation folder only.
    request_logger.LLM_LOG_FILE = output / "requests.jsonl"
    request_logger.set_llm_log_level("standard")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    data = ASRData.from_subtitle_file(str(source))
    indices = cue_indices(args.cues, len(data.segments))
    runtime = get_llm_provider_runtime_config("deepseek")
    client = create_client(base_url=runtime.base_url, api_key=runtime.api_key)
    contexts = {}
    try:
        if args.baseline_context:
            baseline = TranslationContext(**json.loads(args.baseline_context.read_text()))
        else:
            baseline = _build_translation_context(
                data,
                model="deepseek-flash",
                target_language=TargetLanguage.SIMPLIFIED_CHINESE,
                llm_client=client,
                use_cache=False,
            )
        contexts["baseline"] = baseline
        contexts["web"] = research_translation_terms(
            baseline,
            transcript=_compact_transcript(s.text for s in data.segments),
            model="deepseek-flash",
            target_language="简体中文",
            client=client,
        )
        for name, context in contexts.items():
            (output / f"{name}-context.json").write_text(
                json.dumps(asdict(context), ensure_ascii=False, indent=2) + "\n"
            )
        results = {}
        durations = {}
        for name, context in contexts.items():
            start = time.monotonic()
            sample = ASRData(copy.deepcopy([data.segments[i] for i in indices]))
            for segment in sample.segments:
                segment.translated_text = ""
            lock = lock_source_segments(sample)
            translator = LLMTranslator(
                thread_num=3,
                batch_num=8,
                target_language=TargetLanguage.SIMPLIFIED_CHINESE,
                model="deepseek-flash",
                custom_prompt="",
                is_reflect=False,
                update_callback=None,
                use_cache=False,
                translation_context=context,
                llm_client=client,
            )
            print(f"Translating {name}: {len(indices)} cues", flush=True)
            translated = translator.translate_subtitle(sample)
            validate_bilingual_result(translated, lock)
            results[name] = translated
            translated.save(
                str(output / f"{name}.srt"),
                layout=SubtitleLayoutEnum.TRANSLATE_ON_TOP,
                speaker_style="none",
            )
            durations[name] = round(time.monotonic() - start, 2)
        rows = []
        for position, original_index in enumerate(indices):
            original = data.segments[original_index]
            baseline_text = results["baseline"].segments[position].translated_text
            web_text = results["web"].segments[position].translated_text
            rows.append(
                {
                    "cue": original_index + 1,
                    "source": original.text,
                    "reference_translation": original.translated_text,
                    "baseline": baseline_text,
                    "web": web_text,
                    "source_digits": re.findall(r"\d+(?:\.\d+)?", original.text),
                    "web_digits": re.findall(r"\d+(?:\.\d+)?", web_text),
                }
            )
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        report = {
            "model": "deepseek-flash",
            "source_file": str(source),
            "source_sha256": source_hash,
            "source_unchanged": True,
            "source_cue_count": len(data.segments),
            "tested_cues": len(indices),
            "source_and_timing_validation": "passed for both arms",
            "durations_seconds": durations,
            "web_research": contexts["web"].research_report,
            "rows": rows,
        }
        (output / "comparison.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        lines = [
            "# 联网术语核实对照",
            "",
            f"模型：deepseek-flash；样本：{len(indices)} 条字幕。",
            "",
            "原翻译仅作参考，不作为人工金标准。两组使用相同模型、源字幕与生产翻译流程，仅联网术语上下文不同。",
            "",
        ]
        for row in rows:
            lines.extend(
                [
                    f"## 字幕 {row['cue']}",
                    "",
                    f"原文：{row['source']}",
                    "",
                    f"普通翻译：{row['baseline']}",
                    "",
                    f"联网核实：{row['web']}",
                    "",
                ]
            )
        (output / "comparison.md").write_text("\n".join(lines))
        print(
            json.dumps(
                {k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2
            )
        )
    finally:
        close_client(client)


if __name__ == "__main__":
    main()
