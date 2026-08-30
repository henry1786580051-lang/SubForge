"""Deterministic, text-free snapshots of legacy Chinese boundary signals."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from subforge.core.split.boundary import assess_english_boundary
from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.quality.boundary_registry import boundary_rule_for_message

from .manifest import CorpusManifest
from .srt import SrtDocument, parse_srt


def _source_rule_ids(signal: str, left: str, right: str) -> tuple[str, ...]:
    direct_rule = boundary_rule_for_message(signal)
    if direct_rule is not None:
        return (direct_rule.rule_id,)
    assessment = assess_english_boundary(left, right)
    expected = "; ".join(assessment.reasons) or "unstable source boundary"
    if assessment.unstable and signal == expected:
        return tuple(item.rule_id for item in assessment.contributions)
    return ()


def snapshot_chinese_boundary_document(document: SrtDocument) -> dict[str, Any]:
    translator = object.__new__(LLMTranslator)
    translator._gap_after_index = {
        left.index: max(0, right.start_ms - left.end_ms)
        for left, right in zip(document.cues, document.cues[1:])
    }

    decisions: list[dict[str, Any]] = []
    source_signal_counts: Counter[str] = Counter()
    target_signal_counts: Counter[str] = Counter()
    source_rule_counts: Counter[str] = Counter()
    target_rule_counts: Counter[str] = Counter()
    source_signal_count = 0
    target_signal_count = 0
    syntax_signal_count = 0
    display_signal_count = 0
    unregistered_source_signal_count = 0
    unregistered_target_signal_count = 0
    skipped_empty_source = 0
    skipped_empty_target = 0

    for left, right in zip(document.cues, document.cues[1:]):
        if not left.source.strip() or not right.source.strip():
            skipped_empty_source += 1
            source_signal = ""
            source_rule_ids: tuple[str, ...] = ()
        else:
            source_signal = LLMTranslator._source_boundary_signal(
                left.source,
                right.source,
                left.target,
                right.target,
            )
            source_rule_ids = _source_rule_ids(source_signal, left.source, right.source)

        if not left.target.strip() or not right.target.strip():
            skipped_empty_target += 1
            syntax_signal = ""
            display_signal = ""
        else:
            syntax_signal = LLMTranslator._chinese_boundary_signal(left.target, right.target)
            display_signal = translator._long_gap_chinese_boundary_signal(
                left.index,
                left.target,
                right.target,
            )
        target_signal = display_signal or syntax_signal
        target_rule = boundary_rule_for_message(target_signal)
        target_rule_id = target_rule.rule_id if target_rule is not None else ""

        source_signal_count += int(bool(source_signal))
        target_signal_count += int(bool(target_signal))
        syntax_signal_count += int(bool(syntax_signal))
        display_signal_count += int(bool(display_signal))
        unregistered_source_signal_count += int(bool(source_signal) and not source_rule_ids)
        unregistered_target_signal_count += int(bool(target_signal) and not target_rule_id)
        source_signal_counts.update([source_signal] if source_signal else [])
        target_signal_counts.update([target_signal] if target_signal else [])
        source_rule_counts.update(source_rule_ids)
        target_rule_counts.update([target_rule_id] if target_rule_id else [])
        decisions.append(
            {
                "left": left.index,
                "right": right.index,
                "gap_ms": translator._gap_after_index[left.index],
                "source_signal": source_signal,
                "source_rule_ids": source_rule_ids,
                "syntax_signal": syntax_signal,
                "display_signal": display_signal,
                "target_signal": target_signal,
                "target_rule_id": target_rule_id,
            }
        )

    canonical = json.dumps(
        decisions,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "boundary_count": len(decisions),
        "skipped_empty_source": skipped_empty_source,
        "skipped_empty_target": skipped_empty_target,
        "source_signal_count": source_signal_count,
        "target_signal_count": target_signal_count,
        "syntax_signal_count": syntax_signal_count,
        "display_signal_count": display_signal_count,
        "unregistered_source_signal_count": unregistered_source_signal_count,
        "unregistered_target_signal_count": unregistered_target_signal_count,
        "source_signal_counts": dict(sorted(source_signal_counts.items())),
        "target_signal_counts": dict(sorted(target_signal_counts.items())),
        "source_rule_counts": dict(sorted(source_rule_counts.items())),
        "target_rule_counts": dict(sorted(target_rule_counts.items())),
        "decision_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _aggregate(samples: list[dict[str, Any]], field: str) -> dict[str, Any]:
    documents = [sample[field] for sample in samples]
    canonical = json.dumps(
        [(sample["sample_id"], sample[field]["decision_sha256"]) for sample in samples],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "sample_count": len(documents),
        "boundary_count": sum(document["boundary_count"] for document in documents),
        "source_signal_count": sum(document["source_signal_count"] for document in documents),
        "target_signal_count": sum(document["target_signal_count"] for document in documents),
        "syntax_signal_count": sum(document["syntax_signal_count"] for document in documents),
        "display_signal_count": sum(document["display_signal_count"] for document in documents),
        "unregistered_source_signal_count": sum(
            document["unregistered_source_signal_count"] for document in documents
        ),
        "unregistered_target_signal_count": sum(
            document["unregistered_target_signal_count"] for document in documents
        ),
        "decision_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def evaluate_chinese_boundary_snapshot(
    manifest: CorpusManifest,
    root: Path,
    *,
    manifest_hash: str,
    splits: set[str] | None = None,
) -> dict[str, Any]:
    selected = [sample for sample in manifest.samples if splits is None or sample.split in splits]
    samples: list[dict[str, Any]] = []
    for sample in selected:
        machine = parse_srt(root / sample.machine_srt, layout="target_above")
        gold = parse_srt(root / sample.gold_srt, layout="target_above")
        samples.append(
            {
                "sample_id": sample.id,
                "split": sample.split,
                "machine": snapshot_chinese_boundary_document(machine),
                "gold": snapshot_chinese_boundary_document(gold),
            }
        )
    return {
        "corpus_id": manifest.corpus_id,
        "manifest_hash": manifest_hash,
        "aggregate": {
            "machine": _aggregate(samples, "machine"),
            "gold": _aggregate(samples, "gold"),
        },
        "samples": samples,
    }


def write_chinese_boundary_snapshot(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
