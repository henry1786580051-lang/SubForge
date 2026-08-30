"""Deterministic, text-free snapshots of legacy English boundary decisions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from subforge.core.split.boundary import assess_english_boundary

from .manifest import CorpusManifest
from .srt import SrtDocument, parse_srt


def snapshot_document(document: SrtDocument) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    unregistered_reason_counts: Counter[str] = Counter()
    registered_rule_counts: Counter[str] = Counter()
    total_risk = 0
    registered_risk = 0
    unstable_count = 0
    skipped_empty_source = 0

    for left, right in zip(document.cues, document.cues[1:]):
        if not left.source.strip() or not right.source.strip():
            skipped_empty_source += 1
            continue
        assessment = assess_english_boundary(left.source, right.source)
        decisions.append(
            {
                "left": left.index,
                "right": right.index,
                "risk": assessment.risk,
                "reasons": assessment.reasons,
            }
        )
        total_risk += assessment.risk
        registered_risk += assessment.registered_risk
        unstable_count += int(assessment.unstable)
        reason_counts.update(assessment.reasons)
        contribution_reasons = {item.reason for item in assessment.contributions}
        unregistered_reason_counts.update(
            reason for reason in assessment.reasons if reason not in contribution_reasons
        )
        registered_rule_counts.update(item.rule_id for item in assessment.contributions)

    canonical = json.dumps(
        decisions,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "boundary_count": len(decisions),
        "skipped_empty_source": skipped_empty_source,
        "unstable_count": unstable_count,
        "total_risk": total_risk,
        "registered_risk": registered_risk,
        "unregistered_risk": total_risk - registered_risk,
        "reason_counts": dict(sorted(reason_counts.items())),
        "unregistered_reason_counts": dict(sorted(unregistered_reason_counts.items())),
        "registered_rule_counts": dict(sorted(registered_rule_counts.items())),
        "decision_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def evaluate_boundary_snapshot(
    manifest: CorpusManifest,
    root: Path,
    *,
    manifest_hash: str,
    splits: set[str] | None = None,
) -> dict[str, Any]:
    selected = [sample for sample in manifest.samples if splits is None or sample.split in splits]
    samples: list[dict[str, Any]] = []
    for sample in selected:
        document = parse_srt(root / sample.machine_srt, layout="target_above")
        samples.append(
            {
                "sample_id": sample.id,
                "split": sample.split,
                **snapshot_document(document),
            }
        )

    aggregate_source = json.dumps(
        [(sample["sample_id"], sample["decision_sha256"]) for sample in samples],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "corpus_id": manifest.corpus_id,
        "manifest_hash": manifest_hash,
        "aggregate": {
            "sample_count": len(samples),
            "boundary_count": sum(sample["boundary_count"] for sample in samples),
            "unstable_count": sum(sample["unstable_count"] for sample in samples),
            "total_risk": sum(sample["total_risk"] for sample in samples),
            "registered_risk": sum(sample["registered_risk"] for sample in samples),
            "unregistered_risk": sum(sample["unregistered_risk"] for sample in samples),
            "decision_sha256": hashlib.sha256(aggregate_source).hexdigest(),
        },
        "samples": samples,
    }


def write_boundary_snapshot(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
