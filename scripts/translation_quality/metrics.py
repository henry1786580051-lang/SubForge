"""Deterministic translation-quality metrics for static corpus evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from subforge.core.translate.quality import (
    contains_reasoning_leak,
    is_source_copy,
)

from .manifest import CorpusManifest, CorpusSample
from .srt import SrtCue, SrtDocument, parse_srt

_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _is_placeholder(text: str) -> bool:
    """Detect standalone workflow notes without matching normal Chinese words."""
    value = str(text or "").strip()
    if not value:
        return True
    compact = re.sub(r"\s+", "", value).strip(
        "()（）[]【】<>《》“”\"'。，、；;：:！!?"
    )
    previous_refs = r"上一句|上句|上一条|上条|前一句|前一条|前文|前面"
    patterns = (
        r"(?:此|本)句.*(?:合并|并入|省略|略去|无需翻译|不单独翻译).*",
        rf"(?:已)?(?:合并|并入|接上|延续|已译|包含).*(?:{previous_refs})",
        rf"(?:{previous_refs}).*(?:合并|包含|已译|并入|已经翻译)",
        r"(?:最终版本|最终字幕).*(?:合并|省略)",
        r"(?:内容)?(?:同上|见上|略|省略|无需翻译|不单独翻译)",
        r"merged(?:with|into)?(?:the)?(?:previous|above)",
        r"sameasabove",
        r"(?:translation)?(?:missing|omitted)",
        r"(?:待翻译|未翻译|无法翻译)",
    )
    if any(re.fullmatch(pattern, compact, flags=re.IGNORECASE) for pattern in patterns):
        return True
    return bool(
        re.search(
            r"(?:\(|（|\[|【)\s*(?:应为|疑似|译注|注\s*[:：]|原文(?:应为)?|可能是)"
            r"[^\)）\]】]*(?:\)|）|\]|】)",
            value,
            flags=re.IGNORECASE,
        )
    )


def _compact(text: str) -> str:
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", str(text or "").casefold())


def _similarity(left: str, right: str) -> float:
    normalized_left = _compact(left)
    normalized_right = _compact(right)
    if not normalized_left and not normalized_right:
        return 1.0
    if not normalized_left or not normalized_right:
        return 0.0
    return SequenceMatcher(None, normalized_left, normalized_right, autojunk=False).ratio()


def _is_source_copy(cue: SrtCue) -> bool:
    return is_source_copy(cue.target, cue.source)


def _is_untranslated(cue: SrtCue) -> bool:
    if not cue.target.strip():
        return False
    return not _HAN.search(cue.target) and bool(re.search(r"[A-Za-z]{2}", cue.source))


def _indices(cues: Iterable[SrtCue], predicate: Any) -> list[int]:
    return [cue.index for cue in cues if predicate(cue)]


def _adjacent_duplicate_risks(cues: tuple[SrtCue, ...]) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    for left, right in zip(cues, cues[1:]):
        if len(_compact(left.target)) < 8 or len(_compact(right.target)) < 8:
            continue
        target_similarity = _similarity(left.target, right.target)
        source_similarity = _similarity(left.source, right.source)
        if target_similarity >= 0.82 and target_similarity >= source_similarity + 0.18:
            risks.append(
                {
                    "left": left.index,
                    "right": right.index,
                    "target_similarity": round(target_similarity, 4),
                    "source_similarity": round(source_similarity, 4),
                }
            )
    return risks


def _document_quality(document: SrtDocument) -> dict[str, Any]:
    cues = document.cues
    return {
        "cue_count": len(cues),
        "empty_targets": _indices(cues, lambda cue: not cue.target.strip()),
        "placeholder_targets": _indices(cues, lambda cue: _is_placeholder(cue.target)),
        "reasoning_leaks": _indices(cues, lambda cue: contains_reasoning_leak(cue.target)),
        "source_copy_targets": _indices(cues, _is_source_copy),
        "untranslated_targets": _indices(cues, _is_untranslated),
        "adjacent_duplicate_risks": _adjacent_duplicate_risks(cues),
        "encoding": document.encoding,
        "newline": document.newline,
        "has_bom": document.has_bom,
    }


def _structure(machine: SrtDocument, gold: SrtDocument) -> dict[str, Any]:
    machine_keys = [cue.index for cue in machine.cues]
    gold_keys = [cue.index for cue in gold.cues]
    paired = list(zip(machine.cues, gold.cues))
    timeline_mismatches = [
        machine_cue.index
        for machine_cue, gold_cue in paired
        if machine_cue.timeline != gold_cue.timeline
    ]
    source_mismatches = [
        machine_cue.index
        for machine_cue, gold_cue in paired
        if machine_cue.source != gold_cue.source
    ]
    return {
        "cue_count_equal": len(machine.cues) == len(gold.cues),
        "key_set_equal": set(machine_keys) == set(gold_keys),
        "key_order_equal": machine_keys == gold_keys,
        "timeline_mismatches": timeline_mismatches,
        "source_mismatches": source_mismatches,
        "exact": (
            len(machine.cues) == len(gold.cues)
            and machine_keys == gold_keys
            and not timeline_mismatches
            and not source_mismatches
        ),
    }


def _gold_comparison(machine: SrtDocument, gold: SrtDocument) -> dict[str, Any]:
    if len(machine.cues) != len(gold.cues):
        return {
            "comparable": False,
            "reason": "cue_count_mismatch",
            "changed_cues": [],
            "exact_match_rate": None,
            "mean_similarity": None,
            "median_similarity": None,
            "lowest_similarity_indices": [],
        }
    similarities: list[tuple[int, float]] = []
    changed: list[int] = []
    for machine_cue, gold_cue in zip(machine.cues, gold.cues):
        score = _similarity(machine_cue.target, gold_cue.target)
        similarities.append((machine_cue.index, score))
        if machine_cue.target != gold_cue.target:
            changed.append(machine_cue.index)
    sorted_similarities = sorted(similarities, key=lambda item: item[1])
    values = [score for _, score in similarities]
    return {
        "comparable": True,
        "changed_cues": changed,
        "changed_rate": round(len(changed) / len(machine.cues), 4) if machine.cues else 0.0,
        "exact_match_rate": round(1 - len(changed) / len(machine.cues), 4)
        if machine.cues
        else 1.0,
        "mean_similarity": round(mean(values), 4) if values else 1.0,
        "median_similarity": round(median(values), 4) if values else 1.0,
        "lowest_similarity_indices": [index for index, _ in sorted_similarities[:20]],
    }


@dataclass(frozen=True)
class SampleEvaluation:
    sample_id: str
    title: str
    split: str
    source: dict[str, Any]
    machine: dict[str, Any]
    gold: dict[str, Any]
    structure: dict[str, Any]
    gold_comparison: dict[str, Any]
    hard_failure_count: int

    def to_dict(self, *, redact_details: bool = False) -> dict[str, Any]:
        payload = {
            "sample_id": self.sample_id,
            "title": self.title,
            "split": self.split,
            "source": self.source,
            "machine": self.machine,
            "gold": self.gold,
            "structure": self.structure,
            "gold_comparison": self.gold_comparison,
            "hard_failure_count": self.hard_failure_count,
        }
        if redact_details:
            payload["machine"] = _redact_quality_details(self.machine)
            payload["gold"] = _redact_quality_details(self.gold)
            payload["structure"] = _redact_structure_details(self.structure)
            payload["gold_comparison"] = _redact_gold_details(self.gold_comparison)
            payload["details_redacted"] = True
        return payload


@dataclass(frozen=True)
class EvaluationReport:
    corpus_id: str
    manifest_hash: str
    comparison_identity: str
    samples: tuple[SampleEvaluation, ...]
    aggregate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "manifest_hash": self.manifest_hash,
            "comparison_identity": self.comparison_identity,
            "aggregate": self.aggregate,
            "samples": [
                sample.to_dict(redact_details=sample.split == "holdout")
                for sample in self.samples
            ],
        }


def _redact_quality_details(quality: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(quality)
    for key in (
        "empty_targets",
        "placeholder_targets",
        "reasoning_leaks",
        "source_copy_targets",
        "untranslated_targets",
        "adjacent_duplicate_risks",
    ):
        values = redacted.pop(key, [])
        redacted[f"{key}_count"] = len(values)
    return redacted


def _redact_structure_details(structure: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(structure)
    for key in ("timeline_mismatches", "source_mismatches"):
        values = redacted.pop(key, [])
        redacted[f"{key}_count"] = len(values)
    return redacted


def _redact_gold_details(comparison: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(comparison)
    changed = redacted.pop("changed_cues", [])
    redacted["changed_cue_count"] = len(changed)
    redacted.pop("lowest_similarity_indices", None)
    return redacted


def _sample_paths(sample: CorpusSample, root: Path) -> tuple[Path, Path, Path]:
    return (
        root / sample.source_srt,
        root / sample.machine_srt,
        root / sample.gold_srt,
    )


def evaluate_sample(sample: CorpusSample, root: Path) -> SampleEvaluation:
    source_path, machine_path, gold_path = _sample_paths(sample, root)
    source_doc = parse_srt(source_path, layout="source_only")
    machine_doc = parse_srt(machine_path, layout="target_above")
    gold_doc = parse_srt(gold_path, layout="target_above")
    source_summary = {
        "cue_count": len(source_doc.cues),
        "duration_ms": max((cue.end_ms for cue in source_doc.cues), default=0),
        "encoding": source_doc.encoding,
        "newline": source_doc.newline,
        "has_bom": source_doc.has_bom,
    }
    machine_quality = _document_quality(machine_doc)
    gold_quality = _document_quality(gold_doc)
    structure = _structure(machine_doc, gold_doc)
    declared_structure = str(sample.alignment.get("cue_structure", "exact"))
    structure["declared_cue_structure"] = declared_structure
    if declared_structure == "exact":
        comparison = _gold_comparison(machine_doc, gold_doc)
    else:
        comparison = {
            "comparable": False,
            "reason": f"declared_{declared_structure}",
            "changed_cues": [],
            "exact_match_rate": None,
            "mean_similarity": None,
            "median_similarity": None,
            "lowest_similarity_indices": [],
        }
    hard_failure_count = sum(
        (
            0 if structure["exact"] or declared_structure != "exact" else 1,
            len(machine_quality["empty_targets"]),
            len(machine_quality["placeholder_targets"]),
            len(machine_quality["reasoning_leaks"]),
        )
    )
    return SampleEvaluation(
        sample_id=sample.id,
        title=sample.title,
        split=sample.split,
        source=source_summary,
        machine=machine_quality,
        gold=gold_quality,
        structure=structure,
        gold_comparison=comparison,
        hard_failure_count=hard_failure_count,
    )


def evaluate_manifest(
    manifest: CorpusManifest,
    root: Path,
    *,
    manifest_hash: str,
    splits: set[str] | None = None,
) -> EvaluationReport:
    selected = [sample for sample in manifest.samples if splits is None or sample.split in splits]
    evaluations = tuple(evaluate_sample(sample, root) for sample in selected)
    comparison_identity = hashlib.sha256(
        json.dumps(
            [
                {
                    "id": sample.id,
                    "split": sample.split,
                    "source_sha256": sample.hashes.get("source_sha256", ""),
                    "gold_sha256": sample.hashes.get("gold_sha256", ""),
                    "alignment": sample.alignment,
                }
                for sample in selected
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    aggregate = {
        "sample_count": len(evaluations),
        "split_counts": {
            split: sum(sample.split == split for sample in evaluations)
            for split in ("development", "validation", "holdout")
        },
        "source_cue_count": sum(sample.source["cue_count"] for sample in evaluations),
        "machine_cue_count": sum(sample.machine["cue_count"] for sample in evaluations),
        "gold_cue_count": sum(sample.gold["cue_count"] for sample in evaluations),
        "structurally_exact_samples": sum(sample.structure["exact"] for sample in evaluations),
        "hard_failure_count": sum(sample.hard_failure_count for sample in evaluations),
        "empty_targets": sum(len(sample.machine["empty_targets"]) for sample in evaluations),
        "placeholder_targets": sum(
            len(sample.machine["placeholder_targets"]) for sample in evaluations
        ),
        "reasoning_leaks": sum(len(sample.machine["reasoning_leaks"]) for sample in evaluations),
        "source_copy_targets": sum(
            len(sample.machine["source_copy_targets"]) for sample in evaluations
        ),
        "untranslated_targets": sum(
            len(sample.machine["untranslated_targets"]) for sample in evaluations
        ),
        "adjacent_duplicate_risks": sum(
            len(sample.machine["adjacent_duplicate_risks"]) for sample in evaluations
        ),
        "human_changed_cues": sum(
            len(sample.gold_comparison["changed_cues"]) for sample in evaluations
        ),
        "requires_alignment_samples": sum(
            sample.structure.get("declared_cue_structure") != "exact"
            for sample in evaluations
        ),
    }
    return EvaluationReport(
        corpus_id=manifest.corpus_id,
        manifest_hash=manifest_hash,
        comparison_identity=comparison_identity,
        samples=evaluations,
        aggregate=aggregate,
    )
