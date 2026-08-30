#!/usr/bin/env python3
"""Compare one Kimi K3 subtitle result with V4 and human-edited references."""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subforge.core.asr.asr_data import ASRData  # noqa: E402

PLACEHOLDER_PATTERN = re.compile(
    r"(?:待翻译|未翻译|合并至|merged\s+(?:with|into)|same\s+as\s+above|"
    r"translation\s+(?:missing|omitted)|此句.*合并)",
    re.IGNORECASE,
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
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def _load(path: Path) -> list[dict[str, Any]]:
    data = ASRData.from_subtitle_file(str(path))
    return [
        {
            "index": index,
            "start": segment.start_time,
            "end": segment.end_time,
            "source": segment.text.strip(),
            "target": segment.translated_text.strip(),
        }
        for index, segment in enumerate(data.segments, 1)
    ]


def _align_by_time(
    candidate_cues: list[dict[str, Any]],
    reference_cues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project split/merged reference cues onto candidate timing windows."""
    aligned = []
    left = 0
    for candidate in candidate_cues:
        while left < len(reference_cues) and reference_cues[left]["end"] <= candidate["start"]:
            left += 1
        matches = []
        cursor = left
        while cursor < len(reference_cues) and reference_cues[cursor]["start"] < candidate["end"]:
            reference = reference_cues[cursor]
            overlap = min(candidate["end"], reference["end"]) - max(
                candidate["start"], reference["start"]
            )
            if overlap > 0:
                matches.append(reference)
            cursor += 1
        if matches:
            aligned.append(
                {
                    "index": candidate["index"],
                    "start": min(item["start"] for item in matches),
                    "end": max(item["end"] for item in matches),
                    "source": " ".join(item["source"] for item in matches).strip(),
                    "target": " ".join(item["target"] for item in matches).strip(),
                    "reference_indices": [item["index"] for item in matches],
                }
            )
        else:
            aligned.append(
                {
                    "index": candidate["index"],
                    "start": candidate["start"],
                    "end": candidate["end"],
                    "source": "",
                    "target": "",
                    "reference_indices": [],
                }
            )
    return aligned


def evaluate(candidate: Path, v4_reference: Path, human_reference: Path) -> dict[str, Any]:
    candidate_cues = _load(candidate)
    v4_cues = _load(v4_reference)
    human_cues = _load(human_reference)
    comparable_count = min(len(candidate_cues), len(v4_cues))
    human_aligned = _align_by_time(candidate_cues[:comparable_count], human_cues)

    human_similarities: list[float] = []
    v4_similarities: list[float] = []
    source_matches = 0
    timestamp_matches = 0
    missing_targets: list[int] = []
    placeholders: list[int] = []
    untranslated: list[int] = []
    ownership_shift_risks: list[dict[str, Any]] = []
    duplicate_risks: list[dict[str, Any]] = []
    differences: list[dict[str, Any]] = []

    for offset in range(comparable_count):
        current = candidate_cues[offset]
        v4 = v4_cues[offset]
        human = human_aligned[offset]
        source_matches += _compact(current["source"]) == _compact(v4["source"])
        timestamp_matches += (current["start"], current["end"]) == (
            v4["start"],
            v4["end"],
        )
        target = current["target"]
        if not target:
            missing_targets.append(current["index"])
        if PLACEHOLDER_PATTERN.search(target):
            placeholders.append(current["index"])
        if target and _similarity(target, current["source"]) >= 0.92:
            untranslated.append(current["index"])

        human_score = _similarity(target, human["target"])
        v4_score = _similarity(target, v4["target"])
        human_similarities.append(human_score)
        v4_similarities.append(v4_score)
        differences.append(
            {
                "index": current["index"],
                "human_similarity": round(human_score, 4),
                "source": current["source"],
                "candidate": target,
                "v4": v4["target"],
                "human": human["target"],
            }
        )

        neighbor_scores: list[tuple[int, float]] = []
        for neighbor_offset in (offset - 1, offset + 1):
            if 0 <= neighbor_offset < comparable_count:
                neighbor_scores.append(
                    (
                        candidate_cues[neighbor_offset]["index"],
                        _similarity(target, human_aligned[neighbor_offset]["target"]),
                    )
                )
        if neighbor_scores:
            neighbor_index, neighbor_score = max(neighbor_scores, key=lambda item: item[1])
            if neighbor_score >= 0.72 and neighbor_score >= human_score + 0.2:
                ownership_shift_risks.append(
                    {
                        "index": current["index"],
                        "closer_to_human_key": neighbor_index,
                        "own_similarity": round(human_score, 4),
                        "neighbor_similarity": round(neighbor_score, 4),
                    }
                )

        if offset:
            previous = candidate_cues[offset - 1]
            candidate_overlap = _similarity(previous["target"], target)
            human_overlap = _similarity(human_aligned[offset - 1]["target"], human["target"])
            source_overlap = _similarity(previous["source"], current["source"])
            if (
                len(_compact(previous["target"])) >= 6
                and len(_compact(target)) >= 6
                and candidate_overlap >= 0.76
                and candidate_overlap >= max(human_overlap, source_overlap) + 0.2
            ):
                duplicate_risks.append(
                    {
                        "left": previous["index"],
                        "right": current["index"],
                        "candidate_similarity": round(candidate_overlap, 4),
                        "human_similarity": round(human_overlap, 4),
                    }
                )

    differences.sort(key=lambda item: item["human_similarity"])
    return {
        "candidate": str(candidate),
        "v4_reference": str(v4_reference),
        "human_reference": str(human_reference),
        "cue_counts": {
            "candidate": len(candidate_cues),
            "v4": len(v4_cues),
            "human": len(human_cues),
            "comparable": comparable_count,
        },
        "structure": {
            "source_match_rate": round(source_matches / comparable_count, 4)
            if comparable_count
            else 0.0,
            "timestamp_match_rate": round(timestamp_matches / comparable_count, 4)
            if comparable_count
            else 0.0,
            "missing_targets": missing_targets,
            "placeholder_targets": placeholders,
            "untranslated_targets": untranslated,
            "ownership_shift_risks": ownership_shift_risks,
            "adjacent_duplicate_risks": duplicate_risks,
        },
        "reference_similarity": {
            "human_mean": round(mean(human_similarities), 4) if human_similarities else 0.0,
            "v4_mean": round(mean(v4_similarities), 4) if v4_similarities else 0.0,
        },
        "lowest_human_similarity": differences[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("v4_reference", type=Path)
    parser.add_argument("human_reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = evaluate(
        args.candidate.expanduser().resolve(),
        args.v4_reference.expanduser().resolve(),
        args.human_reference.expanduser().resolve(),
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.expanduser().resolve().write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
