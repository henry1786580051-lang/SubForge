"""Machine-readable and human-readable evaluation report writers."""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import EvaluationReport


def write_report(report: EvaluationReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    (output_dir / "aggregate.json").write_text(
        json.dumps(payload["aggregate"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "sample-metrics.jsonl").open("w", encoding="utf-8") as stream:
        for sample in payload["samples"]:
            stream.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")


def _markdown(report: EvaluationReport) -> str:
    aggregate = report.aggregate
    lines = [
        "# Translation Quality Baseline",
        "",
        f"- Corpus: `{report.corpus_id}`",
        f"- Manifest SHA-256: `{report.manifest_hash}`",
        f"- Comparison identity: `{report.comparison_identity}`",
        f"- Samples: {aggregate['sample_count']}",
        f"- Machine cues: {aggregate['machine_cue_count']}",
        f"- Human gold cues: {aggregate['gold_cue_count']}",
        f"- Structurally exact samples: {aggregate['structurally_exact_samples']}",
        f"- Samples requiring alignment: {aggregate['requires_alignment_samples']}",
        f"- Hard failures: {aggregate['hard_failure_count']}",
        f"- Empty translations: {aggregate['empty_targets']}",
        f"- Placeholder translations: {aggregate['placeholder_targets']}",
        f"- Reasoning leaks: {aggregate['reasoning_leaks']}",
        f"- Source-copy risks: {aggregate['source_copy_targets']}",
        f"- Untranslated risks: {aggregate['untranslated_targets']}",
        f"- Adjacent duplicate risks: {aggregate['adjacent_duplicate_risks']}",
        f"- Human-edited cues: {aggregate['human_changed_cues']}",
        "",
        "## Samples",
        "",
        "| Sample | Split | Machine cues | Exact structure | Hard failures | Human changes |",
        "| --- | --- | ---: | :---: | ---: | ---: |",
    ]
    for sample in report.samples:
        lines.append(
            "| "
            + " | ".join(
                (
                    sample.title.replace("|", "\\|"),
                    sample.split,
                    str(sample.machine["cue_count"]),
                    "yes" if sample.structure["exact"] else "no",
                    str(sample.hard_failure_count),
                    str(len(sample.gold_comparison["changed_cues"])),
                )
            )
            + " |"
        )
    lines.extend(("", "Detailed cue text is intentionally omitted from the tracked report.", ""))
    return "\n".join(lines)
