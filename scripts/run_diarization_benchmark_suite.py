#!/usr/bin/env python3
"""Run fixed-count and automatic Community-1 baselines from a prepared manifest."""

from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import logging
import sys
import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_diarization import _highest_disagreement_windows, _run
from subforge.core.utils.cache import disable_cache


def _aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        groups[(report["dataset"], report["mode"])].append(report)

    result: dict[str, Any] = {}
    for (dataset, mode), items in sorted(groups.items()):
        speaker_time = sum(
            float(item["accuracy"]["reference_speaker_time_seconds"]) for item in items
        )
        component_totals = {
            component: sum(float(item["accuracy"]["der_components"][component]) for item in items)
            for component in ("missed detection", "false alarm", "confusion")
        }
        weighted_der = sum(component_totals.values()) / speaker_time if speaker_time else 0.0
        result[f"{dataset}:{mode}"] = {
            "recordings": len(items),
            "runtime_seconds": sum(float(item["runtime_seconds"]) for item in items),
            "audio_seconds": sum(float(item["audio_duration_seconds"]) for item in items),
            "real_time_factor": (
                sum(float(item["runtime_seconds"]) for item in items)
                / sum(float(item["audio_duration_seconds"]) for item in items)
            ),
            "strict_regular_der": weighted_der,
            "regular_macro_jer": sum(float(item["accuracy"]["jer"]) for item in items)
            / len(items),
            "regular_der_component_rates": {
                key: value / speaker_time if speaker_time else 0.0
                for key, value in component_totals.items()
            },
            "regular_boundary_f1_250ms_diagnostic": sum(
                float(item["accuracy"]["boundary_250ms"]["f1"]) for item in items
            )
            / len(items),
            "regular_boundary_f1_500ms_diagnostic": sum(
                float(item["accuracy"]["boundary_500ms"]["f1"]) for item in items
            )
            / len(items),
            "speaker_count_exact": sum(
                item["detected_speakers"] == item["reference_speakers"] for item in items
            ),
            "short_islands_1500ms": sum(
                int(item["diagnostics"]["short_islands_1500ms"]) for item in items
            ),
            "execution_devices": sorted({str(item["execution_device"]) for item in items}),
        }
        alternative_items = [item for item in items if "alternative_accuracy" in item]
        if alternative_items:
            alternative_speaker_time = sum(
                float(item["alternative_accuracy"]["reference_speaker_time_seconds"])
                for item in alternative_items
            )
            alternative_errors = sum(
                sum(
                    float(value)
                    for value in item["alternative_accuracy"]["der_components"].values()
                )
                for item in alternative_items
            )
            result[f"{dataset}:{mode}"]["exclusive_diagnostic_der"] = (
                alternative_errors / alternative_speaker_time if alternative_speaker_time else 0.0
            )
            result[f"{dataset}:{mode}"]["exclusive_boundary_f1_250ms_diagnostic"] = sum(
                float(item["alternative_accuracy"]["boundary_250ms"]["f1"])
                for item in alternative_items
            ) / len(alternative_items)
            result[f"{dataset}:{mode}"]["exclusive_boundary_f1_500ms_diagnostic"] = sum(
                float(item["alternative_accuracy"]["boundary_500ms"]["f1"])
                for item in alternative_items
            ) / len(alternative_items)
    return result


def _audio_duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(path)
    return info.frames / info.samplerate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/diarization/baseline"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--mode", action="append", choices=("known", "auto"))
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--max-recordings", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    disable_cache()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    recordings = manifest.get("recordings", [])
    if args.dataset:
        recordings = [item for item in recordings if item["dataset"] in args.dataset]
    if args.max_recordings:
        recordings = recordings[: args.max_recordings]
    modes = args.mode or ["known", "auto"]
    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    suite_started = time.perf_counter()

    for recording in recordings:
        for mode in modes:
            uri = recording["uri"]
            dataset = recording["dataset"]
            report_path = output_root / dataset / f"{uri}-{mode}.json"
            hypothesis = report_path.with_suffix(".rttm")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            run_args = Namespace(
                audio=Path(recording["audio"]),
                hypothesis=hypothesis,
                reference=Path(recording["reference_rttm"]),
                uem=None,
                uri=uri,
                model=args.model,
                model_dir=args.model_dir,
                num_speakers=int(recording["speakers"]) if mode == "known" else None,
                min_speakers=2,
                max_speakers=10,
                no_cache=True,
                representation="regular",
            )
            logging.info("Running %s %s in %s mode", dataset, uri, mode)
            try:
                report = _run(run_args)
                report.update(
                    {
                        "dataset": dataset,
                        "mode": mode,
                        "reference_speakers": int(recording["speakers"]),
                        "detected_speakers": int(report["diagnostics"]["speakers"]),
                        "audio_duration_seconds": _audio_duration(run_args.audio),
                    }
                )
                report["highest_error_windows"] = _highest_disagreement_windows(
                    run_args.reference,
                    hypothesis,
                    uri,
                    window_seconds=30.0,
                    limit=10,
                )
                report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                reports.append(report)
            except Exception as exc:
                logging.exception("Benchmark failed for %s %s", uri, mode)
                failures.append({"dataset": dataset, "uri": uri, "mode": mode, "error": str(exc)})

    suite = {
        "format_version": 1,
        "manifest": str(args.manifest.resolve()),
        "wall_seconds": time.perf_counter() - suite_started,
        "reports": reports,
        "failures": failures,
        "aggregate": _aggregate(reports),
    }
    suite_path = output_root / "suite-report.json"
    suite_path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    print(suite_path)
    print(f"completed={len(reports)} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
