#!/usr/bin/env python3
"""Run or score SubForge speaker diarization with reproducible reports."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subforge.core.asr.diarization_benchmark import (  # noqa: E402
    boundary_f1,
    diagnose_turns,
    load_rttm,
    write_rttm,
)


def _single_recording(recordings: dict[str, list], uri: str | None) -> tuple[str, list]:
    if uri:
        return uri, recordings[uri]
    if len(recordings) != 1:
        raise ValueError("RTTM contains multiple recordings; pass --uri")
    return next(iter(recordings.items()))


def _load_pyannote_annotation(path: Path, uri: str):
    from pyannote.database.util import load_rttm

    annotations = load_rttm(path)
    if uri not in annotations:
        raise ValueError(f"RTTM does not contain recording {uri!r}")
    return annotations[uri]


def _load_uem(path: Path | None, uri: str):
    if path is None:
        return None
    from pyannote.database.util import load_uem

    timelines = load_uem(path)
    if uri not in timelines:
        raise ValueError(f"UEM does not contain recording {uri!r}")
    return timelines[uri]


def _highest_disagreement_windows(
    baseline_path: Path,
    candidate_path: Path,
    uri: str,
    *,
    window_seconds: float,
    limit: int = 20,
) -> list[dict[str, float | str]]:
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    baseline = _load_pyannote_annotation(baseline_path, uri)
    candidate = _load_pyannote_annotation(candidate_path, uri)
    extent = baseline.get_timeline().union(candidate.get_timeline()).extent()
    if not extent or window_seconds <= 0:
        return []
    windows: list[dict[str, float | str]] = []
    start = float(extent.start)
    while start < extent.end:
        end = min(float(extent.end), start + window_seconds)
        evaluation_map = Timeline([Segment(start, end)], uri=uri)
        metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
        details: Any = metric(
            baseline,
            candidate,
            detailed=True,
            uem=evaluation_map,
        )
        total = float(details.get("total", 0.0))
        components = {
            key: float(details.get(key, 0.0))
            for key in ("missed detection", "false alarm", "confusion")
        }
        component_rates = {
            key: value / total if total else 0.0 for key, value in components.items()
        }
        disagreement = sum(component_rates.values())
        dominant_error = max(component_rates, key=component_rates.get)
        windows.append(
            {
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "disagreement": disagreement,
                "missed_detection": component_rates["missed detection"],
                "false_alarm": component_rates["false alarm"],
                "confusion": component_rates["confusion"],
                "dominant_error": dominant_error,
            }
        )
        start = end
    return sorted(windows, key=lambda item: item["disagreement"], reverse=True)[:limit]


def _score(reference_path: Path, hypothesis_path: Path, uri: str | None, uem: Path | None):
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

    reference_records = load_rttm(reference_path, uri=uri)
    hypothesis_records = load_rttm(hypothesis_path, uri=uri)
    recording_uri, reference_turns = _single_recording(reference_records, uri)
    if recording_uri not in hypothesis_records:
        raise ValueError(f"Hypothesis RTTM does not contain recording {recording_uri!r}")
    hypothesis_turns = hypothesis_records[recording_uri]
    reference = _load_pyannote_annotation(reference_path, recording_uri)
    hypothesis = _load_pyannote_annotation(hypothesis_path, recording_uri)
    evaluation_map = _load_uem(uem, recording_uri)
    der_metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    jer_metric = JaccardErrorRate(collar=0.0, skip_overlap=False)
    detailed: Any = der_metric(reference, hypothesis, detailed=True, uem=evaluation_map)
    total = float(detailed.get("total", 0.0))
    components = {
        key: float(detailed.get(key, 0.0))
        for key in ("missed detection", "false alarm", "confusion")
    }
    jer_result: Any = jer_metric(reference, hypothesis, uem=evaluation_map)
    return {
        "uri": recording_uri,
        "scoring": "strict: collar=0, overlap included",
        "der": float(abs(der_metric)),
        "jer": float(jer_result),
        "reference_speaker_time_seconds": total,
        "der_components": components,
        "der_component_rates": {
            key: value / total if total else 0.0 for key, value in components.items()
        },
        "reference": diagnose_turns(reference_turns).to_dict(),
        "hypothesis": diagnose_turns(hypothesis_turns).to_dict(),
        "boundary_250ms": boundary_f1(reference_turns, hypothesis_turns, tolerance_ms=250),
        "boundary_500ms": boundary_f1(reference_turns, hypothesis_turns, tolerance_ms=500),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    from subforge.core.asr.speaker_diarization import diarize_audio

    if args.no_cache:
        from subforge.core.utils.cache import disable_cache

        disable_cache()

    started = time.perf_counter()
    turns = diarize_audio(
        str(args.audio),
        model=args.model,
        model_dir=args.model_dir,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers if args.num_speakers is None else None,
        max_speakers=args.max_speakers if args.num_speakers is None else None,
    )
    elapsed = time.perf_counter() - started
    uri = args.uri or args.audio.stem
    representation = getattr(args, "representation", "exclusive")
    regular_turns = getattr(turns, "regular_turns", [])
    selected_turns = regular_turns if representation == "regular" and regular_turns else turns
    write_rttm(args.hypothesis, {uri: selected_turns})
    report: dict[str, Any] = {
        "uri": uri,
        "runtime_seconds": elapsed,
        "representation": representation,
        "diagnostics": diagnose_turns(selected_turns).to_dict(),
        "exclusive_diagnostics": diagnose_turns(turns).to_dict(),
        "regular_diagnostics": diagnose_turns(regular_turns).to_dict(),
        "device_requested": os.environ.get("SUBFORGE_DIARIZATION_DEVICE", "auto"),
        "execution_device": getattr(turns, "execution_device", "unknown"),
        "hypothesis_rttm": str(args.hypothesis),
    }
    if args.reference:
        report["accuracy"] = _score(args.reference, args.hypothesis, uri, args.uem)
        if regular_turns:
            alternative_representation = "exclusive" if representation == "regular" else "regular"
            alternative_turns = (
                turns if alternative_representation == "exclusive" else regular_turns
            )
            alternative_path = args.hypothesis.with_name(
                f"{args.hypothesis.stem}-{alternative_representation}{args.hypothesis.suffix}"
            )
            write_rttm(alternative_path, {uri: alternative_turns})
            report["alternative_representation"] = alternative_representation
            report["alternative_hypothesis_rttm"] = str(alternative_path)
            report["alternative_accuracy"] = _score(
                args.reference,
                alternative_path,
                uri,
                args.uem,
            )
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="score an existing RTTM")
    score.add_argument("reference", type=Path)
    score.add_argument("hypothesis", type=Path)
    score.add_argument("--uem", type=Path)
    score.add_argument("--uri")
    score.add_argument("--report", type=Path)

    stability = subparsers.add_parser(
        "stability",
        help="compare two unlabeled runs; values measure agreement, not accuracy",
    )
    stability.add_argument("baseline", type=Path)
    stability.add_argument("candidate", type=Path)
    stability.add_argument("--uem", type=Path)
    stability.add_argument("--uri")
    stability.add_argument("--window-seconds", type=float, default=30.0)
    stability.add_argument("--report", type=Path)

    run = subparsers.add_parser("run", help="run Community-1 and optionally score it")
    run.add_argument("audio", type=Path)
    run.add_argument("hypothesis", type=Path)
    run.add_argument("--reference", type=Path)
    run.add_argument("--uem", type=Path)
    run.add_argument("--uri")
    run.add_argument("--model", default="pyannote/speaker-diarization-community-1")
    run.add_argument("--model-dir", type=Path)
    count = run.add_mutually_exclusive_group()
    count.add_argument("--num-speakers", type=int)
    count.add_argument("--auto-speakers", action="store_true")
    run.add_argument("--min-speakers", type=int, default=2)
    run.add_argument("--max-speakers", type=int, default=10)
    run.add_argument("--report", type=Path)
    run.add_argument(
        "--representation",
        choices=("regular", "exclusive"),
        default="regular",
        help="RTTM representation used for the primary accuracy score",
    )
    run.add_argument(
        "--no-cache",
        action="store_true",
        help="disable result caching for runtime and device diagnostics",
    )

    args = parser.parse_args()
    if args.command == "score":
        report = _score(args.reference, args.hypothesis, args.uri, args.uem)
    elif args.command == "stability":
        baseline_records = load_rttm(args.baseline, uri=args.uri)
        recording_uri, _ = _single_recording(baseline_records, args.uri)
        report = {
            "notice": "Agreement against the baseline run; this is not ground-truth accuracy.",
            "agreement": _score(
                args.baseline,
                args.candidate,
                recording_uri,
                args.uem,
            ),
            "highest_disagreement_windows": _highest_disagreement_windows(
                args.baseline,
                args.candidate,
                recording_uri,
                window_seconds=args.window_seconds,
            ),
        }
    else:
        if args.num_speakers is not None and not 2 <= args.num_speakers <= 10:
            parser.error("--num-speakers must be between 2 and 10")
        if args.min_speakers > args.max_speakers:
            parser.error("--min-speakers must not exceed --max-speakers")
        report = _run(args)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
