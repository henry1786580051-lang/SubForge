#!/usr/bin/env python3
"""Replay captured translation batches offline and write text-free results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.translation_quality.experiments import (  # noqa: E402
    EXPERIMENTS,
    translation_experiments,
)
from scripts.translation_quality.replay import replay_agent, stable_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", action="append", choices=EXPERIMENTS, default=[])
    args = parser.parse_args()
    paths = sorted(args.fixtures.glob("*.json"))
    if not paths:
        parser.error("No replay fixtures found")
    reports = []
    for path in paths:
        fixture = json.loads(path.read_text(encoding="utf-8"))
        with translation_experiments(tuple(args.experiment)):
            first = replay_agent(fixture)
            second = replay_agent(fixture)
        if stable_json(first) != stable_json(second):
            raise RuntimeError("Non-deterministic replay detected")
        reports.append(first)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(stable_json({"fixtures": reports, "experiments": args.experiment}))
    matched = sum(report["matches_capture"] for report in reports)
    print(f"replayed={len(reports)} matched={matched} deterministic={len(reports)}")
    # For experiments, changed outcomes are evidence to audit, not automatic
    # quality successes. The report records every difference against the capture.
    return 0 if args.experiment or matched == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
