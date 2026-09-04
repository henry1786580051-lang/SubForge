#!/usr/bin/env python3
"""Discover, validate, and evaluate the local translation-quality corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.translation_quality.boundary_audit import (  # noqa: E402
    audit_boundary_file,
    write_boundary_audit,
)
from scripts.translation_quality.boundary_snapshot import (  # noqa: E402
    evaluate_boundary_snapshot,
    write_boundary_snapshot,
)
from scripts.translation_quality.chinese_boundary_audit import (  # noqa: E402
    audit_chinese_boundary_file,
    write_chinese_boundary_audit,
)
from scripts.translation_quality.chinese_boundary_snapshot import (  # noqa: E402
    evaluate_chinese_boundary_snapshot,
    write_chinese_boundary_snapshot,
)
from scripts.translation_quality.comparison import (  # noqa: E402
    compare_evaluation_reports,
    load_evaluation_report,
    write_comparison_report,
)
from scripts.translation_quality.efficiency import (  # noqa: E402
    aggregate_efficiency_payloads,
    load_efficiency_payload,
    write_efficiency_report,
)
from scripts.translation_quality.manifest import (  # noqa: E402
    discover_corpus,
    load_manifest,
    validate_manifest,
    write_manifest,
)
from scripts.translation_quality.metrics import evaluate_manifest  # noqa: E402
from scripts.translation_quality.report import write_report  # noqa: E402


def _root(value: str | None) -> Path:
    raw = value or os.environ.get("SUBFORGE_TRANSLATION_CORPUS_ROOT")
    if not raw:
        raise ValueError(
            "Set --root or SUBFORGE_TRANSLATION_CORPUS_ROOT to the local corpus directory"
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _write_environment(output_dir: Path) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_status": _git_value("status", "--short").splitlines(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    (output_dir / "environment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _discover(args: argparse.Namespace) -> int:
    manifest = discover_corpus(
        _root(args.root),
        created_at=datetime.now(timezone.utc).isoformat(),
        holdout_ids=set(args.holdout_id),
        validation_ids=set(args.validation_id),
    )
    write_manifest(manifest, args.output)
    print(f"WROTE_MANIFEST={args.output}")
    print(f"SAMPLES={len(manifest.samples)}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    validate_manifest(manifest, _root(args.root), verify_hashes=not args.skip_hashes)
    print("MANIFEST_VALID=yes")
    print(f"SAMPLES={len(manifest.samples)}")
    return 0


def _baseline(args: argparse.Namespace) -> int:
    root = _root(args.root)
    manifest = load_manifest(args.manifest)
    validate_manifest(manifest, root)
    report = evaluate_manifest(
        manifest,
        root,
        manifest_hash=_sha256(args.manifest),
        splits=set(args.split) if args.split else None,
    )
    write_report(report, args.output_dir)
    _write_environment(args.output_dir)
    run_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "splits": args.split or ["development", "validation", "holdout"],
        "mode": "static",
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE_REPORT={args.output_dir / 'report.md'}")
    print(json.dumps(report.aggregate, ensure_ascii=False, sort_keys=True))
    if args.fail_on_hard and report.aggregate["hard_failure_count"]:
        return 2
    return 0


def _compare(args: argparse.Namespace) -> int:
    from scripts.translation_quality.admission import assess_admission, load_admission_policy

    if bool(args.policy) != bool(args.policy_sha256):
        raise ValueError("Supply --policy and its frozen --policy-sha256 together")
    policy = load_admission_policy(args.policy, args.policy_sha256) if args.policy else None
    legacy_efficiency = (
        load_efficiency_payload(args.legacy_efficiency)
        if args.legacy_efficiency
        else None
    )
    candidate_efficiency = (
        load_efficiency_payload(args.candidate_efficiency)
        if args.candidate_efficiency
        else None
    )
    legacy = load_evaluation_report(args.legacy_report)
    candidate = load_evaluation_report(args.candidate_report)
    payload = compare_evaluation_reports(
        legacy,
        candidate,
        legacy_efficiency=legacy_efficiency,
        candidate_efficiency=candidate_efficiency,
    )
    if policy is not None:
        payload["admission"] = assess_admission(
            payload, legacy, candidate, policy,
            legacy_efficiency=legacy_efficiency,
            candidate_efficiency=candidate_efficiency,
        )
        payload["admission"]["policy_sha256"] = args.policy_sha256
    write_comparison_report(payload, args.output_dir)
    print(f"WROTE_COMPARISON={args.output_dir / 'comparison.md'}")
    print(json.dumps(payload["metrics"], ensure_ascii=False, sort_keys=True))
    if policy is not None:
        decision = payload["admission"]["decision"]
        print(f"ADMISSION_SCREENING={decision}; PRODUCTION_ADOPTION=not_assessed")
        return {"review": 0, "observe": 2, "blocked": 3}[decision] if args.fail_on_regression else 0
    return 2 if args.fail_on_regression and not payload["accepted"] else 0


def _aggregate_efficiency(args: argparse.Namespace) -> int:
    payload = aggregate_efficiency_payloads(
        load_efficiency_payload(path) for path in args.snapshot
    )
    write_efficiency_report(payload, args.output)
    print(f"WROTE_EFFICIENCY_REPORT={args.output}")
    print(json.dumps(payload["aggregate"], ensure_ascii=False, sort_keys=True))
    return 0


def _boundary_snapshot(args: argparse.Namespace) -> int:
    root = _root(args.root)
    manifest = load_manifest(args.manifest)
    validate_manifest(manifest, root)
    payload = evaluate_boundary_snapshot(
        manifest,
        root,
        manifest_hash=_sha256(args.manifest),
        splits=set(args.split) if args.split else None,
    )
    write_boundary_snapshot(payload, args.output)
    print(f"WROTE_BOUNDARY_SNAPSHOT={args.output}")
    print(json.dumps(payload["aggregate"], ensure_ascii=False, sort_keys=True))
    return 0


def _boundary_audit(args: argparse.Namespace) -> int:
    payload = audit_boundary_file(PROJECT_ROOT / "subforge/core/split/boundary.py")
    write_boundary_audit(payload, args.output)
    print(f"WROTE_BOUNDARY_AUDIT={args.output}")
    summary_keys = (
        "registered_definition_count",
        "registered_call_count",
        "dynamic_registered_call_count",
        "unknown_registered_rule_ids",
        "legacy_site_count",
        "legacy_reason_count",
        "unpaired_legacy_site_count",
        "legacy_family_counts",
        "inventory_sha256",
    )
    print(json.dumps({key: payload[key] for key in summary_keys}, sort_keys=True))
    return int(
        bool(payload["unknown_registered_rule_ids"]) or bool(payload["unpaired_legacy_site_count"])
    )


def _chinese_boundary_snapshot(args: argparse.Namespace) -> int:
    root = _root(args.root)
    manifest = load_manifest(args.manifest)
    validate_manifest(manifest, root)
    payload = evaluate_chinese_boundary_snapshot(
        manifest,
        root,
        manifest_hash=_sha256(args.manifest),
        splits=set(args.split) if args.split else None,
    )
    write_chinese_boundary_snapshot(payload, args.output)
    print(f"WROTE_CHINESE_BOUNDARY_SNAPSHOT={args.output}")
    print(json.dumps(payload["aggregate"], ensure_ascii=False, sort_keys=True))
    return 0


def _chinese_boundary_audit(args: argparse.Namespace) -> int:
    payload = audit_chinese_boundary_file(
        PROJECT_ROOT / "subforge/core/translate/llm_translator.py"
    )
    write_chinese_boundary_audit(payload, args.output)
    print(f"WROTE_CHINESE_BOUNDARY_AUDIT={args.output}")
    summary_keys = (
        "registered_definition_count",
        "literal_message_site_count",
        "emitted_message_count",
        "dynamic_signal_return_counts",
        "unknown_emitted_messages",
        "unreferenced_registered_messages",
        "signal_call_count",
        "unexpected_signal_call_count",
        "diagnostic_adapter_call_count",
        "inventory_sha256",
        "layout_sha256",
        "call_flow_sha256",
    )
    print(json.dumps({key: payload[key] for key in summary_keys}, ensure_ascii=False, sort_keys=True))
    return int(
        bool(payload["missing_functions"])
        or bool(payload["unknown_emitted_messages"])
        or bool(payload["unreferenced_registered_messages"])
        or bool(payload["unexpected_signal_call_count"])
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="Discover local three-file groups")
    discover.add_argument("--root")
    discover.add_argument("--output", type=Path, required=True)
    discover.add_argument("--holdout-id", action="append", default=[])
    discover.add_argument("--validation-id", action="append", default=[])
    discover.set_defaults(handler=_discover)

    validate = subparsers.add_parser("validate", help="Validate a corpus manifest")
    validate.add_argument("--root")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--skip-hashes", action="store_true")
    validate.set_defaults(handler=_validate)

    baseline = subparsers.add_parser("baseline", help="Write deterministic static metrics")
    baseline.add_argument("--root")
    baseline.add_argument("--manifest", type=Path, required=True)
    baseline.add_argument("--output-dir", type=Path, required=True)
    baseline.add_argument(
        "--split",
        action="append",
        choices=("development", "validation", "holdout"),
    )
    baseline.add_argument("--fail-on-hard", action="store_true")
    baseline.set_defaults(handler=_baseline)

    compare = subparsers.add_parser(
        "compare",
        help="Compare isolated legacy and candidate evaluation reports",
    )
    compare.add_argument("--legacy-report", type=Path, required=True)
    compare.add_argument("--candidate-report", type=Path, required=True)
    compare.add_argument("--legacy-efficiency", type=Path)
    compare.add_argument("--candidate-efficiency", type=Path)
    compare.add_argument("--policy", type=Path, help="Prospectively frozen schema-2 admission policy")
    compare.add_argument("--policy-sha256", help="Policy hash recorded before candidate testing")
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument("--fail-on-regression", action="store_true")
    compare.set_defaults(handler=_compare)

    efficiency = subparsers.add_parser(
        "aggregate-efficiency",
        help="Aggregate task-scoped LLM efficiency snapshots",
    )
    efficiency.add_argument("--snapshot", type=Path, action="append", required=True)
    efficiency.add_argument("--output", type=Path, required=True)
    efficiency.set_defaults(handler=_aggregate_efficiency)

    boundary_snapshot = subparsers.add_parser(
        "boundary-snapshot",
        help="Write text-free hashes of legacy English boundary decisions",
    )
    boundary_snapshot.add_argument("--root")
    boundary_snapshot.add_argument("--manifest", type=Path, required=True)
    boundary_snapshot.add_argument("--output", type=Path, required=True)
    boundary_snapshot.add_argument(
        "--split",
        action="append",
        choices=("development", "validation", "holdout"),
    )
    boundary_snapshot.set_defaults(handler=_boundary_snapshot)

    boundary_audit = subparsers.add_parser(
        "boundary-audit",
        help="Inventory registered and legacy English boundary score branches",
    )
    boundary_audit.add_argument("--output", type=Path, required=True)
    boundary_audit.set_defaults(handler=_boundary_audit)

    chinese_boundary_snapshot = subparsers.add_parser(
        "chinese-boundary-snapshot",
        help="Write text-free hashes of legacy Chinese boundary signals",
    )
    chinese_boundary_snapshot.add_argument("--root")
    chinese_boundary_snapshot.add_argument("--manifest", type=Path, required=True)
    chinese_boundary_snapshot.add_argument("--output", type=Path, required=True)
    chinese_boundary_snapshot.add_argument(
        "--split",
        action="append",
        choices=("development", "validation", "holdout"),
    )
    chinese_boundary_snapshot.set_defaults(handler=_chinese_boundary_snapshot)

    chinese_boundary_audit = subparsers.add_parser(
        "chinese-boundary-audit",
        help="Inventory registered Chinese boundary signals and their call flow",
    )
    chinese_boundary_audit.add_argument("--output", type=Path, required=True)
    chinese_boundary_audit.set_defaults(handler=_chinese_boundary_audit)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        return int(args.handler(args))
    except Exception as exc:
        print(f"ERROR={exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
