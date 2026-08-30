#!/usr/bin/env python3
"""Run an isolated, development-only translation-quality shadow workload."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import shutil
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for import_root in (PROJECT_ROOT, BACKEND_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.translation_quality.efficiency import load_efficiency_payload  # noqa: E402
from scripts.translation_quality.experiments import (  # noqa: E402
    EXPERIMENTS,
    translation_experiments,
)
from scripts.translation_quality.manifest import (  # noqa: E402
    CorpusManifest,
    CorpusSample,
    load_manifest,
    sha256_file,
    validate_manifest,
    write_manifest,
)
from scripts.translation_quality.srt import parse_srt  # noqa: E402


def select_development_samples(
    manifest: CorpusManifest,
    requested_ids: tuple[str, ...] = (),
) -> tuple[CorpusSample, ...]:
    """Select only development samples and reject ambiguous requests."""
    development = tuple(sample for sample in manifest.samples if sample.split == "development")
    if not requested_ids:
        return development
    requested = set(requested_ids)
    known = {sample.id: sample for sample in development}
    invalid = sorted(requested - known.keys())
    if invalid:
        raise ValueError(
            "Shadow workloads may use development samples only; unavailable ids: "
            + ", ".join(invalid)
        )
    return tuple(sample for sample in development if sample.id in requested)


def select_shadow_samples(
    manifest: CorpusManifest,
    *,
    split: str,
    requested_ids: tuple[str, ...] = (),
    blind_holdout: bool = False,
) -> tuple[CorpusSample, ...]:
    """Select one evaluation split while preventing holdout cherry-picking."""
    if split == "development":
        return select_development_samples(manifest, requested_ids)
    if split != "holdout":
        raise ValueError(f"Unsupported shadow split: {split}")
    if not blind_holdout:
        raise ValueError("Holdout runs require the explicit --blind-holdout safeguard")
    if requested_ids:
        raise ValueError("Blind holdout runs must include the complete frozen holdout split")
    return tuple(sample for sample in manifest.samples if sample.split == "holdout")


def _copy_workload_files(
    sample: CorpusSample,
    *,
    corpus_root: Path,
    workload_root: Path,
) -> tuple[Path, Path]:
    sample_dir = workload_root / sample.id
    sample_dir.mkdir(parents=True, exist_ok=False)
    source = sample_dir / "source.srt"
    gold = sample_dir / "gold.srt"
    shutil.copy2(corpus_root / sample.source_srt, source)
    shutil.copy2(corpus_root / sample.gold_srt, gold)
    return source, gold


@contextmanager
def isolated_settings_source(path: Path | None) -> Iterator[None]:
    """Choose a saved app profile explicitly; never rewrite or silently fall back."""
    if path is None:
        yield
        return
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("Settings source must be a file")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Settings source must contain an object")
    config_api = importlib.import_module("app.api.config")
    original = config_api._read_settings
    # A frozen snapshot prevents a UI provider switch from changing a running test.
    def read_settings() -> dict:
        import copy

        return copy.deepcopy(data)

    setattr(config_api, "_read_settings", read_settings)
    config_api.invalidate_config_cache()
    try:
        yield
    finally:
        setattr(config_api, "_read_settings", original)
        config_api.invalidate_config_cache()


@contextmanager
def isolated_runtime_settings(
    *,
    output_dir: Path,
    concurrency: int,
    batch_size: int,
) -> Iterator[None]:
    """Override non-secret task settings without modifying persisted settings."""
    config_api = importlib.import_module("app.api.config")
    original = config_api.get_config_value
    overrides: dict[str, Any] = {
        "work_dir": str(output_dir),
        "thread_num": concurrency,
        "batch_size": batch_size,
    }

    def get_config_value(key: str, default: Any) -> Any:
        return overrides[key] if key in overrides else original(key, default)

    setattr(config_api, "get_config_value", get_config_value)
    try:
        yield
    finally:
        setattr(config_api, "get_config_value", original)


@contextmanager
def suppress_console_output(enabled: bool) -> Iterator[None]:
    """Silence task internals so blind holdout text cannot leak through logs."""
    if not enabled:
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(sys.stdout.fileno())
    saved_stderr = os.dup(sys.stderr.fileno())
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, sys.stdout.fileno())
        os.dup2(null_fd, sys.stderr.fileno())
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, sys.stdout.fileno())
        os.dup2(saved_stderr, sys.stderr.fileno())
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(null_fd)


def sanitize_task_record_details(
    *,
    blind_holdout: bool,
    warnings: Any,
    error: str | None,
) -> tuple[list[Any], str]:
    """Prevent task diagnostics from persisting blind subtitle content."""
    if blind_holdout:
        return [], "redacted for blind holdout"
    return list(warnings or []), error or "unknown error"


async def run_shadow_sample(
    sample: CorpusSample,
    *,
    source_path: Path,
    output_dir: Path,
    revision: str,
    provider: str,
    model: str,
    concurrency: int,
    batch_size: int,
    reflect: bool,
    blind_holdout: bool = False,
) -> dict[str, Any]:
    """Run one candidate task through the production backend worker."""
    from subforge.core.translate.quality.pipeline_identity import (
        QUALITY_PIPELINE_FLAG,
        QUALITY_PIPELINE_REVISION,
        resolve_translation_pipeline_identity,
    )

    config_api = importlib.import_module("app.api.config")
    subtitle_api = importlib.import_module("app.api.subtitle")
    task_api = importlib.import_module("app.core.task_manager")
    runtime = config_api.get_llm_runtime_config()
    if runtime.provider != provider or runtime.model != model:
        raise RuntimeError(
            "Active LLM profile does not match the requested shadow workload: "
            f"expected {provider}/{model}, got {runtime.provider}/{runtime.model}"
        )
    if not runtime.api_key:
        raise RuntimeError("The active LLM profile has no usable API key")

    identity = resolve_translation_pipeline_identity(
        {
            QUALITY_PIPELINE_FLAG: "1",
            QUALITY_PIPELINE_REVISION: revision,
        }
    )
    task = task_api.task_manager.create_task("translation_quality_shadow")
    request = subtitle_api.SubtitleRequest(
        subtitle_file=str(source_path),
        target_language="chinese",
        translator="llm",
        need_optimize=True,
        need_translate=True,
        need_reflect=reflect,
        llm_provider=provider,
        llm_model=model,
        custom_prompt="",
    )
    with (
        isolated_runtime_settings(
            output_dir=output_dir,
            concurrency=concurrency,
            batch_size=batch_size,
        ),
        suppress_console_output(blind_holdout),
    ):
        await subtitle_api._run_subtitle(task.id, request, identity)

    sample_label = "[blind]" if blind_holdout else sample.id
    completed = task_api.task_manager.get_task(task.id)
    if completed is None:
        raise RuntimeError(f"Shadow task disappeared: {sample_label}")
    result = completed.result if isinstance(completed.result, dict) else {}
    succeeded = completed.status == task_api.TaskStatus.COMPLETED
    output_value = result.get("subtitle_file") if succeeded else result.get("recovery_file")
    output_path = Path(str(output_value or ""))
    telemetry_path = Path(str(result.get("telemetry_file") or ""))
    if not output_path.is_file() or not telemetry_path.is_file():
        outcome = "completed" if succeeded else "failed"
        error_detail = "redacted for blind holdout"
        if not blind_holdout:
            error_detail = completed.error or "none"
        raise RuntimeError(
            f"Shadow task {outcome} without complete artifacts: {sample_label}; "
            f"error={error_detail}"
        )
    telemetry = load_efficiency_payload(telemetry_path)
    if telemetry.get("pipeline") != identity.result_metadata():
        raise RuntimeError(f"Shadow telemetry identity mismatch: {sample_label}")
    warnings, error = sanitize_task_record_details(
        blind_holdout=blind_holdout,
        warnings=result.get("warnings"),
        error=completed.error,
    )
    return {
        "sample_id": sample.id,
        "task_id": task.id,
        "status": "completed" if succeeded else "failed",
        "output": str(output_path),
        "telemetry": str(telemetry_path),
        "warnings": warnings,
        **({"error": error} if not succeeded else {}),
    }


def build_candidate_manifest(
    original: CorpusManifest,
    *,
    selected: tuple[CorpusSample, ...],
    workload_root: Path,
    records: tuple[dict[str, Any], ...],
    revision: str,
    model: str,
    concurrency: int,
    batch_size: int,
    reflect: bool,
) -> CorpusManifest:
    """Build a self-contained manifest for static candidate evaluation."""
    record_by_id = {str(record["sample_id"]): record for record in records}
    staged: list[CorpusSample] = []
    for sample in selected:
        record = record_by_id.get(sample.id)
        if record is None:
            raise ValueError(f"Missing shadow output for {sample.id}")
        sample_dir = workload_root / sample.id
        source = sample_dir / "source.srt"
        gold = sample_dir / "gold.srt"
        machine = Path(str(record["output"]))
        machine_doc = parse_srt(machine, layout="target_above")
        gold_doc = parse_srt(gold, layout="target_above")
        exact = (
            [(cue.index, cue.timeline, cue.source) for cue in machine_doc.cues]
            == [(cue.index, cue.timeline, cue.source) for cue in gold_doc.cues]
        )
        staged.append(
            replace(
                sample,
                source_srt=source.relative_to(workload_root).as_posix(),
                machine_srt=machine.relative_to(workload_root).as_posix(),
                gold_srt=gold.relative_to(workload_root).as_posix(),
                source_media=None,
                machine_model=model,
                algorithm_version=revision,
                alignment={
                    **sample.alignment,
                    "cue_structure": "exact" if exact else "requires_alignment",
                    "comparison_basis": "fresh_segmentation_vs_frozen_gold",
                    "timing_changed": (
                        [(cue.index, cue.timeline) for cue in machine_doc.cues]
                        != [(cue.index, cue.timeline) for cue in gold_doc.cues]
                    ),
                    # No automatic deletion inference across independently split outputs.
                    "advertisements_removed": None,
                    "removed_ranges": [],
                },
                configuration={
                    **sample.configuration,
                    "concurrency": concurrency,
                    "batch_size": batch_size,
                    "reflect": reflect,
                    "pipeline_variant": "candidate",
                    "pipeline_revision": revision,
                },
                hashes={
                    **sample.hashes,
                    "source_sha256": sha256_file(source),
                    "machine_sha256": sha256_file(machine),
                    "gold_sha256": sha256_file(gold),
                },
            )
        )
    return CorpusManifest(
        schema_version=original.schema_version,
        corpus_id=original.corpus_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        data_root_env=original.data_root_env,
        samples=tuple(staged),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _recover_unrecorded_failure(
    sample: CorpusSample,
    *,
    workload_root: Path,
    revision: str,
) -> dict[str, Any] | None:
    """Recover a fully persisted failed task after the runner itself was interrupted."""
    sample_dir = workload_root / sample.id
    if not sample_dir.is_dir():
        return None
    recoveries = tuple(sample_dir.glob(f"*_candidate_{revision}_*_recovery.srt"))
    if not recoveries:
        return None
    if len(recoveries) != 1:
        raise RuntimeError(f"Ambiguous recovery artifacts for {sample.id}")
    output_path = recoveries[0]
    telemetry_path = output_path.with_suffix(".telemetry.json")
    telemetry = load_efficiency_payload(telemetry_path)
    if telemetry.get("pipeline") != {"variant": "candidate", "revision": revision}:
        raise RuntimeError(f"Recovered telemetry identity mismatch: {sample.id}")
    return {
        "sample_id": sample.id,
        "task_id": str(telemetry.get("task_id") or ""),
        "status": "failed",
        "output": str(output_path),
        "telemetry": str(telemetry_path),
        "warnings": [],
        "error": "Recovered failed task from its isolated recovery artifact",
    }


def _validate_resume_payload(payload: dict[str, Any], args: argparse.Namespace) -> None:
    expected = {
        "pipeline": {"variant": "candidate", "revision": args.revision},
        "provider": args.provider,
        "model": args.model,
        "concurrency": args.concurrency,
        "batch_size": args.batch_size,
        "reflect": args.reflect,
        "split": args.split,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if payload.get("experiments", []) != list(getattr(args, "experiment", [])):
        mismatches.append("experiments")
    if mismatches:
        raise RuntimeError(
            "Resume configuration does not match the frozen run: " + ", ".join(mismatches)
        )


async def _run(args: argparse.Namespace) -> int:
    corpus_root = args.root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest, corpus_root)
    selected = select_shadow_samples(
        manifest,
        split=args.split,
        requested_ids=tuple(args.sample_id),
        blind_holdout=args.blind_holdout,
    )
    if not selected:
        raise ValueError(f"The manifest has no selected {args.split} samples")
    output_exists = output_dir.exists() and any(output_dir.iterdir())
    if output_exists and not args.resume:
        raise FileExistsError(f"Shadow output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    workload_root = output_dir / "workloads"
    workload_root.mkdir(exist_ok=args.resume)

    run_path = output_dir / "run.json"
    if output_exists:
        run_payload = json.loads(run_path.read_text(encoding="utf-8"))
        if not isinstance(run_payload, dict):
            raise RuntimeError("The saved shadow run is not a JSON object")
        _validate_resume_payload(run_payload, args)
        if run_payload.get("sample_ids") != [sample.id for sample in selected]:
            raise RuntimeError("Resume sample set does not match the frozen run")
        records = list(run_payload.get("records") or [])
    else:
        run_payload = {
            "schema_version": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "pipeline": {"variant": "candidate", "revision": args.revision},
            "provider": args.provider,
            "model": args.model,
            "concurrency": args.concurrency,
            "batch_size": args.batch_size,
            "reflect": args.reflect,
            "experiments": list(getattr(args, "experiment", [])),
            "split": args.split,
            "sample_ids": [sample.id for sample in selected],
            "records": [],
        }
        records = []
    run_payload["status"] = "running"
    run_payload.pop("finished_at", None)
    run_payload.pop("error", None)
    _write_json(run_path, run_payload)

    try:
        for index, sample in enumerate(selected, start=1):
            recorded = next(
                (record for record in records if record.get("sample_id") == sample.id),
                None,
            )
            if recorded is not None:
                sample_label = sample.id if args.split == "development" else "[blind]"
                print(
                    f"SHADOW_SAMPLE={index}/{len(selected)}:{sample_label}:SKIPPED",
                    flush=True,
                )
                continue
            recovered = _recover_unrecorded_failure(
                sample,
                workload_root=workload_root,
                revision=args.revision,
            )
            if recovered is not None:
                records.append(recovered)
                run_payload["records"] = records
                _write_json(run_path, run_payload)
                sample_label = sample.id if args.split == "development" else "[blind]"
                print(
                    f"SHADOW_SAMPLE={index}/{len(selected)}:{sample_label}:RECOVERED",
                    flush=True,
                )
                continue
            source, _gold = _copy_workload_files(
                sample,
                corpus_root=corpus_root,
                workload_root=workload_root,
            )
            sample_label = sample.id if args.split == "development" else "[blind]"
            print(f"SHADOW_SAMPLE={index}/{len(selected)}:{sample_label}", flush=True)
            from scripts.translation_quality.replay import capture_agent_replays

            capture = (
                capture_agent_replays(
                    output_dir / "replay", provider=args.provider, revision=args.revision,
                )
                if getattr(args, "capture_replay", False) else nullcontext([])
            )
            with translation_experiments(tuple(getattr(args, "experiment", []))), capture as capture_errors:
                record = await run_shadow_sample(
                    sample,
                    source_path=source,
                    output_dir=source.parent,
                    revision=args.revision,
                    provider=args.provider,
                    model=args.model,
                    concurrency=args.concurrency,
                    batch_size=args.batch_size,
                    reflect=args.reflect,
                    blind_holdout=args.split == "holdout",
                )
            records.append(record)
            run_payload["records"] = records
            _write_json(run_path, run_payload)
            if capture_errors:
                raise RuntimeError(f"Replay capture incomplete: {len(capture_errors)} errors")
    except Exception as exc:
        run_payload["status"] = "failed"
        run_payload["finished_at"] = datetime.now(timezone.utc).isoformat()
        run_payload["error"] = str(exc)
        _write_json(run_path, run_payload)
        raise

    candidate_manifest = build_candidate_manifest(
        manifest,
        selected=selected,
        workload_root=workload_root,
        records=tuple(records),
        revision=args.revision,
        model=args.model,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        reflect=args.reflect,
    )
    candidate_manifest_path = output_dir / "candidate-manifest.json"
    write_manifest(candidate_manifest, candidate_manifest_path)
    validate_manifest(candidate_manifest, workload_root)
    run_payload["status"] = (
        "completed_with_failures"
        if any(record.get("status") == "failed" for record in records)
        else "completed"
    )
    run_payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    run_payload["candidate_manifest"] = str(candidate_manifest_path)
    _write_json(run_path, run_payload)
    print(f"SHADOW_RUN={run_path}")
    print(f"CANDIDATE_MANIFEST={candidate_manifest_path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--reflect", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument(
        "--split",
        choices=("development", "holdout"),
        default="development",
    )
    parser.add_argument(
        "--blind-holdout",
        action="store_true",
        help="Acknowledge that the complete holdout split will run without sample selection",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--capture-replay", action="store_true")
    parser.add_argument("--experiment", action="append", choices=EXPERIMENTS, default=[])
    parser.add_argument(
        "--settings-file", type=Path,
        help="Read this saved app configuration in isolation; never modify it",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.concurrency < 1 or args.batch_size < 1:
        raise SystemExit("Concurrency and batch size must be positive")
    try:
        with isolated_settings_source(args.settings_file):
            return asyncio.run(_run(args))
    except Exception as exc:
        print(f"ERROR={exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
