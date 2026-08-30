import json
import sys
from pathlib import Path

import pytest

from scripts.run_translation_quality_shadow import (
    _validate_resume_payload,
    build_candidate_manifest,
    isolated_runtime_settings,
    isolated_settings_source,
    sanitize_task_record_details,
    select_development_samples,
    select_shadow_samples,
    suppress_console_output,
)
from scripts.translation_quality.manifest import CorpusManifest, CorpusSample, sha256_file


def _sample(sample_id: str, split: str = "development") -> CorpusSample:
    return CorpusSample(
        id=sample_id,
        title=sample_id,
        split=split,  # type: ignore[arg-type]
        domain="other",
        speaker_mode="unknown",
        speaker_count=None,
        source_languages=("en",),
        target_language="zh-CN",
        duration_ms=1000,
        source_srt=f"{sample_id}/source.srt",
        machine_srt=f"{sample_id}/old.srt",
        gold_srt=f"{sample_id}/gold.srt",
        source_media=None,
        machine_model="old-model",
        algorithm_version="legacy",
        configuration={},
        provenance={},
        alignment={"cue_structure": "exact"},
        known_issues=(),
        notes="",
        hashes={},
    )


def test_select_development_samples_never_admits_holdout():
    manifest = CorpusManifest(
        1,
        "test",
        "now",
        "ROOT",
        (_sample("dev"), _sample("secret", "holdout")),
    )

    assert [sample.id for sample in select_development_samples(manifest)] == ["dev"]
    with pytest.raises(ValueError, match="development samples only"):
        select_development_samples(manifest, ("secret",))


def test_blind_holdout_selection_requires_explicit_full_split():
    manifest = CorpusManifest(
        1,
        "test",
        "now",
        "ROOT",
        (
            _sample("dev"),
            _sample("secret-1", "holdout"),
            _sample("secret-2", "holdout"),
        ),
    )

    with pytest.raises(ValueError, match="--blind-holdout"):
        select_shadow_samples(manifest, split="holdout")
    with pytest.raises(ValueError, match="complete frozen holdout"):
        select_shadow_samples(
            manifest,
            split="holdout",
            requested_ids=("secret-1",),
            blind_holdout=True,
        )

    selected = select_shadow_samples(
        manifest,
        split="holdout",
        blind_holdout=True,
    )
    assert [sample.id for sample in selected] == ["secret-1", "secret-2"]


def test_shadow_selection_preserves_development_filtering():
    manifest = CorpusManifest(
        1,
        "test",
        "now",
        "ROOT",
        (_sample("dev-1"), _sample("dev-2"), _sample("secret", "holdout")),
    )

    selected = select_shadow_samples(
        manifest,
        split="development",
        requested_ids=("dev-2",),
    )

    assert [sample.id for sample in selected] == ["dev-2"]


def test_isolated_runtime_settings_restores_backend_config(monkeypatch, tmp_path: Path):
    import importlib

    config_api = importlib.import_module("app.api.config")

    original = config_api.get_config_value
    monkeypatch.setattr(config_api, "get_config_value", lambda key, default: f"saved:{key}")
    saved = config_api.get_config_value

    with isolated_runtime_settings(
        output_dir=tmp_path,
        concurrency=20,
        batch_size=20,
    ):
        assert config_api.get_config_value("work_dir", "") == str(tmp_path)
        assert config_api.get_config_value("thread_num", 3) == 20
        assert config_api.get_config_value("other", "") == "saved:other"

    assert config_api.get_config_value is saved
    monkeypatch.setattr(config_api, "get_config_value", original)


def test_blind_console_suppression_hides_task_output_and_restores_streams(capfd):
    print("before")
    with suppress_console_output(True):
        print("secret subtitle text")
        print("secret validation detail", file=sys.stderr)
    print("after")

    captured = capfd.readouterr()
    assert captured.out.splitlines() == ["before", "after"]
    assert captured.err == ""


def test_blind_task_record_details_drop_warnings_and_redact_error():
    warnings, error = sanitize_task_record_details(
        blind_holdout=True,
        warnings=["subtitle text leaked through validator"],
        error="failed around a secret subtitle line",
    )

    assert warnings == []
    assert error == "redacted for blind holdout"


def test_development_task_record_details_remain_diagnostic():
    warnings, error = sanitize_task_record_details(
        blind_holdout=False,
        warnings=["retry used"],
        error="provider failed",
    )

    assert warnings == ["retry used"]
    assert error == "provider failed"


def test_build_candidate_manifest_uses_staged_files_and_new_hashes(tmp_path: Path):
    workload_root = tmp_path / "workloads"
    sample_dir = workload_root / "dev"
    sample_dir.mkdir(parents=True)
    source = sample_dir / "source.srt"
    gold = sample_dir / "gold.srt"
    machine = sample_dir / "candidate.srt"
    source.write_text("source", encoding="utf-8")
    gold.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好 世界\nHello world.\n", encoding="utf-8")
    machine.write_text("1\n00:00:00,000 --> 00:00:00,500\n你好\nHello\n\n"
                       "2\n00:00:00,500 --> 00:00:01,000\n世界\nworld.\n", encoding="utf-8")
    sample = _sample("dev")
    original = CorpusManifest(1, "test", "now", "ROOT", (sample,))

    candidate = build_candidate_manifest(
        original,
        selected=(sample,),
        workload_root=workload_root,
        records=({"sample_id": "dev", "output": str(machine)},),
        revision="phase8-shadow-baseline",
        model="deepseek-v4-flash",
        concurrency=20,
        batch_size=20,
        reflect=True,
    )

    staged = candidate.samples[0]
    assert staged.source_srt == "dev/source.srt"
    assert staged.machine_srt == "dev/candidate.srt"
    assert staged.gold_srt == "dev/gold.srt"
    assert staged.machine_model == "deepseek-v4-flash"
    assert staged.configuration["pipeline_revision"] == "phase8-shadow-baseline"
    assert staged.alignment["cue_structure"] == "requires_alignment"
    assert staged.alignment["timing_changed"] is True
    assert staged.alignment["advertisements_removed"] is None
    assert staged.hashes == {
        "source_sha256": sha256_file(source),
        "machine_sha256": sha256_file(machine),
        "gold_sha256": sha256_file(gold),
    }


def test_resume_rejects_changed_workload_configuration():
    from argparse import Namespace

    args = Namespace(
        revision="phase8",
        provider="deepseek",
        model="deepseek-v4-flash",
        concurrency=20,
        batch_size=20,
        reflect=True,
        split="development",
    )
    payload = {
        "pipeline": {"variant": "candidate", "revision": "phase8"},
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "concurrency": 10,
        "batch_size": 20,
        "reflect": True,
        "split": "development",
    }

    with pytest.raises(RuntimeError, match="concurrency"):
        _validate_resume_payload(payload, args)


def test_explicit_settings_source_is_frozen_and_restored(tmp_path: Path):
    from app.api import config

    path = tmp_path / "settings.json"
    payload = {"llm_provider": "zhipu", "llm_model": "glm-5.3-flash", "thread_num": 20}
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    original = config._read_settings
    with pytest.raises(RuntimeError, match="test failure"):
        with isolated_settings_source(path):
            assert config.get_config_value("thread_num", 1) == 20
            data = config._read_settings()
            data["llm_provider"] = "other"
            assert config._read_settings()["llm_provider"] == "zhipu"
            assert path.read_bytes() == before
            raise RuntimeError("test failure")
    assert config._read_settings is original
    assert path.read_bytes() == before


def test_missing_settings_source_never_falls_back(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        with isolated_settings_source(tmp_path / "missing.json"):
            pytest.fail("must not read a different profile")
