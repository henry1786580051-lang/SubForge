import json
from pathlib import Path

from scripts.translation_quality.manifest import CorpusManifest, CorpusSample, sha256_file
from scripts.translation_quality.metrics import _is_placeholder, evaluate_manifest
from scripts.translation_quality.report import write_report


def _write(path: Path, target_one: str, target_two: str) -> Path:
    path.write_text(
        f"""1
00:00:00,000 --> 00:00:01,000
{target_one}
First source.

2
00:00:01,000 --> 00:00:02,000
{target_two}
Second source.
""",
        encoding="utf-8",
    )
    return path


def test_evaluate_manifest_reports_human_changes_without_exposing_text(tmp_path: Path):
    source = tmp_path / "source.srt"
    source.write_text(
        """1
00:00:00,000 --> 00:00:01,000
First source.

2
00:00:01,000 --> 00:00:02,000
Second source.
""",
        encoding="utf-8",
    )
    machine = _write(tmp_path / "machine.srt", "第一条机器译文", "第二条机器译文")
    gold = _write(tmp_path / "gold.srt", "第一条人工译文", "第二条机器译文")
    sample = CorpusSample(
        id="sample",
        title="Sample",
        split="development",
        domain="other",
        speaker_mode="unknown",
        speaker_count=None,
        source_languages=("en",),
        target_language="zh-CN",
        duration_ms=2000,
        source_srt=source.name,
        machine_srt=machine.name,
        gold_srt=gold.name,
        source_media=None,
        machine_model="deepseek-v4-flash",
        algorithm_version="test",
        configuration={},
        provenance={},
        alignment={"cue_structure": "exact"},
        known_issues=(),
        notes="",
        hashes={
            "source_sha256": sha256_file(source),
            "machine_sha256": sha256_file(machine),
            "gold_sha256": sha256_file(gold),
        },
    )
    manifest = CorpusManifest(1, "test", "now", "ROOT", (sample,))

    report = evaluate_manifest(manifest, tmp_path, manifest_hash="abc")

    assert report.aggregate["sample_count"] == 1
    assert report.aggregate["human_changed_cues"] == 1
    assert report.aggregate["hard_failure_count"] == 0
    assert report.samples[0].structure["exact"] is True

    output = tmp_path / "report"
    write_report(report, output)
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "第一条机器译文" not in markdown
    assert json.loads((output / "aggregate.json").read_text())["human_changed_cues"] == 1


def test_evaluate_manifest_counts_placeholders_and_reasoning_leaks(tmp_path: Path):
    source = tmp_path / "source.srt"
    source.write_text(
        """1
00:00:00,000 --> 00:00:01,000
First source.

2
00:00:01,000 --> 00:00:02,000
Second source.
""",
        encoding="utf-8",
    )
    machine = _write(tmp_path / "machine.srt", "待翻译", "<think>分析</think>")
    gold = _write(tmp_path / "gold.srt", "第一条", "第二条")
    sample = CorpusSample(
        id="sample",
        title="Sample",
        split="development",
        domain="other",
        speaker_mode="unknown",
        speaker_count=None,
        source_languages=("en",),
        target_language="zh-CN",
        duration_ms=2000,
        source_srt=source.name,
        machine_srt=machine.name,
        gold_srt=gold.name,
        source_media=None,
        machine_model="unknown",
        algorithm_version="unknown",
        configuration={},
        provenance={},
        alignment={"cue_structure": "exact"},
        known_issues=(),
        notes="",
        hashes={
            "source_sha256": sha256_file(source),
            "machine_sha256": sha256_file(machine),
            "gold_sha256": sha256_file(gold),
        },
    )

    report = evaluate_manifest(
        CorpusManifest(1, "test", "now", "ROOT", (sample,)),
        tmp_path,
        manifest_hash="abc",
    )

    assert report.aggregate["placeholder_targets"] == 1
    assert report.aggregate["reasoning_leaks"] == 1
    assert report.aggregate["hard_failure_count"] == 2


def test_report_redacts_holdout_cue_indices(tmp_path: Path):
    source = tmp_path / "source.srt"
    source.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nFirst source.\n", encoding="utf-8"
    )
    machine = tmp_path / "machine.srt"
    machine.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n待翻译\nFirst source.\n", encoding="utf-8"
    )
    gold = tmp_path / "gold.srt"
    gold.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n人工译文\nFirst source.\n", encoding="utf-8"
    )
    sample = CorpusSample(
        id="holdout",
        title="Holdout",
        split="holdout",
        domain="other",
        speaker_mode="unknown",
        speaker_count=None,
        source_languages=("en",),
        target_language="zh-CN",
        duration_ms=1000,
        source_srt=source.name,
        machine_srt=machine.name,
        gold_srt=gold.name,
        source_media=None,
        machine_model="unknown",
        algorithm_version="unknown",
        configuration={},
        provenance={},
        alignment={"cue_structure": "exact"},
        known_issues=(),
        notes="",
        hashes={
            "source_sha256": sha256_file(source),
            "machine_sha256": sha256_file(machine),
            "gold_sha256": sha256_file(gold),
        },
    )

    payload = evaluate_manifest(
        CorpusManifest(1, "test", "now", "ROOT", (sample,)),
        tmp_path,
        manifest_hash="abc",
    ).to_dict()

    holdout = payload["samples"][0]
    assert holdout["details_redacted"] is True
    assert "placeholder_targets" not in holdout["machine"]
    assert holdout["machine"]["placeholder_targets_count"] == 1
    assert "changed_cues" not in holdout["gold_comparison"]


def test_placeholder_detection_does_not_match_normal_words_containing_lue():
    assert _is_placeholder("（此句合并至上一句）") is True
    assert _is_placeholder("内容同上") is True
    assert _is_placeholder("比最终造价略便宜一些") is False
    assert _is_placeholder("这是更广泛旅游战略的一部分") is False


def test_reasoning_metric_does_not_treat_normal_result_narration_as_private_reasoning(
    tmp_path: Path,
):
    source = tmp_path / "source.srt"
    source.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHere are the results.\n",
        encoding="utf-8",
    )
    machine = tmp_path / "machine.srt"
    machine.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n以下是结果\nHere are the results.\n",
        encoding="utf-8",
    )
    gold = tmp_path / "gold.srt"
    gold.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n结果如下\nHere are the results.\n",
        encoding="utf-8",
    )
    sample = CorpusSample(
        id="normal-result",
        title="Normal Result",
        split="development",
        domain="other",
        speaker_mode="unknown",
        speaker_count=None,
        source_languages=("en",),
        target_language="zh-CN",
        duration_ms=1000,
        source_srt=source.name,
        machine_srt=machine.name,
        gold_srt=gold.name,
        source_media=None,
        machine_model="unknown",
        algorithm_version="unknown",
        configuration={},
        provenance={},
        alignment={"cue_structure": "exact"},
        known_issues=(),
        notes="",
        hashes={
            "source_sha256": sha256_file(source),
            "machine_sha256": sha256_file(machine),
            "gold_sha256": sha256_file(gold),
        },
    )

    report = evaluate_manifest(
        CorpusManifest(1, "test", "now", "ROOT", (sample,)),
        tmp_path,
        manifest_hash="abc",
    )

    assert report.aggregate["reasoning_leaks"] == 0

