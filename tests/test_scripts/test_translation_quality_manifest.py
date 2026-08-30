from pathlib import Path

import pytest

from scripts.translation_quality.manifest import (
    CorpusManifest,
    CorpusSample,
    ManifestError,
    _choose_triplet,
    discover_corpus,
    load_manifest,
    sha256_file,
    validate_manifest,
    write_manifest,
)


def _srt(target: str | None = None) -> str:
    text = "Source sentence."
    body = f"{target}\n{text}" if target is not None else text
    return f"1\n00:00:00,000 --> 00:00:01,000\n{body}\n"


def _sample(root: Path, relative: str = "source.srt") -> CorpusSample:
    source = root / "source.srt"
    machine = root / "machine.srt"
    gold = root / "gold.srt"
    source.write_text(_srt(), encoding="utf-8")
    machine.write_text(_srt("机器译文"), encoding="utf-8")
    gold.write_text(_srt("人工译文"), encoding="utf-8")
    return CorpusSample(
        id="sample",
        title="Sample",
        split="development",
        domain="other",
        speaker_mode="unknown",
        speaker_count=None,
        source_languages=("en",),
        target_language="zh-CN",
        duration_ms=1000,
        source_srt=relative,
        machine_srt="machine.srt",
        gold_srt="gold.srt",
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


def test_manifest_round_trip_and_hash_validation(tmp_path: Path):
    sample = _sample(tmp_path)
    manifest = CorpusManifest(
        schema_version=1,
        corpus_id="test",
        created_at="2026-08-29T00:00:00+00:00",
        data_root_env="SUBFORGE_TRANSLATION_CORPUS_ROOT",
        samples=(sample,),
    )
    path = tmp_path / "manifest.json"

    write_manifest(manifest, path)
    loaded = load_manifest(path)
    validate_manifest(loaded, tmp_path)

    assert loaded == manifest


def test_manifest_rejects_path_escape(tmp_path: Path):
    sample = _sample(tmp_path, relative="../source.srt")
    manifest = CorpusManifest(1, "test", "now", "ROOT", (sample,))

    with pytest.raises(ManifestError, match="escapes data root"):
        validate_manifest(manifest, tmp_path)


def test_discover_corpus_classifies_triplet_without_renaming(tmp_path: Path):
    triplet = tmp_path / "A Test Video" / "三件套"
    triplet.mkdir(parents=True)
    (triplet / "A Test Video.srt").write_text(_srt(), encoding="utf-8")
    (triplet / "A Test Video_processed.srt").write_text(_srt("机器译文"), encoding="utf-8")
    (triplet / "A%20Test%20Video_processed_chatgpt_edited.srt").write_text(
        _srt("人工译文"), encoding="utf-8"
    )

    manifest = discover_corpus(tmp_path, created_at="now")

    assert len(manifest.samples) == 1
    sample = manifest.samples[0]
    assert sample.id == "a-test-video"
    assert sample.provenance["gold_editor"] == "chatgpt_pro"
    assert sample.alignment["cue_structure"] == "exact"
    validate_manifest(manifest, tmp_path)


def test_discover_corpus_uses_explicit_speaker_metadata(tmp_path: Path):
    triplet = tmp_path / "Dialogue" / "三件套"
    triplet.mkdir(parents=True)
    source = """1
00:00:00,000 --> 00:00:01,000
[Speaker 1] Hello.

2
00:00:01,000 --> 00:00:02,000
[Speaker 2] Hi.
"""
    bilingual = """1
00:00:00,000 --> 00:00:01,000
你好
[Speaker 1] Hello.

2
00:00:01,000 --> 00:00:02,000
嗨
[Speaker 2] Hi.
"""
    (triplet / "Dialogue.srt").write_text(source, encoding="utf-8")
    (triplet / "Dialogue_processed.srt").write_text(bilingual, encoding="utf-8")
    (triplet / "Dialogue_processed_chatgpt_edited.srt").write_text(
        bilingual, encoding="utf-8"
    )

    sample = discover_corpus(tmp_path, created_at="now").samples[0]

    assert sample.speaker_mode == "dialogue"
    assert sample.speaker_count == 2


@pytest.mark.parametrize("editor", ["GPT-5.6-Pro", "sonnet5", "CHATGPT"])
def test_model_qualified_gold_filenames_are_supported(tmp_path: Path, editor: str):
    triplet = tmp_path / "New Sample" / "三件套"
    triplet.mkdir(parents=True)
    source = triplet / "New Sample.srt"
    machine = triplet / "New Sample_processed.srt"
    gold = triplet / f"New%20Sample_processed_{editor}_edited.srt"
    source.write_text(_srt(), encoding="utf-8")
    machine.write_text(_srt("机器译文"), encoding="utf-8")
    gold.write_text(_srt("人工译文"), encoding="utf-8")
    assert _choose_triplet(triplet.iterdir()) == (source, machine, gold)
    sample = discover_corpus(tmp_path, created_at="now").samples[0]
    assert sample.provenance["verified_by_user"] is None
    assert sample.provenance["gold_editor"] == (
        "chatgpt_pro" if editor == "CHATGPT" else "unknown"
    )


def test_ambiguous_gold_files_are_not_silently_selected():
    files = map(Path, ["a.srt", "a_processed.srt", "a_gpt_edited.srt", "a_glm_edited.srt"])
    with pytest.raises(ManifestError, match="found 2"):
        _choose_triplet(files)


def test_edited_word_in_title_does_not_identify_gold():
    files = map(Path, ["chatgpt_edited_video.srt", "a_processed.srt", "a.srt"])
    with pytest.raises(ManifestError, match="found 0"):
        _choose_triplet(files)
