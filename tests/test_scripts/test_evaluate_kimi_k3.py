from pathlib import Path

from scripts.evaluate_kimi_k3 import evaluate


def _write_srt(path: Path, targets: list[str]) -> None:
    blocks = []
    sources = ["The first source.", "The second source.", "The third source."]
    for index, (source, target) in enumerate(zip(sources, targets, strict=True), 1):
        blocks.append(f"{index}\n00:00:0{index},000 --> 00:00:0{index + 1},000\n{target}\n{source}")
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8-sig")


def test_evaluator_reports_missing_placeholder_and_neighbor_shift(tmp_path):
    human = tmp_path / "human.srt"
    v4 = tmp_path / "v4.srt"
    candidate = tmp_path / "candidate.srt"
    _write_srt(human, ["第一句完整译文", "第二句完全不同", "第三句完整译文"])
    _write_srt(v4, ["第一句机器译文", "第二句机器译文", "第三句机器译文"])
    _write_srt(candidate, ["第二句完全不同", "待翻译", "第三句完整译文"])

    report = evaluate(candidate, v4, human)

    assert report["cue_counts"]["candidate"] == 3
    assert report["structure"]["placeholder_targets"] == [2]
    assert report["structure"]["ownership_shift_risks"] == [
        {
            "index": 1,
            "closer_to_human_key": 2,
            "own_similarity": 0.4286,
            "neighbor_similarity": 1.0,
        }
    ]
    assert report["structure"]["source_match_rate"] == 1.0
    assert report["structure"]["timestamp_match_rate"] == 1.0


def test_evaluator_aligns_a_split_human_reference_by_time(tmp_path):
    candidate = tmp_path / "candidate.srt"
    v4 = tmp_path / "v4.srt"
    human = tmp_path / "human.srt"
    candidate.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n完整译文\nComplete source.\n",
        encoding="utf-8-sig",
    )
    v4.write_text(candidate.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
    human.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n完整\nComplete\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\n译文\nsource.\n",
        encoding="utf-8-sig",
    )

    report = evaluate(candidate, v4, human)

    assert report["cue_counts"] == {
        "candidate": 1,
        "v4": 1,
        "human": 2,
        "comparable": 1,
    }
    assert report["reference_similarity"]["human_mean"] == 1.0
    assert report["structure"]["ownership_shift_risks"] == []
