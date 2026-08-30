from pathlib import Path

from scripts.translation_quality.chinese_boundary_snapshot import (
    snapshot_chinese_boundary_document,
)
from scripts.translation_quality.srt import SrtCue, SrtDocument


def _cue(index: int, start_ms: int, end_ms: int, source: str, target: str) -> SrtCue:
    return SrtCue(
        index=index,
        timeline="",
        start_ms=start_ms,
        end_ms=end_ms,
        source=source,
        target=target,
        text_lines=(target, source),
    )


def test_chinese_boundary_snapshot_is_deterministic_and_contains_no_subtitle_text() -> None:
    document = SrtDocument(
        path=Path("sample.srt"),
        cues=(
            _cue(1, 0, 1_000, "Inside the tunnel", "在隧道内部"),
            _cue(2, 1_100, 2_000, "supports will be installed", "会安装支撑结构"),
        ),
        encoding="utf-8",
        newline="lf",
        has_bom=False,
    )

    first = snapshot_chinese_boundary_document(document)
    second = snapshot_chinese_boundary_document(document)

    assert first == second
    assert first["boundary_count"] == 1
    assert first["target_signal_count"] == 1
    assert first["unregistered_target_signal_count"] == 0
    assert first["target_rule_counts"] == {
        "translation.boundary.target.locative_predicate": 1
    }
    assert "tunnel" not in str(first).lower()
    assert "隧道" not in str(first)


def test_chinese_boundary_snapshot_attributes_dynamic_source_signal_to_english_rule() -> None:
    document = SrtDocument(
        path=Path("sample.srt"),
        cues=(
            _cue(1, 0, 1_000, "in October", "项目将在十月"),
            _cue(2, 1_100, 2_000, "2026, the project opens", "2026年启动"),
        ),
        encoding="utf-8",
        newline="lf",
        has_bom=False,
    )

    snapshot = snapshot_chinese_boundary_document(document)

    assert snapshot["source_signal_count"] == 1
    assert snapshot["unregistered_source_signal_count"] == 0
    assert snapshot["source_rule_counts"] == {
        "split.boundary.english.numeric.calendar_month_year": 1
    }
