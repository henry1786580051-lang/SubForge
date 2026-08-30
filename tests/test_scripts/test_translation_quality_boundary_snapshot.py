from pathlib import Path

from scripts.translation_quality.boundary_snapshot import snapshot_document
from scripts.translation_quality.srt import SrtCue, SrtDocument


def _cue(index: int, source: str) -> SrtCue:
    return SrtCue(
        index=index,
        timeline=f"00:00:0{index - 1},000 --> 00:00:0{index},000",
        start_ms=(index - 1) * 1000,
        end_ms=index * 1000,
        source=source,
        target="译文",
        text_lines=("译文", source),
    )


def test_boundary_snapshot_is_deterministic_and_contains_no_subtitle_text() -> None:
    document = SrtDocument(
        path=Path("sample.srt"),
        cues=(
            _cue(1, "The meeting is in March"),
            _cue(2, "2026 before the opening."),
            _cue(3, "The project then continued."),
        ),
        encoding="utf-8",
        newline="lf",
        has_bom=False,
    )

    first = snapshot_document(document)
    second = snapshot_document(document)

    assert first == second
    assert first["boundary_count"] == 2
    assert first["unstable_count"] == 1
    assert first["registered_risk"] == 42
    assert first["unregistered_reason_counts"] == {}
    assert first["registered_rule_counts"] == {
        "split.boundary.english.numeric.calendar_month_year": 1
    }
    assert first["decision_sha256"] == second["decision_sha256"]
    assert "meeting" not in str(first).lower()
