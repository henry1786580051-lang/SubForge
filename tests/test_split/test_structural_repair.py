import json
from pathlib import Path

import pytest

from subforge.core.asr.asr_data import ASRDataSeg, ASRWord
from subforge.core.split.bilingual_repair import repair_bilingual_boundaries
from subforge.core.split.mapped_boundary import map_cleaned_words, normalize_mapped_boundaries
from subforge.core.split.structural import structural_dependency


def cues(ids):
    data = json.loads((Path(__file__).parent / "fixtures/ioniq_structural.json").read_text())
    return [
        ASRDataSeg(
            c["text"],
            c["start"],
            c["end"],
            speaker_id=c["speaker"],
            words=[ASRWord(**w) for w in c["words"]],
        )
        for c in data
        if c["id"] in ids
    ]


def test_deleted_filler_at_neighbor_boundary_is_preserved_exactly_once():
    before = cues([684, 685, 686, 687, 688, 689, 690, 691, 692])
    after = normalize_mapped_boundaries(before)
    assert [w for c in after for w in c.words] == [w for c in before for w in c.words]
    assert any(w.text == "um" and w.start_time == 2300082 for c in after for w in c.words)
    again = normalize_mapped_boundaries(after)
    assert [(c.text, c.start_time, c.end_time) for c in again] == [
        (c.text, c.start_time, c.end_time) for c in after
    ]


def test_custom_length_limit_prevents_otherwise_valid_merge():
    before = cues([518, 519])
    after = normalize_mapped_boundaries(before, soft_max_words=6, hard_max_words=10)
    assert len(after) == 2
    assert max(len(c.text.split()) for c in after) <= 10


def test_word_speakers_protect_boundary_when_cue_speakers_are_missing():
    before = cues([518, 519])
    for i, cue in enumerate(before):
        cue.speaker_id = ""
        for word in cue.words:
            word.speaker_id = str(i + 1)
    assert normalize_mapped_boundaries(before) == before


def test_structural_rule_does_not_infer_diarization_glitches():
    from subforge.core.split.boundary import normalize_boundaries

    before = cues([47, 48])
    before[1].speaker_id = "other"
    for word in before[1].words:
        word.speaker_id = "other"
    assert normalize_boundaries(before) == before


def test_historical_repair_budget_does_not_call_provider_or_change_original():
    before = cues([47, 48])
    diagnostics = []

    def forbidden(_):
        pytest.fail("Exhausted repair budget must not call the model")

    after = repair_bilingual_boundaries(
        before, forbidden, lambda c, t: True, diagnostics=diagnostics, max_repair_windows=0
    )
    assert after == before
    assert diagnostics[-1]["reason"] == "repair window budget exhausted"


def test_optimizer_repair_is_before_translation_and_uses_configured_limits(monkeypatch):
    from subforge.core.asr.asr_data import ASRData
    from subforge.core.optimize.optimize import SubtitleOptimizer

    optimizer = SubtitleOptimizer(1, 20, "unused", "", use_cache=False, max_word_count_english=6)
    monkeypatch.setattr(
        optimizer,
        "_parallel_optimize",
        lambda chunks: {str(i): c.text for i, c in enumerate(cues([518, 519]), 1)},
    )
    try:
        result = optimizer.optimize_subtitle(ASRData(cues([518, 519])))
        assert len(result.segments) == 2
        assert max(len(c.text.split()) for c in result.segments) <= 10
        assert not any(
            d["reason"] == "unresolved dependency" for d in optimizer.boundary_diagnostics
        )
    finally:
        optimizer.stop()


@pytest.mark.parametrize("n", [29, 47, 177, 237, 451, 518, 609])
def test_actual_dependencies_are_detected(n):
    a, b = cues([n, n + 1])
    assert structural_dependency(a.text, b.text)


@pytest.mark.parametrize(
    "left,right",
    [
        ("I like this.", "Sort of thing we discussed."),
        ("We finished the pattern.", "And we left."),
        ("What I enjoyed.", "About five years ago."),
        ("I cannot drive after two months,", "figure it out yourself."),
        ("because it,", "they can go"),
        ("It is very impressive", "and works well."),
        ("There are two.", "And a half dozen later."),
        ("no one owns this", "hauling equipment."),
    ],
)
def test_normal_sentences_and_unproven_structures_not_flagged(left, right):
    assert structural_dependency(left, right) is None


def test_47_moves_text_and_time_without_dropping_words_and_is_idempotent():
    before = cues([47, 48])
    after = normalize_mapped_boundaries(before)
    assert after[0].text.endswith("sort of thing,")
    assert after[1].text.startswith("and we have")
    assert (after[0].end_time, after[1].start_time) == (174831, 175152)
    assert [w for c in after for w in c.words] == [w for c in before for w in c.words]
    again = normalize_mapped_boundaries(after)
    assert [(c.text, c.start_time, c.end_time) for c in again] == [
        (c.text, c.start_time, c.end_time) for c in after
    ]


@pytest.mark.parametrize("n", [29, 177, 237, 609])
def test_unsolved_dependencies_are_reported_instead_of_creating_new_fragments(n):
    before = cues([n, n + 1])
    diagnostics = []
    after = normalize_mapped_boundaries(before, diagnostics=diagnostics)
    assert [c.text for c in before] == [c.text for c in after]
    assert any(d["reason"] == "unresolved dependency" for d in diagnostics)


def test_short_restarted_predicate_can_merge():
    before = cues([518, 519])
    after = normalize_mapped_boundaries(before)
    assert len(after) == 1
    assert "it sharpens" in after[0].text


@pytest.mark.parametrize("n", [188, 241, 386])
def test_mixed_speaker_evidence_is_never_merged(n):
    before = cues([n, n + 1])
    after = normalize_mapped_boundaries(before)
    assert after == before


def test_cleaned_filler_mapping_preserves_raw_words_and_repeated_model_context():
    before = cues([5, 19, 596, 597])
    mapping = map_cleaned_words(before[-1])
    assert not mapping.error and mapping.deleted_indices
    after = normalize_mapped_boundaries(before)
    assert any("IONIQ 5" in c.text for c in after[2:])
    assert "is is" not in " ".join(c.text for c in after[2:])
    assert [w for c in before for w in c.words] == [w for c in after for w in c.words]


@pytest.mark.parametrize("case", ["substitution", "deletion", "speaker", "estimated", "overlap"])
def test_unreliable_mapping_is_rejected(case):
    cue = cues([47])[0]
    if case == "substitution":
        cue.text = cue.text.replace("camo", "wood")
    if case == "deletion":
        cue.text = cue.text.replace("front ", "")
    if case == "speaker":
        cue.words[-1].speaker_id = "other"
    if case == "estimated":
        cue.words[-1].timing_source = "estimated"
    if case == "overlap":
        cue.words[-1].start_time = cue.words[-2].start_time
    assert map_cleaned_words(cue).error


def test_bilingual_normalizer_guard():
    before = cues([47, 48])
    before[0].translated_text = "旧译文"
    assert normalize_mapped_boundaries(before) == before


@pytest.mark.parametrize("case", ["empty", "missing", "validation", "provider", "mutation"])
def test_bilingual_window_rolls_back_atomically(case):
    before = cues([47, 48])
    for c in before:
        c.translated_text = "原有译文"

    def translate(window):
        if case == "provider":
            raise RuntimeError("unavailable")
        if case == "mutation":
            window[0].words[0].text = "CORRUPTED"
        return [""] * 2 if case == "empty" else ["one"] if case == "missing" else ["one", "two"]

    diagnostics = []
    after = repair_bilingual_boundaries(
        before,
        translate,
        lambda c, t: case not in {"validation", "mutation"},
        diagnostics=diagnostics,
    )
    assert after == before
    assert before[0].words[0].text == "The"
    assert diagnostics[-1]["reason"] == "rolled back"


def test_bilingual_window_success_and_cancellation():
    before = cues([47, 48])
    before[0].translated_text = "旧"
    after = repair_bilingual_boundaries(
        before, lambda c: ["迷彩风格", "红色拖车钩"], lambda c, t: True, diagnostics=[]
    )
    assert after[0].translated_text == "迷彩风格" and after[0].end_time == 174831
    assert before[0].translated_text == "旧" and before[0].end_time == 174270

    def cancel(_):
        raise InterruptedError()

    with pytest.raises(InterruptedError):
        repair_bilingual_boundaries(before, cancel, lambda c, t: True, diagnostics=[])
