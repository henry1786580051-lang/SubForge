import json
from pathlib import Path

import pytest

from subforge.core.entities import SubtitleProcessData
from subforge.core.llm.client import LLMRequestCancelled
from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.quality.references import physical_motion_reference
from subforge.core.translate.quality.text import is_untranslated_output
from subforge.core.translate.types import TargetLanguage


@pytest.fixture
def translator():
    t = LLMTranslator(
        1, 20, TargetLanguage.SIMPLIFIED_CHINESE, "offline", "", True, None, use_cache=False
    )
    yield t
    t.stop()


@pytest.mark.parametrize("first", [9, 155, 221, 226, 227, 234, 419, 555, 600])
def test_source_backed_breaks_are_mandatory_even_with_no_display_gap(translator, first):
    data = json.loads(
        (Path(__file__).parents[1] / "fixtures/asr/porsche_structural_boundaries.json").read_text()
    )
    cues = {c["id"]: c for w in data["windows"] for c in w["cues"]}
    left, right = cues[first], cues[first + 1]
    source = [SubtitleProcessData(i, c["source"], c["target"]) for i, c in [(1, left), (2, right)]]
    translator._all_source_by_index = {s.index: s.original_text for s in source}
    translator._gap_after_index = {1: 0}
    assert (
        translator._target_boundary_signal(1, left["target"], right["target"])
        == "source-backed Chinese syntax unit is split"
    )
    assert translator._mandatory_chinese_fluency_candidates(
        source, {s.index: s for s in source}
    ) == [1]


@pytest.mark.parametrize(
    "source,target",
    [
        (
            "S, the Carrera T, the GTS, the GT3, the GT3 Touring.",
            "S、Carrera T、GTS、GT3、GT3 Touring。",
        ),
        ("Carrera S, Carrera T, GT3 Touring.", "Carrera S、Carrera T、GT3 Touring。"),
        ("GT3, GTS, Turbo S", "GT3、GTS、Turbo S"),
    ],
)
def test_owned_model_lists_are_valid_chinese_subtitle_content(source, target):
    assert not is_untranslated_output(target, source, TargetLanguage.SIMPLIFIED_CHINESE, "en")


@pytest.mark.parametrize(
    "source,target",
    [
        ("I, YOU, WE", "I, YOU, WE"),
        ("The GT3 is very fast.", "The GT3 is very fast."),
        ("Carrera S, GT3", "Carrera S、GT4"),
        ("Carrera S, GT3", "GT3、Carrera S"),
    ],
)
def test_list_exception_does_not_accept_english_or_changed_models(source, target):
    assert is_untranslated_output(target, source, TargetLanguage.SIMPLIFIED_CHINESE, "en")


@pytest.mark.parametrize(
    "source,following,left,right",
    [
        ("This is made by Porsche.", "It is a hybrid.", "这是保时捷造的", "它是混动"),
        ("It is soft.", "It feels nice.", "扶手箱是软的", "手感很好"),
        ("No.", "But you can try.", "不会", "不过可以试试"),
        ("Start playback.", "Then stop.", "开始播放", "然后停止"),
    ],
)
def test_normal_complete_chinese_is_not_a_source_backed_hard_split(
    translator, source, following, left, right
):
    translator._all_source_by_index = {1: source, 2: following}
    assert (
        translator._target_boundary_signal(1, left, right)
        != "source-backed Chinese syntax unit is split"
    )


def test_complete_qualification_does_not_raise_mandatory_review(translator):
    source = [
        SubtitleProcessData(
            1,
            "because it means that if you're pulling into your driveway in a hurry, or something,",
            "因为这意味着 如果你急匆匆地开进车道什么的",
        ),
        SubtitleProcessData(
            2,
            "the car will already preemptively start lifting the front.",
            "车子就会提前开始抬升前桥",
        ),
    ]
    translator._all_source_by_index = {s.index: s.original_text for s in source}
    assert (
        translator._mandatory_chinese_fluency_candidates(source, {s.index: s for s in source}) == []
    )
    translator._record_structure_review(source, source)
    assert translator.structure_review["status"] == "passed"


def test_motion_pronoun_uses_only_nearby_explicit_source_context(translator):
    source = "If you want this to go down prematurely,"
    translator._all_source_by_index = {1: "Now the front end lift.", 3: source}
    assert translator._physical_motion_reference("3", source)["direction"] == "down"
    valid, feedback = translator._validate_contextual_automotive_terms(
        {"3": "如果你想让这提前发生"},
        {"3": source},
        str,
    )
    assert not valid and "physical" in feedback
    assert translator._validate_contextual_automotive_terms(
        {"3": "如果你想让它提前降下来"},
        {"3": source},
        str,
    )[0]
    assert not physical_motion_reference(3, source, {1: "Front end lift."}, {1: 30_000})
    assert not physical_motion_reference(20, source, {1: "Front end lift."}, {})
    assert not physical_motion_reference(
        3, "I wish I saw this go down on camera.", {1: "Front end lift."}, {}
    )


def test_rechecking_a_region_clears_only_its_previous_review_flags(translator, monkeypatch):
    source = [SubtitleProcessData(i, f"Sentence {i}.", "完整句子") for i in (1, 2)]
    translator.structure_review = {"checked_boundaries": [1, 9], "unresolved_boundaries": [1, 9]}
    monkeypatch.setattr(translator, "_mandatory_chinese_fluency_candidates", lambda *a: [])
    translator._record_structure_review(source, source)
    assert translator.structure_review["unresolved_boundaries"] == [9]


def test_failed_auditor_is_reported_as_needing_review(translator, monkeypatch):
    source = [SubtitleProcessData(i, f"Sentence {i}.", "完整句子") for i in (1, 2)]
    translator._all_source_by_index = {s.index: s.original_text for s in source}
    monkeypatch.setattr(translator, "_chinese_fluency_candidates", lambda *a: [1])
    monkeypatch.setattr(translator, "_mandatory_chinese_fluency_candidates", lambda *a: [])

    def unavailable(*args):
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(translator, "_request_chinese_fluency_flags", unavailable)
    translator._repair_chinese_boundary_fluency(source, {s.index: s for s in source})
    translator._record_structure_review(source, source)
    assert translator.structure_review["unresolved_boundaries"] == [1]


def test_cancelled_structural_rewrite_does_not_issue_fallback_requests(translator, monkeypatch):
    source = [SubtitleProcessData(i, f"Sentence {i}.", "完整句子") for i in (1, 2)]
    monkeypatch.setattr(
        translator, "_deterministic_chinese_fluency_fallback", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        translator, "_should_reason_about_chinese_fluency_window", lambda *a, **kw: False
    )
    calls = []

    def cancel(*args, **kwargs):
        calls.append(True)
        raise LLMRequestCancelled("cancelled")

    monkeypatch.setattr(translator, "_rewrite_chinese_fluency_window", cancel)
    with pytest.raises(LLMRequestCancelled):
        translator._repair_chinese_fluency_window_with_retries(source, source)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "left,right",
    [
        ("我觉得这些还算温和", "对 我们不需要前桥抬升"),
        ("你得把它留在后备箱里", "再拿出来"),
        ("先把它放在储物格中", "然后合上盖子"),
    ],
)
def test_complete_words_and_disposal_predicates_are_not_hard_splits(translator, left, right):
    assert translator._target_boundary_signal(1, left, right) not in {
        "connective stranded at previous subtitle end",
        "ba construction is separated from its predicate",
    }


def test_disposal_object_without_a_predicate_is_still_rejected(translator):
    assert (
        translator._target_boundary_signal(1, "你得把那个导流板", "放在后备箱里")
        == "ba construction is separated from its predicate"
    )


def test_standalone_spoken_hesitation_is_allowed_only_at_its_real_pause(translator):
    translator._all_source_by_index = {1: "I mean,", 2: "Like, yeah, it sounds cool."}
    translator._gap_after_index = {1: 2532}
    assert not translator._target_boundary_signal(1, "我是说", "对 它听起来很酷")
    translator._gap_after_index = {1: 0}
    assert translator._target_boundary_signal(1, "我是说", "对 它听起来很酷")


@pytest.mark.parametrize(
    "source,following,left,right",
    [
        ("this is always here", "it's always a button push away", "这个一直都在", "永远一键即达"),
        (
            "Would it be comfortable for long drives? No.",
            "But maybe your partner has something else.",
            "长途乘坐会舒服吗？不会",
            "但也许你的伴侣有别的车",
        ),
    ],
)
def test_source_proves_complete_short_answers_and_existence(
    translator, source, following, left, right
):
    translator._all_source_by_index = {1: source, 2: following}
    assert not translator._target_boundary_signal(1, left, right)


def test_open_source_locative_still_requires_its_complement(translator):
    translator._all_source_by_index = {1: "It's always in", 2: "the trunk."}
    assert translator._target_boundary_signal(1, "这个一直都在", "后备箱里")


@pytest.mark.parametrize("unsafe", ["", "speaker", "language", "too_long", "source", "quoted"])
def test_deferred_connective_is_scoped_and_moves_no_material_fact(translator, unsafe):
    source = [
        SubtitleProcessData(
            1, "The Turbo S wafts unlike anything before, but", "Turbo S的巡航感从未有过 但"
        ),
        SubtitleProcessData(
            2, "I think if we ever consider teleporting,", "我觉得如果我们考虑瞬移"
        ),
    ]
    translator._gap_after_index = {1: 6278}
    if unsafe == "speaker":
        translator._all_speaker_by_index = {1: "A", 2: "B"}
    if unsafe == "language":
        source[0].source_language, source[1].source_language = "en", "fr"
    if unsafe == "too_long":
        translator._gap_after_index[1] = 30_000
    if unsafe == "source":
        source[0].original_text = "The Turbo S wafts unlike anything before."
    if unsafe == "quoted":
        source[0].translated_text = "他说的是“不但”"
    result = translator._defer_paused_source_connective(source, source)
    if unsafe:
        assert result is None
    else:
        assert [s.translated_text for s in result] == [
            "Turbo S的巡航感从未有过",
            "但我觉得如果我们考虑瞬移",
        ]
        assert [s.original_text for s in result] == [s.original_text for s in source]
        assert source[0].translated_text.endswith("但")
