import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from subforge.core.asr import final_coverage as fc
from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.asr.whisperx_asr import _critical_aligned_speech_gaps


def words(items):
    return ASRData([ASRDataSeg(t, a, b, timestamp_granularity="word") for t, a, b in items])


def timeline():
    return words([(f"word{i}", i * 500, i * 500 + 350) for i in range(60)])


class Context:
    audio_path = "original.wav"

    def samples(self):
        return np.zeros(40 * 16000, dtype=np.float32)


@pytest.fixture
def setup(monkeypatch):
    context = Context()
    config = SimpleNamespace(cancel_event=None, whisperx_model="local-test")
    speech = [(10000, 20000)]

    def audit(data, samples):
        return _critical_aligned_speech_gaps(
            {"segments": [{"words": fc._words(data)}]}, speech, len(samples) / 16000
        )

    monkeypatch.setattr(fc, "_audit", audit)
    records = []
    monkeypatch.setattr(fc, "_save_evidence", lambda r: records.extend(r))
    return context, config, records


def run(data, setup):
    context, config, _ = setup
    return fc.repair_final_coverage(data, "original.wav", config, lambda *a: None, None, context)


def damaged():
    data = timeline()
    data.segments = [s for s in data.segments if not 10000 <= s.start_time < 20000]
    # A bad neighboring phrase must be replaced, not preserved by gap-only insertion.
    data.segments[16].text = "misplaced"
    return data


def test_clean_timeline_never_decodes(monkeypatch, setup):
    monkeypatch.setattr(fc, "_decode", lambda *a: pytest.fail("unexpected decode"))
    result = run(timeline(), setup)
    assert not result.coverage_issues
    assert not setup[2]


def test_confirmed_recovery_replaces_wrong_neighbor_and_preserves_edges(monkeypatch, setup):
    data = damaged()
    original = copy.deepcopy(data)
    calls = []

    def decode(*args):
        calls.append((args[1], args[2]))
        return timeline()

    monkeypatch.setattr(fc, "_decode", decode)
    result = run(data, setup)
    assert len(calls) == 2
    assert calls[0] != calls[1]
    assert not result.coverage_issues
    assert "misplaced" not in [s.text for s in result.segments]
    assert fc._words(result) == fc._words(timeline())
    assert result.segments[0].__dict__ == original.segments[0].__dict__
    assert setup[2][0]["status"] == "recovered"


def test_disagreeing_decodes_leave_original_intact_and_flag_review(monkeypatch, setup):
    data = damaged()
    before = fc._words(data)
    calls = 0

    def decode(*args):
        nonlocal calls
        calls += 1
        candidate = timeline()
        if calls >= 2:
            candidate.segments[25].text = f"contradiction{calls}"
        return candidate

    monkeypatch.setattr(fc, "_decode", decode)
    result = run(data, setup)
    assert fc._words(result) == before
    assert result.coverage_issues[0]["reason"] == "final_export_gap"


def test_repeated_or_missing_anchors_do_not_overwrite(monkeypatch, setup):
    data = damaged()
    candidate = timeline()
    for seg in candidate.segments:
        seg.text = "unrelated"
    monkeypatch.setattr(fc, "_decode", lambda *a: candidate)
    before = fc._words(data)
    assert fc._words(run(data, setup)) == before
    assert data.coverage_issues


def test_overlong_word_cannot_prove_coverage_after_duration_cap(setup):
    data = damaged()
    preceding = next(s for s in data.segments if s.start_time == 9500)
    preceding.end_time = 20000
    assert not fc._audit(data, setup[0].samples())
    data.cap_abnormal_word_durations()
    assert fc._audit(data, setup[0].samples())


def test_budget_exhaustion_preserves_and_flags(monkeypatch, setup):
    monkeypatch.setattr(fc, "MAX_DECODE_SECONDS", 1)
    monkeypatch.setattr(fc, "_decode", lambda *a: pytest.fail("budget ignored"))
    data = damaged()
    before = fc._words(data)
    assert fc._words(run(data, setup)) == before
    assert data.coverage_issues
    assert setup[2][0]["status"] == "budget_exhausted"


def test_cancellation_is_not_swallowed(monkeypatch, setup):
    import threading

    event = threading.Event()
    setup[1].cancel_event = event

    def decode(*args):
        event.set()
        raise RuntimeError("stopped")

    monkeypatch.setattr(fc, "_decode", decode)
    with pytest.raises(RuntimeError, match="cancelled"):
        run(damaged(), setup)


def test_audit_failure_does_not_silently_succeed(monkeypatch, setup):
    monkeypatch.setattr(fc, "_audit", lambda *a: (_ for _ in ()).throw(OSError("no VAD")))
    data = damaged()
    data.media_duration_ms = 30000
    assert run(data, setup).coverage_issues == [
        {"start": 0, "end": 30, "reason": "final_audit_unavailable"}
    ]


def test_existing_unrelated_issues_survive_success(monkeypatch, setup):
    data = damaged()
    issue = {"start": 0, "end": 0.5, "reason": "foreign_language"}
    data.coverage_issues = [issue]
    monkeypatch.setattr(fc, "_decode", lambda *a: timeline())
    assert run(data, setup).coverage_issues == [issue]


def test_diagnostic_failure_does_not_discard_result(monkeypatch, setup):
    monkeypatch.setattr(fc, "_decode", lambda *a: timeline())
    monkeypatch.setattr(fc, "_save_evidence", lambda *a: (_ for _ in ()).throw(OSError("disk")))
    assert not run(damaged(), setup).coverage_issues


def test_recovery_that_leaves_hole_is_rejected(monkeypatch, setup):
    monkeypatch.setattr(fc, "_decode", lambda *a: damaged())
    data = damaged()
    before = fc._words(data)
    assert fc._words(run(data, setup)) == before
    assert data.coverage_issues


def test_defender_export_gap_fixture():
    # Observed Defender timings and TEN-VAD intervals, relative to 11:25.
    data = words([("bar", 10724, 12566), ("go", 21930, 22580)])
    gaps = _critical_aligned_speech_gaps(
        {"segments": [{"words": fc._words(data)}]},
        [(13078, 19606), (19862, 21654)],
        38,
    )
    assert len(gaps) == 1
    assert gaps[0].speech_seconds == pytest.approx(8.32)
    assert gaps[0].speech_ratio == pytest.approx(0.8885091841)


def test_confirmed_foreign_ranges_are_not_recovered(monkeypatch):
    from subforge.core.asr import whisperx_asr as wx
    from subforge.core.asr.chunk_merger import ChunkMerger

    first = words([("hello", 0, 350)])
    second = words([("world", 11000, 11350)])
    second.excluded_speech_ranges = [{"start": 0, "end": 10, "detected_language": "fr"}]
    data = ChunkMerger().merge_chunks([first, second], [0, 10000])
    assert data.excluded_speech_ranges[0]["start"] == 10
    assert data.excluded_speech_ranges[0]["end"] == 20
    gap = SimpleNamespace(start=10, end=20)
    monkeypatch.setattr(wx, "_critical_aligned_speech_gaps", lambda *a: [gap])
    monkeypatch.setattr(wx, "_detect_speech_in_mlx_gaps", lambda *a, **k: [])
    assert fc._audit(data, np.zeros(1)) == []


def test_final_guard_runs_after_all_timeline_mutations(monkeypatch):
    import importlib

    from subforge.core.entities import TranscribeConfig, TranscribeModelEnum

    tr = importlib.import_module("subforge.core.asr.transcribe")
    from subforge.core.asr import speech_vad

    data = damaged()
    monkeypatch.setattr(
        tr, "_create_asr_instance", lambda *a, **k: SimpleNamespace(run=lambda **k: data)
    )
    monkeypatch.setattr(speech_vad, "is_available", lambda: False)
    mutations = []
    for name in (
        "cap_abnormal_word_durations",
        "fix_boundary_overlaps",
        "filter_hallucinations",
        "deduplicate_alignment_echoes",
        "deduplicate_adjacent_text",
    ):
        monkeypatch.setattr(
            ASRData, name, lambda self, *a, _name=name, **k: mutations.append(_name)
        )

    def guard(result, *a, **k):
        assert mutations[-1] == "fix_boundary_overlaps"
        assert "cap_abnormal_word_durations" in mutations
        assert "deduplicate_adjacent_text" in mutations
        mutations.append("final_audit")
        return result

    monkeypatch.setattr(fc, "repair_final_coverage", guard)
    tr.transcribe(
        "missing.wav",
        TranscribeConfig(
            transcribe_model=TranscribeModelEnum.WHISPERX,
            transcribe_language="en",
            enable_audio_enhancement=False,
        ),
    )
    assert mutations[-1] == "final_audit"


def test_evidence_retention_is_bounded_and_contains_no_audio(monkeypatch, tmp_path):
    from subforge.core.utils import logger

    monkeypatch.setattr(logger, "get_active_log_file", lambda: tmp_path / "app.log")
    for i in range(23):
        fc._save_evidence([{"stage": "test", "index": i}])
    files = list((tmp_path / "speech-coverage").iterdir())
    assert len(files) == 20
    assert all(f.suffix == ".json" for f in files)


def test_unavailable_audit_message_does_not_claim_measured_gap():
    from subforge.core.asr.speech_gap_repair import coverage_issue_message

    message = coverage_issue_message([{"start": 0, "end": 0, "reason": "final_audit_unavailable"}])
    assert "unavailable" in message
    assert "00:00:00" not in message


def test_decode_caps_before_offset_and_removes_temporary_audio(tmp_path):
    from pydub import AudioSegment

    config = SimpleNamespace(cancel_event=None)
    context = SimpleNamespace(audio_segment=lambda: AudioSegment.silent(duration=5000))
    inputs = []

    def factory(path, config):
        from pathlib import Path

        assert Path(path).is_file()
        inputs.append(Path(path))
        return SimpleNamespace(run=lambda **k: words([("bar", 0, 2500), ("go", 2700, 2950)]))

    decoded = fc._decode(context, 1000, 4000, config, factory, lambda *a: None)
    assert decoded.segments[0].start_time == 1000
    assert decoded.segments[0].end_time == 1900
    assert decoded.segments[0].words[0].end_time == 1900
    assert not inputs[0].exists()


def test_recovery_cannot_put_words_back_into_confirmed_nonspeech(monkeypatch, setup):
    data = damaged()
    data.confirmed_non_speech_ranges = [{"start": 12, "end": 13}]
    before = fc._words(data)
    monkeypatch.setattr(fc, "_decode", lambda *a: timeline())
    assert fc._words(run(data, setup)) == before
    assert data.coverage_issues


@pytest.mark.parametrize(
    "start,end,speech",
    [
        (139246, 143002, [(139534, 141102), (141614, 142366)]),
        (715230, 720418, [(715774, 716654), (717182, 719806)]),
    ],
)
def test_vietnam_pauses_do_not_hide_missing_speech(monkeypatch, start, end, speech):
    from subforge.core.asr import whisperx_asr as wx

    data = words([("before", start - 300, start), ("after", end, end + 300)])
    monkeypatch.setattr(wx, "_detect_speech_in_mlx_gaps", lambda *a, **k: speech)
    gaps = fc._audit(data, np.zeros((end // 1000 + 2) * 16000))
    assert len(gaps) == 1
    assert gaps[0].start == pytest.approx(start / 1000)
    assert gaps[0].end == pytest.approx(end / 1000)


def test_stretched_word_is_not_proof_of_transcription(monkeypatch):
    from subforge.core.asr import whisperx_asr as wx

    data = words([("do.", 834200, 834574), ("In", 834738, 838190), ("practice", 838626, 839330)])
    before = copy.deepcopy(data)
    monkeypatch.setattr(wx, "_detect_speech_in_mlx_gaps", lambda *a, **k: [(835800, 838100)])
    gaps = fc._audit(data, np.zeros(841 * 16000))
    assert any(g.start < 836 and g.end > 838 for g in gaps)
    assert fc._words(data) == fc._words(before)


@pytest.mark.parametrize("token", ["1,541-kilometre-long", "275/35 R19", "ありがとうございます"])
def test_shadow_audit_does_not_truncate_compounds_or_nonlatin(monkeypatch, token):
    from subforge.core.asr import whisperx_asr as wx

    data = words([("before", 0, 500), (token, 600, 4500), ("after", 4500, 4800)])
    monkeypatch.setattr(wx, "_detect_speech_in_mlx_gaps", lambda *a, **k: [(1500, 4400)])
    assert fc._audit(data, np.zeros(5 * 16000)) == []


def test_short_music_transient_is_not_a_recovery_candidate(monkeypatch):
    from subforge.core.asr import whisperx_asr as wx

    data = words([("before", 0, 500), ("after", 4500, 4800)])
    monkeypatch.setattr(wx, "_detect_speech_in_mlx_gaps", lambda *a, **k: [(2000, 2638)])
    assert fc._audit(data, np.zeros(5 * 16000)) == []


def test_relaxed_candidates_respect_confirmed_nonspeech(monkeypatch):
    from subforge.core.asr import whisperx_asr as wx

    data = words([("before", 0, 500), ("after", 4500, 4800)])
    data.confirmed_non_speech_ranges = [{"start": 1, "end": 4.5}]
    monkeypatch.setattr(wx, "_detect_speech_in_mlx_gaps", lambda *a, **k: [(1000, 4400)])
    assert fc._audit(data, np.zeros(5 * 16000)) == []


def test_common_anchors_allow_context_edge_variation(monkeypatch, setup):
    data = damaged()
    calls = 0

    def decode(*args):
        nonlocal calls
        calls += 1
        candidate = timeline()
        if calls == 2:
            candidate.segments[0].text = "different"
            candidate.segments[-1].text = "ending"
        return candidate

    monkeypatch.setattr(fc, "_decode", decode)
    result = run(data, setup)
    assert not result.coverage_issues
    assert fc._words(result) == fc._words(timeline())


@pytest.mark.parametrize("number,expected", [("1,541km", True), ("1,451km", False)])
def test_numeric_unit_evidence_preserves_value(number, expected):
    first = [
        {"word": "1541", "start": 137.85, "end": 139.25},
        {"word": "kilometer", "start": 140.12, "end": 140.64},
        {"word": "long", "start": 140.74, "end": 141.04},
        {"word": "rail", "start": 141.8, "end": 142.08},
    ]
    second = [
        {"word": number, "start": 137.857, "end": 139.257},
        {"word": "long", "start": 140.743, "end": 141.044},
        {"word": "rail", "start": 141.805, "end": 142.086},
    ]
    assert fc._corroborates_repair(first, second) is expected


def test_repair_consensus_does_not_ignore_pronoun_or_negation():
    first = [{"word": "It's", "start": 1, "end": 1.2}]
    second = [{"word": "That's", "start": 1, "end": 1.2}]
    assert not fc._corroborates_repair(first, second)
    assert not fc._corroborates_repair(first, [*first, {"word": "not", "start": 1.2, "end": 1.4}])


@pytest.mark.parametrize("changed", ["15.41km", "-1541km", "1,54km"])
def test_numeric_consensus_keeps_decimal_sign_and_grouping(changed):
    first = [{"word": "1541km", "start": 1, "end": 2}]
    second = [{"word": changed, "start": 1, "end": 2}]
    assert not fc._corroborates_repair(first, second)


@pytest.mark.parametrize(
    "amount,expected", [("$80,000-ish", True), ("$80,000", False), ("$18,000-ish", False)]
)
def test_money_consensus_preserves_scale_and_approximation(amount, expected):
    first = [
        {"word": "80-ish", "start": 15.01, "end": 15.49},
        {"word": "thousand", "start": 15.53, "end": 15.87},
        {"word": "dollar", "start": 15.89, "end": 16.13},
        {"word": "vehicle", "start": 16.17, "end": 16.55},
    ]
    second = [
        {"word": amount, "start": 15.01, "end": 16.13},
        {"word": "vehicle", "start": 16.17, "end": 16.55},
    ]
    assert fc._corroborates_repair(first, second) is expected


def test_third_context_decode_can_confirm_recovery(monkeypatch, setup):
    data = damaged()
    first = timeline()
    first.segments[25].text = "wrong"
    decodes = iter([first, timeline(), timeline()])
    calls = []

    def decode(*args):
        calls.append((args[1], args[2]))
        return next(decodes)

    monkeypatch.setattr(fc, "_decode", decode)
    result = run(data, setup)
    assert len(set(calls)) == 3
    assert not result.coverage_issues
    assert fc._words(result) == fc._words(timeline())
    assert "retry_window" in setup[2][0]


def test_third_decode_respects_total_audio_budget(monkeypatch, setup):
    data = damaged()
    calls = 0

    def decode(*args):
        nonlocal calls
        calls += 1
        candidate = timeline()
        candidate.segments[25].text = f"different{calls}"
        return candidate

    monkeypatch.setattr(fc, "_decode", decode)
    # Two windows cost 61.7s; a third would exceed this budget.
    monkeypatch.setattr(fc, "MAX_DECODE_SECONDS", 62)
    before = fc._words(data)
    assert fc._words(run(data, setup)) == before
    assert calls == 2
    assert data.coverage_issues


@pytest.mark.parametrize("speech", [[(2175198, 2176606)], [(2175006, 2176702)]])
def test_lexus_two_second_sentence_is_a_final_recovery_candidate(monkeypatch, speech):
    from subforge.core.asr import whisperx_asr as wx

    # Original GX export gap: 36:14.686-36:16.780; strict TEN-VAD voiced 1.408s.
    data = words([("I", 2173595, 2174686), ("I", 2176780, 2177260)])
    monkeypatch.setattr(wx, "_detect_speech_in_mlx_gaps", lambda *a, **k: speech)
    gaps = fc._audit(data, np.zeros(2178 * 16000))
    assert len(gaps) == 1
    assert gaps[0].start == pytest.approx(2174.686)
    assert gaps[0].end == pytest.approx(2176.780)


@pytest.mark.parametrize("change", [None, "negation", "unstable_anchor"])
def test_lexus_consensus_is_localized_without_ignoring_word_disagreements(monkeypatch, change):
    from subforge.core.asr import whisperx_asr as wx

    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures/asr/lexus_gx_short_gaps.json").read_text()
    )
    offset = 1178

    def load(key):
        return words([
            (w["word"], round((w["start"] - offset) * 1000), round((w["end"] - offset) * 1000))
            for w in fixture[key]
        ])

    data, first, second = [load(k) for k in ("before", "decoded", "confirmation")]
    if change == "negation":
        next(s for s in second.segments if s.text == "perk").text = "not"
    elif change == "unstable_anchor":
        # The nearby right anchor must agree acoustically in both decodes.
        next(s for s in second.segments if s.text == "remember").end_time += 250
    before = copy.deepcopy(data)
    decodes = iter([first, second, words([("unrelated", 1000, 1200)])])
    monkeypatch.setattr(fc, "_decode", lambda *a: next(decodes))
    speech = [(12942, 15806), (16366, 16798)]
    monkeypatch.setattr(
        wx, "_detect_speech_in_mlx_gaps",
        lambda samples, sr, ranges, **k: [
            (a, b) for a, b in speech
            if any(lo * 1000 <= a and b <= hi * 1000 for lo, hi in ranges)
        ],
    )
    records = []
    monkeypatch.setattr(fc, "_save_evidence", records.extend)
    context = SimpleNamespace(audio_path="original.wav", samples=lambda: np.zeros(30 * 16000))
    config = SimpleNamespace(cancel_event=None, whisperx_model="local-test")
    result = fc.repair_final_coverage(data, "original.wav", config, lambda *a: None, None, context)
    if change:
        assert fc._words(result) == fc._words(before)
        assert result.coverage_issues
    else:
        assert not result.coverage_issues
        assert records[0]["localized_consensus"]
        assert "think the biggest perk" in " ".join(s.text for s in result.segments)
        # The following sentence, including its source timing, stays intact.
        assert [s.__dict__ for s in result.segments if s.start_time >= 17446] == [
            s.__dict__ for s in before.segments if s.start_time >= 17446
        ]


def test_lexus_money_gap_requires_two_decodes_preserving_ish(monkeypatch):
    from subforge.core.asr import whisperx_asr as wx

    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures/asr/lexus_gx_money_gap.json").read_text()
    )

    def load(key):
        return words([
            (w["word"], round(w["start"] * 1000), round(w["end"] * 1000))
            for w in fixture[key]
        ])

    original = load("before")
    first, second, third = [load(k) for k in ("decoded", "confirmation", "retry")]
    assert fc._consensus_proposal(original, first, second, 14.686, 16.78) is None
    decodes = iter([first, second, third])
    monkeypatch.setattr(fc, "_decode", lambda *a: next(decodes))
    speech = [(15198, 16606)]
    monkeypatch.setattr(
        wx, "_detect_speech_in_mlx_gaps",
        lambda samples, sr, ranges, **k: [
            (a, b) for a, b in speech
            if any(lo * 1000 <= a and b <= hi * 1000 for lo, hi in ranges)
        ],
    )
    records = []
    monkeypatch.setattr(fc, "_save_evidence", records.extend)
    context = SimpleNamespace(audio_path="original.wav", samples=lambda: np.zeros(30 * 16000))
    config = SimpleNamespace(cancel_event=None, whisperx_model="local-test")
    repaired = fc.repair_final_coverage(
        original, "original.wav", config, lambda *a: None, None, context
    )
    assert not repaired.coverage_issues
    assert records[0]["status"] == "recovered"
    assert "retry" in records[0]
    assert "80-ish thousand dollar vehicle" in " ".join(s.text for s in repaired.segments)
