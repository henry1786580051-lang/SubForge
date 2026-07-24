import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import subforge.core.asr.speaker_diarization as diarization_module
from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.asr.speaker_diarization import (
    SpeakerTurn,
    _deserialize_cached_turns,
    _select_diarization_device,
    assign_speakers,
    is_diarization_model_dir,
    require_local_diarization_model,
    resolve_diarization_model,
    smooth_speaker_assignments,
)


class _MpsBackend:
    @staticmethod
    def is_available() -> bool:
        return True


def _write_community_model(model_dir: Path) -> None:
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("pipeline: {}" * 20, encoding="utf-8")
    for relative in (
        "segmentation/pytorch_model.bin",
        "embedding/pytorch_model.bin",
    ):
        path = model_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * (1024 * 1024))
    for relative in ("plda/plda.npz", "plda/xvec_transform.npz"):
        path = model_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 1024)


def test_select_diarization_device_prefers_mps_on_apple_silicon(monkeypatch):
    fake_torch = SimpleNamespace(backends=SimpleNamespace(mps=_MpsBackend()))
    monkeypatch.setattr(diarization_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(diarization_module.platform, "machine", lambda: "arm64")
    monkeypatch.delenv("SUBFORGE_DIARIZATION_DEVICE", raising=False)

    assert _select_diarization_device(fake_torch) == "mps"


def test_select_diarization_device_honors_cpu_override(monkeypatch):
    fake_torch = SimpleNamespace(backends=SimpleNamespace(mps=_MpsBackend()))
    monkeypatch.setenv("SUBFORGE_DIARIZATION_DEVICE", "cpu")

    assert _select_diarization_device(fake_torch) == "cpu"


def test_diarize_audio_reloads_pipeline_on_mps_failure(tmp_path, monkeypatch):
    model_dir = tmp_path / "pyannote-speaker-diarization-community-1"
    _write_community_model(model_dir)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    devices: list[str] = []
    load_count = 0

    class FakePipeline:
        def __init__(self, fail: bool):
            self.fail = fail

        def to(self, device):
            devices.append(str(device))

        def __call__(self, _audio, **_kwargs):
            if self.fail:
                raise RuntimeError("unsupported MPS operator")
            segment = SimpleNamespace(start=0.0, end=1.0)
            return [(segment, "raw-speaker")]

    class PipelineFactory:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            nonlocal load_count
            load_count += 1
            return FakePipeline(fail=load_count == 1)

    fake_audio_module = ModuleType("pyannote.audio")
    fake_audio_module.Pipeline = PipelineFactory
    fake_torch_module = ModuleType("torch")
    fake_torch_module.device = lambda name: name
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio_module)
    monkeypatch.setitem(sys.modules, "torch", fake_torch_module)
    monkeypatch.setattr(diarization_module, "_load_waveform", lambda _path: object())
    monkeypatch.setattr(
        diarization_module,
        "_select_diarization_device",
        lambda _torch: "mps",
    )
    import subforge.core.utils.cache as cache_module

    monkeypatch.setattr(cache_module, "is_cache_enabled", lambda: False)

    turns = diarization_module.diarize_audio(
        str(audio_path),
        model_dir=tmp_path,
        num_speakers=2,
    )

    assert load_count == 2
    assert devices == ["mps", "cpu"]
    assert turns == [SpeakerTurn(0, 1_000, "Speaker 1")]


def test_assign_speakers_preserves_text_and_timestamps():
    data = ASRData(
        [
            ASRDataSeg("Hello", 100, 400),
            ASRDataSeg("there", 420, 700),
            ASRDataSeg("Hi", 900, 1_100),
        ]
    )
    original = [(segment.text, segment.start_time, segment.end_time) for segment in data]

    assign_speakers(
        data,
        [
            SpeakerTurn(0, 800, "Speaker 1"),
            SpeakerTurn(850, 1_300, "Speaker 2"),
        ],
    )

    assert [(segment.text, segment.start_time, segment.end_time) for segment in data] == original
    assert [segment.speaker_id for segment in data] == [
        "Speaker 1",
        "Speaker 1",
        "Speaker 2",
    ]


def test_assign_speakers_uses_largest_overlap_at_change_point():
    data = ASRData([ASRDataSeg("handoff", 900, 1_300)])
    turns = [
        SpeakerTurn(0, 1_000, "Speaker 1"),
        SpeakerTurn(1_000, 2_000, "Speaker 2"),
    ]

    assign_speakers(data, turns)

    assert data.segments[0].speaker_id == "Speaker 2"


def test_assign_speakers_does_not_fill_distant_silence():
    data = ASRData([ASRDataSeg("uncertain", 2_000, 2_200)])

    assign_speakers(data, [SpeakerTurn(0, 1_000, "Speaker 1")])

    assert data.segments[0].speaker_id == ""


def test_assign_speakers_suppresses_only_short_isolated_flip():
    data = ASRData(
        [
            ASRDataSeg("one", 0, 200),
            ASRDataSeg("brief", 210, 410),
            ASRDataSeg("again", 420, 650),
        ]
    )
    turns = [
        SpeakerTurn(0, 205, "Speaker 1"),
        SpeakerTurn(205, 415, "Speaker 2"),
        SpeakerTurn(415, 700, "Speaker 1"),
    ]

    assign_speakers(data, turns)

    assert [segment.speaker_id for segment in data] == ["Speaker 1"] * 3


def test_smooth_speaker_assignments_repairs_island_and_boundary_prefix():
    data = ASRData(
        [
            ASRDataSeg("It", 0, 100, speaker_id="Speaker 1"),
            ASRDataSeg("is", 120, 220, speaker_id="Speaker 2"),
            ASRDataSeg("great.", 240, 500, speaker_id="Speaker 1"),
            ASRDataSeg("It's", 700, 900, speaker_id="Speaker 1"),
            ASRDataSeg("fast.", 920, 1_200, speaker_id="Speaker 2"),
        ]
    )

    smooth_speaker_assignments(data)

    assert [segment.speaker_id for segment in data] == [
        "Speaker 1",
        "Speaker 1",
        "Speaker 1",
        "Speaker 2",
        "Speaker 2",
    ]


def test_smooth_speaker_assignments_fills_short_unlabeled_edge():
    data = ASRData(
        [
            ASRDataSeg("Oh", 0, 100),
            ASRDataSeg("yes", 120, 300, speaker_id="Speaker 2"),
        ]
    )

    smooth_speaker_assignments(data)

    assert [segment.speaker_id for segment in data] == ["Speaker 2", "Speaker 2"]


def test_smooth_speaker_assignments_repairs_longer_incomplete_island():
    data = ASRData(
        [
            ASRDataSeg("that is helping us to", 0, 900, speaker_id="Speaker 1"),
            ASRDataSeg("adapt and to", 940, 1_883, speaker_id="Speaker 2"),
            ASRDataSeg("engage better tomorrow.", 2_204, 3_100, speaker_id="Speaker 1"),
        ]
    )

    smooth_speaker_assignments(data)

    assert [segment.speaker_id for segment in data] == ["Speaker 1"] * 3


def test_smooth_speaker_assignments_preserves_complete_short_interjection():
    data = ASRData(
        [
            ASRDataSeg("I kept explaining", 0, 900, speaker_id="Speaker 1"),
            ASRDataSeg("Yeah.", 940, 1_500, speaker_id="Speaker 2"),
            ASRDataSeg("until the end.", 1_540, 2_400, speaker_id="Speaker 1"),
        ]
    )

    smooth_speaker_assignments(data)

    assert [segment.speaker_id for segment in data] == [
        "Speaker 1",
        "Speaker 2",
        "Speaker 1",
    ]


def test_smooth_speaker_assignments_repairs_near_boundary_question_island():
    data = ASRData(
        [
            ASRDataSeg("Is he going to succeed?", 0, 599, speaker_id="Speaker 2"),
            ASRDataSeg("Do you think?", 821, 1_197, speaker_id="Speaker 1"),
            ASRDataSeg("I mean, can he?", 1_459, 2_000, speaker_id="Speaker 2"),
        ]
    )

    smooth_speaker_assignments(data)

    assert [segment.speaker_id for segment in data] == ["Speaker 2"] * 3


def test_smooth_speaker_assignments_repairs_short_subject_phrase_at_boundary():
    data = ASRData(
        [
            ASRDataSeg("blood and soil—", 0, 480, speaker_id="Speaker 1"),
            ASRDataSeg("He", 500, 560, speaker_id="Speaker 1"),
            ASRDataSeg("certainly", 580, 841, speaker_id="Speaker 1"),
            ASRDataSeg("delivered", 881, 1_200, speaker_id="Speaker 2"),
            ASRDataSeg("a strong message.", 1_220, 1_800, speaker_id="Speaker 2"),
        ]
    )

    smooth_speaker_assignments(data)

    assert [segment.speaker_id for segment in data] == [
        "Speaker 1",
        "Speaker 2",
        "Speaker 2",
        "Speaker 2",
        "Speaker 2",
    ]


def test_smooth_speaker_assignments_preserves_demonstrative_turn_ending():
    data = ASRData(
        [
            ASRDataSeg("in a whole weird", 0, 800, speaker_id="Speaker 2"),
            ASRDataSeg("that", 820, 900, speaker_id="Speaker 2"),
            ASRDataSeg("way,", 920, 1_020, speaker_id="Speaker 2"),
            ASRDataSeg("yeah", 1_040, 1_220, speaker_id="Speaker 2"),
            ASRDataSeg("yeah,", 1_420, 1_600, speaker_id="Speaker 1"),
            ASRDataSeg("I mean", 1_620, 1_900, speaker_id="Speaker 1"),
        ]
    )

    smooth_speaker_assignments(data)

    assert [segment.speaker_id for segment in data] == [
        "Speaker 2",
        "Speaker 2",
        "Speaker 2",
        "Speaker 2",
        "Speaker 1",
        "Speaker 1",
    ]


def test_resolve_diarization_model_prefers_managed_snapshot(tmp_path: Path):
    local_model = tmp_path / "pyannote-speaker-diarization-community-1"
    _write_community_model(local_model)

    assert is_diarization_model_dir(local_model)
    assert resolve_diarization_model("pyannote/speaker-diarization-community-1", tmp_path) == str(
        local_model
    )


def test_require_local_diarization_model_fails_before_asr(tmp_path: Path):
    try:
        require_local_diarization_model("pyannote/speaker-diarization-community-1", tmp_path)
    except RuntimeError as exc:
        assert "not downloaded" in str(exc)
    else:
        raise AssertionError("Expected a missing-model error")


def test_cached_diarization_turns_are_strictly_validated():
    cached = [
        {"start_ms": 0, "end_ms": 500, "speaker_id": "Speaker 1"},
        {"start_ms": 510, "end_ms": 900, "speaker_id": "Speaker 2"},
    ]

    assert _deserialize_cached_turns(cached) == [
        SpeakerTurn(0, 500, "Speaker 1"),
        SpeakerTurn(510, 900, "Speaker 2"),
    ]
    assert _deserialize_cached_turns([{"start_ms": 500, "end_ms": 100}]) == []
