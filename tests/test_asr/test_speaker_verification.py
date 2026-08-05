import sys
from types import ModuleType

import numpy as np
import pytest

import subforge.core.asr.speaker_verification as verification
from subforge.core.asr.asr_data import ASRData, ASRDataSeg


def _speaker_data() -> ASRData:
    return ASRData(
        [
            ASRDataSeg("host reference", 0, 2_500, speaker_id="Speaker 1"),
            ASRDataSeg("guest reference", 3_000, 5_500, speaker_id="Speaker 2"),
            ASRDataSeg("uncertain", 6_000, 6_500, speaker_id="Speaker 1"),
        ]
    )


def _read_audio(start_ms: int, _end_ms: int) -> np.ndarray:
    if start_ms < 3_000:
        return np.array([1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 1.0], dtype=np.float32)


def _identity_embedding(samples: np.ndarray) -> np.ndarray:
    return samples / np.linalg.norm(samples)


def test_acoustic_verifier_accepts_only_supported_label_proposal(monkeypatch):
    data = _speaker_data()
    monkeypatch.setattr(
        verification,
        "_proposed_labels",
        lambda _data: ["Speaker 1", "Speaker 2", "Speaker 2"],
    )

    stats = verification.verify_speaker_assignment_proposals(
        data,
        read_audio=_read_audio,
        embedding=_identity_embedding,
    )

    assert [segment.speaker_id for segment in data.segments] == [
        "Speaker 1",
        "Speaker 2",
        "Speaker 2",
    ]
    assert stats.proposals == 1
    assert stats.accepted == 1


def test_acoustic_verifier_rejects_proposal_without_margin(monkeypatch):
    data = _speaker_data()
    monkeypatch.setattr(
        verification,
        "_proposed_labels",
        lambda _data: ["Speaker 1", "Speaker 2", "Speaker 2"],
    )

    def current_speaker_audio(start_ms: int, _end_ms: int) -> np.ndarray:
        if start_ms < 3_000 or start_ms >= 6_000:
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    stats = verification.verify_speaker_assignment_proposals(
        data,
        read_audio=current_speaker_audio,
        embedding=_identity_embedding,
    )

    assert data.segments[-1].speaker_id == "Speaker 1"
    assert stats.accepted == 0


def test_acoustic_verifier_requires_confirmation_model_consensus(monkeypatch):
    data = _speaker_data()
    monkeypatch.setattr(
        verification,
        "_proposed_labels",
        lambda _data: ["Speaker 1", "Speaker 2", "Speaker 2"],
    )

    def distinguish_intervals(start_ms: int, _end_ms: int) -> np.ndarray:
        if start_ms < 3_000:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if start_ms < 6_000:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def primary_embedding(samples: np.ndarray) -> np.ndarray:
        return np.array([samples[0], samples[1] + samples[2]], dtype=np.float32)

    def disagreeing_embedding(samples: np.ndarray) -> np.ndarray:
        return np.array([samples[0] + samples[2], samples[1]], dtype=np.float32)

    stats = verification.verify_speaker_assignment_proposals(
        data,
        read_audio=distinguish_intervals,
        embedding=primary_embedding,
        confirmation_embedding=disagreeing_embedding,
    )

    assert data.segments[-1].speaker_id == "Speaker 1"
    assert stats.accepted == 0
    assert stats.skipped_consensus == 1


def test_acoustic_verifier_accepts_when_both_models_agree(monkeypatch):
    data = _speaker_data()
    monkeypatch.setattr(
        verification,
        "_proposed_labels",
        lambda _data: ["Speaker 1", "Speaker 2", "Speaker 2"],
    )

    stats = verification.verify_speaker_assignment_proposals(
        data,
        read_audio=_read_audio,
        embedding=_identity_embedding,
        confirmation_embedding=_identity_embedding,
    )

    assert data.segments[-1].speaker_id == "Speaker 2"
    assert stats.accepted == 1
    assert stats.skipped_consensus == 0


def test_acoustic_verifier_rescues_near_threshold_when_confirmation_is_strong(monkeypatch):
    data = _speaker_data()
    monkeypatch.setattr(
        verification,
        "_proposed_labels",
        lambda _data: ["Speaker 1", "Speaker 2", "Speaker 2"],
    )

    def distinguish_intervals(start_ms: int, _end_ms: int) -> np.ndarray:
        if start_ms < 3_000:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if start_ms < 6_000:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def near_threshold(samples: np.ndarray) -> np.ndarray:
        if samples[2] > 0:
            vector = np.array([0.66, 0.75], dtype=np.float32)
            return vector / np.linalg.norm(vector)
        return samples[:2]

    def strong_confirmation(samples: np.ndarray) -> np.ndarray:
        return np.array([samples[0], samples[1] + samples[2]], dtype=np.float32)

    stats = verification.verify_speaker_assignment_proposals(
        data,
        read_audio=distinguish_intervals,
        embedding=near_threshold,
        confirmation_embedding=strong_confirmation,
    )

    assert data.segments[-1].speaker_id == "Speaker 2"
    assert stats.accepted == 1


def test_acoustic_verifier_never_changes_overlap_region(monkeypatch):
    data = _speaker_data()
    monkeypatch.setattr(
        verification,
        "_proposed_labels",
        lambda _data: ["Speaker 1", "Speaker 2", "Speaker 2"],
    )

    stats = verification.verify_speaker_assignment_proposals(
        data,
        read_audio=_read_audio,
        embedding=_identity_embedding,
        overlap_regions=[(5_900, 6_100)],
    )

    assert data.segments[-1].speaker_id == "Speaker 1"
    assert stats.accepted == 0
    assert stats.skipped_overlap == 1


def test_proposal_generation_restores_labels_when_smoothing_fails(monkeypatch):
    data = _speaker_data()

    def fail_after_mutation(asr_data, **_kwargs):
        asr_data.segments[0].speaker_id = "wrong"
        raise RuntimeError("failed")

    import subforge.core.asr.speaker_diarization as diarization

    monkeypatch.setattr(diarization, "smooth_speaker_assignments", fail_after_mutation)

    with pytest.raises(RuntimeError, match="failed"):
        verification._proposed_labels(data)

    assert [segment.speaker_id for segment in data.segments] == [
        "Speaker 1",
        "Speaker 2",
        "Speaker 1",
    ]


def test_pipeline_verifier_reads_only_needed_audio_and_applies_gate(tmp_path, monkeypatch):
    import soundfile as sf

    data = _speaker_data()
    samples = np.zeros(7 * 16_000, dtype=np.float32)
    samples[: 2_500 * 16] = 0.1
    samples[3_000 * 16 : 5_500 * 16] = 0.9
    samples[6_000 * 16 : 6_500 * 16] = 0.9
    audio_path = tmp_path / "speakers.wav"
    sf.write(audio_path, samples, 16_000)
    monkeypatch.setattr(
        verification,
        "_proposed_labels",
        lambda _data: ["Speaker 1", "Speaker 2", "Speaker 2"],
    )

    class FakeEmbedding:
        sample_rate = 16_000
        min_num_samples = 1

        def to(self, _device):
            return self

        def __call__(self, batch):
            mean = float(batch.mean())
            return [np.array([1.0, 0.0]) if mean < 0.5 else np.array([0.0, 1.0])]

    pipeline = type("Pipeline", (), {"_embedding": FakeEmbedding()})()
    fake_torch_module = ModuleType("torch")
    fake_torch_module.device = lambda name: name
    fake_torch_module.from_numpy = lambda samples: samples
    monkeypatch.setitem(sys.modules, "torch", fake_torch_module)
    stats = verification.verify_speakers_with_pipeline(
        data,
        str(audio_path),
        pipeline=pipeline,
        device="cpu",
    )

    assert data.segments[-1].speaker_id == "Speaker 2"
    assert stats.accepted == 1
