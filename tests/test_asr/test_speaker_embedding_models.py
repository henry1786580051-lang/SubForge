from pathlib import Path

import subforge.core.asr.speaker_embedding_models as models


def test_managed_speaker_verification_model_requires_complete_weight(tmp_path: Path):
    model_path = models.speaker_verification_model_path(tmp_path)
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"incomplete")

    assert not models.is_speaker_verification_model_ready(tmp_path)


def test_managed_speaker_verification_model_accepts_full_weight(tmp_path: Path, monkeypatch):
    model_path = models.speaker_verification_model_path(tmp_path)
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"x")
    monkeypatch.setattr(models, "_MINIMUM_MODEL_SIZE", 1)

    assert models.is_speaker_verification_model_ready(tmp_path)
