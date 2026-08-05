"""Managed independent speaker-embedding models for diarization verification."""

from __future__ import annotations

import os
import platform
import stat
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

DEFAULT_SPEAKER_VERIFICATION_MODEL = "Wespeaker/wespeaker-ecapa-tdnn512-LM"
SPEAKER_VERIFICATION_MODEL_REVISION = "a2f3dcb1c8702caccc7a55ceb57f5e8d1842112b"
LOCAL_SPEAKER_VERIFICATION_DIR = "wespeaker-voxceleb-ecapa-tdnn512-LM"
SPEAKER_VERIFICATION_MODEL_FILE = "voxceleb_ECAPA512_LM.onnx"
_MINIMUM_MODEL_SIZE = 20 * 1024 * 1024
_DARWIN_DATALESS_FLAG = 0x40000000


def speaker_verification_model_path(models_dir: str | Path) -> Path:
    """Return the managed ECAPA512-LM model file path."""
    return (
        Path(models_dir).expanduser()
        / LOCAL_SPEAKER_VERIFICATION_DIR
        / SPEAKER_VERIFICATION_MODEL_FILE
    )


def is_speaker_verification_model_ready(models_dir: str | Path) -> bool:
    """Return whether the managed ONNX weight is fully available offline."""
    path = speaker_verification_model_path(models_dir)
    try:
        metadata = path.stat()
    except OSError:
        return False
    flags = int(getattr(metadata, "st_flags", 0) or 0)
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_size >= _MINIMUM_MODEL_SIZE
        and not (platform.system() == "Darwin" and flags & _DARWIN_DATALESS_FLAG)
    )


class ECAPASpeakerEmbedding:
    """Extract normalized WeSpeaker ECAPA embeddings with ONNX Runtime."""

    sample_rate = 16_000
    minimum_samples = 8_000

    def __init__(self, model_path: str | Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        self._session = ort.InferenceSession(
            str(Path(model_path).expanduser()),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def __call__(self, samples: Any):
        import numpy as np
        import torch
        import torchaudio.compliance.kaldi as kaldi

        waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
        if waveform.size < self.minimum_samples:
            missing = self.minimum_samples - waveform.size
            left = missing // 2
            waveform = np.pad(waveform, (left, missing - left))
        tensor = torch.from_numpy(waveform).reshape(1, -1) * (1 << 15)
        features = kaldi.fbank(
            tensor,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            sample_frequency=self.sample_rate,
            window_type="hamming",
            use_energy=False,
        )
        features = features - torch.mean(features, dim=0)
        batch = features.unsqueeze(0).numpy().astype(np.float32, copy=False)
        outputs = cast(
            list[Any],
            self._session.run(output_names=["embs"], input_feed={"feats": batch}),
        )
        vector = np.asarray(outputs[0][0], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise RuntimeError("ECAPA speaker embedding is invalid")
        return vector / norm


@lru_cache(maxsize=1)
def load_speaker_verification_embedding(models_dir: str | Path) -> ECAPASpeakerEmbedding:
    """Load and cache the managed independent embedding model."""
    if not is_speaker_verification_model_ready(models_dir):
        raise FileNotFoundError(
            "WeSpeaker ECAPA-TDNN512-LM is not downloaded; independent speaker verification is unavailable"
        )
    return ECAPASpeakerEmbedding(speaker_verification_model_path(models_dir))
