from pathlib import Path
from types import SimpleNamespace

import subforge.core.asr.alignment_models as alignment_models


def test_alignment_model_rejects_dataless_cloud_placeholder(monkeypatch, tmp_path: Path):
    spec = alignment_models.alignment_model_for_language("en")
    assert spec is not None
    model_path = alignment_models.alignment_model_path(spec, tmp_path)
    model_path.write_bytes(b"weights")
    original_stat = Path.stat

    def _stat(path: Path, *args, **kwargs):
        metadata = original_stat(path, *args, **kwargs)
        if path == model_path:
            return SimpleNamespace(
                st_size=metadata.st_size,
                st_flags=alignment_models._DARWIN_DATALESS_FLAG,
                st_mode=metadata.st_mode,
            )
        return metadata

    monkeypatch.setattr(Path, "stat", _stat)

    assert not alignment_models.is_alignment_model_ready(spec, tmp_path)
