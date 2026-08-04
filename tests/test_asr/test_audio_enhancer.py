import importlib
import sys
import types

import pytest

audio_enhancer = importlib.import_module("subforge.core.asr.audio_enhancer")


class FakeTorch:
    def __init__(self, threads: int = 5):
        self.threads = threads

    def get_num_threads(self) -> int:
        return self.threads

    def set_num_threads(self, threads: int) -> None:
        self.threads = threads


def test_apple_silicon_cpu_uses_benchmarked_thread_count(monkeypatch):
    torch = FakeTorch()
    monkeypatch.setattr(audio_enhancer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(audio_enhancer.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(audio_enhancer.os, "cpu_count", lambda: 14)
    monkeypatch.delenv("SUBFORGE_DENOISE_THREADS", raising=False)

    audio_enhancer._configure_apple_silicon_cpu(torch)

    assert torch.threads == 8


def test_apple_silicon_cpu_allows_thread_override(monkeypatch):
    torch = FakeTorch()
    monkeypatch.setattr(audio_enhancer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(audio_enhancer.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(audio_enhancer.os, "cpu_count", lambda: 14)
    monkeypatch.setenv("SUBFORGE_DENOISE_THREADS", "6")

    audio_enhancer._configure_apple_silicon_cpu(torch)

    assert torch.threads == 6


def test_apple_silicon_cpu_clamps_excessive_thread_override(monkeypatch):
    torch = FakeTorch()
    monkeypatch.setattr(audio_enhancer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(audio_enhancer.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(audio_enhancer.os, "cpu_count", lambda: 10)
    monkeypatch.setenv("SUBFORGE_DENOISE_THREADS", "999")

    audio_enhancer._configure_apple_silicon_cpu(torch)

    assert torch.threads == 10


def test_other_platforms_keep_torch_thread_configuration(monkeypatch):
    torch = FakeTorch()
    monkeypatch.setattr(audio_enhancer.platform, "system", lambda: "Windows")
    monkeypatch.setattr(audio_enhancer.platform, "machine", lambda: "AMD64")

    audio_enhancer._configure_apple_silicon_cpu(torch)

    assert torch.threads == 5


def test_deepfilter_system_exit_becomes_recoverable_error(monkeypatch):
    def aborting_init_df(**_kwargs):
        raise SystemExit(1)

    monkeypatch.setattr(audio_enhancer, "_df_model", None)
    monkeypatch.setattr(audio_enhancer, "_df_state", None)
    monkeypatch.setattr(audio_enhancer, "_df_device", None)
    monkeypatch.setitem(
        sys.modules,
        "df.enhance",
        types.SimpleNamespace(init_df=aborting_init_df),
    )

    with pytest.raises(RuntimeError, match="checkpoint could not be loaded"):
        audio_enhancer._load_model(device="cpu")

    assert audio_enhancer._df_model is None
    assert audio_enhancer._df_state is None


def test_incomplete_default_model_cache_is_removed(tmp_path):
    model_dir = tmp_path / "DeepFilterNet3"
    model_dir.mkdir()
    (model_dir / "config.ini").write_text("partial", encoding="utf-8")
    module = types.SimpleNamespace(
        DEFAULT_MODEL="DeepFilterNet3",
        get_model_basedir=lambda _model: model_dir,
    )

    removed = audio_enhancer._repair_default_model_cache(module)

    assert removed is True
    assert not model_dir.exists()


def test_complete_default_model_cache_is_preserved(tmp_path):
    model_dir = tmp_path / "DeepFilterNet3"
    checkpoints = model_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    (model_dir / "config.ini").write_text("complete", encoding="utf-8")
    (checkpoints / "model.ckpt.best").write_bytes(b"model")
    module = types.SimpleNamespace(
        DEFAULT_MODEL="DeepFilterNet3",
        get_model_basedir=lambda _model: model_dir,
    )

    removed = audio_enhancer._repair_default_model_cache(module)

    assert removed is False
    assert model_dir.exists()


def test_corrupt_default_model_cache_is_refreshed_after_load_abort(
    tmp_path,
    monkeypatch,
):
    model_dir = tmp_path / "DeepFilterNet3"
    checkpoints = model_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    (model_dir / "config.ini").write_text("corrupt", encoding="utf-8")
    (checkpoints / "model.ckpt.best").write_bytes(b"corrupt")
    calls = 0

    def init_df(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SystemExit(1)
        assert not model_dir.exists()
        return object(), object(), None

    module = types.SimpleNamespace(
        DEFAULT_MODEL="DeepFilterNet3",
        get_model_basedir=lambda _model: model_dir,
        init_df=init_df,
    )
    monkeypatch.setitem(sys.modules, "df.enhance", module)
    monkeypatch.setattr(audio_enhancer, "_df_model", None)
    monkeypatch.setattr(audio_enhancer, "_df_state", None)
    monkeypatch.setattr(audio_enhancer, "_df_device", None)
    monkeypatch.setattr(audio_enhancer, "_move_model_to_device", lambda _device: None)

    audio_enhancer._load_model(device="cpu")

    assert calls == 2


def test_packaged_windows_enhancement_uses_isolated_worker(monkeypatch, tmp_path):
    source = tmp_path / "audio.wav"
    source.touch()
    expected = tmp_path / "enhanced.wav"
    captured = {}
    monkeypatch.setattr(audio_enhancer.platform, "system", lambda: "Windows")
    monkeypatch.setattr(audio_enhancer.sys, "frozen", True, raising=False)

    def fake_worker(input_path, output_path, atten_lim_db, cancel_event):
        captured.update(
            {
                "input_path": input_path,
                "output_path": output_path,
                "atten_lim_db": atten_lim_db,
                "cancel_event": cancel_event,
            }
        )
        return str(expected)

    monkeypatch.setattr(audio_enhancer, "_enhance_in_packaged_worker", fake_worker)

    result = audio_enhancer.enhance_audio(
        str(source),
        str(expected),
        atten_lim_db=12.0,
    )

    assert result == str(expected)
    assert captured["input_path"] == str(source)
    assert captured["atten_lim_db"] == 12.0
