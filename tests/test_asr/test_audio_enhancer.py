import hashlib
import importlib
import io
import sys
import types
import zipfile

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


def _model_module(tmp_path, **kwargs):
    values = {
        "DEFAULT_MODEL": "DeepFilterNet3",
        "get_cache_dir": lambda: tmp_path,
    }
    values.update(kwargs)
    return types.SimpleNamespace(**values)


def _write_complete_model(model_dir):
    checkpoints = model_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    (model_dir / "config.ini").write_text("complete", encoding="utf-8")
    (checkpoints / "model.ckpt.best").write_bytes(b"model")


def test_deepfilter_system_exit_becomes_recoverable_error(monkeypatch, tmp_path):
    def aborting_init_df(**_kwargs):
        raise SystemExit(1)

    model_dir = tmp_path / "DeepFilterNet3"
    _write_complete_model(model_dir)
    monkeypatch.setattr(audio_enhancer, "_df_model", None)
    monkeypatch.setattr(audio_enhancer, "_df_state", None)
    monkeypatch.setattr(audio_enhancer, "_df_device", None)
    monkeypatch.setitem(
        sys.modules,
        "df.enhance",
        _model_module(tmp_path, init_df=aborting_init_df),
    )

    with pytest.raises(RuntimeError, match="checkpoint could not be loaded"):
        audio_enhancer._load_model(device="cpu")

    assert audio_enhancer._df_model is None
    assert audio_enhancer._df_state is None


def test_incomplete_default_model_cache_is_removed(tmp_path):
    model_dir = tmp_path / "DeepFilterNet3"
    model_dir.mkdir()
    (model_dir / "config.ini").write_text("partial", encoding="utf-8")
    module = _model_module(tmp_path)

    removed = audio_enhancer._repair_default_model_cache(module)

    assert removed is True
    assert not model_dir.exists()


def test_complete_default_model_cache_is_preserved(tmp_path):
    model_dir = tmp_path / "DeepFilterNet3"
    _write_complete_model(model_dir)
    module = _model_module(tmp_path)

    removed = audio_enhancer._repair_default_model_cache(module)

    assert removed is False
    assert model_dir.exists()


def test_corrupt_default_model_cache_is_removed_after_load_abort(
    tmp_path,
    monkeypatch,
):
    model_dir = tmp_path / "DeepFilterNet3"
    checkpoints = model_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    (model_dir / "config.ini").write_text("corrupt", encoding="utf-8")
    (checkpoints / "model.ckpt.best").write_bytes(b"corrupt")
    def init_df(**_kwargs):
        raise SystemExit(1)

    module = _model_module(tmp_path, init_df=init_df)
    monkeypatch.setitem(sys.modules, "df.enhance", module)
    monkeypatch.setattr(audio_enhancer, "_df_model", None)
    monkeypatch.setattr(audio_enhancer, "_df_state", None)
    monkeypatch.setattr(audio_enhancer, "_df_device", None)
    monkeypatch.setattr(audio_enhancer, "_move_model_to_device", lambda _device: None)

    with pytest.raises(RuntimeError, match="checkpoint could not be loaded"):
        audio_enhancer._load_model(device="cpu")

    assert not model_dir.exists()


def test_incomplete_download_is_removed_without_calling_implicit_downloader(tmp_path):
    archive = tmp_path / "DeepFilterNet3.zip"
    archive.write_bytes(b"partial")
    implicit_download_called = False

    def implicit_download(_model):
        nonlocal implicit_download_called
        implicit_download_called = True
        raise AssertionError("implicit downloader must not run")

    module = _model_module(tmp_path, get_model_basedir=implicit_download)

    assert audio_enhancer._repair_default_model_cache(module) is True
    assert not archive.exists()
    assert implicit_download_called is False


def test_model_download_is_verified_and_extracted(monkeypatch, tmp_path):
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w") as bundle:
        bundle.writestr("DeepFilterNet3/config.ini", "config")
        bundle.writestr("DeepFilterNet3/checkpoints/model.ckpt.best", b"checkpoint")
    archive_bytes = archive_buffer.getvalue()

    class Response:
        headers = {"Content-Length": str(len(archive_bytes))}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            return (
                archive_bytes[offset : offset + chunk_size]
                for offset in range(0, len(archive_bytes), chunk_size)
            )

    requests = types.SimpleNamespace(get=lambda *_args, **_kwargs: Response())
    monkeypatch.setitem(sys.modules, "requests", requests)
    monkeypatch.setattr(audio_enhancer, "_DEFAULT_MODEL_DOWNLOAD_BYTES", len(archive_bytes))
    monkeypatch.setattr(
        audio_enhancer,
        "_DEFAULT_MODEL_SHA256",
        hashlib.sha256(archive_bytes).hexdigest(),
    )
    progress = []

    model_dir = audio_enhancer._ensure_default_model(
        _model_module(tmp_path),
        progress_callback=lambda value, message: progress.append((value, message)),
    )

    assert audio_enhancer._model_cache_ready(model_dir)
    assert progress
    assert any("Downloading" in message for _, message in progress)
    assert not (tmp_path / "DeepFilterNet3.zip").exists()


def test_failed_model_download_removes_partial_file(monkeypatch, tmp_path):
    class Response:
        headers = {"Content-Length": "100"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"partial"
            raise TimeoutError("stalled")

    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(get=lambda *_args, **_kwargs: Response()),
    )

    with pytest.raises(TimeoutError, match="stalled"):
        audio_enhancer._ensure_default_model(_model_module(tmp_path))

    assert not list(tmp_path.glob("*.part-*"))


def test_packaged_desktop_enhancement_uses_isolated_worker(monkeypatch, tmp_path):
    source = tmp_path / "audio.wav"
    source.touch()
    expected = tmp_path / "enhanced.wav"
    captured = {}
    monkeypatch.setattr(audio_enhancer.sys, "frozen", True, raising=False)

    def fake_worker(input_path, output_path, atten_lim_db, cancel_event, progress_callback):
        captured.update(
            {
                "input_path": input_path,
                "output_path": output_path,
                "atten_lim_db": atten_lim_db,
                "cancel_event": cancel_event,
                "progress_callback": progress_callback,
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
