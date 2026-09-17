import multiprocessing
from contextlib import contextmanager
from types import SimpleNamespace

from huggingface_hub import HfApi, _snapshot_download, constants, file_download

from subforge.core.utils.hub_progress import observe_hub_download


def test_hub_multi_file_cache_resume_and_recursive_retry(monkeypatch, tmp_path):
    shared = multiprocessing.get_context("spawn").Array("d", [0, -1, 0])
    files = [
        SimpleNamespace(rfilename="model.bin", size=1000),
        SimpleNamespace(rfilename="config.json", size=100),
        SimpleNamespace(rfilename="excluded.txt", size=200),
    ]
    calls = []

    def info(self, *args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(siblings=files)

    @contextmanager
    def bar(**kwargs):
        yield kwargs.get("_tqdm_bar") or SimpleNamespace(update=lambda amount: None)

    def download(*args, filename, **kwargs):
        if filename == "model.bin":
            with file_download._get_progress_bar_context(initial=400) as progress:
                assert shared[0] == 400
                progress.update(200)
                with file_download._get_progress_bar_context(
                    initial=600, _tqdm_bar=progress
                ) as retry:
                    retry.update(400)
        path = tmp_path / filename
        path.write_bytes(b"x" * (1000 if filename == "model.bin" else 100))
        return str(path)

    monkeypatch.setattr(HfApi, "repo_info", info)
    monkeypatch.setattr(file_download, "_get_progress_bar_context", bar)
    monkeypatch.setattr(_snapshot_download, "hf_hub_download", download)
    with observe_hub_download(shared, ["*.bin", "*.json"]):
        HfApi().repo_info("example/model", token=False)
        assert shared[1] == 1100
        _snapshot_download.hf_hub_download("example/model", filename="model.bin")
        _snapshot_download.hf_hub_download("example/model", filename="config.json")
        assert list(shared) == [1100, 1100, 600]  # neither cache nor resume counted as speed
    assert HfApi.repo_info is info
    assert file_download._get_progress_bar_context is bar
    assert _snapshot_download.hf_hub_download is download
    assert calls[0]["files_metadata"] is True
    assert calls[0]["token"] is False


def test_observer_restores_library_on_failure(monkeypatch):
    shared = multiprocessing.get_context("spawn").Array("d", [0, -1, 0])
    original = HfApi.repo_info
    chunk_size = constants.DOWNLOAD_CHUNK_SIZE
    try:
        with observe_hub_download(shared):
            assert constants.DOWNLOAD_CHUNK_SIZE == 64 * 1024
            raise RuntimeError("cancel")
    except RuntimeError:
        pass
    assert HfApi.repo_info is original
    assert constants.DOWNLOAD_CHUNK_SIZE == chunk_size
