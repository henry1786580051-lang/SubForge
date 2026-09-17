"""Observe Hub HTTP/Xet byte callbacks only inside the isolated download worker.

Keep the library's caching, authentication, revision and retry behavior intact.
Shared counters avoid a queue of progress events blocking cancellation or exit.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def observe_hub_download(shared, allow_patterns=None, ignore_patterns=None):
    from huggingface_hub import HfApi, _snapshot_download, constants, file_download
    from huggingface_hub.utils._paths import filter_repo_objects

    original_chunk_size = constants.DOWNLOAD_CHUNK_SIZE
    original_info = HfApi.repo_info
    original_file = getattr(_snapshot_download, "hf_hub_download")
    original_bar = getattr(file_download, "_get_progress_bar_context", None)
    local = threading.local()
    positions: dict[str, int] = {}
    lock = threading.RLock()

    def publish(key, position, network=0):
        with lock, shared.get_lock():
            positions[key] = max(0, int(position))
            shared[0] = sum(positions.values())
            shared[2] += max(0, int(network))

    def repo_info(self, *args, **kwargs):
        kwargs["files_metadata"] = True
        info = original_info(self, *args, **kwargs)
        selected = list(
            filter_repo_objects(
                info.siblings or [],
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                key=lambda item: item.rfilename,
            )
        )
        sizes = [item.size for item in selected]
        with shared.get_lock():
            shared[1] = (
                sum(s for s in sizes if s is not None)
                if sizes and all(s is not None for s in sizes)
                else -1
            )
        return info

    def download(*args, **kwargs):
        local.key = kwargs.get("filename") or args[1]
        try:
            result = original_file(*args, **kwargs)
            publish(local.key, Path(result).stat().st_size)
            return result
        finally:
            local.key = None

    @contextmanager
    def progress_context(**kwargs):
        key = getattr(local, "key", None)
        assert original_bar is not None
        with original_bar(**kwargs) as bar:
            if key is None or kwargs.get("_tqdm_bar") is not None:
                yield bar
                return
            position = int(kwargs.get("initial") or 0)
            publish(key, position)

            class Progress:
                def update(self, amount):
                    nonlocal position
                    position += amount
                    publish(key, position, amount)
                    return bar.update(amount)

                def __getattr__(self, name):
                    return getattr(bar, name)

            yield Progress()

    # Hub defaults to 10 MB HTTP callbacks, which look stalled on slow links.
    if not constants.HF_HUB_ENABLE_HF_TRANSFER:
        constants.DOWNLOAD_CHUNK_SIZE = 64 * 1024
    HfApi.repo_info = repo_info
    setattr(_snapshot_download, "hf_hub_download", download)
    if original_bar is not None:
        setattr(file_download, "_get_progress_bar_context", progress_context)
    try:
        yield
    finally:
        constants.DOWNLOAD_CHUNK_SIZE = original_chunk_size
        HfApi.repo_info = original_info
        setattr(_snapshot_download, "hf_hub_download", original_file)
        if original_bar is not None:
            setattr(file_download, "_get_progress_bar_context", original_bar)
