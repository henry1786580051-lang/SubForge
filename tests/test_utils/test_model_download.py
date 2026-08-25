import threading

import pytest

from subforge.core.utils import model_download


class _FakeQueue:
    def get_nowait(self):
        raise AssertionError("cancelled download must not read a result")

    def close(self):
        pass

    def join_thread(self):
        pass


class _FakeProcess:
    def __init__(self):
        self.alive = True
        self.terminated = False
        self.exitcode = None

    def start(self):
        pass

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False

    def kill(self):
        self.alive = False

    def join(self, timeout=None):
        pass


def test_cancellable_model_download_terminates_worker(monkeypatch):
    process = _FakeProcess()

    class _Context:
        def Queue(self, maxsize):
            assert maxsize == 1
            return _FakeQueue()

        def Process(self, **kwargs):
            assert kwargs["name"] == "subforge-huggingface_snapshot"
            return process

    monkeypatch.setattr(model_download.multiprocessing, "get_context", lambda _method: _Context())
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(RuntimeError, match="cancelled"):
        model_download.run_cancellable_model_download(
            "huggingface_snapshot",
            {"repo_id": "example/model"},
            cancel_event,
        )

    assert process.terminated is True
