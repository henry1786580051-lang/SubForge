import subprocess
import threading

from subforge.core.utils import video_utils


def test_video2audio_terminates_ffmpeg_when_cancelled(tmp_path, monkeypatch):
    source = tmp_path / "video.mp4"
    output = tmp_path / "audio.wav"
    source.touch()
    cancel_event = threading.Event()
    cancel_event.set()

    class FakeProcess:
        returncode = -15

        def __init__(self):
            self.calls = 0
            self.terminated = False

        def communicate(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            return "", "cancelled"

        def terminate(self):
            self.terminated = True

        def kill(self):
            raise AssertionError("graceful termination should succeed")

    process = FakeProcess()
    monkeypatch.setattr(video_utils.subprocess, "Popen", lambda *_args, **_kwargs: process)

    assert video_utils.video2audio(
        str(source),
        str(output),
        cancel_event=cancel_event,
    ) is False
    assert process.terminated is True
