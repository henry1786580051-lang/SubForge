import importlib

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
