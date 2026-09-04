import pytest

from subforge.core.asr import base as asr_base
from subforge.core.asr.asr_data import ASRDataSeg
from subforge.core.asr.base import BaseASR


class FakeCache:
    def __init__(self):
        self.get_calls = 0
        self.set_calls = 0
        self.values = {}

    def get(self, key, default=None):
        self.get_calls += 1
        return self.values.get(key, default)

    def set(self, key, value, expire=None):
        self.set_calls += 1
        self.values[key] = value


class DummyASR(BaseASR):
    def _run(self, callback=None, **kwargs):
        return "fresh"

    def _make_segments(self, resp_data):
        return [ASRDataSeg(str(resp_data), 0, 1000)]


def test_base_asr_does_not_read_or_write_cache_when_disabled(monkeypatch, tmp_path):
    from pydub.generators import Sine

    audio_path = tmp_path / "audio.wav"
    Sine(440).to_audio_segment(duration=500).export(audio_path, format="wav").close()
    fake_cache = FakeCache()
    fake_cache.values["asr-v2:DummyASR:any"] = "cached"

    monkeypatch.setattr(asr_base, "get_asr_cache", lambda: fake_cache)
    monkeypatch.setattr(asr_base, "is_cache_enabled", lambda: True)

    asr = DummyASR(str(audio_path), use_cache=False)
    asr._get_key = lambda: "any"
    result = asr.run()

    assert result.segments[0].text == "fresh"
    assert fake_cache.get_calls == 0
    assert fake_cache.set_calls == 0


def test_base_asr_uses_cache_only_when_enabled(monkeypatch, tmp_path):
    from pydub.generators import Sine

    audio_path = tmp_path / "audio.wav"
    Sine(440).to_audio_segment(duration=500).export(audio_path, format="wav").close()
    fake_cache = FakeCache()
    fake_cache.values["asr-v2:DummyASR:any"] = "cached"

    monkeypatch.setattr(asr_base, "get_asr_cache", lambda: fake_cache)
    monkeypatch.setattr(asr_base, "is_cache_enabled", lambda: True)

    asr = DummyASR(str(audio_path), use_cache=True)
    asr._get_key = lambda: "any"
    result = asr.run()

    assert result.segments[0].text == "cached"
    assert fake_cache.get_calls == 1
    assert fake_cache.set_calls == 0


@pytest.mark.parametrize("cached", [False, True])
def test_base_asr_preserves_coverage_issues(monkeypatch, tmp_path, cached):
    from pydub.generators import Sine

    audio = tmp_path / "audio.wav"
    Sine(440).to_audio_segment(duration=50).export(audio, format="wav").close()
    payload = {"coverage_issues": [{"start": 10, "end": 14.2, "reason": "decode_budget"}]}
    cache = FakeCache()
    cache.values["asr-v2:DummyASR:any"] = payload
    monkeypatch.setattr(asr_base, "get_asr_cache", lambda: cache)
    monkeypatch.setattr(asr_base, "is_cache_enabled", lambda: True)
    asr = DummyASR(str(audio), use_cache=cached)
    asr._get_key = lambda: "any"
    asr._run = lambda *args, **kwargs: payload
    assert asr.run().coverage_issues == payload["coverage_issues"]
