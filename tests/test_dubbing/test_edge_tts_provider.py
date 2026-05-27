from pathlib import Path

from subforge.core.speech import (
    EdgeTTSSpeechSynthesizer,
    SpeechProviderConfig,
    SynthesisRequest,
)


class FakeCommunicate:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)

    async def save(self, audio_fname):
        Path(audio_fname).write_bytes(b"fake-mp3")


def test_edge_tts_synthesizer_writes_mp3(tmp_path, monkeypatch):
    monkeypatch.setattr("subforge.core.speech.providers.edge_tts.Communicate", FakeCommunicate)

    config = SpeechProviderConfig(
        provider="edge",
        api_key="",
        model="edge-tts",
        default_voice="zh-CN-XiaoxiaoNeural",
        speed=1.2,
        gain=-10,
    )
    result = EdgeTTSSpeechSynthesizer(config).synthesize(
        SynthesisRequest(text="你好", output_path=str(tmp_path / "line.wav"))
    )

    assert result.output_path.endswith(".mp3")
    assert Path(result.output_path).read_bytes() == b"fake-mp3"
    assert result.voice == "zh-CN-XiaoxiaoNeural"
    assert FakeCommunicate.calls[-1]["rate"] == "+20%"
    assert FakeCommunicate.calls[-1]["volume"] == "-10%"
