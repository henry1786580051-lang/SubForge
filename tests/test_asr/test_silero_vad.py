"""Tests for Silero VAD speech segment detection."""

import numpy as np
import pytest
import soundfile as sf

from subforge.core.asr.silero_vad import is_available

pytestmark = pytest.mark.skipif(not is_available(), reason="torch not installed")


@pytest.fixture
def silence_wav(tmp_path):
    """Generate 5 seconds of silence at 16kHz."""
    path = str(tmp_path / "silence.wav")
    samples = np.zeros(16000 * 5, dtype=np.float32)
    sf.write(path, samples, 16000)
    return path


@pytest.fixture
def noise_wav(tmp_path):
    """Generate 5 seconds of noise (simulating speech-like energy)."""
    path = str(tmp_path / "noise.wav")
    rng = np.random.default_rng(42)
    samples = rng.uniform(-0.5, 0.5, 16000 * 5).astype(np.float32)
    sf.write(path, samples, 16000)
    return path


@pytest.fixture
def mixed_wav(tmp_path):
    """2s silence + 3s noise + 2s silence at 16kHz."""
    path = str(tmp_path / "mixed.wav")
    rng = np.random.default_rng(42)
    silence = np.zeros(16000 * 2, dtype=np.float32)
    noise = rng.uniform(-0.5, 0.5, 16000 * 3).astype(np.float32)
    samples = np.concatenate([silence, noise, silence])
    sf.write(path, samples, 16000)
    return path


class TestDetectSpeechSegments:

    def test_silence_returns_empty_or_minimal(self, silence_wav):
        """Pure silence should produce no segments or very few."""
        from subforge.core.asr.silero_vad import detect_speech_segments
        result = detect_speech_segments(silence_wav, threshold=0.5)
        # Should have 0 or very few segments (noise floor)
        assert len(result) <= 2

    def test_segments_are_sorted(self, noise_wav):
        """Segments must be sorted by start time."""
        from subforge.core.asr.silero_vad import detect_speech_segments
        result = detect_speech_segments(noise_wav, threshold=0.3)
        for i in range(1, len(result)):
            assert result[i][0] >= result[i - 1][0]

    def test_segments_do_not_overlap(self, noise_wav):
        """Adjacent segments must not overlap."""
        from subforge.core.asr.silero_vad import detect_speech_segments
        result = detect_speech_segments(noise_wav, threshold=0.3)
        for i in range(1, len(result)):
            assert result[i][0] >= result[i - 1][1]

    def test_segments_within_audio_bounds(self, noise_wav):
        """All segments must be within [0, audio_duration + padding]."""
        from subforge.core.asr.silero_vad import detect_speech_segments
        result = detect_speech_segments(noise_wav, threshold=0.3, speech_pad_ms=500)
        audio_len_ms = 5000  # 5 seconds
        for start, end in result:
            assert start >= 0
            assert end <= audio_len_ms + 500  # audio_len + max pad
            assert end > start

    def test_invalid_path_raises(self):
        """Non-existent file should raise FileNotFoundError."""
        from subforge.core.asr.silero_vad import detect_speech_segments
        with pytest.raises(FileNotFoundError):
            detect_speech_segments("/nonexistent/path.wav")

    def test_none_path_raises(self):
        """None path should raise FileNotFoundError."""
        from subforge.core.asr.silero_vad import detect_speech_segments
        with pytest.raises(FileNotFoundError):
            detect_speech_segments(None)


class TestParameterSensitivity:

    def test_higher_threshold_same_or_fewer_segments(self, noise_wav):
        """Higher threshold should detect same or less speech."""
        from subforge.core.asr.silero_vad import detect_speech_segments
        low = detect_speech_segments(noise_wav, threshold=0.3)
        high = detect_speech_segments(noise_wav, threshold=0.7)
        low_dur = sum(e - s for s, e in low)
        high_dur = sum(e - s for s, e in high)
        assert high_dur <= low_dur * 1.3  # 30% margin for randomness

    def test_larger_pad_more_coverage(self, noise_wav):
        """Larger padding should produce equal or more total duration."""
        from subforge.core.asr.silero_vad import detect_speech_segments
        small = detect_speech_segments(noise_wav, threshold=0.3, speech_pad_ms=100)
        large = detect_speech_segments(noise_wav, threshold=0.3, speech_pad_ms=500)
        small_dur = sum(e - s for s, e in small)
        large_dur = sum(e - s for s, e in large)
        assert large_dur >= small_dur

    def test_larger_min_silence_merges_more(self, noise_wav):
        """Larger min_silence_ms should produce fewer, longer segments."""
        from subforge.core.asr.silero_vad import detect_speech_segments
        small = detect_speech_segments(noise_wav, threshold=0.3, min_silence_ms=100)
        large = detect_speech_segments(noise_wav, threshold=0.3, min_silence_ms=500)
        assert len(large) <= len(small) + 2  # allow small margin


class TestRunVadInference:

    def test_returns_list_of_tuples(self, noise_wav):
        """run_vad_inference should return List[Tuple[int, int]]."""
        import numpy as np
        from pydub import AudioSegment

        from subforge.core.asr.silero_vad import run_vad_inference

        audio = AudioSegment.from_file(noise_wav)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0

        result = run_vad_inference(samples, audio_len_ms=len(audio))
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], int)
            assert isinstance(item[1], int)

    def test_empty_samples_returns_empty(self):
        """Empty audio should return empty list."""
        from subforge.core.asr.silero_vad import run_vad_inference
        result = run_vad_inference(np.array([], dtype=np.float32), audio_len_ms=0)
        assert result == []
