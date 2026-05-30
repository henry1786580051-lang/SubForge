from subforge.core.asr.asr_data import ASRDataSeg
from subforge.core.optimize.optimize import SubtitleOptimizer


def test_repair_subtitle_preserves_original_keys_without_guessing_alignment():
    original = {
        "8": "so this is kind of our biggest departure since",
        "9": "And boy, does it make a statement,",
        "10": "especially today in this Touring trim",
    }
    optimized = {
        "8": "So this is kind of our biggest departure since.",
        "10": "Especially today in this Touring trim.",
        "11": "Extra line that should not be inserted.",
    }

    repaired = SubtitleOptimizer._repair_subtitle(original, optimized)

    assert list(repaired.keys()) == ["8", "9", "10"]
    assert repaired["8"] == "So this is kind of our biggest departure since."
    assert repaired["9"] == original["9"]
    assert repaired["10"] == "Especially today in this Touring trim."
    assert "11" not in repaired


def test_repair_subtitle_rejects_empty_or_non_string_values():
    original = {"1": "Hello there.", "2": "Welcome back."}
    optimized = {"1": "", "2": {"text": "Welcome back."}}

    repaired = SubtitleOptimizer._repair_subtitle(original, optimized)

    assert repaired == original


def test_create_segments_preserves_translation_and_speaker_metadata():
    original_segments = [
        ASRDataSeg(
            text="hello",
            start_time=0,
            end_time=1000,
            translated_text="你好",
            speaker_id="Speaker 1",
        )
    ]

    result = SubtitleOptimizer._create_segments(
        original_segments,
        {"1": "Hello."},
    )

    assert result[0].text == "Hello."
    assert result[0].translated_text == "你好"
    assert result[0].speaker_id == "Speaker 1"
