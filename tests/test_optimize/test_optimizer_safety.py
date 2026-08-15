from subforge.core.asr.asr_data import ASRData, ASRDataSeg, ASRWord
from subforge.core.optimize.optimize import SubtitleOptimizer, _lexical_edit_violations


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
            words=[
                ASRWord(
                    "hello",
                    0,
                    1000,
                    speaker_id="Speaker 1",
                    timing_source="forced_alignment",
                )
            ],
            timestamp_granularity="sentence",
            timing_source="forced_alignment",
        )
    ]

    result = SubtitleOptimizer._create_segments(
        original_segments,
        {"1": "Hello."},
    )

    assert result[0].text == "Hello."
    assert result[0].translated_text == "你好"
    assert result[0].speaker_id == "Speaker 1"
    assert result[0].timing_source == "forced_alignment"
    assert result[0].words[0].speaker_id == "Speaker 1"


def test_validation_rejects_copying_next_subtitle_into_current_key():
    optimizer = SubtitleOptimizer.__new__(SubtitleOptimizer)
    original = {
        "1": (
            "So anything you'd like to ask, let me know and I will get back "
            "to you, but"
        ),
        "2": "That'll wrap it up. Take care, guys.",
    }
    optimized = {
        "1": (
            "So anything you'd like to ask, let me know and I will get back to "
            "you, but that'll wrap it up. Take care, guys."
        ),
        "2": "That'll wrap it up. Take care, guys.",
    }

    valid, error = optimizer._validate_optimization_result(original, optimized)

    assert not valid
    assert "copied the start of adjacent key" in error


def test_validation_allows_local_recognition_correction():
    optimizer = SubtitleOptimizer.__new__(SubtitleOptimizer)
    original = {
        "1": "I like the seven series very mutch",
        "2": "That'll wrap it up. Take care, guys.",
    }
    optimized = {
        "1": "I like the 7 Series very much.",
        "2": "That'll wrap it up. Take care, guys.",
    }

    valid, error = optimizer._validate_optimization_result(original, optimized)

    assert valid
    assert not error


def test_lexical_guard_rejects_deleting_meaningful_discourse_marker():
    violations = _lexical_edit_violations(
        "After World War II, basically in the aftermath of the war",
        "After World War II, in the aftermath of the war",
    )

    assert violations


def test_lexical_guard_allows_narrow_time_phrase_correction():
    assert not _lexical_edit_violations(
        "At the last 30 years",
        "In the last 30 years",
    )


def test_lexical_guard_rejects_generic_short_preposition_change():
    violations = _lexical_edit_violations(
        "Meet me at the station",
        "Meet me in the station",
    )

    assert violations


def test_lexical_guard_allows_explicit_fillers_and_adjacent_duplicates():
    assert not _lexical_edit_violations(
        "You know the the plant plants are operating",
        "The plants are operating",
    )


def test_lexical_guard_allows_joining_split_spelling_without_changing_letters():
    assert not _lexical_edit_violations(
        "I had a Black berry Pearl",
        "I had a Blackberry Pearl",
    )


def test_lexical_guard_still_rejects_semantic_multiword_replacement():
    assert _lexical_edit_violations(
        "I had a black berry pie",
        "I had a blueberry pie",
    )


def test_global_ownership_check_repairs_copy_across_batch_boundary(monkeypatch):
    optimizer = SubtitleOptimizer.__new__(SubtitleOptimizer)
    optimizer.batch_num = 1
    optimizer.is_running = False
    optimizer.executor = None
    source = ASRData(
        [
            ASRDataSeg(
                "So anything you'd like to ask, let me know and I will get back to you, but",
                0,
                2000,
            ),
            ASRDataSeg("That'll wrap it up. Take care, guys.", 2100, 3000),
        ]
    )
    monkeypatch.setattr(
        optimizer,
        "_parallel_optimize",
        lambda _chunks: {
            "1": (
                "So anything you'd like to ask, let me know and I will get back "
                "to you, but that'll wrap it up. Take care, guys."
            ),
            "2": "That'll wrap it up. Take care, everyone.",
        },
    )

    result = optimizer.optimize_subtitle(source)

    assert result.segments[0].text == source.segments[0].text
    assert result.segments[1].text == "That'll wrap it up. Take care, everyone."
