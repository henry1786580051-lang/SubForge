from subforge.core.asr.asr_data import ASRDataSeg, ASRWord
from subforge.core.split.boundary import (
    assess_english_boundary,
    normalize_boundaries,
)


def _cues(texts: list[str], *, speakers: list[str] | None = None) -> list[ASRDataSeg]:
    cursor = 0
    result = []
    speakers = speakers or ["S1"] * len(texts)
    for text, speaker in zip(texts, speakers):
        words = []
        for token in text.split():
            words.append(
                ASRWord(
                    token,
                    cursor,
                    cursor + 100,
                    speaker_id=speaker,
                    timing_source="forced_alignment",
                )
            )
            cursor += 120
        result.append(
            ASRDataSeg(
                text,
                words[0].start_time,
                words[-1].end_time,
                speaker_id=speaker,
                words=words,
                timestamp_granularity="sentence",
                timing_source="forced_alignment",
            )
        )
    return result


def test_assessment_rejects_user_reported_dangling_boundaries():
    boundaries = [
        ("potential solutions because", "it can deliver electricity"),
        ("Most people can say that they", "either know somebody"),
        ("that's changed. Now", "our nuclear industry"),
        ("it provides really good", "jobs for the town"),
        ("But for them, it", "could really be worth it"),
        ("before they can flip the", "switch and start producing"),
        ("it was actually a red state", "that launched the first program"),
        ("there are a lot of challenges", "for building nuclear power"),
        ("that drastically limits the number", "of sites that are available"),
        ("where to build a nuclear", "power plant in this town"),
        ("It will take at least a decade. But for them,", "it could be worth it"),
    ]

    assert all(assess_english_boundary(left, right).unstable for left, right in boundaries)


def test_assessment_rejects_real_spoken_english_dependency_boundaries():
    boundaries = [
        ("But the problem is they've", "made the taillight narrow"),
        ("this has all just been the same", "sort of size of SUV"),
        ("you have this sort of serrated", "edge to help load stuff"),
        ("they do give", "you a place to mount one"),
        ("It's just", "engine and CVT and electric motors"),
        ("I don't think it's", "making a huge statement"),
        ("the most important thing back here", "is space"),
        ("there are buttons to get", "out of this"),
        ("kind of a pretty", "standard all-season tire"),
        ("different than the one", "you get in a Fiat"),
        ("the biggest thing I like in this interior", "is the packaging"),
        ("This is, I think,", "the same steering wheel"),
        ("you've got this massive", "hexagon in front of you"),
        ("This has a 14 gallon tank. So", "it is doing the best it can"),
        ("Jeep's first attempt at a traditional", "hybrid"),
        ("the whole windshield is just", "this pattern from the dashboard"),
        ("they don't", "have that hybrid brake feel"),
        ("It just gets", "really annoyed"),
        ("I was going to fall", "out of it"),
        ("being helped by my", "sweaty back"),
        ("apart from a higher end", "car like a Volvo"),
        ("and a better", "sound system"),
        ("aiming for this higher", "echelon than before"),
        ("But I think the big", "picture on this is that it does feel"),
    ]

    assert all(assess_english_boundary(left, right).unstable for left, right in boundaries)


def test_assessment_allows_complete_comma_clauses_ending_in_pronouns():
    boundaries = [
        ("Any way you slice it,", "we have to replace the infrastructure"),
        ("If journalism like this is important to you,", "then join the community"),
    ]

    assert all(
        not assess_english_boundary(left, right).unstable
        for left, right in boundaries
    )


def test_normalizer_moves_because_to_the_following_clause():
    cues = _cues(
        [
            "Nuclear energy has come up as one of the potential solutions because",
            "it can deliver a huge amount of electricity 24 hours a day and without producing greenhouse gas emissions.",
        ]
    )

    repaired = normalize_boundaries(cues)

    assert repaired[0].text.endswith("potential solutions")
    assert repaired[1].text.startswith("because it can deliver")
    assert repaired[0].end_time < repaired[1].start_time


def test_normalizer_repairs_consecutive_subject_boundaries():
    cues = _cues(
        [
            "Most people can say that they",
            "either know somebody who works at the plants, or they",
            "work there themselves. 100 years ago or so, the city of Oswego was",
            "a huge port here on the Great Lake,",
        ]
    )

    repaired = normalize_boundaries(cues)

    assert repaired[0].text == (
        "Most people can say that they either know somebody who works at the plants, "
        "or they work there themselves."
    )
    assert repaired[1].text.startswith("100 years ago")
    assert not repaired[1].text.endswith("was")


def test_normalizer_moves_new_sentence_prefix_and_modifier_head():
    cues = _cues(
        [
            "in industry and manufacturing. And over the course of time, that's changed. Now",
            "our nuclear industry is our number one employer.",
            "Not only does it provide jobs, it provides really good",
            "jobs. There's thousands of construction jobs that'll come with each build.",
        ]
    )

    repaired = normalize_boundaries(cues)

    assert repaired[0].text.endswith("that's changed.")
    assert repaired[1].text.startswith("Now our nuclear industry")
    assert repaired[2].text.endswith("really good jobs.")
    assert repaired[3].text.startswith("There's thousands")


def test_normalizer_keeps_a_distant_speaker_boundary():
    cues = _cues(
        ["But for them, it", "could really be worth the headache."],
        speakers=["S1", "S2"],
    )
    shift = 500
    cues[1].start_time += shift
    cues[1].end_time += shift
    for word in cues[1].words:
        word.start_time += shift
        word.end_time += shift

    repaired = normalize_boundaries(cues)

    assert [cue.text for cue in repaired] == [cue.text for cue in cues]


def test_normalizer_repairs_a_short_diarization_flip_inside_a_sentence():
    cues = _cues(
        [
            "Not only does it provide jobs, it provides really good",
            "jobs. There's thousands of construction jobs.",
        ],
        speakers=["S1", "S2"],
    )

    repaired = normalize_boundaries(cues)

    assert repaired[0].text.endswith("really good jobs.")
    assert repaired[1].text.startswith("There's thousands")


def test_normalizer_keeps_relative_clause_with_its_antecedent():
    cues = _cues(
        [
            "But I was surprised to find out that it was actually a red state",
            "that was the first to launch a statewide program in the modern era.",
        ]
    )

    repaired = normalize_boundaries(cues)

    assert repaired[0].text == "But I was surprised to find out"
    assert repaired[1].text.startswith("that it was actually a red state")


def test_normalizer_preserves_every_atomic_word_in_order():
    cues = _cues(
        [
            "Not only does it provide jobs, it provides really good",
            "jobs. There's thousands of construction jobs that'll come with each build.",
        ]
    )
    original = [word.text for cue in cues for word in cue.words]

    repaired = normalize_boundaries(cues)

    assert [word.text for cue in repaired for word in cue.words] == original


def test_normalizer_removes_cross_boundary_singular_false_start():
    cues = _cues(
        [
            "Most people know somebody who works at the plant,",
            "plants, or they work there themselves.",
        ]
    )

    repaired = normalize_boundaries(cues)

    assert len(repaired) == 1
    assert repaired[0].text == (
        "Most people know somebody who works at the plants, or they work there themselves."
    )


def test_normalizer_removes_in_subtitle_singular_false_start():
    cues = _cues(
        ["They know somebody who works at the plant, plants, or they work there."]
    )

    repaired = normalize_boundaries(cues)

    assert repaired[0].text == (
        "They know somebody who works at the plants, or they work there."
    )


def test_normalizer_merges_compact_pair_without_a_stable_internal_boundary():
    cues = _cues(
        [
            "But there are a lot of challenges",
            "for building nuclear in the U.S.",
        ],
        speakers=["S1", "S2"],
    )

    repaired = normalize_boundaries(cues)

    assert len(repaired) == 1
    assert repaired[0].text == (
        "But there are a lot of challenges for building nuclear in the U.S."
    )


def test_normalizer_moves_but_for_them_to_the_following_sentence():
    cues = _cues(
        [
            "It will take at least a decade. But for them,",
            "it could really be worth the headache.",
        ]
    )

    repaired = normalize_boundaries(cues)

    assert repaired[0].text == "It will take at least a decade."
    assert repaired[1].text == "But for them, it could really be worth the headache."
