import pytest

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


def test_assessment_keeps_rev_matching_technical_term_together():
    assessment = assess_english_boundary(
        "which is their fancy way of saying rev",
        "matching which helps in city driving",
    )

    assert assessment.unstable
    assert "split lexical unit 'rev matching'" in assessment.reasons


def test_assessment_keeps_separated_take_away_construction_together():
    assessment = assess_english_boundary(
        "because this car sort of takes that ability",
        "away from you and encourages you to push it",
    )

    assert assessment.unstable
    assert "split phrasal construction 'take ... away'" in assessment.reasons


def test_assessment_keeps_revised_component_with_its_noun():
    assessment = assess_english_boundary(
        "Why don't we show you this newly revised",
        "nine speaker JBL sound system",
    )

    assert assessment.unstable
    assert "dangling modifier 'revised'" in assessment.reasons


def test_assessment_keeps_ordinal_gear_term_together():
    assessment = assess_english_boundary(
        "and you do have the torque, even here in fourth",
        "gear at 30 miles an hour, half throttle",
    )

    assert assessment.unstable is True
    assert "split lexical unit 'fourth gear'" in assessment.reasons


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "predicted the rise of a figure like",
            "Donald Trump that would use populist messaging",
            "comparative marker separated from its example",
        ),
        (
            "teachers said they assigned between zero",
            "and four books a year",
            "numeric range split at conjunction",
        ),
        (
            "And so I think that we see,",
            "a lot of changes in schools",
            "transitive predicate separated from its object",
        ),
        (
            "they did not have the same experimental",
            "standards that we have now",
            "split lexical unit 'experimental standards'",
        ),
    ],
)
def test_assessment_rejects_dialogue_translation_sensitive_boundaries(
    left, right, reason
):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert reason in assessment.reasons


def test_high_value_dialogue_modifier_reaches_strong_risk_threshold():
    assessment = assess_english_boundary(
        "And I think that would need to be a really,",
        "large scale shift for people to make.",
    )

    assert assessment.risk >= 30
    assert "dangling modifier 'really'" in assessment.reasons


def test_assessment_rejects_charger_recovery_dependency_boundaries():
    boundaries = [
        ("European influence here with this first", "generation of the new era Charger."),
        ("I don't know whether", "the Charger is a future classic."),
        ("This was driven by an old", "man who took great care of it."),
        ("and it's the same", "as an S class, so that's cool."),
        ("it really doesn't feel like", "this was put together in 2007."),
    ]

    assert all(assess_english_boundary(left, right).unstable for left, right in boundaries)


def test_assessment_rejects_negated_comparison_split_from_complement():
    assessment = assess_english_boundary(
        "But it isn't quite",
        "as juvenile and crazy as the Elantra.",
    )

    assert assessment.unstable
    assert "negated comparison split from its complement" in assessment.reasons


def test_assessment_rejects_charger_full_run_dependency_boundaries():
    boundaries = [
        ("there was quite a bit of European influence", "here with this first generation"),
        ("a performance pack to give you an intake", "and exhaust that boosted power"),
        ("these days 48 grand damn near", "gets you an RT392 Durango"),
        ("The easiest way to tell if they were a V6 or a V8", "was by the exhaust tips"),
        ("and because of that, the turn", "signals are one touch"),
        ("this is what's so great about", "this is you have the torque"),
        ("instead of spending", "like eighty thousand dollars"),
        ("quite a bit of European", "influence here with this generation"),
        ("get a bit of a performance", "pack to give you an intake"),
        ("The easiest way to tell if they were a V6", "or a V8 was by the exhaust tips"),
        ("these days 48 grand damn", "near gets you an RT392 Durango"),
        ("tell by how many exhaust", "tips they had"),
        ("and this is what's", "so great about this car"),
        ("I don't see why people avoid these cars. I mean,", "instead of spending more"),
        ("I don't see why people avoid these cars, I mean", "instead of spending more"),
        ("I mean, probably the closest", "of the sedans mentioned here"),
        ("any of the other big body American", "sedans mentioned in this video"),
        ("and this is what's so", "great about this car"),
        ("the other big body", "American sedans mentioned here"),
        ("I was a car-obsessed kid, which", "I was at age 13"),
        ("this quiet V8. I mean, that's", "what this car is all about"),
        ("And that's because at the time,", "Chrysler was involved"),
        ("I knew at this time Mercedes", "was involved in the platform"),
        ("Gene Buttman Ford in Ypsilanti,", "Michigan for tossing me the keys"),
    ]

    assert all(assess_english_boundary(left, right).unstable for left, right in boundaries)


def test_assessment_allows_complete_comma_clauses_ending_in_pronouns():
    boundaries = [
        ("Any way you slice it,", "we have to replace the infrastructure"),
        ("If journalism like this is important to you,", "then join the community"),
    ]

    assert all(not assess_english_boundary(left, right).unstable for left, right in boundaries)


def test_assessment_allows_location_inside_subject_before_predicate():
    assessment = assess_english_boundary(
        "And maybe part of the reason blue zones resonate so deeply in America",
        "is because so much of modern American life is designed against longevity.",
    )

    assert not assessment.unstable
    assert "proper-name subject separated from its predicate" not in assessment.reasons


def test_assessment_rejects_degree_complement_split_from_predicate():
    boundaries = [
        ("And maybe part of the reason blue zones resonate", "so deeply in America"),
        ("And loneliness and isolation became", "so widespread that it was an epidemic"),
    ]

    assert all(assess_english_boundary(left, right).unstable for left, right in boundaries)


def test_assessment_rejects_cross_language_sensitive_vehicle_boundaries():
    boundaries = [
        ("Gene Buttman Ford in Ypsilanti,", "Michigan, we're driving this Charger."),
        ("We actually saw this evolve", "before our very eyes, from SRT8 to Hellcat."),
        ("what the Charger was like", "when it was reborn in 2006."),
        ("these days 48 grand gets you an RT392", "Durango with more equipment."),
        ("maybe you're going to go out after this video", "and go purchase it."),
        (
            "I think 90% of what you're going to use",
            "this truck for, like the steering rack, is a good thing.",
        ),
        (
            "And then if you're having trouble trying to follow this little puny",
            "RPM gauge over here on the left, you can change the view.",
        ),
    ]

    assert all(assess_english_boundary(left, right).unstable for left, right in boundaries)


def test_assessment_rejects_hyphenated_attributive_split_from_noun():
    assessment = assess_english_boundary(
        "But what I wanted to show was the ease of use in day-to-day",
        "life for one of these Toyota electric vehicles,",
    )

    assert assessment.unstable
    assert any("hyphenated attributive" in reason for reason in assessment.reasons)


def test_assessment_rejects_internal_toyota_vehicle_phrase_boundaries():
    boundaries = [
        ("for one of these", "Toyota electric vehicles,"),
        ("for one of these Toyota", "electric vehicles,"),
        ("for one of these Toyota electric", "vehicles,"),
    ]

    assert all(assess_english_boundary(left, right).unstable for left, right in boundaries)


def test_assessment_allows_complete_ease_of_use_before_daily_context():
    assessment = assess_english_boundary(
        "But what I really wanted to show was mostly the ease of use",
        "in day-to-day life for one of these Toyota electric vehicles,",
    )

    assert not assessment.unstable


def test_normalizer_keeps_day_to_day_life_together():
    cues = _cues(
        [
            "But what I really wanted to show was mostly the ease of use in day-to-day",
            "life for one of these Toyota electric vehicles,",
            "because most of the things that you interact with in here are fine.",
        ]
    )

    repaired = normalize_boundaries(cues)

    assert any("day-to-day life" in cue.text for cue in repaired)
    assert all(not cue.text.endswith("day-to-day") for cue in repaired)
    assert all(not cue.text.endswith("wanted to show") for cue in repaired)
    assert all(
        not assess_english_boundary(left.text, right.text).unstable
        for left, right in zip(repaired, repaired[1:])
    )


def test_assessment_does_not_confuse_complete_use_question_with_split_relative_object():
    assessment = assess_english_boundary(
        "What tool do you use",
        "This truck is useful for hauling.",
    )

    assert not assessment.unstable


def test_assessment_rejects_raptor_recovery_dependency_boundaries():
    boundaries = [
        (
            "not so nice for other things. I think",
            "90% of what you're going to use this truck for is a good thing",
        ),
        (
            "but first want to show you this five",
            "and a half foot bed with the optional liner",
        ),
        (
            "it would make this thing so much more excellent than it already",
            "is and don't get me wrong",
        ),
        (
            "a taste of what it's like to live with a 2026",
            "Ford F-150 Raptor R",
        ),
        (
            "a taste of what it's like to live with a 2026",
            "ford f-150 raptor r",
        ),
        (
            "And what really sets it aside",
            "and makes the extra cost worthwhile",
        ),
        (
            "it is better than before and don't get me",
            "wrong the old version still has its place",
        ),
    ]

    assert all(assess_english_boundary(left, right).unstable for left, right in boundaries)


def test_assessment_allows_complete_see_how_it_does_clause():
    assessment = assess_english_boundary(
        "Let's get this off the line and see how she does",
        "Oh yes",
    )

    assert not assessment.unstable


def test_normalizer_moves_opinion_marker_and_model_year_to_next_cue():
    cues = _cues(
        [
            "not so nice for other things. I think",
            "90% of what you're going to use this truck for is a good thing.",
            "A taste of what it's like to live with a 2026",
            "Ford F-150 Raptor R. Thanks for watching.",
        ]
    )

    repaired = normalize_boundaries(cues)

    assert repaired[0].text.endswith("other things.")
    assert repaired[1].text.startswith("I think 90%")
    assert any("2026 Ford F-150 Raptor R." in cue.text for cue in repaired)
    assert all(not cue.text.endswith("a 2026") for cue in repaired)


def test_normalizer_repairs_consecutive_degree_and_fixed_phrase_boundaries():
    cues = _cues(
        [
            "I think that would elevate this experience so much like it would make this thing",
            "so much better than it already is and don't get me",
            "wrong the 10 speed still has a time and a place.",
        ]
    )

    repaired = normalize_boundaries(cues)

    assert repaired[0].text.endswith("experience so much")
    assert repaired[1].text.startswith("like it would make this thing")
    assert repaired[1].text.endswith("don't get me wrong")
    assert all(
        not assess_english_boundary(left.text, right.text).unstable
        for left, right in zip(repaired, repaired[1:])
    )
    assert all(not cue.text.startswith("wrong") for cue in repaired)


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


def test_normalizer_repairs_a_strong_dependency_across_a_320ms_speaker_flip():
    cues = _cues(
        ["They're much better quality than most other", "socks that I've found."],
        speakers=["S1", "S2"],
    )
    shift = 300
    cues[1].start_time += shift
    cues[1].end_time += shift
    for word in cues[1].words:
        word.start_time += shift
        word.end_time += shift

    repaired = normalize_boundaries(cues)

    assert len(repaired) == 1
    assert "most other socks" in repaired[0].text


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
    cues = _cues(["They know somebody who works at the plant, plants, or they work there."])

    repaired = normalize_boundaries(cues)

    assert repaired[0].text == ("They know somebody who works at the plants, or they work there.")


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


def test_normalizer_merges_what_use_for_construction_from_reported_failure():
    cues = _cues(
        [
            "I think 90% of what you're going to use",
            "this truck for, like the steering rack, is a good thing",
        ]
    )

    repaired = normalize_boundaries(cues)

    assert len(repaired) == 1
    assert repaired[0].text == (
        "I think 90% of what you're going to use this truck for, like the steering rack, "
        "is a good thing"
    )


def test_normalizer_moves_boundary_after_complete_puny_rpm_gauge_phrase():
    cues = _cues(
        [
            "And then if you're having trouble trying to follow this little puny",
            (
                "RPM gauge over here on the left, you can actually put this into Porsche "
                "911 Performance View Mode."
            ),
        ]
    )

    repaired = normalize_boundaries(cues)

    assert len(repaired) == 2
    assert repaired[0].text == (
        "And then if you're having trouble trying to follow this little puny RPM gauge "
        "over here on the left,"
    )
    assert repaired[1].text == (
        "you can actually put this into Porsche 911 Performance View Mode."
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


def test_assessment_keeps_adjacent_proper_name_tokens_together():
    assessment = assess_english_boundary(
        "A lot of them predicted the rise of a figure like Donald",
        "Trump that would use populist messaging",
    )

    assert assessment.unstable
    assert "proper name split between adjacent tokens" in assessment.reasons


def test_normalizer_moves_boundary_out_of_a_proper_name():
    cues = _cues(
        [
            "A lot of them predicted the rise of a figure like Donald",
            "Trump that would use populist messaging and succeed in politics.",
        ]
    )

    repaired = normalize_boundaries(cues)

    assert all(
        not (left.text.endswith("Donald") and right.text.startswith("Trump"))
        for left, right in zip(repaired, repaired[1:])
    )
    assert "Donald Trump" in " ".join(cue.text for cue in repaired)


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        ("the way I grew up and my past", "experiences shaped me", "past experiences"),
        ("if that's the only", "way people take in information", "only way"),
        ("the broader phenomenon occurring right", "now is that text", "right now"),
    ],
)
def test_assessment_keeps_translation_sensitive_lexical_units_together(
    left,
    right,
    reason,
):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert any(reason in item for item in assessment.reasons)


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "He contradicts himself as though there's no",
            "record of his previous words.",
            "negative existential separated from its complement",
        ),
        (
            "A figure like Donald Trump that would, you know,",
            "use populist messaging.",
            "incomplete predicate before trailing discourse filler",
        ),
        (
            "In many ways, it's actually,",
            "sometimes a more effective way to get information.",
            "predicate separated after a discourse modifier",
        ),
        (
            "time spent devoting and kind of getting,",
            "really connected to a complex text.",
            "open complement after 'getting'",
        ),
    ],
)
def test_assessment_looks_past_discourse_fillers_and_open_complements(
    left,
    right,
    reason,
):
    assessment = assess_english_boundary(left, right)

    assert assessment.risk >= 34
    assert reason in assessment.reasons


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "One person described how when",
            "you are trying to set something up",
            "dangling function word 'when'",
        ),
        (
            "It shaped my past experiences that it definitely",
            "is something I value.",
            "dangling modifier 'definitely'",
        ),
        (
            "I am filling in for Sean. Today,",
            "I am talking with Rose.",
            "sentence-opening time marker belongs to the next cue",
        ),
    ],
)
def test_assessment_keeps_dialogue_time_and_modifier_units_together(left, right, reason):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert reason in assessment.reasons


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "text is no longer kind of the main",
            "way that people transmit information",
            "main way",
        ),
        (
            "it really matters",
            "what we're choosing over and over again",
            "matters what",
        ),
        (
            "there will be more emphasis on grabbing",
            "attention in the first 10 seconds",
            "grabbing attention",
        ),
        (
            "we have seen books and text",
            "become much more widely available",
            "coordinated subject separated from its predicate",
        ),
        (
            "I think at the same time,",
            "it's interesting that books are more available",
            "discourse frame separated from its following clause",
        ),
    ],
)
def test_assessment_keeps_multispeaker_translation_units_together(left, right, reason):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert any(reason in item for item in assessment.reasons)


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "And so I think that we see, you know,",
            "a lot of changes in schools",
            "transitive predicate separated before trailing discourse filler",
        ),
        (
            "In many ways, it's actually, you know,",
            "sometimes a more effective way",
            "predicate separated after a discourse modifier",
        ),
    ],
)
def test_assessment_looks_through_trailing_fillers_for_translation_units(
    left, right, reason
):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert reason in assessment.reasons
