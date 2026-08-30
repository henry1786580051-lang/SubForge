import pytest

from subforge.core.asr.asr_data import ASRDataSeg, ASRWord
from subforge.core.split.boundary import (
    assess_english_boundary,
    normalize_boundaries,
)
from subforge.core.split.boundary_registry import (
    BOUNDARY_SCORE_RULES,
    BoundaryScoreSeverity,
    boundary_score_rule,
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


def test_numeric_boundary_registry_has_unique_stable_ids() -> None:
    assert len({rule.rule_id for rule in BOUNDARY_SCORE_RULES}) == len(BOUNDARY_SCORE_RULES)
    assert all(boundary_score_rule(rule.rule_id) is rule for rule in BOUNDARY_SCORE_RULES)
    assert all(rule.explanation_template for rule in BOUNDARY_SCORE_RULES)
    assert all(rule.source_languages == ("en",) for rule in BOUNDARY_SCORE_RULES)
    assert boundary_score_rule(
        "split.boundary.english.numeric.calendar_month_year"
    ).severity == BoundaryScoreSeverity.ERROR
    assert boundary_score_rule(
        "split.boundary.english.grammar.attributive_or_comparative_modifier"
    ).severity == BoundaryScoreSeverity.WARNING


@pytest.mark.parametrize(
    ("left", "right", "rule_id", "weight"),
    [
        (
            "The meeting is in March",
            "2026 before the opening.",
            "split.boundary.english.numeric.calendar_month_year",
            42,
        ),
        (
            "The tunnel is between five",
            "and six kilometres long.",
            "split.boundary.english.numeric.range_conjunction",
            38,
        ),
        (
            "The wheel is about",
            "500 millimetres wide.",
            "split.boundary.english.numeric.approximate_magnitude",
            38,
        ),
    ],
)
def test_numeric_boundary_rules_emit_observable_legacy_equivalent_contributions(
    left: str,
    right: str,
    rule_id: str,
    weight: int,
) -> None:
    assessment = assess_english_boundary(left, right)
    contribution = next(item for item in assessment.contributions if item.rule_id == rule_id)

    assert contribution.weight == weight
    assert contribution.reason in assessment.reasons
    assert assessment.registered_risk >= weight
    assert assessment.unregistered_risk == assessment.risk - assessment.registered_risk


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


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "The additional runway at Gatwick would have cost",
            "less, but the facility handles fewer flights.",
            "scalar predicate separated from its comparative complement",
        ),
        (
            "because we're in for some turbulent",
            "British politics.",
            "determiner and adjective separated from their head noun",
        ),
        (
            "That's up from 480,000 flights and 84.5 million passengers",
            "today.",
            "sentence-final temporal adverb separated from its clause",
        ),
    ],
)
def test_assessment_catches_airport_sample_dependencies(left, right, reason):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert reason in assessment.reasons


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (
            "the sphere could move from a one-off megaproject to a truly",
            "global network of venues.",
        ),
        ("This is a very", "good car."),
        ("That was an exceptionally", "difficult decision."),
    ],
)
def test_assessment_keeps_attributive_degree_modifier_with_its_head(left, right):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert "determiner and degree modifier separated from their head noun" in assessment.reasons


def test_normalizer_repairs_attributive_degree_modifier_boundary_without_word_loss():
    source = [
        "In other words, the sphere could move from a one-off or even two-off megaproject to a truly",
        "global network of venues.",
        "That would change the economics.",
    ]

    repaired = normalize_boundaries(_cues(source))

    assert " ".join(segment.text for segment in repaired).split() == " ".join(source).split()
    assert "a truly | global network" not in " | ".join(segment.text for segment in repaired)
    assert all(len(segment.text.split()) <= 22 for segment in repaired)


def test_normalizer_repairs_airport_sample_dependencies_across_label_glitches():
    cues = _cues(
        [
            "The additional runway at Gatwick would have cost",
            "less, but the facility handles fewer flights.",
            "because we're in for some turbulent",
            "British politics. Please return to your seats.",
        ],
        speakers=["S1", "S1", "S3", "S1"],
    )

    repaired = normalize_boundaries(cues)
    combined = " | ".join(segment.text for segment in repaired)

    assert "cost | less" not in combined
    assert "turbulent | British" not in combined
    assert "cost less" in combined
    assert "turbulent British politics" in combined


def test_assessment_accepts_complete_scalar_complement_before_contrast():
    assessment = assess_english_boundary(
        "The additional runway at Gatwick would have cost less,",
        "but the facility handles fewer flights.",
    )

    assert not assessment.unstable


def test_assessment_does_not_treat_time_noun_as_dangling_adjective():
    assessment = assess_english_boundary(
        "That's going through a consultation period at the moment",
        "and it's hoped it'll become final policy.",
    )

    assert "determiner and adjective separated from their head noun" not in assessment.reasons


def test_normalizer_attaches_isolated_trailing_today_across_label_glitch():
    cues = _cues(
        [
            "That's up from 480,000 flights and 84.5 million passengers",
            "today.",
            "The idea starts here.",
        ],
        speakers=["S1", "S3", "S1"],
    )
    shift = 3_000 - (cues[1].start_time - cues[0].end_time)
    for cue in cues[1:]:
        cue.start_time += shift
        cue.end_time += shift
        for word in cue.words:
            word.start_time += shift
            word.end_time += shift

    repaired = normalize_boundaries(cues)

    assert repaired[0].text.endswith("passengers today.")
    assert repaired[1].text == "The idea starts here."


def test_assessment_keeps_rev_matching_technical_term_together():
    assessment = assess_english_boundary(
        "which is their fancy way of saying rev",
        "matching which helps in city driving",
    )

    assert assessment.unstable
    assert "split lexical unit 'rev matching'" in assessment.reasons


def test_assessment_keeps_power_steering_technical_term_together():
    assessment = assess_english_boundary(
        "but look at this under the Honda badge power",
        "steering this was a big deal in 1982",
    )

    assert assessment.unstable
    assert "split lexical unit 'power steering'" in assessment.reasons


def test_assessment_keeps_calendar_month_with_its_year():
    assessment = assess_english_boundary(
        "projects cancelled in the year leading up to January",
        "2026, basically we're now looking at a mixed-use world.",
    )

    assert assessment.unstable
    assert "calendar month separated from its year" in assessment.reasons


def test_assessment_keeps_irregular_attributive_adjective_with_head_noun():
    assessment = assess_english_boundary(
        "the buildings are responding to a very different",
        "world generating very different demands.",
    )

    assert assessment.unstable
    assert "dangling attributive 'different'" in assessment.reasons


def test_assessment_moves_sentence_adverb_to_its_following_clause():
    assessment = assess_english_boundary(
        "projects were cancelled by January 2026, basically",
        "we're now looking at a mixed-use world.",
    )

    assert assessment.unstable
    assert "sentence adverb separated from its finite predicate" in assessment.reasons


def test_assessment_keeps_auxiliary_adverb_with_progressive_predicate():
    assessment = assess_english_boundary(
        "projects were cancelled by January 2026, basically we're now",
        "looking at a mixed-use world.",
    )

    assert assessment.unstable
    assert "auxiliary and sentence adverb separated from their predicate" in assessment.reasons


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("projects were cancelled in the year leading", "up to January 2026"),
        ("projects were cancelled in the year leading up", "to January 2026"),
    ],
)
def test_assessment_keeps_leading_up_to_phrase_together(left, right):
    assert assess_english_boundary(left, right).unstable


def test_assessment_keeps_year_with_its_defining_participial_phrase():
    assessment = assess_english_boundary(
        "18 projects were cancelled in the city in the year",
        "leading up to January 2026.",
    )

    assert assessment.unstable
    assert "time frame separated from its defining participial phrase" in assessment.reasons


def test_assessment_keeps_separated_take_away_construction_together():
    assessment = assess_english_boundary(
        "because this car sort of takes that ability",
        "away from you and encourages you to push it",
    )

    assert assessment.unstable
    assert "split phrasal construction 'take ... away'" in assessment.reasons


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "the terminal finally opened after experiencing",
            "some turbulence of its own",
            "prepositional gerund separated from its complement",
        ),
        (
            "the terminal finally opened after",
            "experiencing some turbulence of its own",
            "preposition separated from its gerund phrase",
        ),
        (
            "the country was divided into East",
            "and West after the war",
            "paired directional names split at conjunction",
        ),
        (
            "the terminal is finally up",
            "and running after years of work",
            "fixed state phrase split inside 'up and running'",
        ),
        (
            "the question",
            "is whether the project succeeded",
            "trailing noun subject separated from its finite predicate",
        ),
        (
            "the terminal finally",
            "up and running after years of work",
            "aspect marker split from 'up and running'",
        ),
        (
            "the terminal",
            "finally up and running after years of work",
            "state predicate 'up and running' separated from its subject",
        ),
    ],
)
def test_assessment_keeps_generic_dependency_pairs_together(left, right, reason):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert reason in assessment.reasons


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "Dubai itself now",
            "looks a little more uncertain.",
            "sentence adverb separated from its finite predicate",
        ),
        (
            "a very thick,",
            "continuous reinforced concrete core",
            "dangling modifier 'thick'",
        ),
        (
            "your very own luxury",
            "apartment in a super slender tower",
            "dangling modifier 'luxury'",
        ),
        (
            "while it wasn't the first",
            "Super Slender Skyscraper ever built",
            "determiner separated from its 'first' head noun",
        ),
    ],
)
def test_assessment_catches_general_architecture_dependency_boundaries(left, right, reason):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert reason in assessment.reasons


def test_assessment_keeps_complete_now_clause_before_a_new_sentence():
    assessment = assess_english_boundary(
        "We need to leave now",
        "Tomorrow will be too late.",
    )

    assert not assessment.unstable


def test_assessment_accepts_complete_of_its_own_phrase():
    assessment = assess_english_boundary(
        "the terminal experienced some turbulence of its own,",
        "the question is whether it succeeded",
    )

    assert not assessment.unstable


def test_assessment_keeps_revised_component_with_its_noun():
    assessment = assess_english_boundary(
        "Why don't we show you this newly revised",
        "nine speaker JBL sound system",
    )

    assert assessment.unstable
    assert "dangling modifier 'revised'" in assessment.reasons


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "Do you draw a meaningful distinction between constraints on the one hand",
            "and impediments on the other?",
            "paired contrast frame split before its counterpart",
        ),
        (
            "which means I'm going to be doing a lot of long distance",
            "walking with a small child",
            "dangling attributive 'distance'",
        ),
        (
            "he is the same person in his personal",
            "life as in his work",
            "dangling attributive 'personal'",
        ),
        (
            "he is the same person in his personal life",
            "as in his work",
            "comparison frame separated from its counterpart",
        ),
        (
            "These ideas are useless",
            "if you do not find ways to apply them",
            "condition separated from the predicate it qualifies",
        ),
        (
            "if you do not find ways to concretize",
            "them",
            "transitive predicate separated from its pronoun object",
        ),
        (
            "these ideas are useless if you don't find",
            "ways to concretize them",
            "transitive predicate separated from its object",
        ),
        (
            "ways to concretize them in your own",
            "life so they create change",
            "dangling attributive 'own'",
        ),
    ],
)
def test_assessment_catches_generic_modifier_and_condition_dependencies(
    left,
    right,
    reason,
):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert reason in assessment.reasons


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
def test_assessment_rejects_dialogue_translation_sensitive_boundaries(left, right, reason):
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


@pytest.mark.parametrize(
    ("left", "right", "modifier"),
    [
        ("which is an interesting", "choice for this cabin", "interesting"),
        ("it has a medium", "firmness to it", "medium"),
    ],
)
def test_assessment_detects_common_adjective_noun_splits(left, right, modifier):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert f"dangling modifier '{modifier}'" in assessment.reasons


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        ("I don't think you're sacrificing a whole", "lot here", "split lexical unit"),
        ("the suspension has medium", "firmness", "split lexical unit"),
        ("the vents are up", "here above the screen", "directional phrase split"),
        (
            "it has medium firmness, which I think",
            "is unnecessary for this car",
            "parenthetical opinion separated",
        ),
        (
            "something that would ignite excitement",
            "within me while driving",
            "dependent phrase beginning with 'within'",
        ),
    ],
)
def test_assessment_detects_general_dependency_boundaries(left, right, reason):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert any(reason in item for item in assessment.reasons)


@pytest.mark.parametrize(
    "texts",
    [
        [
            "I don't think you're sacrificing a whole",
            "lot with this roof design.",
        ],
        [
            "the climate vents are up",
            "here which is an interesting choice for this cabin.",
        ],
        [
            "It has a medium firmness to it, which I think",
            "is unnecessary for this car.",
        ],
        [
            "something that would ignite some excitement",
            "within me while driving this.",
        ],
    ],
)
def test_normalizer_repairs_general_dependency_boundaries(texts):
    repaired = normalize_boundaries(_cues(texts), soft_max_words=18, hard_max_words=22)

    assert all(
        not assess_english_boundary(left.text, right.text).unstable
        for left, right in zip(repaired, repaired[1:])
    )


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


def test_assessment_rejects_auxiliary_adverb_split_from_participle():
    assessment = assess_english_boundary(
        "and college graduates and women have historically",
        "been the groups that read the most and declined as well.",
    )

    assert assessment.unstable
    assert "auxiliary phrase separated from its participle" in assessment.reasons


def test_assessment_rejects_numeric_compound_modifier_before_product_name():
    assessment = assess_english_boundary(
        "why don't we show you this newly revised nine speaker",
        "jbl with the old school sound system test song",
    )

    assert assessment.unstable
    assert "numeric compound modifier separated from its head noun" in assessment.reasons


def test_assessment_rejects_compound_member_split_in_reported_subject():
    assessment = assess_english_boundary(
        "I note in the piece that retirees and college",
        "graduates and women have historically been the groups that read the most.",
    )

    assert assessment.unstable
    assert "compound member split inside a coordinated reported subject" in assessment.reasons


def test_normalizer_repairs_auxiliary_adverb_participle_boundary():
    cues = _cues(
        [
            "I note in the piece that retirees",
            "and college graduates and women have historically",
            "been the groups that read the most and they have seen collapses as well.",
        ]
    )

    repaired = normalize_boundaries(cues, soft_max_words=16, hard_max_words=20)

    assert len(repaired) == 2
    assert repaired[0].text.endswith("retirees and college graduates and women")
    assert repaired[1].text.startswith("have historically been")
    assert "college graduates" in " ".join(cue.text for cue in repaired)
    assert all(
        "auxiliary phrase separated from its participle"
        not in assess_english_boundary(left.text, right.text).reasons
        for left, right in zip(repaired, repaired[1:])
    )
    assert "have historically been" in " ".join(cue.text for cue in repaired)


def test_assessment_allows_completed_superlative_before_coordinated_predicate():
    assessment = assess_english_boundary(
        "have historically been the groups that read the most",
        "and they have seen declines as well.",
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


def test_normalizer_keeps_calendar_month_and_year_in_one_cue():
    cues = _cues(
        [
            "and with 18 condo projects cancelled in the city in the year leading up to January",
            "2026, basically we're now looking at a mixed-use world.",
        ]
    )

    repaired = normalize_boundaries(cues)

    assert any("January 2026" in cue.text for cue in repaired)
    assert all(not cue.text.endswith("January") for cue in repaired)
    assert all(not cue.text.startswith("2026") for cue in repaired)


def test_normalizer_keeps_attributive_head_and_rehomes_sentence_adverb():
    cues = _cues(
        [
            "the buildings are responding to a very different",
            "world generating very different demands.",
            "projects were cancelled by January 2026, basically",
            "we're now looking at a mixed-use world.",
        ]
    )

    repaired = normalize_boundaries(cues)

    assert any("different world" in cue.text for cue in repaired)
    assert all(not cue.text.rstrip(" ,").endswith("basically") for cue in repaired)
    assert any("basically we're" in cue.text for cue in repaired)


def test_normalizer_keeps_long_leading_up_to_date_clause_readable():
    cues = _cues(
        [
            "So, with demand for offices having slipped dramatically,",
            "and with 18 condo projects cancelled in the city in the year leading",
            "up to January 2026, basically we're now looking at a mixed-use world.",
        ]
    )

    repaired = normalize_boundaries(cues)

    assert all(
        not cue.text.rstrip(" ,").endswith(("leading", "leading up", "basically"))
        for cue in repaired
    )
    assert all(not cue.text.startswith(("up to", "to January", "2026")) for cue in repaired)
    assert any("leading up to January 2026" in cue.text for cue in repaired)


def test_normalizer_keeps_full_year_leading_up_to_frame_together():
    cues = _cues(
        [
            "and with 18 condo projects cancelled in the city in the year",
            "leading up to January 2026, basically we're now looking at a mixed-use world.",
        ]
    )

    repaired = normalize_boundaries(cues)

    assert all(not cue.text.rstrip(" ,").endswith("in the year") for cue in repaired)
    assert any("in the year leading up to January 2026" in cue.text for cue in repaired)


def test_normalizer_moves_auxiliary_adverb_to_progressive_predicate():
    cues = _cues(
        [
            "and with 18 condo projects cancelled in the city in the year leading up to "
            "January 2026, basically we're now",
            "looking at a mixed-use world.",
        ]
    )

    repaired = normalize_boundaries(cues)

    assert all(not cue.text.rstrip(" ,").endswith("we're now") for cue in repaired)
    assert any("we're now looking" in cue.text for cue in repaired)


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


def test_normalizer_repairs_modifier_noun_across_tiny_diarization_flip():
    cues = _cues(
        ["It's a critical", "aspect of the design."],
        speakers=["S1", "S2"],
    )

    repaired = normalize_boundaries(cues)

    assert len(repaired) == 1
    assert repaired[0].text == "It's a critical aspect of the design."


def test_normalizer_merges_measurement_with_comparative_relation():
    cues = _cues(
        ["which is around 400 metres", "closer to sea level than Addis Ababa."],
    )

    repaired = normalize_boundaries(cues)

    assert len(repaired) == 1
    assert "400 metres closer to sea level" in repaired[0].text


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "established hubs welcoming",
            "20 to 30 million passengers each year",
            "transitive participle separated from its quantified object",
        ),
        (
            "it needs to substantially",
            "redevelop its existing airports",
            "infinitive adverb separated from its predicate",
        ),
        (
            "the hub welcomes more than 20 million passengers",
            "each year",
            "frequency phrase separated from its quantified statement",
        ),
        (
            "Which brings us to Bishoftu. About 35 kilometres away,",
            "the new site is less densely populated",
            "distance modifier separated from its location noun",
        ),
        (
            "airline capacity rose by 6.7%, which is around",
            "24 million seats",
            "approximate magnitude separated from its numeric value",
        ),
        (
            "those terminals that will hopefully",
            "process millions of passengers",
            "modal adverb separated from its predicate",
        ),
        (
            "but for a plane hurtling down the runway,",
            "a lack of density makes takeoff harder",
            "contrastive prepositional frame separated from its main clause",
        ),
    ],
)
def test_assessment_catches_quantified_and_location_dependencies(left, right, reason):
    assert reason in assess_english_boundary(left, right).reasons


def test_normalizer_repairs_modifier_noun_across_363ms_diarization_flip():
    cues = _cues(
        ["those stats are on a serious", "upward trajectory."],
        speakers=["S1", "S2"],
    )
    shift = 343
    cues[1].start_time += shift
    cues[1].end_time += shift
    for word in cues[1].words:
        word.start_time += shift
        word.end_time += shift

    repaired = normalize_boundaries(cues)

    assert len(repaired) == 1
    assert "serious upward trajectory" in repaired[0].text


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
    assert repaired[1].text == ("you can actually put this into Porsche 911 Performance View Mode.")


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
        ("It is nothing to write home", "about, but it is decent.", "home about"),
        ("It is nothing to write", "home about, but it is decent.", "write home"),
        ("Test the six-speaker sound", "system in this car.", "sound system"),
        ("None of the bass", "systems are exceptional.", "bass systems"),
        ("They can tune it and put", "power wherever it is needed.", "open complement after 'put'"),
    ],
)
def test_assessment_keeps_fixed_idiom_and_open_put_complement_together(left, right, reason):
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


def test_assessment_keeps_coordinated_complement_subject_with_have_predicate():
    assessment = assess_english_boundary(
        "I note in the piece that retirees and college graduates and women",
        "have historically been the groups that read the most.",
    )

    assert assessment.unstable
    assert "coordinated subject separated from its predicate" in assessment.reasons


def test_assessment_keeps_case_studies_compound_together():
    assessment = assess_english_boundary(
        "And he wrote about these kind of case",
        "studies that this neuropsychologist had done in the Soviet Union.",
    )

    assert assessment.unstable
    assert "split lexical unit 'case studies'" in assessment.reasons


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
def test_assessment_looks_through_trailing_fillers_for_translation_units(left, right, reason):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert reason in assessment.reasons


def test_assessment_detects_complement_before_trailing_filler():
    assessment = assess_english_boundary(
        "And so it is interesting that, you know,",
        "reading and writing require continued effort.",
    )

    assert assessment.unstable
    assert "dangling function word 'that' before trailing discourse filler" in assessment.reasons


def test_assessment_detects_relative_subject_split_before_predicate():
    assessment = assess_english_boundary(
        "This was uncomfortable to people who you know previously",
        "had a monopoly on sharing information.",
    )

    assert assessment.unstable
    assert "relative-clause subject separated from its predicate" in assessment.reasons


def test_assessment_detects_open_expect_to_always_continuation():
    assessment = assess_english_boundary(
        "We could just kind of expect to always",
        "continue without effort.",
    )

    assert assessment.unstable
    assert "open continuation after 'expect to always'" in assessment.reasons


def test_assessment_keeps_important_role_together():
    assessment = assess_english_boundary(
        "Print played a really important",
        "role in the revolution.",
    )

    assert assessment.unstable
    assert "dangling modifier 'important'" in assessment.reasons


def test_assessment_keeps_short_noun_subject_with_predicate():
    assessment = assess_english_boundary(
        "The founding fathers",
        "used newspapers to spread their message.",
    )

    assert assessment.unstable
    assert "short noun subject separated from its predicate" in assessment.reasons


def test_assessment_keeps_connective_clause_subject_with_copular_predicate():
    assessment = assess_english_boundary(
        "Obviously we have four-corner air suspension, so ride quality",
        "is good. The road here is smooth.",
    )

    assert assessment.unstable
    assert "clause-final subject separated from its finite predicate" in assessment.reasons


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "people adjusted as more",
            "and more people gained access",
            "repeated degree phrase split inside 'more and more'",
        ),
        (
            "they can have any value",
            "judgment on the change that they want",
            "split lexical unit 'value judgment'",
        ),
        (
            "there may be advantages I wanted to point",
            "out to the audience",
            "split lexical unit 'point out'",
        ),
        (
            "print played an important role the founding fathers",
            "used newspapers and pamphlets",
            "trailing noun subject separated from its finite predicate",
        ),
        (
            "we saw in the revolution that print",
            "played a central role",
            "trailing noun subject separated from its finite predicate",
        ),
        (
            "changes that would almost further",
            "people reading less",
            "dangling modifier 'further'",
        ),
    ],
)
def test_assessment_catches_full_run_dependency_failures(left, right, reason):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert reason in assessment.reasons


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "we have definitely seen,",
            "tablets and laptops entering classrooms",
            "perfect reporting predicate separated after its adverb",
        ),
        (
            "we often think of literacy",
            "and reading as something inevitable",
            "coordinated noun subject split before its shared predicate",
        ),
        (
            "we have seen books",
            "and text become more widely available",
            "coordinated noun subject split before its shared predicate",
        ),
        (
            "tablets, Chromebooks and laptops,",
            "entering the classroom much more",
            "coordinated noun list separated from its progressive predicate",
        ),
        (
            "the founders thought that print and newspapers",
            "and were crucial to an informed public",
            "reported subject separated from its predicate",
        ),
    ],
)
def test_assessment_catches_remaining_full_run_structures(left, right, reason):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert reason in assessment.reasons


def test_assessment_keeps_so_many_degree_phrase_together():
    assessment = assess_english_boundary(
        "It is something so",
        "many people take for granted.",
    )

    assert assessment.unstable
    assert "degree phrase split inside 'so many'" in assessment.reasons


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "we've already kind of seen them, both",
            "responding to this and making changes",
            "dangling modifier 'both'",
        ),
        (
            "particularly towards what,",
            "college is for now for people",
            "interrogative complement separated after 'what'",
        ),
        (
            "And so, you know,",
            "but I do think there are lessons from that time",
            "standalone discourse bridge belongs to the following clause",
        ),
        (
            "the point that I was trying to make",
            "was that repeated choices matter",
            "reporting frame separated from its copular content",
        ),
        (
            "I cannot say whether I think",
            "which way this will go",
            "embedded question frame separated from its complement",
        ),
    ],
)
def test_assessment_keeps_discourse_and_reporting_units_together(left, right, reason):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert reason in assessment.reasons


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "And we might be reading",
            "more words than ever",
            "progressive predicate separated from its object",
        ),
        (
            "when you think about the number of like",
            "emails and text messages",
            "quantifying phrase separated from its noun",
        ),
        (
            "one thing that I think is interesting is, with schools,",
            "we've already seen them responding",
            "topic frame separated from its following predicate",
        ),
        (
            "But at the same time, you know,",
            "it was not a purely rosy picture",
            "standalone contrast frame belongs to the following clause",
        ),
        (
            "I think as we continue to see,",
            "new inventions proliferating",
            "reporting predicate separated from its content",
        ),
    ],
)
def test_assessment_keeps_additional_translation_sensitive_units_together(left, right, reason):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert reason in assessment.reasons


def test_assessment_looks_past_leading_filler_in_embedded_question():
    assessment = assess_english_boundary(
        "I cannot say whether I think,",
        "you know, which way this will go.",
    )

    assert assessment.unstable
    assert "embedded question frame separated from its complement" in assessment.reasons


def test_assessment_keeps_perfect_reporting_predicate_with_content():
    assessment = assess_english_boundary(
        "At the same time that we've seen,",
        "books leaving the classroom.",
    )

    assert assessment.unstable
    assert "perfect reporting predicate separated from its content" in assessment.reasons


def test_normalizer_rehomes_sentence_fragment_after_internal_terminal():
    words = [
        ASRWord("current", 0, 100, speaker_id="S1"),
        ASRWord("environment.", 120, 300, speaker_id="S1"),
        ASRWord("He", 400, 500, speaker_id="S1"),
        ASRWord("speaks", 520, 700, speaker_id="S1"),
        ASRWord("and", 720, 800, speaker_id="S1"),
        ASRWord("contradicts", 820, 1_100, speaker_id="S1"),
        ASRWord("himself.", 1_120, 1_400, speaker_id="S1"),
    ]
    segments = [
        ASRDataSeg.from_segments(
            [
                ASRDataSeg(
                    word.text,
                    word.start_time,
                    word.end_time,
                    speaker_id=word.speaker_id,
                    words=[word],
                    timestamp_granularity="word",
                )
                for word in words[:4]
            ],
            text="current environment. He speaks",
            speaker_id="S1",
        ),
        ASRDataSeg.from_segments(
            [
                ASRDataSeg(
                    word.text,
                    word.start_time,
                    word.end_time,
                    speaker_id=word.speaker_id,
                    words=[word],
                    timestamp_granularity="word",
                )
                for word in words[4:]
            ],
            text="and contradicts himself.",
            speaker_id="S1",
        ),
    ]

    normalized = normalize_boundaries(segments)

    assert [segment.text for segment in normalized] == [
        "current environment.",
        "He speaks and contradicts himself.",
    ]


def test_normalizer_repairs_open_linking_complement_from_reported_failure():
    segments = _cues(
        [
            "So while CERN may have become",
            "one of the world's best-known scientific organisations thanks to the LHC,",
            "it was Brookhaven where most of the early breakthroughs were made.",
        ]
    )

    normalized = normalize_boundaries(segments)

    assert [segment.text for segment in normalized] == [
        "So while CERN may have become one of the world's best-known scientific organisations thanks to the LHC,",
        "it was Brookhaven where most of the early breakthroughs were made.",
    ]


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "they reach temperatures that were around 250,000",
            "times higher than the core of our sun.",
            "numeric value separated from its multiplier or unit",
        ),
        (
            "let's get back to the world of particle",
            "physics.",
            "single-word completion stranded in the next subtitle",
        ),
        (
            "The main",
            "one being how the quarks and gluons are structured.",
            "dangling modifier 'main'",
        ),
    ],
)
def test_assessment_catches_general_open_completions(left, right, reason):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert reason in assessment.reasons


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "But that's the beauty of the collection. You never",
            "know what you're going to find next.",
            "subject and adverb separated from its predicate",
        ),
        (
            "Next is another car that's probably never going",
            "anywhere after this season.",
            "progressive predicate separated from its complement",
        ),
        (
            "I need a better solution for actually",
            "holding all of these keys.",
            "adverb separated from its gerund",
        ),
        (
            "Next up is another car that's",
            "probably never going anywhere.",
            "subject and auxiliary stranded at 'that's'",
        ),
        (
            "I have never had any previous",
            "owners contact me.",
            "dangling attributive 'previous'",
        ),
        (
            "I have a plate that says",
            "Big Sexy on the front.",
            "reporting predicate separated from its quoted object",
        ),
        (
            "I",
            "Think the biggest update is the broken power top.",
            "standalone subject separated from its predicate",
        ),
    ],
)
def test_assessment_catches_predicate_dependencies_after_adverbs(left, right, reason):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert reason in assessment.reasons


def test_assessment_keeps_completed_progressive_before_new_sentence():
    assessment = assess_english_boundary(
        "I've been driving",
        "And the car remains reliable.",
    )

    assert "progressive predicate separated from its complement" not in assessment.reasons


def test_assessment_does_not_treat_sentence_initial_just_as_a_name():
    assessment = assess_english_boundary(
        "I went to get something for Emily",
        "Just so I could drive the truck.",
    )

    assert "proper name split between adjacent tokens" not in assessment.reasons


def test_normalizer_rejoins_sentence_suffix_across_short_speaker_flip():
    left_words = [
        ASRWord(token, index * 120, index * 120 + 100, speaker_id="S1")
        for index, token in enumerate("But that's the beauty of the collection. You never".split())
    ]
    right_words = [
        ASRWord(
            token,
            left_words[-1].end_time + 20 + index * 120,
            left_words[-1].end_time + 120 + index * 120,
            speaker_id="S2",
        )
        for index, token in enumerate("know what you're going to find next.".split())
    ]
    segments = [
        ASRDataSeg(
            " ".join(word.text for word in left_words),
            left_words[0].start_time,
            left_words[-1].end_time,
            speaker_id="S1",
            words=left_words,
            timestamp_granularity="sentence",
        ),
        ASRDataSeg(
            " ".join(word.text for word in right_words),
            right_words[0].start_time,
            right_words[-1].end_time,
            speaker_id="S2",
            words=right_words,
            timestamp_granularity="sentence",
        ),
    ]

    normalized = normalize_boundaries(segments)

    assert [segment.text for segment in normalized] == [
        "But that's the beauty of the collection.",
        "You never know what you're going to find next.",
    ]
    assert [segment.speaker_id for segment in normalized] == ["S1", "S1"]


def test_normalizer_moves_progressive_relative_clause_to_readable_cue():
    left_tokens = "Next up is another car that's probably never going".split()
    right_tokens = (
        "anywhere and also It also has a very special first video attached to it,".split()
    )
    left_words = [
        ASRWord(token, index * 120, index * 120 + 100, speaker_id="S1")
        for index, token in enumerate(left_tokens)
    ]
    right_words = [
        ASRWord(
            token,
            left_words[-1].end_time + 60 + index * 120,
            left_words[-1].end_time + 160 + index * 120,
            speaker_id="S1",
        )
        for index, token in enumerate(right_tokens)
    ]
    segments = [
        ASRDataSeg(
            " ".join(left_tokens),
            left_words[0].start_time,
            left_words[-1].end_time,
            speaker_id="S1",
            words=left_words,
            timestamp_granularity="sentence",
        ),
        ASRDataSeg(
            " ".join(right_tokens),
            right_words[0].start_time,
            right_words[-1].end_time,
            speaker_id="S1",
            words=right_words,
            timestamp_granularity="sentence",
        ),
    ]

    normalized = normalize_boundaries(segments)

    assert [segment.text for segment in normalized] == [
        "Next up is another car",
        "that's probably never going anywhere and also It also has a very special first "
        "video attached to it,",
    ]


def test_normalizer_preserves_reply_after_cross_speaker_question_completion():
    left_words = [
        ASRWord("do", 0, 80, speaker_id="S1"),
        ASRWord("you", 100, 240, speaker_id="S1"),
        ASRWord("still", 260, 360, speaker_id="S1"),
    ]
    right_words = [
        ASRWord("love", 380, 460, speaker_id="S2"),
        ASRWord("her", 480, 560, speaker_id="S2"),
        ASRWord("I", 600, 720, speaker_id="S2"),
        ASRWord("love", 740, 900, speaker_id="S2"),
        ASRWord("this", 920, 1080, speaker_id="S2"),
        ASRWord("car.", 1100, 1500, speaker_id="S2"),
    ]
    segments = [
        ASRDataSeg(
            "do you still",
            0,
            360,
            speaker_id="S1",
            words=left_words,
            timestamp_granularity="sentence",
        ),
        ASRDataSeg(
            "love her I love this car.",
            380,
            1500,
            speaker_id="S2",
            words=right_words,
            timestamp_granularity="sentence",
        ),
    ]

    normalized = normalize_boundaries(segments)

    assert [(segment.speaker_id, segment.text) for segment in normalized] == [
        ("S1", "do you still love her"),
        ("S2", "I love this car."),
    ]


def test_assessment_rejects_capitalized_standalone_of_phrase():
    assessment = assess_english_boundary(
        "that you guys are seeing from the thumbnail.",
        "Of this video of driving every car in my collection.",
    )

    assert assessment.unstable
    assert "standalone 'of' phrase separated from its governing noun" in assessment.reasons


def test_normalizer_returns_noun_before_short_cross_speaker_reply():
    normalized = normalize_boundaries(_cues(["A little", "car, yeah."], speakers=["S2", "S1"]))

    assert [(segment.speaker_id, segment.text) for segment in normalized] == [
        ("S2", "A little car,"),
        ("S1", "yeah."),
    ]

    assert normalize_boundaries(normalized) == normalized


def test_normalizer_repairs_single_word_completion_across_short_speaker_flip():
    left_words = [
        ASRWord(token, index * 120, index * 120 + 100, speaker_id="S1")
        for index, token in enumerate("let's get back to the world of particle".split())
    ]
    right_word = ASRWord(
        "physics.",
        left_words[-1].end_time + 300,
        left_words[-1].end_time + 700,
        speaker_id="S2",
    )
    segments = [
        ASRDataSeg(
            "let's get back to the world of particle",
            left_words[0].start_time,
            left_words[-1].end_time,
            speaker_id="S1",
            words=left_words,
            timestamp_granularity="sentence",
        ),
        ASRDataSeg(
            "physics.",
            right_word.start_time,
            right_word.end_time,
            speaker_id="S2",
            words=[right_word],
            timestamp_granularity="sentence",
        ),
    ]

    normalized = normalize_boundaries(segments)

    assert [segment.text for segment in normalized] == [
        "let's get back to the world of particle physics."
    ]


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "And it is ironic to write about how nobody",
            "reads in an almost 9,000 word piece.",
            "dangling subject 'nobody'",
        ),
        (
            "The point that I was",
            "trying to make was that repeated choices matter.",
            "incomplete predicate 'was'",
        ),
        (
            "Obviously, they did not have the exact",
            "same experimental standards that we have now.",
            "dangling modifier 'exact'",
        ),
        (
            "Ong's larger point was that literate",
            "cultures value sustained argumentation.",
            "dangling modifier 'literate'",
        ),
        (
            "but that he is really well suited",
            "and he has figured out how to reach people",
            "context-dependent adjective separated from its continuation",
        ),
        (
            "we have definitely seen tablets and Chromebooks",
            "and laptops entering the classroom much more",
            "final coordinated noun separated with its progressive predicate",
        ),
        (
            "but to hold our digital age",
            "and our print age at the same time",
            "coordinated noun phrase split at conjunction",
        ),
        (
            "not valuing or choosing to access",
            "what is widely available to us",
            "transitive predicate separated from its nominal clause",
        ),
        (
            "And so, you know, but yeah, I think it's, you know,",
            "reading was fundamental to the way I grew up",
            "filler-only discourse frame belongs to the following clause",
        ),
        (
            "And so, but yeah, I think it's reading",
            "was fundamental to the way I grew up",
            "gerund subject separated from its finite predicate",
        ),
        (
            "my past experiences gave me a fondness for it",
            "and wanted to work on this piece",
            "omitted subject separated from its coordinated predicate",
        ),
    ],
)
def test_assessment_catches_full_quality_audit_dependency_chains(left, right, reason):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert reason in assessment.reasons


def test_assessment_keeps_thank_you_together():
    assessment = assess_english_boundary("Thank", "you.")

    assert assessment.unstable
    assert "split lexical unit 'thank you'" in assessment.reasons


def test_assessment_accepts_this_as_a_complete_demonstrative_object():
    assessment = assess_english_boundary(
        "I really don't want to hear this",
        "Emily was riding in the car yesterday.",
    )

    assert assessment.unstable is False


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "This is such a luxury feature that you would never",
            "get in a modern car and see it.",
            "negative auxiliary is separated from its complement",
        ),
        (
            "Once you learn the controls, you would really have no",
            "problem driving it.",
            "negative auxiliary is separated from its complement",
        ),
        (
            "Once you learn the controls, you'd",
            "really have no problem driving it.",
            "subject and auxiliary stranded at 'you'd'",
        ),
    ],
)
def test_assessment_rejects_negative_auxiliary_split_from_complement(left, right, reason):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable is True
    assert reason in assessment.reasons


def test_assessment_allows_complete_visibility_phrase_before_reason_clause():
    assessment = assess_english_boundary(
        "Cars in the 1980s were easy to see out of",
        "because we did not have thick pillars.",
    )

    assert assessment.unstable is False


def test_assessment_treats_steering_as_a_noun_before_new_interjection():
    assessment = assess_english_boundary(
        "You would have no problem with the steering",
        "My goodness, it is beautifully light.",
    )

    assert assessment.unstable is False


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("The concrete core is quite small,", "but it carries the full load."),
        ("It is a typical fatter skyscraper", "and this tower uses less concrete."),
        ("This is a super slender tower,", "you can see the core from here."),
    ],
)
def test_assessment_does_not_treat_complete_adjectives_or_er_nouns_as_dangling(
    left,
    right,
):
    assert assess_english_boundary(left, right).unstable is False


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        ("I think it's...", "actually more practical.", "subject and auxiliary stranded at 'it's'"),
        ("They haven't", "changed the design.", "incomplete predicate 'haven't'"),
        (
            "It is less expensive than",
            "it first appears.",
            "comparative clause separated after 'than'",
        ),
        (
            "The system was defined",
            "by the earlier standard.",
            "participle separated from its complement",
        ),
        ("We are telling", "other teams what changed.", "participle separated from its complement"),
        (
            "The package has 6",
            "speakers in total.",
            "numeric value separated from its unit or noun",
        ),
        (
            "It provides a slightly lower",
            "center of gravity.",
            "attributive or comparative modifier separated from its head",
        ),
        (
            "The facade is a sea of glass",
            "and metal.",
            "coordinated noun phrase split at conjunction",
        ),
        ("The price goes all the way", "up to $50,000.", "incomplete multi-word phrase"),
        ("It is still", "possible to include a spare.", "dangling modifier 'still'"),
        (
            "It is less of a ridiculous train",
            "wreck than it appears.",
            "comparative noun phrase separated before 'than'",
        ),
    ],
)
def test_assessment_catches_general_lexical_dependencies(left, right, reason):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert reason in assessment.reasons


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            "but we are going to start with the 94",
            "F-150. One of my favorite keys.",
            "model year separated from its vehicle name",
        ),
        (
            "I'd offer it to the museum. You guys",
            "saw me drive their cars earlier this year.",
            "sentence-final subject belongs to the following predicate",
        ),
        (
            "I've tried to arrange this like",
            "my garage in Grand Theft Auto 5.",
            "comparison frame separated from its object",
        ),
        (
            "where else will I find a turbocharged",
            "Cobalt SS with low mileage?",
            "powertrain modifier separated from its vehicle name",
        ),
        (
            "man did I think",
            "I was the coolest person in the world.",
            "emphatic inversion separated from its complement",
        ),
        (
            "They had that really cool,",
            "like, LFA-style one that slid back and forth.",
            "filler separated from the demonstrative head noun",
        ),
        (
            "I was driving it pretty steadily, actually,",
            "Until the power top stopped working.",
            "dependent adverbial clause beginning with 'until'",
        ),
        (
            "The video has not gone live yet",
            "at the time of this recording.",
            "dependent phrase beginning with 'at'",
        ),
    ],
)
def test_assessment_catches_latest_full_run_boundaries(left, right, reason):
    assessment = assess_english_boundary(left, right)

    assert assessment.unstable
    assert reason in assessment.reasons


def test_normalizer_repairs_latest_full_run_boundaries_without_losing_words():
    source = [
        "but we are going to start with the 94",
        "F-150. One of my favorite keys.",
        "I'd offer it to the museum. You guys",
        "saw me drive their cars earlier this year.",
        "They had that really cool,",
        "like, LFA-style one that slid back and forth.",
    ]

    repaired = normalize_boundaries(_cues(source))

    assert " ".join(segment.text for segment in repaired).split() == " ".join(source).split()
    assert all(
        not assess_english_boundary(left.text, right.text).unstable
        for left, right in zip(repaired, repaired[1:])
    )


def _japanese_cues(texts: list[str]) -> list[ASRDataSeg]:
    cursor = 0
    result = []
    for text in texts:
        words = []
        for character in text:
            words.append(
                ASRWord(
                    character,
                    cursor,
                    cursor + 80,
                    timing_source="forced_alignment",
                    language_code="ja",
                )
            )
            cursor += 100
        result.append(
            ASRDataSeg(
                text,
                words[0].start_time,
                words[-1].end_time,
                words=words,
                timestamp_granularity="sentence",
                timing_source="forced_alignment",
                language_code="ja",
            )
        )
    return result


def test_normalizer_repairs_japanese_katakana_and_particle_boundaries():
    source = [
        "これを持ってトンネルの力をトンネ",
        "ルのですね",
        "土の圧力や水の圧力を",
        "これで受けていくと",
    ]

    repaired = normalize_boundaries(_japanese_cues(source), hard_max_cjk_chars=25)

    assert "".join(segment.text for segment in repaired) == "".join(source)
    assert all("トンネ" not in segment.text[-3:] for segment in repaired[:-1])
    assert all(
        not segment.text.startswith(("が", "を", "は", "に", "へ", "で", "の"))
        for segment in repaired[1:]
    )
    assert all(not segment.text.endswith("のですね") for segment in repaired[:-1])


def test_normalizer_merges_short_japanese_filler_with_following_phrase():
    repaired = normalize_boundaries(
        _japanese_cues(["こちらのですね", "掘ったトンネルにどんどん"]),
        hard_max_cjk_chars=25,
    )

    assert [segment.text for segment in repaired] == ["こちらのですね掘ったトンネルにどんどん"]


def test_normalizer_does_not_split_japanese_auxiliary_ending():
    repaired = normalize_boundaries(
        _japanese_cues(["こちらの", "ですね掘ったトンネルに"]),
        hard_max_cjk_chars=25,
    )

    assert [segment.text for segment in repaired] == ["こちらのですね掘ったトンネルに"]


def test_normalizer_keeps_japanese_verb_stem_with_inflection():
    repaired = normalize_boundaries(
        _japanese_cues(["こちらのですね掘", "ったトンネルにどんどん"]),
        hard_max_cjk_chars=25,
    )

    assert "".join(segment.text for segment in repaired) == (
        "こちらのですね掘ったトンネルにどんどん"
    )
    assert all(
        not (left.text.endswith("掘") and right.text.startswith("った"))
        for left, right in zip(repaired, repaired[1:])
    )


def test_normalizer_merges_japanese_dangling_coordination():
    repaired = normalize_boundaries(
        _japanese_cues(["トンネルの土の圧力や", "水の圧力を"]),
        hard_max_cjk_chars=25,
    )

    assert [segment.text for segment in repaired] == ["トンネルの土の圧力や水の圧力を"]


def test_normalizer_keeps_small_tsu_with_japanese_inflection():
    repaired = normalize_boundaries(
        _japanese_cues(["こちらのですね掘っ", "たトンネルにどんどん"]),
        hard_max_cjk_chars=25,
    )

    assert all(
        not (left.text.endswith("っ") and right.text.startswith("た"))
        for left, right in zip(repaired, repaired[1:])
    )


def test_normalizer_merges_dangling_japanese_genitive_modifier():
    repaired = normalize_boundaries(
        _japanese_cues(["トンネルのですね土の", "圧力や水の圧力を"]),
        hard_max_cjk_chars=25,
    )

    assert [segment.text for segment in repaired] == ["トンネルのですね土の圧力や水の圧力を"]


def test_normalizer_allows_small_overflow_for_indivisible_japanese_phrase():
    repaired = normalize_boundaries(
        _japanese_cues(["トンネルのですね土の", "圧力や水の圧力を"]),
        hard_max_cjk_chars=16,
    )

    assert [segment.text for segment in repaired] == ["トンネルのですね土の圧力や水の圧力を"]
    assert len(repaired[0].text) == 18


def test_normalizer_keeps_japanese_kanji_compound_intact():
    repaired = normalize_boundaries(
        _japanese_cues(["トンネルの土の圧", "力や水の圧力を"]),
        hard_max_cjk_chars=25,
    )

    assert all(
        not (left.text.endswith("圧") and right.text.startswith("力"))
        for left, right in zip(repaired, repaired[1:])
    )


def test_normalizer_keeps_japanese_attributive_phrase_with_noun():
    repaired = normalize_boundaries(
        _japanese_cues(["こちらのですね掘った", "トンネルにどんどん"]),
        hard_max_cjk_chars=25,
    )

    assert [segment.text for segment in repaired] == ["こちらのですね掘ったトンネルにどんどん"]


def test_normalizer_keeps_japanese_degree_adverb_with_adjective():
    repaired = normalize_boundaries(
        _japanese_cues(["それが非常に", "難しい工事の中身でした"]),
        hard_max_cjk_chars=25,
    )

    assert "".join(segment.text for segment in repaired) == ("それが非常に難しい工事の中身でした")
    assert all(
        not (left.text.endswith("難し") and right.text.startswith("い"))
        for left, right in zip(repaired, repaired[1:])
    )


def test_normalizer_merges_short_japanese_topic_with_its_predicate():
    repaired = normalize_boundaries(
        _japanese_cues(["それが", "非常に難しい工事の中身でした"]),
        hard_max_cjk_chars=16,
    )

    assert [segment.text for segment in repaired] == ["それが非常に難しい工事の中身でした"]


def test_normalizer_does_not_split_japanese_i_adjective_ending():
    repaired = normalize_boundaries(
        _japanese_cues(["それが非常に難し", "い工事の中身でした"]),
        hard_max_cjk_chars=25,
    )

    assert "".join(segment.text for segment in repaired) == ("それが非常に難しい工事の中身でした")
    assert all(
        not (left.text.endswith("難し") and right.text.startswith("い"))
        for left, right in zip(repaired, repaired[1:])
    )


def test_normalizer_uses_one_stable_cue_to_repair_short_number_unit_tail():
    cues = _cues(
        [
            "Where else can I find a turbocharged GS 400 that only has 57,252",
            "miles on it?",
            "Let's fire it up and go drive it.",
        ]
    )
    cursor = 0
    for cue in cues:
        for word in cue.words:
            word.start_time = cursor
            word.end_time = cursor + 500
            cursor += 600
        cue.start_time = cue.words[0].start_time
        cue.end_time = cue.words[-1].end_time

    repaired = normalize_boundaries(
        cues,
        soft_max_words=16,
        hard_max_words=20,
    )

    assert len(repaired) == 3
    assert all(
        not (left.text.rstrip().endswith("57,252") and right.text.startswith("miles"))
        for left, right in zip(repaired, repaired[1:])
    )
    assert "57,252 miles" in " ".join(segment.text for segment in repaired)
