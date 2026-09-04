import pytest

from subforge.core.split.boundary_detectors import grammar
from subforge.core.split.boundary_features import extract_english_boundary_features


@pytest.mark.parametrize(
    ("detector_name", "left", "right", "kwargs"),
    [
        ("dangling_function_word", "because", "we left", {"tail_is_hard_dangling": True}),
        (
            "dangling_function_word_before_filler",
            "because, you know,",
            "we left",
            {"semantic_tail_is_hard_dangling": True},
        ),
        ("dangling_subject", "we", "can continue", {"tail_is_subject": True}),
        ("standalone_subject", "we", "can continue", {"tail_is_subject": True}),
        ("sentence_final_subject", "I agree. We", "can continue", {}),
        (
            "subject_before_adverbial_predicate",
            "we",
            "really can continue",
            {"tail_is_subject": True},
        ),
        (
            "incomplete_predicate",
            "it is",
            "still useful",
            {"tail_is_incomplete_predicate": True},
        ),
        ("contracted_negative_auxiliary", "we haven't", "finished", {}),
        (
            "subject_auxiliary",
            "we're",
            "still working",
            {"tail_is_subject_auxiliary": True},
        ),
        ("adverb_gerund", "about really", "thinking carefully", {}),
        ("dangling_modifier", "really", "useful", {"tail_is_modifier": True}),
        (
            "dangling_attributive",
            "the main",
            "reason",
            {"tail_is_attributive": True},
        ),
        ("filler_noun_modifier", "a kind of", "solution", {}),
        ("determiner_head", "the main", "reason", {"tail_is_attributive": True}),
        ("time_frame_participle", "last year", "leading up to launch", {}),
    ],
)
def test_grammar_detector_activation(
    detector_name: str,
    left: str,
    right: str,
    kwargs: dict[str, bool],
) -> None:
    features = extract_english_boundary_features(left, right)

    assert getattr(grammar, detector_name)(features, **kwargs) is True


def test_incomplete_multiword_uses_explicit_phrase_classification() -> None:
    assert grammar.incomplete_multiword(left_ends_with_dangling_phrase=True)
    assert not grammar.incomplete_multiword(left_ends_with_dangling_phrase=False)


def test_grammar_detector_completion_exceptions_remain_active() -> None:
    visibility = extract_english_boundary_features("hard to see out of", "and continue")
    degree = extract_english_boundary_features("I love it so much", "and recommend it")

    assert not grammar.dangling_function_word(
        visibility,
        tail_is_hard_dangling=True,
    )
    assert not grammar.dangling_modifier(
        degree,
        tail_is_modifier=True,
    )


@pytest.mark.parametrize(
    ("detector_name", "left", "right", "kwargs"),
    [
        ("frequency_quantified_statement", "It runs", "each day", {}),
        ("distance_location_noun", "About 10 miles away,", "a town is growing", {}),
        ("morphological_attributive_modifier", "a useful", "feature", {}),
        ("determiner_adjective_head", "a useful", "feature", {}),
        (
            "determiner_degree_modifier_head",
            "a very",
            "useful feature",
            {"tail_is_attributive_degree_adverb": True},
        ),
        (
            "lexical_comparative_modifier",
            "greater",
            "value",
            {"tail_is_comparative_modifier": True},
        ),
        ("participle_complement", "We are serving", "the city", {}),
        (
            "postpositive_participle_modifier",
            "Workers climbed huge ladders",
            "mounted on wheels to reach the roof",
            {},
        ),
        (
            "postpositive_participle_modifier",
            "It collects information",
            "built up across months and years",
            {},
        ),
        ("coordinated_noun_phrase", "with cameras", "and sensors", {}),
        ("preposition_gerund", "after", "driving home", {}),
        ("open_complement", "getting", "better", {"tail_is_open_complement": True}),
        ("expect_to_always", "We expect to always", "be ready", {}),
        ("quantifying_phrase_noun", "a number of", "projects", {}),
        ("degree_so", "There are so", "many choices", {}),
        ("up_and_running", "The system is up", "and running", {}),
        ("up_and_running_aspect", "The system is finally", "up and running", {}),
        ("single_word_completion", "This is a useful", "feature", {"hard_max_words": 22}),
    ],
)
def test_grammar_detector_second_batch_activation(
    detector_name: str,
    left: str,
    right: str,
    kwargs: dict[str, bool | int],
) -> None:
    features = extract_english_boundary_features(left, right)

    assert getattr(grammar, detector_name)(features, **kwargs) is True


def test_hyphenated_attributive_detector_returns_the_normalized_reason_value() -> None:
    features = extract_english_boundary_features("a real-world", "test")

    assert (
        grammar.hyphenated_attributive_tail(
            features,
            allowed_tails={"real-world"},
        )
        == "real-world"
    )
    assert not grammar.hyphenated_attributive_tail(
        features,
        allowed_tails={"long-term"},
    )


def test_second_batch_completion_exceptions_remain_active() -> None:
    predicative = extract_english_boundary_features("It is useful,", "and durable")
    determiner_participle = extract_english_boundary_features("the serving", "the city")

    assert grammar.morphological_attributive_modifier(predicative) is False
    assert grammar.participle_complement(determiner_participle) is False

    completed = extract_english_boundary_features(
        "Workers climbed huge ladders.",
        "mounted on wheels for storage",
    )
    assert grammar.postpositive_participle_modifier(completed) is False


@pytest.mark.parametrize(
    ("detector_name", "left", "right", "kwargs"),
    [
        ("phrasal_verb", "We take", "away the waste", {"head_is_phrasal_particle": True}),
        ("take_away", "We take the problem", "away", {}),
        ("new_clause_connective", "It worked. And", "we continued", {}),
        ("sentence_opening_time_marker", "It worked. Today", "we continue", {}),
        ("do_not_get_me_wrong", "Do not get me", "wrong here", {}),
        (
            "sentence_opening_time_adverb",
            "Now",
            "we continue",
            {
                "tail_is_sentence_adverb": True,
                "head_is_subject_or_determiner": True,
            },
        ),
        ("contrastive_beneficiary", "It worked but for them", "we changed it", {}),
        ("direction_here", "Come over", "here", {}),
        ("context_dependent_adjective", "It is well suited", "and reliable", {}),
        (
            "reporting_copular_content",
            "The point that I was trying to make",
            "is simple",
            {},
        ),
        ("interrogative_complement", "We know what", "they need", {}),
        (
            "coordinated_noun_phrase_contextual",
            "the retirement age",
            "and the pension age",
            {},
        ),
        ("dependent_phrase", "a view", "of the city", {"head_is_dependent": True}),
        ("standalone_of_phrase", "a view", "Of the city", {}),
        (
            "relative_clause",
            "the system",
            "which works",
            {"head_is_relative": True, "that_starts_complement": False},
        ),
        (
            "dependent_adverbial_clause",
            "We left",
            "when it ended",
            {"head_is_translation_sensitive": True},
        ),
        (
            "clause_final_subject",
            "We expected, but the plan",
            "was changed",
            {"head_is_finite_predicate": True},
        ),
        ("temporal_continuation", "We stopped before lunch", "and go home", {}),
    ],
)
def test_grammar_detector_final_batch_activation(
    detector_name: str,
    left: str,
    right: str,
    kwargs: dict[str, bool],
) -> None:
    features = extract_english_boundary_features(left, right)

    assert getattr(grammar, detector_name)(features, **kwargs) is True


def test_lexical_unit_uses_explicit_dependency_pair_classification() -> None:
    assert grammar.lexical_unit(tail_head_is_dependency_pair=True)
    assert not grammar.lexical_unit(tail_head_is_dependency_pair=False)


def test_final_batch_completion_exceptions_remain_active() -> None:
    of_course = extract_english_boundary_features("A matter", "Of course we can")
    terminal_clause = extract_english_boundary_features("We left.", "when it ended")
    complement = extract_english_boundary_features("We found out", "that it worked")

    assert not grammar.standalone_of_phrase(of_course)
    assert not grammar.dependent_adverbial_clause(
        terminal_clause,
        head_is_translation_sensitive=True,
    )
    assert not grammar.relative_clause(
        complement,
        head_is_relative=True,
        that_starts_complement=True,
    )
