import pytest

from subforge.core.split.boundary_detectors import predicate
from subforge.core.split.boundary_features import extract_english_boundary_features


@pytest.mark.parametrize(
    ("detector_name", "left", "right"),
    [
        ("subject_adverb", "It probably", "works well"),
        ("negative_auxiliary_complement", "It could never", "work"),
        ("linking_verb_complement", "It became", "a problem"),
        ("progressive_complement", "It is running", "smoothly"),
        ("auxiliary_participle", "It has always", "been useful"),
        ("infinitive_adverb", "It needs to quickly", "move forward"),
        ("participle_quantified_object", "It is serving", "about 500 people"),
        ("modal_adverb", "It could quickly", "move forward"),
        (
            "contrastive_prepositional_frame",
            "But for the delay,",
            "the project would be done",
        ),
        ("sentence_final_temporal_adverb", "We are here", "today"),
    ],
)
def test_predicate_detector_activation(detector_name: str, left: str, right: str) -> None:
    features = extract_english_boundary_features(left, right)

    assert getattr(predicate, detector_name)(features) is True


def test_sentence_adverb_detectors_use_caller_lexical_classification() -> None:
    finite = extract_english_boundary_features("It works basically", "it is useful")
    progressive = extract_english_boundary_features("It is basically", "working well")

    assert predicate.sentence_adverb_finite(
        finite,
        tail_is_sentence_adverb=True,
    )
    assert not predicate.sentence_adverb_finite(
        finite,
        tail_is_sentence_adverb=False,
    )
    assert predicate.auxiliary_sentence_adverb(
        progressive,
        tail_is_sentence_adverb=True,
        previous_is_subject_auxiliary=False,
    )


def test_predicate_detector_exclusions_remain_narrow() -> None:
    interrogative = extract_english_boundary_features("what became", "a problem")
    terminal = extract_english_boundary_features("We are here.", "today")

    assert predicate.linking_verb_complement(interrogative) is False
    assert predicate.sentence_final_temporal_adverb(terminal) is False


@pytest.mark.parametrize(
    ("detector_name", "left", "right"),
    [
        ("reporting_quoted_object", "We call", "this the main hall"),
        ("prepositional_gerund_complement", "after driving", "the car"),
        ("condition_qualified_predicate", "It is comfortable", "unless loaded"),
        ("relative_clause_subject", "which currently", "serves the city"),
        ("progressive_object", "It may be reading", "the source incorrectly"),
        (
            "topic_frame",
            "One thing that is interesting is, with cars,",
            "they change quickly",
        ),
        ("up_and_running_subject", "The system", "finally up and running"),
        ("short_noun_subject", "The project", "was delayed"),
        ("omitted_relative_one", "This is the one", "we selected"),
        ("copula_parenthetical_complement", "This is, I think,", "the best option"),
        ("emphatic_inversion_complement", "did I think", "it was useful"),
        ("standalone_contrast_frame", "At the same time,", "we continued"),
        ("negative_existential_complement", "There is no", "clear answer"),
        ("reason_clause_subject", "Because at the time,", "the project was new"),
    ],
)
def test_predicate_detector_second_batch_activation(
    detector_name: str,
    left: str,
    right: str,
) -> None:
    features = extract_english_boundary_features(left, right)

    assert getattr(predicate, detector_name)(features) is True


def test_subject_complement_uses_explicit_tail_classification() -> None:
    features = extract_english_boundary_features("The issue", "is serious")

    assert predicate.subject_complement(features, tail_is_copula_complement=True)
    assert not predicate.subject_complement(features, tail_is_copula_complement=False)


@pytest.mark.parametrize(
    ("detector_name", "left", "right"),
    [
        ("what_is_so_after_demonstrative", "This is what's", "so useful"),
        ("what_is_so_adjective", "What's so", "good about it"),
        ("transitive_object_basic", "I can see", "the issue"),
        ("transitive_object_extended", "We could build", "a bridge"),
        ("perfect_reporting_content", "We've seen", "many changes"),
        ("transitive_pronoun_object", "We could tell", "them today"),
        ("perfect_reporting_after_adverb", "We've clearly seen", "many changes"),
        ("reporting_content", "We started to see", "many changes"),
        ("embedded_question_complement", "whether we know", "what happened"),
        ("reported_subject", "I think that retirees", "and are reading less"),
        ("transitive_nominal_clause", "We know", "what happened"),
        ("what_use_for_object", "what we use", "the truck for"),
        ("dependent_locative_clause", "We met", "here in town"),
        ("way_clause_subject", "the best way", "is clear"),
        ("what_clause_subject", "what this means", "is clear"),
        ("degree_complement", "It became", "so much better"),
    ],
)
def test_predicate_detector_final_batch_activation(
    detector_name: str,
    left: str,
    right: str,
) -> None:
    features = extract_english_boundary_features(left, right)

    assert getattr(predicate, detector_name)(features) is True


def test_subject_shape_detectors_use_explicit_lexical_classification() -> None:
    proper_name = extract_english_boundary_features("Donald Trump", "is speaking")
    trailing_noun = extract_english_boundary_features("and the project", "is ready")
    gerund = extract_english_boundary_features("Reading", "is useful")

    assert predicate.proper_name_subject(
        proper_name,
        previous_is_dependent_head=False,
    )
    assert not predicate.proper_name_subject(
        proper_name,
        previous_is_dependent_head=True,
    )
    assert predicate.trailing_noun_subject(
        trailing_noun,
        head_is_finite_predicate=True,
    )
    assert not predicate.trailing_noun_subject(
        trailing_noun,
        head_is_finite_predicate=False,
    )
    assert predicate.gerund_subject(
        gerund,
        head_is_finite_predicate=True,
    )
