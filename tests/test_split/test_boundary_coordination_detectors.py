import pytest

from subforge.core.split.boundary_detectors import coordination
from subforge.core.split.boundary_features import extract_english_boundary_features


@pytest.mark.parametrize(
    ("detector_name", "left", "right"),
    [
        ("relative_clause_subject", "the people that Alice", "and I interviewed"),
        ("paired_contrast", "on the one hand", "but this is useful"),
        ("directional_names", "north", "and south routes"),
        ("short_predicate_continuation", "we can", "and should continue"),
        ("automotive_intake_exhaust", "the intake", "and exhaust system"),
        (
            "reported_subject_member",
            "I note that retirees and college",
            "graduates and women are reading less",
        ),
        ("subject_predicate", "retirees and women", "are reading less"),
        ("noun_subject_shared_predicate_simple", "retirees", "and women are reading"),
        (
            "noun_subject_shared_predicate_compound",
            "older adults",
            "and retirees and women are reading less",
        ),
        (
            "embedded_question_coordinated_subject",
            "I do not know how lawmakers",
            "and construction firms can modernise the industry",
        ),
        (
            "final_noun_progressive_predicate",
            "students and retirees",
            "and graduates becoming less engaged",
        ),
        ("omitted_subject_predicate", "I finished", "and wanted to continue"),
        (
            "noun_list_progressive_predicate",
            "retirees and graduates",
            "reading less each year",
        ),
        ("what_clause_predicate", "what this means", "and makes possible"),
    ],
)
def test_coordination_detector_activation(
    detector_name: str,
    left: str,
    right: str,
) -> None:
    features = extract_english_boundary_features(left, right)

    assert getattr(coordination, detector_name)(features) is True


def test_coordination_detectors_preserve_sentence_and_pronoun_exclusions() -> None:
    terminal = extract_english_boundary_features("we can.", "and should continue")
    pronoun = extract_english_boundary_features("retirees", "and they are reading")

    assert coordination.short_predicate_continuation(terminal) is False
    assert coordination.noun_subject_shared_predicate_simple(pronoun) is False
    assert (
        coordination.embedded_question_coordinated_subject(
            extract_english_boundary_features(
                "Lawmakers have already published an answer",
                "and construction firms can now respond",
            )
        )
        is False
    )
