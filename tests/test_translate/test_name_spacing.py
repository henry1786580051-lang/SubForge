import pytest

from subforge.core.translate.quality.preservation import exact_latin_spacing_spans


@pytest.mark.parametrize("source,target,expected", [
    ("That redcoat tavern.", "Red Coat Tavern", "Red Coat"),
    ("Honda Matic transmission.", "Hondamatic", "Hondamatic"),
    ("The Silverstone circuit.", "Silver Stone赛道", "Silver Stone"),
    ("New castle museum", "Newcastle博物馆", "Newcastle"),
])
def test_exact_spacing_preserves_whole_source_owned_name(source, target, expected):
    assert any(target[start:end] == expected for start, end in exact_latin_spacing_spans(source, target))


@pytest.mark.parametrize("source,target", [
    ("redcoat", "Red"), ("red coattail", "Red Coat"),
    ("Red. Coat", "Redcoat"), ("redcoat", "Red. Coat"),
    ("redcoat", "Red, Coat"), ("redcoat", "Red Boat"),
    ("the other restaurant", "Red Coat"), ("Honda Magic", "Hondamatic"),
    ("RAV 4", "RAV4"), ("F-150", "F150"),
    ("Redcoat's", "Red Coat"), ("in side", "inside"),
])
def test_spacing_never_accepts_fuzzy_partial_or_cross_sentence_names(source, target):
    assert exact_latin_spacing_spans(source, target) == ()


def test_spacing_evidence_is_occurrence_scoped():
    target = "Red Coat, then Red Boat"
    spans = exact_latin_spacing_spans("redcoat", target)
    assert spans == ((0, 8),)
