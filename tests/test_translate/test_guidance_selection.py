import pytest

from subforge.core.translate.guidance import target_language_style_rules


@pytest.mark.parametrize("source", ["At 50% of the pedal.", "It is 12.5%.", "50%", "An increase of 100% today."])
def test_percentage_symbol_triggers_existing_guidance(source):
    assert "For percentages and use cases" in target_language_style_rules("简体中文", [source])


@pytest.mark.parametrize("source", ["Model A50%X", "1234%", "Ordinary interview."])
def test_percentage_selector_does_not_match_identifiers_or_number_suffixes(source):
    assert "For percentages and use cases" not in target_language_style_rules("简体中文", [source])


def test_percentage_guidance_keeps_non_chinese_targets_unchanged():
    assert target_language_style_rules("英语", ["50% of travel"]) == ""
