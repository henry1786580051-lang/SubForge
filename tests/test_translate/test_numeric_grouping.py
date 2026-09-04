import pytest

from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.quality.numbers import normalize_grouped_numbers
from subforge.core.translate.quality.preservation import inspect_preserved_tokens


def inspect(source, target):
    return inspect_preserved_tokens(
        {"1": target}, {"1": source}, str,
        target_language_value="简体中文",
        localized_magnitude_rendered=LLMTranslator._localized_magnitude_rendered,
    )


@pytest.mark.parametrize("year,count", [(2025, 128), (1999, 512), (2032, 256)])
def test_year_followed_by_quantity_is_not_a_grouped_number(year, count):
    source = f"As of January {year}, {count} plots were opened."
    assert not inspect(source, f"自{year}年1月起开放{count}块地块")


@pytest.mark.parametrize("target", ["1月开放128块地块", "2025年1月开放地块", "2025年1月开放129块地块"])
def test_independent_year_and_quantity_remain_protected(target):
    assert inspect("As of January 2025, 128 plots were opened.", target)


@pytest.mark.parametrize("source,target", [
    ("It costs $2,128.", "售价2128美元"),
    ("It costs $2, 128.", "售价2128美元"),
    ("There are 1,234,567 plots.", "共有1234567块地块"),
    ("There are 1, 234, 567 plots.", "共有1234567块地块"),
    ("It costs $1,234.50.", "售价1234.50美元"),
    ("It covers 20,000 plots.", "占地2万块"),
    ("In 2025, 128, 256 and 512 were recorded.", "2025年记录了128 256和512"),
])
def test_valid_grouping_and_separate_facts_are_preserved(source, target):
    assert not inspect(source, target)


@pytest.mark.parametrize("source,target", [
    ("It costs $2,128.", "售价128美元"),
    ("There are 1,234,567 plots.", "共有234567块地块"),
    ("It covers 20,000 plots.", "占地3万块"),
])
def test_grouped_number_magnitude_cannot_be_dropped(source, target):
    assert inspect(source, target)


def test_invalid_grouping_cannot_license_a_borrowed_chinese_magnitude():
    assert not LLMTranslator._ownership_token_belongs_to_source(
        "200", "In 2000, 000 plots were available.", "共有200万块地块"
    )


@pytest.mark.parametrize("source", ["There are 2,000,000 plots.", "There are 2, 000, 000 plots."])
def test_valid_grouping_licenses_only_exact_localized_magnitude(source):
    assert LLMTranslator._ownership_token_belongs_to_source("200", source, "共有200万块地块")
    assert not LLMTranslator._ownership_token_belongs_to_source("300", source, "共有300万块地块")


@pytest.mark.parametrize("source,expected", [
    ("", ""),
    ("2025, 128, 256 plots", "2025, 128, 256 plots"),
    ("12,34,567", "12,34,567"),
    ("$1,234.50", "$1234.50"),
    ("2, 000, 000", "2000000"),
    ("1,234 and 5,678", "1234 and 5678"),
    ("2,\n128", "2,\n128"),
    ("1.2,345", "1.2,345"),
    ("2025,128", "2025,128"),
    ("金额2,000元", "金额2000元"),
])
def test_group_normalization_is_complete_and_idempotent(source, expected):
    assert normalize_grouped_numbers(source) == expected
    assert normalize_grouped_numbers(expected) == expected
