import pytest

from subforge.core.translate.quality.text import contains_reasoning_leak


@pytest.mark.parametrize("text", ["片名是 '{}' format('追逐')时代", '这里是 "{}" format("文本")'])
def test_malformed_format_expression_is_rejected(text):
    assert contains_reasoning_leak(text)


@pytest.mark.parametrize("text", ["这种 format 很常见", "保留 {} 作为占位符", "The function format(value) is useful."])
def test_ordinary_discussion_of_format_is_not_rejected(text):
    assert not contains_reasoning_leak(text)
