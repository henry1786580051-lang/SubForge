"""Conservative normalization shared by numeric translation validators."""

import re

_COMMA_NUMBER = re.compile(r"(?<![\d.])\d+(?:,[ \t]*\d+)+(?:\.\d+)?")
_THOUSANDS_NUMBER = re.compile(r"\d{1,3}(?:,[ \t]*\d{3})+(?:\.\d+)?")


def normalize_grouped_numbers(text: str) -> str:
    """Collapse valid thousands groups, never a year followed by a quantity.

    Match the complete comma-separated run first, so an invalid prefix cannot
    leave a valid-looking suffix that would silently join two independent facts.
    Spaces after a grouping comma are common in word-timestamp ASR output.
    """
    def normalize(match: re.Match[str]) -> str:
        value = match.group()
        if not _THOUSANDS_NUMBER.fullmatch(value):
            return value
        return re.sub(r"[, \t]", "", value)

    return _COMMA_NUMBER.sub(normalize, text)
