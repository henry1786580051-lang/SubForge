"""Source-backed Chinese rotational-unit equivalence, without model calls."""

from __future__ import annotations

import re
from decimal import Decimal

from .numbers import normalize_grouped_numbers

_NUMBER_CHARS = r"0-9零〇一二两兩三四五六七八九十百千万萬亿億点點."


def bare_rotation_unit_preserved(source: str, token: str, target: str) -> bool:
    """Accept a source RPM quantity as the same adjacent number plus 转/轉.

    Exclude lexical uses such as 转账 and 转弯. Never let a different number's
    rotational unit license this quantity, or require whitespace before Chinese.
    """
    compact_source = normalize_grouped_numbers(source)
    values = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*RPM\b", compact_source, re.I)
    if token.upper() != "RPM":
        match = re.fullmatch(r"(\d+(?:\.\d+)?)rpm", token, re.I)
        if not match or match.group(1) not in values:
            return False
        values = [match.group(1)]
    if not values:
        return False
    text = normalize_grouped_numbers(target)
    for value in values:
        number = format(Decimal(value), "f")
        if "." in number:
            number = number.rstrip("0").rstrip(".")
        if not re.search(
            rf"(?<![{_NUMBER_CHARS}]){re.escape(number)}\s*[转轉]"
            r"(?!账|帳|让|讓|移|弯|彎|化|为|為|到|换|換|发|發|售|动|動|运(?!行)|運(?!行))",
            text,
        ):
            return False
    return True
