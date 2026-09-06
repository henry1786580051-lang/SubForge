"""Exact numeric and entity preservation checks for subtitle translation."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, Literal, Tuple

from subforge.core.translate.quality.diagnostics import QualityDiagnostic
from subforge.core.translate.quality.invariants import (
    PreservedTokenViolation,
    inspect_preserved_token_violations,
)
from subforge.core.translate.quality.localized_quantities import (
    bare_rotation_unit_preserved,
)
from subforge.core.translate.quality.numbers import normalize_grouped_numbers

_LATIN_WORD = re.compile(r"(?<![A-Za-z0-9'’-])[A-Za-z]{2,}(?![A-Za-z0-9'’-])")


def exact_latin_spacing_spans(source: str, target: str) -> tuple[tuple[int, int], ...]:
    """Locate whole Latin names changed only by whitespace within one source cue.

    Match at most four adjacent words, never across punctuation or a cue boundary.
    No spelling correction, partial word, numeric identifier, or glossary evidence
    is admitted. Returning positions prevents a matched occurrence from licensing
    the same word in a different phrase.
    """
    def spans(text: str):
        words = list(_LATIN_WORD.finditer(text))
        for index, first in enumerate(words):
            for last_index in range(index, min(index + 4, len(words))):
                last = words[last_index]
                if last_index > index:
                    previous = words[last_index - 1]
                    if not text[previous.end():last.start()].isspace():
                        break
                value = text[first.start():last.end()]
                yield first.start(), last.end(), value, re.sub(r"\s+", "", value).casefold()

    owned: dict[str, set[tuple[str, ...]]] = {}
    for _start, _end, value, compact in spans(source):
        owned.setdefault(compact, set()).add(tuple(value.casefold().split()))
    matches = []
    for start, end, value, compact in spans(target):
        # Require a proper-name signal in the target, not arbitrary prose joining.
        if not any(part[:1].isupper() for part in value.split()):
            continue
        parts = tuple(value.casefold().split())
        if any(parts != source_parts for source_parts in owned.get(compact, ())):
            matches.append((start, end))
    return tuple(matches)

# Established Chinese names are semantic equivalents, not dropped source
# tokens. Keep this table shared by the normal token validator and the
# alignment-repair validator so the two stages cannot disagree.
_CHINESE_ENTITY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "acura": ("讴歌",),
    "bmw": ("宝马",),
    "buick": ("别克",),
    "civic": ("思域",),
    "ford": ("福特",),
    "gm": ("通用", "通用汽车"),
    "honda": ("本田",),
    "infiniti": ("英菲尼迪",),
    "lexus": ("雷克萨斯",),
    "mazda": ("马自达",),
    "mercedes": ("奔驰", "梅赛德斯"),
    "mercedes-benz": ("奔驰", "梅赛德斯奔驰", "梅赛德斯-奔驰"),
    "nissan": ("日产",),
    "toyota": ("丰田",),
    "zf": ("采埃孚",),
}

# Standard localized forms are valid preservation of the source fact. Keep
# these aliases deterministic so natural Chinese translations do not trigger
# repeated LLM retries merely because they omit the original abbreviation.
_CHINESE_TOKEN_EQUIVALENTS: Dict[str, Tuple[str, ...]] = {
    "AC": (
        "空调",
        "空调系统",
        "冷气",
        "冷气系统",
        "空調",
        "空調系統",
        "冷氣",
        "冷氣系統",
        "交流电",
        "交流電",
    ),
    "ASAP": ("尽快", "尽早", "马上", "立刻", "立即", "立马", "赶紧", "赶快", "第一时间"),
    "CEO": ("首席执行官", "行政总裁"),
    "CO2": ("二氧化碳",),
    "DC": ("华盛顿特区", "哥伦比亚特区", "直流"),
    "DM": ("私信", "私信联系", "直接私信"),
    "EU": ("欧盟", "欧洲联盟"),
    "HD": ("高清", "高精", "高精度"),
    "IAEA": ("国际原子能机构",),
    "IRL": ("现实中", "现实生活中", "现实路况", "实际场景", "实际道路"),
    "IKEA": ("宜家",),
    "JFK": ("肯尼迪", "约翰肯尼迪", "约翰·肯尼迪"),
    "NATO": ("北约", "北大西洋公约组织"),
    "QR": ("二维码", "二维条码"),
    "REM": ("快速眼动",),
    "RPM": ("转/分", "每分钟转数", "转速"),
    "SMR": ("小型模块化反应堆",),
    "TV": ("电视", "电视节目", "影视节目"),
    "UAE": ("阿联酋", "阿拉伯联合酋长国"),
    "UK": ("英国", "联合王国"),
    "UN": ("联合国",),
    "WWII": ("二战", "第二次世界大战"),
}

# Expanded English forms own the same fact as their acronym. This prevents an
# acronym introduced in one cue and expanded in the next from looking like a
# cross-key leak merely because natural Chinese uses the expanded form.
_SOURCE_TOKEN_EQUIVALENTS: Dict[str, Tuple[str, ...]] = {
    "AC": ("air conditioning", "air-conditioning", "a/c", "climate control"),
    "CO2": ("carbon dioxide",),
    "EU": ("european union",),
    "HD": ("high definition", "high-definition"),
    "IAEA": ("international atomic energy agency",),
    "IRL": ("in real life",),
    "JFK": ("john f kennedy", "john fitzgerald kennedy", "kennedy airport"),
    "NATO": ("north atlantic treaty organization",),
    "QR": ("quick response code",),
    "REM": ("rapid eye movement",),
    "RPM": ("revolutions per minute", "engine speed", "revs", "tachometer", "rev counter"),
    "SMR": ("small modular reactor", "small modular reactors"),
    "TV": ("television",),
    "UAE": ("united arab emirates",),
    "UK": ("united kingdom", "britain", "british"),
    "UN": ("united nations",),
    "WWII": ("world war ii", "second world war"),
}


def _contextual_token_equivalents(token: str, source: str) -> tuple[str, ...]:
    """Localize ambiguous abbreviations only with evidence in the owning cue.

    DCT also denotes a discrete cosine transform. Do not add an unconditional
    glossary alias or let a vehicle mentioned elsewhere license this reading.
    """
    if token == "DCT" and re.search(
        r"\b(?:(?:dual|wet|dry)[ -]+clutch\s+DCT|"
        r"DCT\s+(?:cars?|models?|transmissions?|gearbox(?:es)?))\b",
        source,
        flags=re.IGNORECASE,
    ):
        return ("双离合", "雙離合")
    return ()


def inspect_preserved_tokens(  # noqa: C901
    response_dict: Dict[str, Any],
    subtitle_dict: Dict[str, str],
    extract_text: Callable[[Any], str],
    *,
    target_language_value: str,
    localized_magnitude_rendered: Callable[[str, str, str], bool],
) -> tuple[QualityDiagnostic, ...]:
    """Catch likely dropped model names, years, specs, and alphanumeric terms."""
    missing: list[PreservedTokenViolation] = []

    def record_missing(
        key: str,
        token: str,
        *,
        kind: Literal["number", "entity"] | None = None,
    ) -> None:
        if kind is None:
            number_token = re.fullmatch(
                r"\d[\d,]*(?:\.\d+)?(?:st|nd|rd|th|s)?",
                token,
                flags=re.IGNORECASE,
            )
            kind = "number" if number_token else "entity"
        missing.append(
            PreservedTokenViolation(
                cue_key=str(key),
                token=token,
                kind=kind,
            )
        )

    def important_tokens(text: str) -> tuple[str, ...]:
        tokens: dict[str, None] = {}
        uppercase_stopwords = {
            "AM",
            "AS",
            "AT",
            "BE",
            "DO",
            "IF",
            "IN",
            "IS",
            "IT",
            "NO",
            "OF",
            "ON",
            "OR",
            "TO",
            "US",
            "WE",
        }
        spoken_interjections = {
            "HEY",
            "HI",
            "OK",
            "OKAY",
            "WOW",
            "YEAH",
            "YES",
        }
        collapsed_large_numbers = normalize_grouped_numbers(text)
        pattern = (
            r"\b[A-Za-z]+\d+[A-Za-z0-9.-]*\b"
            r"|\b\d+(?:\.\d+)+[A-Za-z]+[A-Za-z0-9.-]*\b"
            r"|(?<![\w.])\d+[A-Za-z]+[A-Za-z0-9.-]*\b"
            r"|\b(?:19|20)\d{2}\b"
            r"|\b\d{2,}\b"
            r"|\b[A-Z]{2,}\b"
        )
        for match in re.finditer(pattern, collapsed_large_numbers):
            token = match.group().strip(".,;:!?()[]{}")
            if len(token) < 2 or token in uppercase_stopwords:
                continue
            if token in spoken_interjections and re.search(
                rf"^\s*{re.escape(token)}\b(?:\s*[,!.?;:]|\s+)",
                collapsed_large_numbers,
                flags=re.IGNORECASE,
            ):
                continue
            tokens.setdefault(token, None)
        return tuple(tokens)

    def normalized_text(text: str) -> str:
        return re.sub(r"[\s,，.。-]+", "", text).lower()

    def _world_war_roman_preserved(
        original: str,
        token: str,
        translated_norm: str,
    ) -> bool:
        roman = token.upper()
        if roman not in {"I", "II"}:
            return False
        if not re.search(
            rf"\bWorld\s+War\s+{roman}\b",
            original,
            flags=re.IGNORECASE,
        ):
            return False
        equivalents = {"一战", "第一次世界大战"} if roman == "I" else {"二战", "第二次世界大战"}
        return any(normalized_text(value) in translated_norm for value in equivalents)

    def _is_decade_token(token: str) -> bool:
        return bool(re.fullmatch(r"(?:\d{2}|\d{4})s", token, flags=re.IGNORECASE))

    def _is_ordinal_token(token: str) -> bool:
        return bool(re.fullmatch(r"\d+(?:st|nd|rd|th)", token, flags=re.IGNORECASE))

    def _ordinal_preserved(token: str, translated_norm: str) -> bool:
        if not _is_ordinal_token(token):
            return False
        number = re.match(r"\d+", token)
        if not number:
            return False
        digits = number.group()
        candidates = {
            digits,
            f"第{digits}",
            *(f"第{form}" for form in _integer_chinese_forms(digits)),
        }
        return any(normalized_text(candidate) in translated_norm for candidate in candidates)

    def _integer_chinese_forms(token: str) -> set[str]:
        if not re.fullmatch(r"\d+", token):
            return set()
        digits = "零一二三四五六七八九"
        digit_form = "".join(digits[int(character)] for character in token)
        value = int(token)
        if value == 0:
            return {"零"}
        if value >= 10000:
            return {digit_form}

        units = ((1000, "千"), (100, "百"), (10, "十"))
        remaining = value
        parts: list[str] = []
        pending_zero = False
        for unit_value, unit_name in units:
            quotient, remaining = divmod(remaining, unit_value)
            if quotient:
                if pending_zero:
                    parts.append("零")
                    pending_zero = False
                if unit_value != 10 or quotient != 1 or parts:
                    parts.append(digits[quotient])
                parts.append(unit_name)
            elif parts and remaining:
                pending_zero = True
        if remaining:
            if pending_zero:
                parts.append("零")
            parts.append(digits[remaining])
        return {digit_form, "".join(parts)}

    def _integer_preserved(token: str, translated_norm: str) -> bool:
        return any(
            normalized_text(candidate) in translated_norm
            for candidate in _integer_chinese_forms(token)
        )

    def _clock_time_preserved(token: str, translated: str) -> bool:
        """Accept natural Chinese renderings of compact times such as ``5am``."""
        match = re.fullmatch(
            r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?"
            r"(?P<period>a\.?m\.?|p\.?m\.?)",
            token,
            flags=re.IGNORECASE,
        )
        if not match:
            return False
        hour = int(match.group("hour"))
        minute = match.group("minute")
        if not 1 <= hour <= 12:
            return False

        period = match.group("period").lower().replace(".", "")
        if period == "am":
            period_pattern = r"(?:凌晨|清晨|早上|上午)"
        elif hour == 12:
            period_pattern = r"(?:中午|下午)"
        elif hour <= 5:
            period_pattern = r"(?:下午|傍晚)"
        else:
            period_pattern = r"(?:傍晚|晚上|夜间|夜里)"

        hour_forms = {str(hour), *_integer_chinese_forms(str(hour))}
        hour_pattern = "|".join(
            map(re.escape, sorted(hour_forms, key=len, reverse=True))
        )
        minute_pattern = ""
        if minute and minute != "00":
            minute_forms = {minute.lstrip("0") or "0", *_integer_chinese_forms(str(int(minute)))}
            minute_pattern = rf"(?:{'|'.join(map(re.escape, minute_forms))})(?:分)?"
        return bool(
            re.search(
                rf"{period_pattern}\s*(?:{hour_pattern})(?:点|时|時){minute_pattern}",
                translated,
            )
        )

    def _absolute_number_candidates(value: Decimal) -> set[str]:
        """Return exact Arabic and natural Chinese magnitude renderings."""

        def decimal_text(number: Decimal) -> str:
            rendered = format(number, "f")
            return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered

        candidates = {decimal_text(value)}
        for divisor, unit in (
            (Decimal(100000000), "亿"),
            (Decimal(10000), "万"),
            (Decimal(1000), "千"),
            (Decimal(100), "百"),
        ):
            coefficient = value / divisor
            exponent = coefficient.as_tuple().exponent
            if coefficient != coefficient.to_integral_value() and not (
                isinstance(exponent, int) and exponent >= -3
            ):
                continue
            rendered = decimal_text(coefficient)
            forms = {rendered}
            if coefficient == coefficient.to_integral_value() and 0 <= coefficient < 10000:
                integer = str(int(coefficient))
                forms.update(_integer_chinese_forms(integer))
                if integer == "2":
                    forms.add("两")
            candidates.update(f"{form}{unit}" for form in forms)
        return candidates

    def _shared_magnitude_range_preserved(
        original: str,
        token: str,
        translated: str,
    ) -> bool:
        """Accept a magnitude or grouped suffix shared by both range bounds."""
        if not re.fullmatch(r"\d+(?:\.\d+)?", token):
            return False

        def exact_candidate_present(candidate: str) -> bool:
            compact_target = re.sub(r"[\s,，。-]+", "", translated)
            compact_candidate = re.sub(r"[\s,，。-]+", "", candidate)
            number_chars = r"0-9零〇一二两三四五六七八九十百千万亿点."
            return bool(
                re.search(
                    rf"(?<![{number_chars}]){re.escape(compact_candidate)}"
                    rf"(?![{number_chars}])",
                    compact_target,
                )
            )

        magnitude_match = re.search(
            r"\b(\d+(?:\.\d+)?)\s*(?:to|or|[-–—])\s*"
            r"(\d+(?:\.\d+)?)\s+"
            r"(hundred|thousand|million|billion|trillion)\b",
            original,
            flags=re.IGNORECASE,
        )
        if magnitude_match and token in magnitude_match.group(1, 2):
            multipliers = {
                "hundred": Decimal(100),
                "thousand": Decimal(1000),
                "million": Decimal(1000000),
                "billion": Decimal(1000000000),
                "trillion": Decimal(1000000000000),
            }
            value = Decimal(token) * multipliers[magnitude_match.group(3).lower()]
            return any(
                exact_candidate_present(candidate)
                for candidate in _absolute_number_candidates(value)
            )

        grouped_match = re.search(
            r"\b(\d{1,3})\s*(?:to|or|[-–—])\s*"
            r"(\d{1,3}),\s*(\d{3})\b",
            original,
            flags=re.IGNORECASE,
        )
        if not grouped_match:
            return False
        left, right_head, right_tail = grouped_match.groups()
        collapsed_right = f"{right_head}{right_tail}"
        if token == left:
            value = Decimal(left) * Decimal(1000)
        elif token == collapsed_right:
            value = Decimal(collapsed_right)
        else:
            return False
        return any(
            exact_candidate_present(candidate)
            for candidate in _absolute_number_candidates(value)
        )

    def _measurement_preserved(
        original: str,
        token: str,
        translated: str,
        translated_norm: str,
    ) -> bool:
        """Accept exact quantities whose abbreviated unit was localized."""
        match = re.fullmatch(
            r"(\d+(?:\.\d+)?)(kmh|kph|km|mph|mpg|rpm|hp|lbs?|ft)",
            token,
            flags=re.IGNORECASE,
        )
        if not match:
            return False
        raw_value, raw_unit = match.groups()
        value_candidates = {raw_value}
        if re.fullmatch(r"\d+", raw_value):
            value_candidates.update(_integer_chinese_forms(raw_value))
        compact_target = re.sub(r"[\s,，。-]+", "", translated)
        number_chars = r"0-9零〇一二两三四五六七八九十百千万亿点."
        value_present = False
        for candidate in value_candidates:
            compact_candidate = re.sub(r"[\s,，。-]+", "", candidate)
            if re.search(
                rf"(?<![{number_chars}]){re.escape(compact_candidate)}"
                rf"(?![{number_chars}])",
                compact_target,
            ):
                value_present = True
                break
        if not value_present:
            return False

        unit = raw_unit.lower()
        if (
            target_language_value in {"简体中文", "繁体中文", "粤语"}
            and unit == "rpm"
            and bare_rotation_unit_preserved(original, token, translated)
        ):
            return True
        unit_patterns = {
            "km": r"(?:公里|千米)",
            "kmh": r"(?:公里|千米)(?:每小时|/小时)?|时速",
            "kph": r"(?:公里|千米)(?:每小时|/小时)?|时速",
            "mph": r"(?:英里)(?:每小时|/小时)?|时速",
            "mpg": r"(?:英里每加仑|英里/加仑|mpg)",
            "rpm": r"(?:转速|转/分|每分钟转数|rpm)",
            "hp": r"(?:马力|hp)",
            "lb": r"(?:磅|lb)",
            "lbs": r"(?:磅|lbs?)",
            "ft": r"(?:英尺|ft)",
        }
        if not re.search(unit_patterns[unit], translated, flags=re.IGNORECASE):
            return False
        if unit == "km" and re.search(
            rf"\b{re.escape(token)}\s+(?:an|per)\s+hour\b",
            original,
            flags=re.IGNORECASE,
        ):
            return bool(re.search(r"(?:时速|每小时|/小时)", translated))
        return True

    def _large_integer_preserved(token: str, translated: str) -> bool:
        """Accept an exactly equivalent Chinese ten-thousand expression."""
        if (
            target_language_value not in {"简体中文", "繁体中文", "粤语"}
            or not re.fullmatch(r"\d+", token)
            or int(token) < 10000
        ):
            return False

        value = Decimal(token)
        ten_thousands = value / Decimal(10000)
        rendered = format(ten_thousands, "f")
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")
        coefficient_forms = {rendered}
        if ten_thousands == ten_thousands.to_integral_value():
            integer = str(int(ten_thousands))
            coefficient_forms.update(_integer_chinese_forms(integer))
            if integer == "2":
                coefficient_forms.add("两")
        return bool(
            re.search(
                rf"(?<![\d.])(?:{'|'.join(map(re.escape, sorted(coefficient_forms, key=len, reverse=True)))})"
                r"\s*万(?![\d万])",
                translated,
            )
        )

    def _spoken_magnitude_preserved(
        original: str,
        token: str,
        translated_norm: str,
    ) -> bool | None:
        """Validate equivalents such as ``200 million`` -> ``两亿`` exactly."""
        if not re.fullmatch(r"\d+(?:\.\d+)?", token):
            return None
        match = re.search(
            rf"\b{re.escape(token)}\s+(hundred|thousand|million|billion|trillion)\b",
            original,
            flags=re.IGNORECASE,
        )
        if not match:
            return None

        multipliers = {
            "hundred": Decimal(100),
            "thousand": Decimal(1000),
            "million": Decimal(1000000),
            "billion": Decimal(1000000000),
            "trillion": Decimal(1000000000000),
        }
        try:
            absolute = Decimal(token) * multipliers[match.group(1).lower()]
        except (InvalidOperation, KeyError):
            return False

        def decimal_text(value: Decimal) -> str:
            rendered = format(value, "f")
            return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered

        def coefficient_forms(value: Decimal) -> set[str]:
            rendered = decimal_text(value)
            forms = {rendered}
            if value == value.to_integral_value() and 0 <= value < 10000:
                forms.update(_integer_chinese_forms(str(int(value))))
                if value == 2:
                    forms.add("两")
            return forms

        candidates = {decimal_text(absolute)}
        for divisor, unit in (
            (Decimal(100000000), "亿"),
            (Decimal(10000), "万"),
            (Decimal(1000), "千"),
            (Decimal(100), "百"),
        ):
            coefficient = absolute / divisor
            exponent = coefficient.as_tuple().exponent
            if coefficient == coefficient.to_integral_value() or (
                isinstance(exponent, int) and exponent >= -3
            ):
                candidates.update(f"{form}{unit}" for form in coefficient_forms(coefficient))
        return any(normalized_text(candidate) in translated_norm for candidate in candidates)

    def _decade_preserved(
        original: str,
        token: str,
        translated: str,
        translated_norm: str,
    ) -> bool:
        if not _is_decade_token(token):
            return False
        digits = token[:-1]
        if normalized_text(token) in translated_norm:
            return True
        if len(digits) == 4:
            century = digits[:2]
            decade = digits[2:]
            candidates = {
                f"{digits}年代",
                f"{century}世纪{decade}年代",
                f"{decade}年代",
                f"{int(decade)}年代",
            }
        else:
            candidates = {
                f"{digits}年代",
                f"{int(digits)}年代",
            }
        chinese_decades = {
            "00": "零零年代",
            "10": "一十年代",
            "20": "二十年代",
            "30": "三十年代",
            "40": "四十年代",
            "50": "五十年代",
            "60": "六十年代",
            "70": "七十年代",
            "80": "八十年代",
            "90": "九十年代",
        }
        decade_key = digits[-2:]
        if decade_key in chinese_decades:
            candidates.add(chinese_decades[decade_key])
        source_decades = [
            match.group(1)[-2:]
            for match in re.finditer(
                r"\b((?:\d{2}|\d{4}))s\b",
                original,
                flags=re.IGNORECASE,
            )
        ]
        if len(source_decades) >= 2 and decade_key in source_decades:
            compact_digits = {
                "00": "零",
                "10": "一",
                "20": "二",
                "30": "三",
                "40": "四",
                "50": "五",
                "60": "六",
                "70": "七",
                "80": "八",
                "90": "九",
            }
            if all(value in compact_digits for value in source_decades):
                combined = "".join(compact_digits[value] for value in source_decades)
                candidates.add(f"{combined}十年代")
        return any(normalized_text(candidate) in translated_norm for candidate in candidates)

    def _age_decade_preserved(
        original: str,
        token: str,
        translated_norm: str,
    ) -> bool:
        """Accept natural Chinese age expressions for possessive age decades."""
        if target_language_value not in {"简体中文", "繁体中文", "粤语"}:
            return False
        match = re.fullmatch(r"(\d{2})s", token, flags=re.IGNORECASE)
        if not match:
            return False
        if not re.search(
            rf"\b(?:in|throughout)\s+(?:my|your|his|her|our|their|one's)\s+"
            rf"(?:(?:early|mid(?:dle)?|late)[ -]?)?{re.escape(token)}\b",
            original,
            flags=re.IGNORECASE,
        ):
            return False

        value = match.group(1)
        age_forms = {f"{value}多岁"}
        for number in _integer_chinese_forms(value):
            age_forms.add(f"{number}多岁")
        return any(normalized_text(candidate) in translated_norm for candidate in age_forms)

    def _standalone_affirmative_percentage_preserved(
        original: str,
        token: str,
        translated_norm: str,
    ) -> bool:
        """Allow ``100%`` as a standalone affirmation, not as a lost quantity."""
        if (
            target_language_value not in {"简体中文", "繁体中文", "粤语"}
            or token != "100"
            or not re.fullmatch(
                r"\s*100\s*(?:%|％|percent)\s*[.!?。！？]*\s*",
                original,
                flags=re.IGNORECASE,
            )
        ):
            return False
        equivalents = (
            "完全如此",
            "完全正确",
            "完全正確",
            "百分之百",
            "当然",
            "當然",
            "没错",
            "沒錯",
            "确实",
            "確實",
            "正是",
            "绝对",
            "絕對",
        )
        return any(normalized_text(value) in translated_norm for value in equivalents)

    def _is_price_band_token(original: str, token: str, translated: str = "") -> bool:
        has_price_context = bool(
            re.search(
                r"\b(?:cost|costs|expensive|pay|price|priced|range|sell|sold|worth)\b"
                r"|[$]",
                original,
                flags=re.IGNORECASE,
            )
            or re.search(
                r"(?:美元|美金|万元?|千元?|\d\s*[万千k])",
                translated,
                flags=re.IGNORECASE,
            )
        )
        return bool(re.fullmatch(r"\d{2}s", token, flags=re.IGNORECASE) and has_price_context)

    def _price_band_preserved(
        original: str,
        token: str,
        translated: str,
        translated_norm: str,
    ) -> bool:
        """Accept plural price bands translated as thousands or ten-thousands."""
        if not _is_price_band_token(original, token, translated):
            return False

        match = re.fullmatch(r"(\d{2})s", token, flags=re.IGNORECASE)
        assert match is not None
        value = int(match.group(1))
        candidates = {
            f"{value}k",
            f"{value}000",
            f"{Decimal(value) / Decimal(10):g}万",
        }
        if value == 20:
            candidates.update({"二万", "两万"})
        return any(
            normalized_text(candidate) in translated_norm for candidate in candidates
        ) or bool(
            re.search(
                rf"(?<!\d){value}(?!\d)\s*(?:千|k)",
                translated,
                flags=re.IGNORECASE,
            )
        )

    def _inflected_alnum_preserved(token: str, translated_norm: str) -> bool:
        if not re.search(r"\d", token):
            return False
        if not re.fullmatch(r"[A-Za-z0-9.-]+s", token):
            return False
        singular = token[:-1]
        return len(singular) >= 2 and normalized_text(singular) in translated_norm

    def _equivalent_token_preserved(token: str, translated_norm: str) -> bool:
        normalized_token = token.strip(".,;:!?()[]{}").upper()
        contextual_equivalent = target_language_value in {"简体中文", "繁体中文", "粤语"} and any(
            normalized_text(equivalent) in translated_norm
            for equivalent in _contextual_token_equivalents(normalized_token, original)
        )
        # The legacy general token extractor skips single-digit quantities.
        # This new localization must not accept a missing/changed gear count
        # merely because the acronym itself now has a valid Chinese rendering.
        if contextual_equivalent and all(
            re.search(
                r"(?<![0-9零一二三四五六七八九十百千])(?:"
                + "|".join(re.escape(form) for form in sorted({speed, *_integer_chinese_forms(speed)}))
                + r")\s*(?:速|挡|檔)",
                translated_norm,
            )
            for speed in re.findall(r"\b(\d+)[ -]+speed\b", original, flags=re.IGNORECASE)
        ):
            return True
        if (
            target_language_value in {"简体中文", "繁体中文", "粤语"}
            and normalized_token == "RPM"
            and bare_rotation_unit_preserved(original, token, translated)
        ):
            return True
        equivalents = _CHINESE_TOKEN_EQUIVALENTS.get(normalized_token)
        if not equivalents:
            normalized_token = token.strip(".,;:!?()[]{}").casefold()
            equivalents = _CHINESE_ENTITY_ALIASES.get(normalized_token, ())
        if not equivalents:
            return False
        return any(normalized_text(equivalent) in translated_norm for equivalent in equivalents)

    def _compound_model_preserved(token: str, translated: str) -> bool:
        if not (re.search(r"[A-Za-z]", token) and re.search(r"\d", token)):
            return False
        token_compact = re.sub(r"[^a-z0-9]", "", token.lower())
        translated_compact = re.sub(r"[^a-z0-9]", "", translated.lower())
        return bool(token_compact and token_compact in translated_compact)

    def _magnitude_preserved(
        original: str,
        translated_norm: str,
        token: str = "",
    ) -> bool:
        """Accept equivalent grand/K notation without allowing a lost magnitude."""

        def decimal_text(value: Decimal) -> str:
            rendered = format(value, "f")
            return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered

        for match in re.finditer(
            r"\b(\d+(?:\.\d+)?)\s*(grand|k)\b",
            original,
            flags=re.IGNORECASE,
        ):
            raw = match.group(0)
            if token and normalized_text(token) not in {
                normalized_text(raw),
                normalized_text(match.group(1)),
            }:
                continue
            try:
                stated = Decimal(match.group(1))
            except InvalidOperation:
                continue
            absolute = stated * 1000
            ten_thousands = absolute / 10000
            candidates = {
                raw,
                decimal_text(absolute),
                f"{decimal_text(ten_thousands)}万",
            }
            if any(normalized_text(candidate) in translated_norm for candidate in candidates):
                return True
        return False

    def _thousand_magnitude_preserved(
        original: str,
        translated_norm: str,
        token: str,
    ) -> bool:
        """Accept natural Chinese equivalents of explicit thousand amounts."""
        if not re.fullmatch(r"\d+(?:\.\d+)?", token):
            return False

        normalized_source = re.sub(r"[-\s]+", " ", original).strip()
        match = re.search(
            rf"\b{re.escape(token)}\s+"
            r"(?:(?:some\s+odd|some|odd)\s+)?thousand\b",
            normalized_source,
            flags=re.IGNORECASE,
        )
        if not match:
            return False

        try:
            absolute = Decimal(token) * 1000
        except InvalidOperation:
            return False

        def decimal_text(value: Decimal) -> str:
            rendered = format(value, "f")
            return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered

        candidates = {decimal_text(absolute)}
        ten_thousands = absolute / 10000
        if ten_thousands == ten_thousands.to_integral_value():
            compact_value = str(int(ten_thousands))
            candidates.add(f"{compact_value}万")
            for chinese_value in _integer_chinese_forms(compact_value):
                candidates.add(f"{chinese_value}万")
            if compact_value == "2":
                candidates.add("两万")
        else:
            candidates.add(f"{decimal_text(ten_thousands)}万")
        return any(normalized_text(candidate) in translated_norm for candidate in candidates)

    def _introductory_101_preserved(
        original: str,
        token: str,
        translated_norm: str,
    ) -> bool:
        """Treat ``subject 101`` as an introductory-concept idiom in Chinese."""
        if token != "101" or target_language_value not in {
            "简体中文",
            "繁体中文",
            "粤语",
        }:
            return False
        if re.search(
            r"\b(?:route|highway|room|suite|flight|model|interstate)\s+101\b",
            original,
            flags=re.IGNORECASE,
        ):
            return False
        if not re.search(
            r"\b[A-Za-z][A-Za-z' -]{1,80}\s+101\b"
            r"(?=\s*(?:[.!?,;:]|$|\bis\b|\bwas\b))",
            original,
            flags=re.IGNORECASE,
        ):
            return False
        return any(
            normalized_text(equivalent) in translated_norm
            for equivalent in ("入门", "基础", "基本常识", "基础知识", "初级", "概论")
        )

    def _asr_formatted_number_preserved(
        original: str,
        token: str,
        translated: str,
        translated_norm: str,
    ) -> bool:
        """Allow only narrow, explicit repairs of ASR-formatted quantities."""
        grouped = re.fullmatch(r"(\d{1,3})000", token)
        if grouped:
            base = grouped.group(1)
            source_pattern = rf"[$]\s*{re.escape(base)},000\b"
            unit_after_pattern = (
                rf"(?<!\d){re.escape(base)}\s*(?:mpg|mph|km/?h|kph|rpm|"
                r"英里每加仑|英里/加仑|英里每小时|公里每小时|马力|"
                r"磅英尺|磅-英尺)\b"
            )
            unit_before_pattern = (
                rf"(?:每加仑|时速|速度|mpg|mph)\D{{0,8}}(?<!\d)"
                rf"{re.escape(base)}(?!\d)"
            )
            has_unit = bool(
                re.search(
                    unit_after_pattern,
                    translated,
                    flags=re.IGNORECASE,
                )
                or re.search(
                    unit_before_pattern,
                    translated,
                    flags=re.IGNORECASE,
                )
            )
            if re.search(source_pattern, original) and has_unit:
                return True

        if re.fullmatch(r"\d{2}", token):
            year = f"20{token}"
            shorthand_pattern = rf"\b(?:for|in|model\s+year)\s+{token}\b"
            if re.search(shorthand_pattern, original, flags=re.IGNORECASE) and (
                year in translated_norm
            ):
                return True
        return False

    def _abandoned_numeric_self_correction(
        original: str,
        token: str,
        translated_norm: str,
    ) -> bool:
        """Allow an explicit false-start number to yield to its corrected value."""
        match = re.search(
            r"\b(\d+(?:\.\d+)?)\s*,\s+(?:a|uh|um|er)\s+(\d+\.\d+)\b",
            original,
            flags=re.IGNORECASE,
        )
        if not match or normalized_text(token) != normalized_text(match.group(1)):
            return False
        return normalized_text(match.group(2)) in translated_norm

    def _discount_preserved(
        original: str,
        token: str,
        translated: str,
        translated_norm: str,
    ) -> bool:
        """Accept exact Chinese discount notation for an ``N% off`` source."""
        if not re.fullmatch(r"\d{1,3}", token):
            return False
        if not re.search(
            rf"\b{re.escape(token)}\s*%\s*off\b",
            original,
            flags=re.IGNORECASE,
        ):
            return False
        discount = int(token)
        if not 0 <= discount <= 100:
            return False
        if discount == 100:
            return "免费" in translated
        payable = Decimal(100 - discount) / Decimal(10)
        payable_text = format(payable, "f")
        if "." in payable_text:
            payable_text = payable_text.rstrip("0").rstrip(".")
        chinese_digits = str.maketrans("0123456789", "零一二三四五六七八九")
        candidates = {
            f"{payable_text}折",
            f"{payable_text.translate(chinese_digits)}折",
        }
        return any(normalized_text(value) in translated_norm for value in candidates)

    def _casual_numeric_range_preserved(
        original: str,
        token: str,
        translated_norm: str,
    ) -> bool:
        """Accept natural Chinese compression of a casual range such as 10 or 15."""
        if not re.fullmatch(r"\d{1,3}", token):
            return False
        match = re.search(
            r"\b(\d{1,3})\s+or\s+(\d{1,3})\s+"
            r"(?:seconds?|minutes?|hours?|days?|years?)\b",
            original,
            flags=re.IGNORECASE,
        )
        if not match or token not in match.groups():
            return False
        lower, upper = map(int, match.groups())
        if lower > upper:
            lower, upper = upper, lower
        if lower == 10 and 11 <= upper <= 19:
            return any(
                normalized_text(candidate) in translated_norm for candidate in ("十几", "十来")
            )
        return False

    for key, original in subtitle_dict.items():
        translated = extract_text(response_dict.get(key, ""))
        translated_norm = normalized_text(translated)
        for token in important_tokens(original):
            token_norm = normalized_text(token)
            spoken_magnitude = _spoken_magnitude_preserved(
                original,
                token,
                translated_norm,
            )
            if spoken_magnitude is not None:
                if spoken_magnitude:
                    continue
                record_missing(key, token)
                continue
            if _abandoned_numeric_self_correction(
                original,
                token,
                translated_norm,
            ):
                continue
            if _discount_preserved(original, token, translated, translated_norm):
                continue
            if _casual_numeric_range_preserved(original, token, translated_norm):
                continue
            if _shared_magnitude_range_preserved(original, token, translated):
                continue
            if _world_war_roman_preserved(original, token, translated_norm):
                continue
            if _age_decade_preserved(original, token, translated_norm):
                continue
            if _standalone_affirmative_percentage_preserved(
                original,
                token,
                translated_norm,
            ):
                continue
            if _is_price_band_token(original, token, translated):
                if _price_band_preserved(
                    original,
                    token,
                    translated,
                    translated_norm,
                ):
                    continue
                record_missing(key, token, kind="number")
                continue
            if _decade_preserved(original, token, translated, translated_norm):
                continue
            if _ordinal_preserved(token, translated_norm):
                continue
            if _large_integer_preserved(token, translated):
                continue
            if _clock_time_preserved(token, translated):
                continue
            if _integer_preserved(token, translated_norm):
                continue
            if _inflected_alnum_preserved(token, translated_norm):
                continue
            if _equivalent_token_preserved(token, translated_norm):
                continue
            if _measurement_preserved(original, token, translated, translated_norm):
                continue
            if localized_magnitude_rendered(original, token, translated):
                continue
            if _compound_model_preserved(token, translated):
                continue
            if _magnitude_preserved(original, translated_norm, token):
                continue
            if _thousand_magnitude_preserved(original, translated_norm, token):
                continue
            if _introductory_101_preserved(original, token, translated_norm):
                continue
            if _asr_formatted_number_preserved(
                original,
                token,
                translated,
                translated_norm,
            ):
                continue
            if token_norm and token_norm not in translated_norm:
                record_missing(key, token)

        spoken_numbers = {
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
        }
        for number_word, value in spoken_numbers.items():
            if not re.search(
                rf"\b{number_word}\s+(?!(?:and|or)\b)[A-Za-z]",
                original,
                re.IGNORECASE,
            ):
                continue
            candidates = {str(value), *_integer_chinese_forms(str(value))}
            if value == 2:
                candidates.add("两")
            if not any(normalized_text(item) in translated_norm for item in candidates):
                record_missing(key, number_word, kind="number")

        if re.search(
            r"\b\d+(?:\.\d+)?\s*(?:grand|k)\b",
            original,
            flags=re.IGNORECASE,
        ) and not _magnitude_preserved(original, translated_norm):
            record_missing(key, "numeric magnitude", kind="number")

    return inspect_preserved_token_violations(missing)
