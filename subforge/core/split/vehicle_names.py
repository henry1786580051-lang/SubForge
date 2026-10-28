"""Source-only vehicle-name spans shared by scoring and cue planning."""

from __future__ import annotations

import re
from dataclasses import dataclass

VEHICLE_BRANDS = frozenset(
    {
        "acura",
        "audi",
        "bmw",
        "cadillac",
        "chevrolet",
        "dodge",
        "ford",
        "gmc",
        "honda",
        "hyundai",
        "jeep",
        "kia",
        "lexus",
        "lincoln",
        "mazda",
        "mercedes",
        "mercedes-benz",
        "mercedes-amg",
        "mini",
        "nissan",
        "porsche",
        "ram",
        "subaru",
        "tesla",
        "toyota",
        "volkswagen",
        "volvo",
    }
)
_GENERIC = frozenset(
    {
        "all",
        "and",
        "car",
        "cars",
        "company",
        "dealer",
        "dealership",
        "drive",
        "electric",
        "hybrid",
        "is",
        "model",
        "models",
        "motor",
        "motors",
        "new",
        "sedan",
        "suv",
        "system",
        "truck",
        "vehicle",
        "vehicles",
        "this",
        "that",
        "it",
        "we",
        "they",
        "she",
        "he",
        "i",
        "welcome",
    }
)
_TRIMS = frozenset(
    {
        "premium",
        "plus",
        "luxury",
        "limited",
        "platinum",
        "ultimate",
        "denali",
        "overtrail",
        "competition",
        "performance",
        "sport",
        "sports",
        "gt",
        "gts",
        "gt-line",
        "f-sport",
        "xrt",
        "at4",
        "at4x",
        "long-range",
        "awd",
        "fwd",
        "rwd",
        "4wd",
        "4matic",
        "4matic+",
        "xdrive",
        "quattro",
    }
)
_ID = re.compile(
    r"(?:[A-Za-z]{1,12}[\d][A-Za-z\d-]*|[A-Za-z]{1,5}-\d+[A-Za-z\d-]*|\d{1,4}[A-Za-z]?)$"
)
_NAMED_IDENTIFIER = re.compile(
    r"\b(?:Carrera\s+(?:S|T|GTS)|Turbo\s+S|GT[234]\s+(?:Touring|RS))"
    r"(?:\s+(?:Cabriolet|Coupe|Spyder))?\b(?![\w-])",
    re.I,
)


@dataclass(frozen=True)
class VehicleName:
    start: int
    core_end: int
    end: int


def _clean(token: str) -> str:
    return token.strip(".,;:!?()[]{}\"'“”")


def vehicle_names(text: str) -> list[VehicleName]:
    """Recognize spoken identifiers; never borrow a name from a title or web page."""
    matches = list(re.finditer(r"\S+", text))
    tokens = [_clean(m.group()) for m in matches]
    names = [VehicleName(m.start(), m.end(), m.end()) for m in _NAMED_IDENTIFIER.finditer(text)]
    for i, brand in enumerate(tokens):
        if brand.casefold() not in VEHICLE_BRANDS or i + 1 >= len(tokens):
            continue
        if re.search(r"[,;:.!?]$", matches[i].group()):
            continue
        j = i + 1
        model = tokens[j]
        if model.casefold() in _GENERIC and not (brand.casefold() == "tesla" and model == "Model"):
            continue
        if _ID.fullmatch(model):
            if model.isdigit() and 1900 <= int(model) <= 2100:
                continue
            j += 1
        elif re.fullmatch(r"[A-Z][A-Za-z-]{0,20}", model):
            j += 1
            if (
                j < len(tokens)
                and not re.search(r"[,;:.!?]$", matches[j - 1].group())
                and _ID.fullmatch(tokens[j])
            ):
                j += 1
            elif model.isupper() and len(model) <= 5:
                continue
        else:
            continue
        if j < len(tokens) and not re.search(r"[,;:.!?]$", matches[j - 1].group()):
            tail = _NAMED_IDENTIFIER.match(text, matches[j].start())
            if tail:
                while j < len(matches) and matches[j].start() < tail.end():
                    j += 1
        # Keep split suffixes such as "ES 350 h" and generation identifiers.
        if (
            j < len(tokens)
            and not re.search(r"[,;:.!?]$", matches[j - 1].group())
            and re.fullmatch(r"[hied]", tokens[j])
        ):
            j += 1
        core_end = j
        while j < len(tokens) and not re.search(r"[,;:.!?]$", matches[j - 1].group()):
            if tokens[j].casefold() in _TRIMS:
                j += 1
            elif (
                tokens[j].casefold() in {"all-wheel", "four-wheel", "rear-wheel", "front-wheel"}
                and j + 1 < len(tokens)
                and tokens[j + 1].casefold() == "drive"
            ):
                j += 2
            else:
                break
            if re.search(r"[,;:.!?]$", matches[j - 1].group()):
                break
        start = i - 1 if i and re.fullmatch(r"(?:19|20)\d{2}", tokens[i - 1]) else i
        names.append(
            VehicleName(matches[start].start(), matches[core_end - 1].end(), matches[j - 1].end())
        )
    return sorted(names, key=lambda n: (n.start, -n.end))


def is_identifier_enumeration(source: str, output: str) -> bool:
    """Accept an exact, language-neutral list, never English prose or added names."""

    def items(text: str) -> list[str]:
        parts = re.split(r"[,，、;；]+|\s+and\s+", text.strip().rstrip(".。!?！?"), flags=re.I)
        return [re.sub(r"^the\s+", "", p.strip(), flags=re.I) for p in parts]

    before, after = items(source), items(output)
    if not 2 <= len(before) <= 24 or len(before) != len(after):
        return False
    if not any(_NAMED_IDENTIFIER.fullmatch(p) or re.search(r"\d", p) for p in before):
        return False
    for original, translated in zip(before, after):
        if re.sub(r"\s+", "", original).casefold() != re.sub(r"\s+", "", translated).casefold():
            return False
        if not (
            _NAMED_IDENTIFIER.fullmatch(original)
            or re.fullmatch(r"(?:[A-Z]{1,6}\d{0,3}|\d{2,4}[A-Za-z]?)", original)
        ):
            return False
    return True


def splits_vehicle_name(left: str, right: str) -> bool:
    left, right = left.strip(), right.strip()
    seam = len(left)
    text = left + " " + right
    if any(n.start <= seam < n.core_end for n in vehicle_names(text)):
        return True
    # A drive label is atomic even when its brand lies in an earlier cue.
    return bool(
        re.search(r"\b(?:all|four|rear|front)-wheel$", left, re.I)
        and re.match(r"^drive\b", right, re.I)
    )


def introduction_seam(left: str, right: str) -> bool:
    """A greeting/channel introduction can close before a presentation clause."""
    return bool(
        re.match(r"^(?:hey|hi|hello|welcome|good\s+(?:morning|afternoon|evening))\b", left, re.I)
        and (
            re.search(r"\bwelcome\s+(?:back\s+)?to\b", left, re.I)
            or re.search(r"\b(?:i['’]m|my name is|this is)\b", left, re.I)
        )
        and re.search(r"[A-Za-z][,;:]$", left)
        and re.match(
            r"^(?:(?:where|and)\s+)?(?:this\s+week|today|this\s+morning|now|i['’]ve|we['’]ve|we['’]re|i['’]m|we\s+are)\b",
            right,
            re.I,
        )
        and re.search(r"\b(?:driv\w*|test\w*|review\w*|living|experienc\w*|look)\b", right, re.I)
        and vehicle_names(right)
    )


def vehicle_apposition(left: str, right: str) -> bool:
    """A complete model followed by a separate descriptive noun phrase."""
    return bool(
        left.rstrip().endswith(",")
        and any(n.end == len(left.rstrip()) for n in vehicle_names(left))
        and re.match(r"^the\s+(?:new|all-new|latest|first|\d+(?:st|nd|rd|th))\b", right, re.I)
        and re.search(
            r"\b(?:generation|sedan|suv|truck|coupe|crossover|hatchback)\b",
            " ".join(right.split()[:7]),
            re.I,
        )
    )
