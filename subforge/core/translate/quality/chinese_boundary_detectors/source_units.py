"""Strong Chinese defects with an independently visible source dependency."""

from __future__ import annotations

import re

from subforge.core.split.syntax_units import unit_dependency
from subforge.core.split.vehicle_names import splits_vehicle_name

from .visible_pause import BoundarySignalMatch, _match


def detect_source_unit_boundary(
    source: str, following: str, left: str, right: str
) -> BoundarySignalMatch | None:
    a, b = (
        re.sub(r"[\s，。！？；：、,.!?;:…]+", "", left),
        re.sub(r"[\s，。！？；：、,.!?;:…]+", "", right),
    )
    unit = unit_dependency(source, following)
    kind = unit[0] if unit else ""
    broken = bool(
        (
            kind == "nominal_head"
            and a.endswith("的")
            and re.match(r"^(?:特点|做法|系统|设计|功能|方式)", b)
        )
        or (
            kind == "conditional_complement"
            and re.search(
                r"(?:如果|假如|要是|若)(?:你|您|我|我们|他们)?(?:想|希望|打算|要|需要)$", a
            )
        )
        or (kind == "trailing_qualification" and re.match(r"^(?:或)?(?:之类|什么的|等等)", b))
        or (
            kind in {"modal_adverb_predicate", "aspect_predicate"}
            and re.search(r"(?:会|将|已经|正在|开始|提前|预先|主动|自动|能够|可以)$", a)
            and re.match(r"^(?:开始|继续|抬升|升起|降下|降低|提升|移动|调整|打开|关闭)", b)
        )
        or (
            kind == "lexical_atom"
            and re.search(r"\bflat\b", source, re.I)
            and a.endswith("水平")
            and b.startswith("对置")
        )
        or (
            kind == "lexical_atom"
            and re.search(r"\bdoor$", source, re.I)
            and a.endswith("门")
            and b.startswith("袋")
        )
        or (
            kind == "lexical_atom"
            and re.search(r"\bhybrid$", source, re.I)
            and a.endswith(("跟", "与", "和", "比"))
        )
        or (
            kind == "lexical_atom"
            and re.search(r"\bfeel$", source, re.I)
            and a.endswith(("我觉得", "我感觉"))
            and b.startswith(("觉得", "感觉"))
        )
        or (
            splits_vehicle_name(source, following)
            and re.search(r"\b(?:Turbo|Carrera)\b", source, re.I)
            and re.match(r"^[ST](?![A-Za-z0-9])", right)
        )
    )
    if broken:
        return _match("source-backed Chinese syntax unit is split")
    return None
