"""Source-backed exclusions for two weak Chinese boundary heuristics."""

from __future__ import annotations

import re

from subforge.core.split.boundary import assess_english_boundary

_NEW_CHINESE_CLAUSE = re.compile(
    r"^(?:(?:不过|但是|而且|所以|然后|其实|另外)[，,]?)?"
    r"(?:我们|你们|他们|她们|它们|我|你|他|她|它|这|那)"
    r"(?:也|还|就|会|有|是|想|可以|能|要|觉得|看|喜欢|打算|准备)"
)
_COMPLETE_DE = re.compile(
    r"(?:(?:挺|蛮|很|非常|相当|确实|真的)"
    r"(?:好用|好开|不错|方便|舒服|舒适|容易|正常|合理|合适|漂亮|实用|顺利|"
    r"安全|清楚|可靠|喜欢|满意|好)|"
    r"(?:这么|那么|这样|那样)(?:做|开|用|想|说|操作))的$"
)
_PRONOUN_OBJECT = re.compile(
    r"(?:告诉|提醒|帮助|帮|等|理解|支持|相信|感谢|喜欢|认识|看着|陪着|看见|找到|联系)"
    r"(?:我们|你们|他们|她们|它们|我|你|他|她|它)$"
)


def is_closed_soft_boundary(
    source: str,
    next_source: str,
    left: str,
    right: str,
    signal: str,
) -> bool:
    """Exclude closed predicates/objects, never strong or source-open boundaries."""
    if signal not in {"possible function-word split", "possible pronoun boundary"}:
        return False
    if not re.search(r"[.!?][\"')\]]*$", source.strip()) or re.search(r"\.{2,}$", source.strip()):
        return False
    if not next_source.strip() or assess_english_boundary(source, next_source).risk:
        return False
    clean_left = re.sub(r"\s+", "", left).rstrip("，,。.!！?？")
    clean_right = re.sub(r"\s+", "", right)
    if not _NEW_CHINESE_CLAUSE.match(clean_right):
        return False
    if signal == "possible function-word split":
        return bool(_COMPLETE_DE.search(clean_left))
    return bool(_PRONOUN_OBJECT.search(clean_left))
