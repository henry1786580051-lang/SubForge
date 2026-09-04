"""Chinese disposal and governing-clause attachment boundary defects."""

from __future__ import annotations

import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_BA_TAIL = re.compile(
    r"(?:^|[，,；;。.!?])"
    r"(?:(?:我|我们|你|你们|他|她|它|他们|她们|它们)"
    r"(?:会|将|要|需要|可以|能够|打算|准备)?|"
    r"(?:会|将|要|需要|可以|能够|打算|准备))?把"
    r"(?P<object>[㐀-鿿A-Za-z0-9·一二两三四五六七八九十百千万]{1,32})$"
)
_BA_ANY_TAIL = re.compile(
    r"(?<![一二两三四五六七八九十百千万几多])把"
    r"(?P<object>[㐀-鿿A-Za-z0-9·一二两三四五六七八九十百千万]{1,32})$"
)
_BA_PREDICATE_IN_OBJECT = re.compile(
    r"(?:关上|打开|放下|拿走|带走|带回|开回|送回|调成|改成|变成|"
    r"装入|装进|装到|装在|安装|放入|放进|放到|放在|移到|送到|"
    r"搭在|靠在|贴在|连到|接到|推进|延伸|穿过|切开|完成|用于|"
    r"塑造成|视为|称为|当作|看成|弄成|收好|锁上|停好|装回|装回去)$"
)
_BA_CONTINUATION = re.compile(
    r"^(?:在|向|往|从|沿|通过|用|给|让|使|被|由|再|并|不断|"
    r"逐一|逐段|一(?:块|件|个|段|步)块?地|穿|切|建|完成|实施|"
    r"安装|装|放|移|送|带|连接|贴|推进|延伸|承受|承担)"
)
_DISPOSAL_TAIL = re.compile(
    r"(?:^|[\s，,；;。.!?])(?:将|把)"
    r"[㐀-鿿A-Za-z0-9·一二两三四五六七八九十百千万]{2,24}$"
)
_DISPOSAL_PREDICATE = re.compile(
    r"(?:穿过|穿越|穿透|突破|切开|建成|完成|实施|安装|放置|移动|送到|"
    r"带到|连接|贴到|推进|延伸|用于|承受)"
)
_DISPOSAL_CONTINUATION = re.compile(
    r"^(?:在|向|往|从|沿|通过|用|给|让|使|被|由|再|并|不断|逐一|逐段|"
    r"穿|切|建|完成|实施|安装|放|移|送|带|连接|贴|推进|延伸|承受|承担)"
)
_PREDICATE_WITHOUT_COMPLEMENT = re.compile(
    r"(?:安装到|安装在|放到|放在|装到|装在|移到|移至|送到|带到|"
    r"连接到|贴到|推进到|延伸到|通向|进入|用于|承受)$"
)
_CAUSATIVE_OBJECT_TAIL = re.compile(
    r"(?:让|使|令)(?:我|我们|你|你们|他|她|它|他们|她们|它们)$"
)
_COMPLETED_PASSIVE_BEARING = re.compile(r"^(?:将)?由.{1,12}(?:来)?(?:承受|承担)$")
_LOCATIVE_PHRASE = re.compile(
    r"(?:在|沿着|沿|向|往|从|通过|借助|利用).{1,24}"
    r"(?:里|中|内|上|下|之间|之中|隧道|墙|墙体|土体|内部|表面)"
)
_LOCATIVE_CONTINUATION = re.compile(
    r"^(?:会|将|就|再|又|还|不断|持续|开始|继续|逐渐|直接|可以|能够|"
    r"逐一|逐段|被|把|贴|装|放|移|送|带|连接|推进|延伸|承受|承担)"
)
_STANDALONE_TEMPORAL_PHRASE = re.compile(r"(?:尤其是|特别是)?在.{2,36}(?:时|时候)")


def detect_governing_attachment_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return the first disposal/governing signal in legacy precedence order."""
    ba_tail = _BA_TAIL.search(features.compact_left) or _BA_ANY_TAIL.search(
        features.compact_left
    )
    ba_has_predicate = bool(
        ba_tail and _BA_PREDICATE_IN_OBJECT.search(ba_tail.group("object"))
    )
    ba_continues = bool(_BA_CONTINUATION.match(features.right))
    if ba_tail and not ba_has_predicate and ba_continues:
        return _match("ba construction is separated from its predicate")

    disposal_tail = _DISPOSAL_TAIL.search(features.left)
    disposal_has_predicate = _DISPOSAL_PREDICATE.search(features.left)
    disposal_continues = _DISPOSAL_CONTINUATION.match(features.right)
    if disposal_tail and not disposal_has_predicate and disposal_continues:
        return _match("disposal construction is separated from its predicate")

    predicate_without_complement = _PREDICATE_WITHOUT_COMPLEMENT.search(features.left)
    completed_passive_bearing = _COMPLETED_PASSIVE_BEARING.search(features.left)
    if predicate_without_complement and not completed_passive_bearing:
        return _match("predicate is separated from its required complement")

    if (
        _CAUSATIVE_OBJECT_TAIL.search(features.left)
        and not features.left_has_terminal_punctuation
    ):
        return _match("predicate is separated from its required complement")

    if _LOCATIVE_PHRASE.fullmatch(features.left) and _LOCATIVE_CONTINUATION.match(
        features.right
    ):
        return _match("locative phrase is separated from its predicate")

    if _STANDALONE_TEMPORAL_PHRASE.fullmatch(features.right):
        return _match("standalone temporal phrase is separated from its governing clause")

    return None
