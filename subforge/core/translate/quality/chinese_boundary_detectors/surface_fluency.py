"""Chinese surface duplication and literal-fluency boundary defects."""

from __future__ import annotations

import difflib
import re

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .visible_pause import BoundarySignalMatch, _match

_DUPLICATED_BEARING_PREFIX = re.compile(r"[㐀-鿿]{2,12}所受(?:到)?的")
_SUPERLATIVE_TAIL = re.compile(r"(?:在)?(?:世界|全球)(?:上)?首次$")
_LITERAL_JAPANESE_DIFFICULTY = re.compile(
    r"(?:(?:很|非常|十分|极其)(?:困难|艰难)|难度(?:很|极|非常|极其)?高)"
    r"的(?:工程|施工)(?:内容|中身)$"
)
_DUPLICATED_CONSTRUCTION_NOMINALIZATION = re.compile(
    r"(?:(?:工程|项目)(?:即为|就是|是)?(?:如此|这样).{0,6}(?:工程|施工)|"
    r"(?:施工工程|工程施工|穿越施工工程))$"
)
_STACKED_CONNECTIVE = re.compile(
    r"^(?:所以|因此|不过|但是|但|而且|并且)[ \t]*"
    r"(?:但|但是|不过|所以|因此|而且|并且|是的)"
)
_ACCIDENTAL_DOUBLE_DE = re.compile(r"(?<!目)的的(?!确)")
_MALFORMED_DEMONSTRATIVE_CLASSIFIER = re.compile(r"这在(?:款|台|辆)(?:车|车型)")
_REPEATED_SHORT_UNITS = frozenset(
    {"不到", "做出", "进行", "采取", "选择", "获取", "使用", "开始", "继续", "变得"}
)
_REPEATED_MEANING_UNITS = frozenset({"阅读", "写作", "书写", "文字", "书籍", "信息"})
_REPEATED_PREDICATE = re.compile(
    r"((?:我|我们|你|你们|他|她|他们|她们|它们).{1,4})$"
)
_REPEATED_LOCATIVE = re.compile(r"在?([㐀-鿿A-Za-z]{2,8}(?:中|里|上|下|方面))")


def detect_surface_fluency_boundary(
    features: ChineseBoundaryFeatures,
) -> BoundarySignalMatch | None:
    """Return the first surface-fluency signal in legacy precedence order."""
    if features.left.endswith("所受的") and _DUPLICATED_BEARING_PREFIX.match(
        features.right
    ):
        return _match("possible duplicated boundary phrase")

    if _SUPERLATIVE_TAIL.search(features.left):
        return _match("superlative modifier is separated from its predicate")

    if any(
        _LITERAL_JAPANESE_DIFFICULTY.search(text)
        for text in (features.left, features.right)
    ):
        return _match("literal Japanese difficulty construction")

    if any(
        _DUPLICATED_CONSTRUCTION_NOMINALIZATION.search(text)
        for text in (features.left, features.right)
    ):
        return _match("duplicated construction nominalization")

    if _STACKED_CONNECTIVE.search(features.left) or _STACKED_CONNECTIVE.search(
        features.right
    ):
        return _match("stacked discourse connectives")

    if _ACCIDENTAL_DOUBLE_DE.search(features.left) or _ACCIDENTAL_DOUBLE_DE.search(
        features.right
    ):
        return _match("accidental duplicated Chinese particle")

    if _MALFORMED_DEMONSTRATIVE_CLASSIFIER.search(
        features.left
    ) or _MALFORMED_DEMONSTRATIVE_CLASSIFIER.search(features.right):
        return _match("malformed demonstrative classifier phrase")

    canonical_left = features.canonical_left
    canonical_right = features.canonical_right
    shared_limit = min(len(canonical_left), len(canonical_right), 18)
    if any(
        canonical_left.endswith(canonical_right[:size])
        for size in range(shared_limit, 2, -1)
    ):
        return _match("possible duplicated boundary phrase")

    if min(len(canonical_left), len(canonical_right)) >= 10:
        left_bigrams = {
            canonical_left[index : index + 2] for index in range(len(canonical_left) - 1)
        }
        right_bigrams = {
            canonical_right[index : index + 2] for index in range(len(canonical_right) - 1)
        }
        shorter_bigrams = min(len(left_bigrams), len(right_bigrams))
        overlap = (
            len(left_bigrams & right_bigrams) / shorter_bigrams if shorter_bigrams else 0.0
        )
        similarity = difflib.SequenceMatcher(
            None,
            canonical_left,
            canonical_right,
        ).ratio()
        if overlap >= 0.72 and similarity >= 0.72:
            return _match("possible duplicated boundary meaning")

    if any(
        canonical_left.endswith(unit) and canonical_right.startswith(unit)
        for unit in _REPEATED_SHORT_UNITS
    ):
        return _match("possible duplicated boundary phrase")

    if any(
        canonical_left.endswith(unit) and canonical_right.startswith(unit)
        for unit in _REPEATED_MEANING_UNITS
    ):
        return _match("possible duplicated boundary phrase")

    repeated_predicate = _REPEATED_PREDICATE.search(canonical_left)
    if repeated_predicate and canonical_right.startswith(repeated_predicate.group(1)):
        return _match("possible duplicated boundary phrase")

    repeated_locative = _REPEATED_LOCATIVE.match(canonical_right)
    if repeated_locative and repeated_locative.group(1) in canonical_left[-24:]:
        return _match("possible duplicated boundary meaning")

    return None
