from dataclasses import dataclass

DEFAULT_CJK_HARD_LIMIT = 25
DEFAULT_ENGLISH_SOFT_LIMIT = 18
ENGLISH_HARD_LIMIT_OVERFLOW = 4


@dataclass(frozen=True)
class SubtitleLengthPolicy:
    """Resolved subtitle limits shared by prompts, validation, and cleanup."""

    cjk_hard_limit: int
    english_soft_limit: int
    english_hard_limit: int


def resolve_length_policy(
    max_word_count_cjk: int = DEFAULT_CJK_HARD_LIMIT,
    max_word_count_english: int = DEFAULT_ENGLISH_SOFT_LIMIT,
) -> SubtitleLengthPolicy:
    cjk_limit = max(1, int(max_word_count_cjk))
    english_soft_limit = max(1, int(max_word_count_english))
    return SubtitleLengthPolicy(
        cjk_hard_limit=cjk_limit,
        english_soft_limit=english_soft_limit,
        english_hard_limit=english_soft_limit + ENGLISH_HARD_LIMIT_OVERFLOW,
    )
