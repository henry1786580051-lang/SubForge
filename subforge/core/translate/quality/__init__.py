"""Pure translation quality checks shared by every provider."""

from subforge.core.translate.quality.text import (
    TranslationCompletenessReport,
    inspect_translation_batch,
    is_placeholder_translation,
    is_untranslated_output,
)

__all__ = [
    "TranslationCompletenessReport",
    "inspect_translation_batch",
    "is_placeholder_translation",
    "is_untranslated_output",
]
