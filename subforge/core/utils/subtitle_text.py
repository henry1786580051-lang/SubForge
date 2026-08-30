"""Pure text finalizers shared by subtitle import and export workflows."""

from __future__ import annotations

import re


def finalize_chinese_translation_punctuation(text: str) -> str:
    """Replace Chinese sentence punctuation without damaging ASCII identifiers."""
    translated = str(text or "")
    if not translated:
        return ""

    has_han = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", translated))
    cleaned: list[str] = []
    for index, character in enumerate(translated):
        if character in "，。":
            cleaned.append(" ")
            continue
        if character in ",." and has_han:
            previous = translated[index - 1] if index > 0 else ""
            following = translated[index + 1] if index + 1 < len(translated) else ""
            if (
                previous.isascii()
                and previous.isalnum()
                and following.isascii()
                and following.isalnum()
            ):
                cleaned.append(character)
            else:
                cleaned.append(" ")
            continue
        cleaned.append(character)

    finalized = "".join(cleaned)
    finalized = re.sub(r"、+\s*$", "", finalized)
    return re.sub(r"[ \t]{2,}", " ", finalized).strip()
