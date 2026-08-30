"""Identity and artifact isolation for translation pipeline evaluations."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

QUALITY_PIPELINE_FLAG = "SUBFORGE_QUALITY_PIPELINE_V2"
QUALITY_PIPELINE_REVISION = "SUBFORGE_QUALITY_PIPELINE_REVISION"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})
_TOKEN_RE = re.compile(r"[^a-z0-9._-]+")


class TranslationPipelineVariant(str, Enum):
    """Product and development translation pipeline variants."""

    LEGACY = "legacy"
    CANDIDATE = "candidate"


@dataclass(frozen=True)
class TranslationPipelineIdentity:
    """Immutable names used by one translation pipeline execution."""

    variant: TranslationPipelineVariant
    revision: str

    @property
    def is_candidate(self) -> bool:
        return self.variant == TranslationPipelineVariant.CANDIDATE

    @property
    def cache_namespace(self) -> str:
        if not self.is_candidate:
            return ""
        return f"translation-quality:candidate:{self.revision}"

    def artifact_suffix(self, kind: str, *, task_id: str = "") -> str:
        """Return a legacy-compatible or collision-resistant candidate suffix."""
        if kind not in {"processed", "recovery"}:
            raise ValueError(f"Unsupported translation artifact kind: {kind}")
        if not self.is_candidate:
            return f"_{kind}"
        task_token = _safe_token(task_id)
        if not task_token:
            raise ValueError("Candidate translation artifacts require a task id")
        return f"_candidate_{self.revision}_{task_token}_{kind}"

    def result_metadata(self) -> dict[str, str]:
        return {
            "variant": self.variant.value,
            "revision": self.revision,
        }


LEGACY_TRANSLATION_PIPELINE = TranslationPipelineIdentity(
    variant=TranslationPipelineVariant.LEGACY,
    revision="legacy",
)


def _safe_token(value: str) -> str:
    normalized = _TOKEN_RE.sub("-", str(value or "").strip().casefold()).strip("-._")
    return normalized[:64]


def resolve_translation_pipeline_identity(
    environ: Mapping[str, str] | None = None,
) -> TranslationPipelineIdentity:
    """Resolve the hidden development pipeline flag, defaulting safely to legacy."""
    values = os.environ if environ is None else environ
    raw_flag = str(values.get(QUALITY_PIPELINE_FLAG, "")).strip().casefold()
    if raw_flag in _FALSE_VALUES:
        return LEGACY_TRANSLATION_PIPELINE
    if raw_flag not in _TRUE_VALUES:
        raise ValueError(
            f"{QUALITY_PIPELINE_FLAG} must be one of: 1, true, yes, on, 0, false, no, off"
        )

    revision = _safe_token(values.get(QUALITY_PIPELINE_REVISION, ""))
    if not revision:
        raise ValueError(
            f"{QUALITY_PIPELINE_REVISION} is required when the candidate pipeline is enabled"
        )
    return TranslationPipelineIdentity(
        variant=TranslationPipelineVariant.CANDIDATE,
        revision=revision,
    )
