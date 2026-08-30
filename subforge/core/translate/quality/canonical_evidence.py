"""Text-free observations about context-proposed ASR canonicalization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Mapping

CANONICAL_EVIDENCE_SCHEMA_VERSION = 1

_ASR_LABEL_RE = re.compile(
    r"(?:asr|phonetic|mishear|recognition|spoken\s+self-correction|"
    r"self-correction|转录|听写|同音|口误|自我修正)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class CanonicalEvidenceSummary:
    """Bounded counters that never retain source text or canonical names."""

    terminology_line_count: int = 0
    asr_labeled_line_count: int = 0
    parseable_mapping_count: int = 0
    mapping_with_source_match_count: int = 0
    source_mapping_match_count: int = 0
    supported_source_mapping_match_count: int = 0
    rejected_source_mapping_match_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CANONICAL_EVIDENCE_SCHEMA_VERSION,
            "counts": {
                "terminology_line_count": self.terminology_line_count,
                "asr_labeled_line_count": self.asr_labeled_line_count,
                "parseable_mapping_count": self.parseable_mapping_count,
                "mapping_with_source_match_count": self.mapping_with_source_match_count,
                "source_mapping_match_count": self.source_mapping_match_count,
                "supported_source_mapping_match_count": (
                    self.supported_source_mapping_match_count
                ),
                "rejected_source_mapping_match_count": (
                    self.rejected_source_mapping_match_count
                ),
            },
        }


def collect_canonical_evidence(
    terminology: str,
    source_by_index: Mapping[int, str],
    *,
    parse_mapping: Callable[[str], tuple[str, str] | None],
    has_document_support: Callable[[str, str, str], bool],
) -> CanonicalEvidenceSummary:
    """Measure the context-to-validator contract without retaining its content."""
    lines = tuple(line.strip() for line in str(terminology or "").splitlines() if line.strip())
    asr_labeled_line_count = sum(bool(_ASR_LABEL_RE.search(line)) for line in lines)
    parseable_mapping_count = 0
    mapping_with_source_match_count = 0
    source_mapping_match_count = 0
    supported_source_mapping_match_count = 0
    rejected_source_mapping_match_count = 0

    for line in lines:
        mapping = parse_mapping(line)
        if mapping is None:
            continue
        parseable_mapping_count += 1
        heard, canonical = mapping
        pattern = re.compile(rf"(?<!\w){re.escape(heard)}(?!\w)", flags=re.IGNORECASE)
        matched_sources = tuple(
            source
            for source in source_by_index.values()
            if source and pattern.search(source)
        )
        if matched_sources:
            mapping_with_source_match_count += 1
        for source in matched_sources:
            source_mapping_match_count += 1
            if has_document_support(heard, canonical, source):
                supported_source_mapping_match_count += 1
            else:
                rejected_source_mapping_match_count += 1

    return CanonicalEvidenceSummary(
        terminology_line_count=len(lines),
        asr_labeled_line_count=asr_labeled_line_count,
        parseable_mapping_count=parseable_mapping_count,
        mapping_with_source_match_count=mapping_with_source_match_count,
        source_mapping_match_count=source_mapping_match_count,
        supported_source_mapping_match_count=supported_source_mapping_match_count,
        rejected_source_mapping_match_count=rejected_source_mapping_match_count,
    )
