import json

from subforge.core.translate.factory import TranslatorFactory
from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.quality.canonical_evidence import (
    collect_canonical_evidence,
)
from subforge.core.translate.quality.pipeline_identity import (
    QUALITY_PIPELINE_FLAG,
    QUALITY_PIPELINE_REVISION,
    resolve_translation_pipeline_identity,
)
from subforge.core.translate.types import TargetLanguage, TranslatorType


def _parse_mapping(line: str) -> tuple[str, str] | None:
    if "ASR" not in line or " -> " not in line:
        return None
    heard, target = line.lstrip("- ").split(" -> ", 1)
    return heard, target.split(" (", 1)[0]


def test_canonical_evidence_counts_contract_without_retaining_text():
    terminology = "\n".join(
        (
            "- Name -> Naim (probable ASR correction)",
            "- Ravuelto -> Revuelto (phonetic ASR variant)",
            "- ordinary -> 普通",
            "- malformed ASR note",
        )
    )
    sources = {
        1: "Name explained the design.",
        2: "The Ravuelto is parked outside.",
        3: "Name returned later.",
    }

    summary = collect_canonical_evidence(
        terminology,
        sources,
        parse_mapping=_parse_mapping,
        has_document_support=(
            lambda _heard, canonical, _source: canonical == "Revuelto"
        ),
    )

    payload = summary.to_dict()
    assert payload == {
        "schema_version": 1,
        "counts": {
            "terminology_line_count": 4,
            "asr_labeled_line_count": 3,
            "parseable_mapping_count": 2,
            "mapping_with_source_match_count": 2,
            "source_mapping_match_count": 3,
            "supported_source_mapping_match_count": 1,
            "rejected_source_mapping_match_count": 2,
        },
    }
    serialized = json.dumps(payload)
    assert "Name" not in serialized
    assert "Naim" not in serialized
    assert "Ravuelto" not in serialized
    assert "Revuelto" not in serialized


def test_canonical_evidence_is_empty_without_context():
    summary = collect_canonical_evidence(
        "",
        {1: "Source text"},
        parse_mapping=_parse_mapping,
        has_document_support=lambda _heard, _canonical, _source: False,
    )

    assert all(value == 0 for value in summary.to_dict()["counts"].values())


def test_factory_collects_canonical_evidence_only_for_candidate_pipeline():
    legacy = TranslatorFactory.create_translator(
        translator_type=TranslatorType.OPENAI,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
    )
    candidate = TranslatorFactory.create_translator(
        translator_type=TranslatorType.OPENAI,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        pipeline_identity=resolve_translation_pipeline_identity(
            {
                QUALITY_PIPELINE_FLAG: "1",
                QUALITY_PIPELINE_REVISION: "phase8-observation-only",
            }
        ),
    )

    assert isinstance(legacy, LLMTranslator)
    assert isinstance(candidate, LLMTranslator)
    assert legacy._collect_canonical_evidence_telemetry is False
    assert candidate._collect_canonical_evidence_telemetry is True
