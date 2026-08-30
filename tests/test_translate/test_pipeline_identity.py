import pytest

from subforge.core.translate.quality.pipeline_identity import (
    LEGACY_TRANSLATION_PIPELINE,
    QUALITY_PIPELINE_FLAG,
    QUALITY_PIPELINE_REVISION,
    TranslationPipelineVariant,
    resolve_translation_pipeline_identity,
)


def test_translation_pipeline_defaults_to_legacy_without_changing_namespaces():
    identity = resolve_translation_pipeline_identity({})

    assert identity is LEGACY_TRANSLATION_PIPELINE
    assert identity.variant == TranslationPipelineVariant.LEGACY
    assert identity.cache_namespace == ""
    assert identity.artifact_suffix("processed") == "_processed"
    assert identity.artifact_suffix("recovery") == "_recovery"


def test_candidate_pipeline_isolates_artifacts_cache_and_metadata():
    identity = resolve_translation_pipeline_identity(
        {
            QUALITY_PIPELINE_FLAG: "true",
            QUALITY_PIPELINE_REVISION: "Phase 8 / R1",
        }
    )

    assert identity.variant == TranslationPipelineVariant.CANDIDATE
    assert identity.revision == "phase-8-r1"
    assert identity.cache_namespace == "translation-quality:candidate:phase-8-r1"
    assert identity.artifact_suffix("processed", task_id="ABC-123") == (
        "_candidate_phase-8-r1_abc-123_processed"
    )
    assert identity.artifact_suffix("recovery", task_id="ABC-123") == (
        "_candidate_phase-8-r1_abc-123_recovery"
    )
    assert identity.result_metadata() == {
        "variant": "candidate",
        "revision": "phase-8-r1",
    }


@pytest.mark.parametrize("flag", ["unexpected", "enabled", "2"])
def test_candidate_pipeline_rejects_ambiguous_feature_flags(flag):
    with pytest.raises(ValueError, match=QUALITY_PIPELINE_FLAG):
        resolve_translation_pipeline_identity({QUALITY_PIPELINE_FLAG: flag})


def test_candidate_pipeline_requires_an_explicit_revision_and_task_id():
    with pytest.raises(ValueError, match=QUALITY_PIPELINE_REVISION):
        resolve_translation_pipeline_identity({QUALITY_PIPELINE_FLAG: "1"})

    identity = resolve_translation_pipeline_identity(
        {
            QUALITY_PIPELINE_FLAG: "1",
            QUALITY_PIPELINE_REVISION: "phase8",
        }
    )
    with pytest.raises(ValueError, match="task id"):
        identity.artifact_suffix("processed")


def test_translation_pipeline_rejects_unknown_artifact_kind():
    with pytest.raises(ValueError, match="Unsupported translation artifact kind"):
        LEGACY_TRANSLATION_PIPELINE.artifact_suffix("preview")
