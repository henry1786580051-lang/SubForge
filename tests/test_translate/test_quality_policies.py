import pytest

from subforge.core.translate.quality import (
    LanguagePolicyKind,
    SpeakerPolicyKind,
    select_translation_mode_policy,
    translation_metadata_guidance,
)


@pytest.mark.parametrize(
    ("speakers", "expected_kind", "expected_suffix", "may_cross"),
    [
        ((), SpeakerPolicyKind.MONOLOGUE, "", True),
        (("S1", "S1"), SpeakerPolicyKind.MONOLOGUE, "", True),
        (("S1", "S2"), SpeakerPolicyKind.DIALOGUE, "_multi", False),
    ],
)
def test_speaker_policy_matches_existing_prompt_selection(
    speakers,
    expected_kind,
    expected_suffix,
    may_cross,
):
    policy = select_translation_mode_policy(speakers=speakers)

    assert policy.speaker == expected_kind
    assert policy.prompt_suffix == expected_suffix
    assert policy.allow_cross_speaker_repair is may_cross


@pytest.mark.parametrize(
    ("languages", "expected"),
    [
        ((), LanguagePolicyKind.UNKNOWN),
        (("en",), LanguagePolicyKind.SINGLE),
        (("en", "en"), LanguagePolicyKind.SINGLE),
        (("en", "ja"), LanguagePolicyKind.MIXED),
        (("mixed",), LanguagePolicyKind.MIXED),
    ],
)
def test_language_policy_is_composable_with_speaker_mode(languages, expected):
    policy = select_translation_mode_policy(
        speakers=("S1", "S2"),
        languages=languages,
    )

    assert policy.speaker == SpeakerPolicyKind.DIALOGUE
    assert policy.language == expected
    assert policy.language_metadata_available is bool(languages)


def test_metadata_guidance_preserves_existing_contract_and_order():
    guidance = translation_metadata_guidance(
        include_speakers=True,
        include_languages=True,
    )

    assert guidance.startswith("\n\n<dialogue_metadata>")
    assert guidance.index("<dialogue_metadata>") < guidance.index(
        "<source_language_metadata>"
    )
    assert "Never merge dialogue turns or move meaning between keys." in guidance
    assert "Language metadata is read-only and must not appear" in guidance


def test_metadata_guidance_is_empty_when_window_has_no_metadata():
    assert (
        translation_metadata_guidance(
            include_speakers=False,
            include_languages=False,
        )
        == ""
    )
