"""Explicit, composable task-mode policies for translation quality work."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from subforge.core.translate.quality.session import TranslationSession


class SpeakerPolicyKind(str, Enum):
    MONOLOGUE = "monologue"
    DIALOGUE = "dialogue"


class LanguagePolicyKind(str, Enum):
    UNKNOWN = "unknown"
    SINGLE = "single_language"
    MIXED = "mixed_language"


@dataclass(frozen=True, slots=True)
class TranslationModePolicy:
    speaker: SpeakerPolicyKind
    language: LanguagePolicyKind
    prompt_suffix: str
    allow_cross_speaker_repair: bool
    language_metadata_available: bool

    @property
    def is_dialogue(self) -> bool:
        return self.speaker == SpeakerPolicyKind.DIALOGUE

    @property
    def is_mixed_language(self) -> bool:
        return self.language == LanguagePolicyKind.MIXED


def select_translation_mode_policy(
    session: TranslationSession | None = None,
    *,
    speakers: Iterable[str] = (),
    languages: Iterable[str] = (),
) -> TranslationModePolicy:
    """Select mode from immutable session data or behavior-compatible metadata."""
    if session is not None:
        speaker_values = {cue.speaker for cue in session.cues if cue.speaker}
        language_values = {cue.language for cue in session.cues if cue.language}
    else:
        speaker_values = {str(value) for value in speakers if str(value)}
        language_values = {str(value) for value in languages if str(value)}

    speaker_kind = (
        SpeakerPolicyKind.DIALOGUE
        if len(speaker_values) > 1
        else SpeakerPolicyKind.MONOLOGUE
    )
    if not language_values:
        language_kind = LanguagePolicyKind.UNKNOWN
    elif "mixed" in language_values or len(language_values) > 1:
        language_kind = LanguagePolicyKind.MIXED
    else:
        language_kind = LanguagePolicyKind.SINGLE

    return TranslationModePolicy(
        speaker=speaker_kind,
        language=language_kind,
        prompt_suffix="_multi" if speaker_kind == SpeakerPolicyKind.DIALOGUE else "",
        allow_cross_speaker_repair=speaker_kind == SpeakerPolicyKind.MONOLOGUE,
        language_metadata_available=bool(language_values),
    )


def translation_metadata_guidance(
    *,
    include_speakers: bool,
    include_languages: bool,
) -> str:
    """Render the existing metadata contract from explicit applicability flags."""
    rules: list[str] = []
    if include_speakers:
        rules.append(
            "<dialogue_metadata>\n"
            "current_subtitles values may include an anonymous speaker field. Use speaker "
            "changes and neighboring turns only to resolve who is responding, pronouns, "
            "ellipsis, intent, tone, and register. The speaker value is metadata, not "
            "subtitle text. Never translate, repeat, rename, or output speaker labels. "
            "Never merge dialogue turns or move meaning between keys.\n"
            "</dialogue_metadata>"
        )
    if include_languages:
        rules.append(
            "<source_language_metadata>\n"
            "source_language is authoritative ASR metadata: en means English, ja means "
            "Japanese, and mixed means that the key contains more than one source language. "
            "Translate every source-language span into the requested target language. "
            "Do not normalize a Japanese span into English, omit it, or leave kana copied "
            "into a Chinese translation. If an isolated Japanese cue ends with a connective "
            "such as ので or のに and the following cue changes language, render it as a "
            "complete natural target-language sentence rather than a dangling 'because' or "
            "'although' fragment. Language metadata is read-only and must not appear "
            "in the output.\n"
            "</source_language_metadata>"
        )
    return "\n\n" + "\n".join(rules) if rules else ""
