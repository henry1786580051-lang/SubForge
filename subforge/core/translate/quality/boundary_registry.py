"""Declarative metadata for translation boundary diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BoundaryRuleScope(str, Enum):
    SOURCE = "source"
    TARGET = "target"
    DISPLAY = "display"


class BoundaryRuleKind(str, Enum):
    STRUCTURE = "structure"
    DUPLICATION = "duplication"
    FLUENCY = "fluency"


class BoundaryRuleLevel(str, Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True, slots=True)
class BoundaryRuleDefinition:
    """Stable metadata for one legacy-compatible boundary signal."""

    rule_id: str
    legacy_message: str
    scope: BoundaryRuleScope
    kind: BoundaryRuleKind
    level: BoundaryRuleLevel
    source_languages: tuple[str, ...]
    speaker_modes: tuple[str, ...]
    required_features: tuple[str, ...]
    exclusions: tuple[str, ...] = ()
    weight: int | None = None


_TARGET_RULE_IDS = {
    "standalone connective": "translation.boundary.target.standalone_connective",
    "demonstrative subject is stranded": "translation.boundary.target.demonstrative_subject",
    "existential predicate is separated from its object": "translation.boundary.target.existential_object",
    "sentence adverb is separated from its predicate": "translation.boundary.target.sentence_adverb",
    "subject and sentence adverb are separated from their predicate": "translation.boundary.target.subject_adverb",
    "coordinated subject is separated from its predicate": "translation.boundary.target.coordinated_subject",
    "relative clause is separated from its head noun": "translation.boundary.target.relative_clause_head",
    "demonstrative relative clause lacks its head noun": "translation.boundary.target.demonstrative_relative_head",
    "comparative object is omitted after a governing verb": "translation.boundary.target.comparative_object",
    "comparison frame is separated from its object": "translation.boundary.target.comparison_frame",
    "vehicle modifier is separated from its model name": "translation.boundary.target.vehicle_modifier",
    "reporting predicate is separated from its complement": "translation.boundary.target.reporting_complement",
    "classifier phrase is stranded": "translation.boundary.target.classifier_phrase",
    "demonstrative modifier is separated from its head noun": "translation.boundary.target.demonstrative_modifier",
    "count classifier lacks its contextual head noun": "translation.boundary.target.count_classifier_head",
    "ba construction is separated from its predicate": "translation.boundary.target.ba_predicate",
    "disposal construction is separated from its predicate": "translation.boundary.target.disposal_predicate",
    "predicate is separated from its required complement": "translation.boundary.target.predicate_complement",
    "locative phrase is separated from its predicate": "translation.boundary.target.locative_predicate",
    "standalone temporal phrase is separated from its governing clause": "translation.boundary.target.temporal_governing_clause",
    "possible duplicated boundary phrase": "translation.boundary.target.possible_duplicate_phrase",
    "superlative modifier is separated from its predicate": "translation.boundary.target.superlative_predicate",
    "literal Japanese difficulty construction": "translation.fluency.target.literal_japanese_difficulty",
    "duplicated construction nominalization": "translation.fluency.target.duplicated_nominalization",
    "stacked discourse connectives": "translation.fluency.target.stacked_connectives",
    "accidental duplicated Chinese particle": "translation.fluency.target.duplicated_particle",
    "malformed demonstrative classifier phrase": "translation.fluency.target.malformed_demonstrative_classifier",
    "adjective complement is missing": "translation.boundary.target.adjective_complement",
    "aspect predicate is separated from its complement": "translation.boundary.target.aspect_complement",
    "comparative noun modifier is stranded": "translation.boundary.target.comparative_noun_modifier",
    "comparison example is stranded": "translation.boundary.target.comparison_example",
    "comparison phrase is stranded": "translation.boundary.target.comparison_phrase",
    "connective stranded at previous subtitle end": "translation.boundary.target.trailing_connective",
    "consequence predicate is missing": "translation.boundary.target.consequence_predicate",
    "coordinated modifier may be stranded": "translation.boundary.target.possible_coordinated_modifier",
    "coordinated subject may be stranded": "translation.boundary.target.possible_coordinated_subject",
    "copular frame is separated from its result": "translation.boundary.target.copular_result",
    "distance modifier is separated from its noun": "translation.boundary.target.distance_modifier",
    "duplicated boundary connective": "translation.boundary.target.duplicated_connective",
    "literal fundamental calque": "translation.fluency.target.literal_fundamental_calque",
    "locative frame is separated from its complement": "translation.boundary.target.locative_complement",
    "material subject may be stranded": "translation.boundary.target.possible_material_subject",
    "motion predicate is separated from its destination": "translation.boundary.target.motion_destination",
    "negated comparison is split from its complement": "translation.boundary.target.negated_comparison",
    "nominal modifier is stranded": "translation.boundary.target.nominal_modifier",
    "nominal subject is separated from its copular predicate": "translation.boundary.target.nominal_copular_predicate",
    "numeric complement is stranded": "translation.boundary.target.numeric_complement",
    "numeric range is split": "translation.boundary.target.numeric_range",
    "particle stranded at next subtitle start": "translation.boundary.target.leading_particle",
    "percentage use-case predicate is stranded": "translation.boundary.target.percentage_use_case",
    "possible copular bridge": "translation.boundary.target.possible_copular_bridge",
    "possible demonstrative split": "translation.boundary.target.possible_demonstrative",
    "possible duplicated boundary meaning": "translation.boundary.target.possible_duplicate_meaning",
    "possible function-word split": "translation.boundary.target.possible_function_word",
    "possible pronoun boundary": "translation.boundary.target.possible_pronoun",
    "possible reporting frame": "translation.boundary.target.possible_reporting_frame",
    "possessive pronoun is separated from its head phrase": "translation.boundary.target.possessive_head",
    "predicate fragment starts at next subtitle": "translation.boundary.target.leading_predicate_fragment",
    "resultative predicate is stranded": "translation.boundary.target.resultative_predicate",
    "semantic frame is incomplete": "translation.boundary.target.semantic_frame",
    "standalone Chinese temporal fragment": "translation.boundary.target.standalone_temporal_fragment",
    "standalone subject is separated from its predicate": "translation.boundary.target.standalone_subject",
    "style modifier is separated from its head noun": "translation.boundary.target.style_modifier",
    "transitive predicate is split from its object": "translation.boundary.target.transitive_object",
    "transitive predicate is split from its quantified object": "translation.boundary.target.transitive_quantified_object",
    "unfinished Chinese adverbial predicate": "translation.boundary.target.unfinished_adverbial_predicate",
    "unfinished Chinese degree phrase": "translation.boundary.target.unfinished_degree_phrase",
    "unfinished Chinese grammatical structure": "translation.boundary.target.unfinished_grammar",
    "unfinished Chinese locative frame": "translation.boundary.target.unfinished_locative_frame",
    "unfinished Chinese locative subject": "translation.boundary.target.unfinished_locative_subject",
    "unfinished Chinese predicate or governing word": "translation.boundary.target.unfinished_predicate",
    "unfinished Chinese reason construction": "translation.boundary.target.unfinished_reason",
    "vague filler-only frame": "translation.boundary.target.vague_filler_frame",
    "number and unit are separated by a visible pause": "translation.boundary.display.number_unit_pause",
    "unfinished predicate or modifier crosses a visible pause": "translation.boundary.display.unfinished_pause",
}

_SOURCE_RULE_IDS = {
    "coordinate phrase crosses the subtitle boundary": "translation.boundary.source.coordinate_phrase",
    "degree complement crosses the subtitle boundary": "translation.boundary.source.degree_complement",
    "short source fragment crosses an unfinished sentence": "translation.boundary.source.short_fragment",
    "source continuation may require different target-language order": "translation.boundary.source.cross_language_order",
    "target-language modifier may be stranded at the next cue": "translation.boundary.source.target_modifier",
    "target-language temporal or governing phrase is unfinished": "translation.boundary.source.target_temporal_frame",
}

_SOFT_MESSAGES = frozenset(
    {
        "possible copular bridge",
        "possible demonstrative split",
        "possible duplicated boundary meaning",
        "possible duplicated boundary phrase",
        "possible function-word split",
        "possible pronoun boundary",
        "possible reporting frame",
        "coordinated modifier may be stranded",
        "coordinated subject may be stranded",
        "material subject may be stranded",
        *_SOURCE_RULE_IDS,
    }
)


def _definition(message: str, rule_id: str, scope: BoundaryRuleScope) -> BoundaryRuleDefinition:
    if rule_id.startswith("translation.fluency"):
        kind = BoundaryRuleKind.FLUENCY
    elif "duplicat" in message:
        kind = BoundaryRuleKind.DUPLICATION
    else:
        kind = BoundaryRuleKind.STRUCTURE
    if rule_id.startswith("translation.boundary.display"):
        scope = BoundaryRuleScope.DISPLAY
    return BoundaryRuleDefinition(
        rule_id=rule_id,
        legacy_message=message,
        scope=scope,
        kind=kind,
        level=(BoundaryRuleLevel.SOFT if message in _SOFT_MESSAGES else BoundaryRuleLevel.HARD),
        source_languages=("en",) if scope == BoundaryRuleScope.SOURCE else ("*",),
        speaker_modes=("monologue", "dialogue"),
        required_features=(
            "source_boundary_pair" if scope == BoundaryRuleScope.SOURCE else "target_boundary_pair",
        ),
    )


BOUNDARY_RULES = tuple(
    [
        *(
            _definition(message, rule_id, BoundaryRuleScope.TARGET)
            for message, rule_id in _TARGET_RULE_IDS.items()
        ),
        *(
            _definition(message, rule_id, BoundaryRuleScope.SOURCE)
            for message, rule_id in _SOURCE_RULE_IDS.items()
        ),
    ]
)

_BOUNDARY_RULE_BY_MESSAGE = {rule.legacy_message: rule for rule in BOUNDARY_RULES}
_BOUNDARY_RULE_BY_ID = {rule.rule_id: rule for rule in BOUNDARY_RULES}

if len(_BOUNDARY_RULE_BY_MESSAGE) != len(BOUNDARY_RULES):
    raise RuntimeError("Duplicate legacy boundary message in registry")
if len(_BOUNDARY_RULE_BY_ID) != len(BOUNDARY_RULES):
    raise RuntimeError("Duplicate boundary rule ID in registry")


def boundary_rule_for_message(message: str) -> BoundaryRuleDefinition | None:
    return _BOUNDARY_RULE_BY_MESSAGE.get(str(message or "").strip())


def boundary_rule_for_id(rule_id: str) -> BoundaryRuleDefinition | None:
    return _BOUNDARY_RULE_BY_ID.get(str(rule_id or "").strip())


def registered_boundary_messages() -> frozenset[str]:
    return frozenset(_BOUNDARY_RULE_BY_MESSAGE)
