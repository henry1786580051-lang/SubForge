"""Conservative subtitle boundary scoring and word-timestamp repair."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence, cast

from subforge.core.asr.asr_data import ASRDataSeg, ASRWord, TimestampSource
from subforge.core.split.boundary_detectors import comparison as comparison_boundary
from subforge.core.split.boundary_detectors import coordination as coordination_boundary
from subforge.core.split.boundary_detectors import discourse as discourse_boundary
from subforge.core.split.boundary_detectors import entity as entity_boundary
from subforge.core.split.boundary_detectors import grammar as grammar_boundary
from subforge.core.split.boundary_detectors import numeric as numeric_boundary
from subforge.core.split.boundary_detectors import predicate as predicate_boundary
from subforge.core.split.boundary_features import (
    CLAUSE_RE as _CLAUSE_RE,
)
from subforge.core.split.boundary_features import (
    TERMINAL_RE as _TERMINAL_RE,
)
from subforge.core.split.boundary_features import (
    EnglishBoundaryFeatures,
    extract_english_boundary_features,
)
from subforge.core.split.boundary_features import (
    tokenize_english as _tokens,
)
from subforge.core.split.boundary_registry import (
    BoundaryScoreContribution,
    record_boundary_score,
)
from subforge.core.utils.logger import setup_logger

logger = setup_logger("subtitle_boundary")

SOFT_MAX_WORDS = 18
HARD_MAX_WORDS = 22
HARD_MAX_CJK_CHARS = 25
MAX_JAPANESE_ATOMIC_OVERFLOW_CHARS = 4
MAX_JAPANESE_ATOMIC_DURATION_MS = 8000
MAX_BOUNDARY_SHIFT_WORDS = 9
MAX_RELOCATABLE_GAP_MS = 1800
MAX_DIARIZATION_GLITCH_GAP_MS = 250
MAX_STRONG_DEPENDENCY_DIARIZATION_GLITCH_GAP_MS = 350
MAX_EXACT_DEPENDENCY_DIARIZATION_GLITCH_GAP_MS = 450
MAX_TRAILING_ADVERB_DIARIZATION_GLITCH_GAP_MS = 3500
MAX_DUPLICATE_CORRECTION_GAP_MS = 600
MIN_REPAIR_IMPROVEMENT = 8.0

_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_KATAKANA_RE = re.compile(r"^[\u30a0-\u30ff\u31f0-\u31ffー]+$")
_JAPANESE_PARTICLE_HEAD_RE = re.compile(
    r"^(?:が|を|は|に|へ|と|で|の|も|や|から|まで|より|ので|のに)"
)
_JAPANESE_FILLER_TAIL_RE = re.compile(
    r"(?:こちら|そちら|あちら)(?:の|の?ですね)$|(?:この|その|あの)$|のですね$"
)
_JAPANESE_SHORT_TOPIC_TAIL_RE = re.compile(r"^(?:これ|それ|あれ|こちら|そちら|あちら)(?:が|は)$")
_JAPANESE_PREFERRED_TAIL_RE = re.compile(
    r"(?:が|を|は|に|へ|と|で|も|や|から|まで|より|ので|のに|[。！？、])$"
)
_JAPANESE_DANGLING_TAIL_RE = re.compile(
    r"(?:の|や|及び|または|そして|しかし|けど|けれど|ので|のに|"
    r"った|いた|した|ている|ていく|ない|れる|られる|といった|しい|"
    r"非常に|とても|かなり|最も|より)$"
)
_JAPANESE_INFLECTION_HEAD_RE = re.compile(
    r"^(?:った|って|いた|いて|れる|られる|ない|ます|する|した|して|せる|させる|たい)"
)
_JAPANESE_BOUND_FORMS = (
    "これが",
    "それが",
    "あれが",
    "こちらが",
    "そちらが",
    "あちらが",
    "った",
    "って",
    "いた",
    "いて",
    "しい",
    "です",
    "ですね",
    "でした",
    "ます",
    "ました",
    "ません",
    "ますので",
    "という",
    "といった",
    "ている",
    "ていく",
    "てきます",
    "となります",
)

_HARD_DANGLING_TAILS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "because",
    "although",
    "though",
    "unless",
    "when",
    "while",
    "whereas",
    "where",
    "how",
    "if",
    "as",
    "at",
    "about",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "over",
    "through",
    "that",
    "to",
    "under",
    "whether",
    "with",
}

_SUBJECT_TAILS = {
    "anybody",
    "anyone",
    "everybody",
    "everyone",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "this",
    "these",
    "those",
    "nobody",
    "noone",
    "somebody",
    "someone",
    "who",
    "which",
}

_RESPONSE_INTERJECTIONS = {
    "absolutely",
    "exactly",
    "okay",
    "ok",
    "right",
    "sure",
    "yeah",
    "yep",
    "yes",
}

_INCOMPLETE_PREDICATE_TAILS = {
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "will",
    "would",
    "not",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "can't",
    "couldn't",
    "won't",
    "wouldn't",
}

_SUBJECT_AUX_TAILS = {
    "that's",
    "this's",
    "who's",
    "which's",
    "i'd",
    "i'm",
    "you're",
    "he's",
    "she's",
    "it's",
    "we're",
    "they're",
    "i've",
    "you've",
    "we've",
    "they've",
    "i'll",
    "you'd",
    "you'll",
    "he'd",
    "he'll",
    "she'd",
    "she'll",
    "it'd",
    "it'll",
    "we'd",
    "we'll",
    "they'd",
    "they'll",
}

_MODIFIER_TAILS = {
    "also",
    "another",
    "any",
    "big",
    "both",
    "closest",
    "cocooned",
    "each",
    "electric",
    "exact",
    "every",
    "few",
    "further",
    "good",
    "great",
    "high",
    "huge",
    "important",
    "interesting",
    "large",
    "larger",
    "less",
    "little",
    "literate",
    "low",
    "luxury",
    "main",
    "many",
    "medium",
    "more",
    "most",
    "much",
    "new",
    "other",
    "our",
    "public",
    "puny",
    "red",
    "revised",
    "drastically",
    "definitely",
    "her",
    "his",
    "its",
    "massive",
    "my",
    "near",
    "really",
    "serrated",
    "several",
    "small",
    "some",
    "specialized",
    "still",
    "traditional",
    "their",
    "thick",
    "very",
    "your",
}

_COMPARATIVE_MODIFIER_TAILS = {
    "better",
    "bigger",
    "broader",
    "brighter",
    "cheaper",
    "closer",
    "closest",
    "faster",
    "fewer",
    "greater",
    "higher",
    "larger",
    "lesser",
    "longer",
    "lower",
    "narrower",
    "newer",
    "older",
    "shorter",
    "slower",
    "smaller",
    "taller",
    "thicker",
    "thinner",
    "wider",
    "worse",
}

_ATTRIBUTIVE_TAILS = {
    "different",
    "distance",
    "first",
    "old",
    "own",
    "personal",
    "previous",
    "same",
    "second",
    "similar",
}

_SENTENCE_ADVERB_TAILS = {
    "basically",
    "effectively",
    "essentially",
    "frankly",
    "generally",
    "honestly",
    "now",
    "overall",
    "technically",
    "today",
    "tonight",
    "ultimately",
    "yesterday",
}

_ATTRIBUTIVE_DEGREE_ADVERBS = {
    "especially",
    "exceptionally",
    "extremely",
    "fairly",
    "genuinely",
    "highly",
    "incredibly",
    "particularly",
    "pretty",
    "quite",
    "rather",
    "really",
    "relatively",
    "remarkably",
    "strikingly",
    "surprisingly",
    "truly",
    "unusually",
    "very",
}

_OPEN_COMPLEMENT_TAILS = {
    "devoting",
    "give",
    "gives",
    "gave",
    "get",
    "gets",
    "getting",
    "got",
    "grabbing",
    "just",
    "put",
    "puts",
    "putting",
    "spend",
    "spending",
    "spent",
}
_PHRASAL_PARTICLES = {"away", "back", "down", "in", "off", "on", "out", "over", "up"}

_DANGLING_PHRASES = {
    ("a", "lot", "of"),
    ("all", "the", "way"),
    ("as", "much", "as"),
    ("because", "of"),
    ("going", "to"),
    ("has", "been"),
    ("have", "been"),
    ("high", "end"),
    ("higher", "end"),
    ("in", "order"),
    ("kind", "of"),
    ("leading", "up", "to"),
    ("need", "to"),
    ("not", "only"),
    ("one", "of"),
    ("one", "of", "these"),
    ("one", "of", "those"),
    ("over", "the", "last"),
    ("start", "generating"),
    ("tends", "to"),
    ("used", "to"),
    ("want", "to"),
}

_DEPENDENCY_PAIRS = {
    ("ago", "or"),
    ("american", "sedans"),
    ("above", "ground"),
    ("bass", "sound"),
    ("bass", "systems"),
    ("better", "sound"),
    ("big", "picture"),
    ("body", "american"),
    ("case", "studies"),
    ("condition", "or"),
    ("damn", "near"),
    ("european", "influence"),
    ("exhaust", "tips"),
    ("experimental", "standards"),
    ("fall", "out"),
    ("flip", "switch"),
    ("first", "gear"),
    ("second", "gear"),
    ("third", "gear"),
    ("fourth", "gear"),
    ("fifth", "gear"),
    ("sixth", "gear"),
    ("generating", "power"),
    ("good", "jobs"),
    ("high", "end"),
    ("higher", "echelon"),
    ("home", "about"),
    ("grabbing", "attention"),
    ("leading", "up"),
    ("nuclear", "plants"),
    ("nuclear", "power"),
    ("only", "way"),
    ("main", "way"),
    ("matters", "what"),
    ("medium", "firmness"),
    ("other", "socks"),
    ("past", "experiences"),
    ("photo", "shoot"),
    ("power", "plants"),
    ("power", "steering"),
    ("point", "out"),
    ("performance", "pack"),
    ("pretty", "standard"),
    ("public", "college"),
    ("rev", "matching"),
    ("rpm", "gauge"),
    ("right", "now"),
    ("same", "sort"),
    ("serrated", "edge"),
    ("so", "much"),
    ("sound", "system"),
    ("specialized", "employees"),
    ("traditional", "hybrid"),
    ("up", "to"),
    ("thank", "you"),
    ("turn", "signals"),
    ("value", "judgment"),
    ("whole", "lot"),
    ("which", "i"),
    ("write", "home"),
    ("that's", "what"),
}

_COPULA_COMPLEMENT_TAILS = {
    "car",
    "here",
    "interior",
    "point",
    "problem",
    "reason",
    "story",
    "thing",
}

_DEPENDENT_RIGHT_HEADS = {
    "a",
    "an",
    "at",
    "the",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "outside",
    "over",
    "than",
    "through",
    "to",
    "under",
    "within",
    "with",
}

_PREFERRED_CLAUSE_HEADS = {
    "although",
    "and",
    "because",
    "but",
    "however",
    "if",
    "now",
    "or",
    "so",
    "that",
    "then",
    "unless",
    "when",
    "while",
}

_RELATIVE_CLAUSE_HEADS = {"that", "which", "who", "whom", "whose"}
_CONTRACTED_RELATIVE_CLAUSE_HEADS = {"that's", "who's"}
_TRANSLATION_SENSITIVE_HEADS = {"after", "before", "until", "when", "where"}
_HYPHENATED_ATTRIBUTIVE_TAILS = {
    "all-wheel",
    "day-to-day",
    "end-to-end",
    "four-wheel",
    "three-legged",
    "front-wheel",
    "high-speed",
    "long-term",
    "low-speed",
    "one-on-one",
    "point-to-point",
    "real-world",
    "rear-wheel",
    "short-term",
}
_THAT_COMPLEMENT_TAILS = {
    ("find", "out"),
    ("found", "out"),
    ("know",),
    ("knew",),
    ("say",),
    ("said",),
    ("think",),
    ("thought",),
}


def _ends_with_phrase(tokens: Sequence[str]) -> bool:
    return any(
        len(tokens) >= len(phrase) and tuple(tokens[-len(phrase) :]) == phrase
        for phrase in _DANGLING_PHRASES
    )


@dataclass(frozen=True)
class BoundaryAssessment:
    risk: int
    reasons: tuple[str, ...]
    contributions: tuple[BoundaryScoreContribution, ...] = ()

    @property
    def unstable(self) -> bool:
        return self.risk >= 20

    @property
    def registered_risk(self) -> int:
        return sum(contribution.weight for contribution in self.contributions)

    @property
    def unregistered_risk(self) -> int:
        return self.risk - self.registered_risk


def _score_english_boundary_foundation(
    features: EnglishBoundaryFeatures,
    reasons: list[str],
    contributions: list[BoundaryScoreContribution],
) -> int:
    left_tokens = list(features.left_tokens)
    tail = features.tail
    previous = features.previous
    semantic_tail = features.semantic_tail
    risk = 0

    if grammar_boundary.incomplete_multiword(
        left_ends_with_dangling_phrase=_ends_with_phrase(left_tokens)
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.incomplete_multiword",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.dangling_function_word(
        features,
        tail_is_hard_dangling=tail in _HARD_DANGLING_TAILS,
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.dangling_function_word",
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": tail},
        )
    if grammar_boundary.dangling_function_word_before_filler(
        features,
        semantic_tail_is_hard_dangling=semantic_tail in _HARD_DANGLING_TAILS,
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.dangling_function_word_before_filler",
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": semantic_tail},
        )
    if grammar_boundary.dangling_subject(
        features,
        tail_is_subject=tail in _SUBJECT_TAILS,
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.dangling_subject",
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": tail},
        )
    if grammar_boundary.standalone_subject(
        features,
        tail_is_subject=tail in _SUBJECT_TAILS,
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.standalone_subject",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.sentence_final_subject(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.sentence_final_subject",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.subject_before_adverbial_predicate(
        features,
        tail_is_subject=tail in _SUBJECT_TAILS,
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.subject_before_adverbial_predicate",
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": tail},
        )
    if predicate_boundary.subject_adverb(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.subject_adverb",
            reasons=reasons,
            contributions=contributions,
        )
    if coordination_boundary.relative_clause_subject(features):
        risk += record_boundary_score(
            "split.boundary.english.coordination.relative_clause_subject",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.incomplete_predicate(
        features,
        tail_is_incomplete_predicate=tail in _INCOMPLETE_PREDICATE_TAILS,
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.incomplete_predicate",
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": tail},
        )
    if grammar_boundary.contracted_negative_auxiliary(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.incomplete_predicate",
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": tail},
        )
    if predicate_boundary.negative_auxiliary_complement(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.negative_auxiliary_complement",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.linking_verb_complement(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.linking_verb_complement",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.subject_auxiliary(
        features,
        tail_is_subject_auxiliary=tail in _SUBJECT_AUX_TAILS,
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.subject_auxiliary",
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": tail},
        )
    if predicate_boundary.progressive_complement(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.progressive_complement",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.adverb_gerund(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.adverb_gerund",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.auxiliary_participle(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.auxiliary_participle",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.dangling_modifier(
        features,
        tail_is_modifier=tail in _MODIFIER_TAILS,
    ):
        rule_id = (
            "split.boundary.english.grammar.dangling_modifier_strong"
            if tail in {"also", "definitely", "main", "really"}
            else "split.boundary.english.grammar.dangling_modifier"
        )
        risk += record_boundary_score(
            rule_id,
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": tail},
        )
    if grammar_boundary.dangling_attributive(
        features,
        tail_is_attributive=tail in _ATTRIBUTIVE_TAILS,
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.dangling_attributive",
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": tail},
        )
    if grammar_boundary.filler_noun_modifier(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.filler_noun_modifier",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.determiner_head(
        features,
        tail_is_attributive=tail in _ATTRIBUTIVE_TAILS,
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.determiner_head",
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": tail},
        )
    if predicate_boundary.sentence_adverb_finite(
        features,
        tail_is_sentence_adverb=tail in _SENTENCE_ADVERB_TAILS,
    ):
        risk += record_boundary_score(
            "split.boundary.english.predicate.sentence_adverb_finite",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.auxiliary_sentence_adverb(
        features,
        tail_is_sentence_adverb=tail in _SENTENCE_ADVERB_TAILS,
        previous_is_subject_auxiliary=previous in _SUBJECT_AUX_TAILS,
    ):
        risk += record_boundary_score(
            "split.boundary.english.predicate.auxiliary_sentence_adverb",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.time_frame_participle(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.time_frame_participle",
            reasons=reasons,
            contributions=contributions,
        )
    return risk


def _score_english_boundary_relations(
    features: EnglishBoundaryFeatures,
    reasons: list[str],
    contributions: list[BoundaryScoreContribution],
) -> int:
    tail = features.tail
    risk = 0

    if comparison_boundary.clause_after_than(features):
        risk += record_boundary_score(
            "split.boundary.english.comparison.clause_after_than",
            reasons=reasons,
            contributions=contributions,
        )
    if comparison_boundary.noun_phrase_before_than(features):
        risk += record_boundary_score(
            "split.boundary.english.comparison.noun_phrase_before_than",
            reasons=reasons,
            contributions=contributions,
        )
    if comparison_boundary.scalar_predicate(features):
        risk += record_boundary_score(
            "split.boundary.english.comparison.scalar_predicate",
            reasons=reasons,
            contributions=contributions,
        )
    if comparison_boundary.frame_object(features):
        risk += record_boundary_score(
            "split.boundary.english.comparison.frame_object",
            reasons=reasons,
            contributions=contributions,
        )
    if numeric_boundary.measurement_comparative(features):
        risk += record_boundary_score(
            "split.boundary.english.numeric.measurement_comparative",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.infinitive_adverb(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.infinitive_adverb",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.participle_quantified_object(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.participle_quantified_object",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.frequency_quantified_statement(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.frequency_quantified_statement",
            reasons=reasons,
            contributions=contributions,
        )
    if numeric_boundary.approximate_magnitude(features):
        risk += record_boundary_score(
            "split.boundary.english.numeric.approximate_magnitude",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.modal_adverb(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.modal_adverb",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.contrastive_prepositional_frame(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.contrastive_prepositional_frame",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.distance_location_noun(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.distance_location_noun",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.morphological_attributive_modifier(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.attributive_or_comparative_modifier",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.determiner_adjective_head(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.determiner_adjective_head",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.determiner_degree_modifier_head(
        features,
        tail_is_attributive_degree_adverb=tail in _ATTRIBUTIVE_DEGREE_ADVERBS,
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.determiner_degree_modifier_head",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.sentence_final_temporal_adverb(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.sentence_final_temporal_adverb",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.lexical_comparative_modifier(
        features,
        tail_is_comparative_modifier=tail in _COMPARATIVE_MODIFIER_TAILS,
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.attributive_or_comparative_modifier",
            reasons=reasons,
            contributions=contributions,
        )
    if entity_boundary.powertrain_vehicle_name(features):
        risk += record_boundary_score(
            "split.boundary.english.entity.powertrain_vehicle_name",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.participle_complement(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.participle_complement",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.postpositive_participle_modifier(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.postpositive_participle_modifier",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.reporting_quoted_object(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.reporting_quoted_object",
            reasons=reasons,
            contributions=contributions,
        )
    if numeric_boundary.value_unit_or_noun(features):
        risk += record_boundary_score(
            "split.boundary.english.numeric.value_unit_or_noun",
            reasons=reasons,
            contributions=contributions,
        )
    if numeric_boundary.calendar_month_year(features):
        risk += record_boundary_score(
            "split.boundary.english.numeric.calendar_month_year",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.coordinated_noun_phrase(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.coordinated_noun_phrase",
            reasons=reasons,
            contributions=contributions,
        )
    if coordination_boundary.paired_contrast(features):
        risk += record_boundary_score(
            "split.boundary.english.coordination.paired_contrast",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.prepositional_gerund_complement(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.prepositional_gerund_complement",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.preposition_gerund(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.preposition_gerund",
            reasons=reasons,
            contributions=contributions,
        )
    if coordination_boundary.directional_names(features):
        risk += record_boundary_score(
            "split.boundary.english.coordination.directional_names",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.condition_qualified_predicate(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.condition_qualified_predicate",
            reasons=reasons,
            contributions=contributions,
        )
    hyphenated_tail = grammar_boundary.hyphenated_attributive_tail(
        features,
        allowed_tails=_HYPHENATED_ATTRIBUTIVE_TAILS,
    )
    if hyphenated_tail:
        risk += record_boundary_score(
            "split.boundary.english.grammar.hyphenated_attributive",
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": hyphenated_tail},
        )
    return risk


def _score_english_boundary_completions(
    features: EnglishBoundaryFeatures,
    reasons: list[str],
    contributions: list[BoundaryScoreContribution],
) -> int:
    tail = features.tail
    head = features.head
    risk = 0

    if entity_boundary.vehicle_brand_model(features):
        risk += record_boundary_score(
            "split.boundary.english.entity.vehicle_brand_model",
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": tail},
        )
    if grammar_boundary.open_complement(
        features,
        tail_is_open_complement=tail in _OPEN_COMPLEMENT_TAILS,
    ):
        rule_id = (
            "split.boundary.english.grammar.open_complement_strong"
            if tail in {"devoting", "getting"}
            else "split.boundary.english.grammar.open_complement"
        )
        risk += record_boundary_score(
            rule_id,
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": tail},
        )
    if grammar_boundary.expect_to_always(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.expect_to_always",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.relative_clause_subject(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.relative_clause_subject",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.progressive_object(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.progressive_object",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.quantifying_phrase_noun(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.quantifying_phrase_noun",
            reasons=reasons,
            contributions=contributions,
        )
    if numeric_boundary.compound_modifier(features):
        risk += record_boundary_score(
            "split.boundary.english.numeric.compound_modifier",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.topic_frame(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.topic_frame",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.degree_so(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.degree_so",
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": head},
        )
    if comparison_boundary.repeated_degree(features):
        risk += record_boundary_score(
            "split.boundary.english.comparison.repeated_degree",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.up_and_running(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.up_and_running",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.up_and_running_aspect(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.up_and_running_aspect",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.up_and_running_subject(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.up_and_running_subject",
            reasons=reasons,
            contributions=contributions,
        )
    if numeric_boundary.multiplier_or_unit(features):
        risk += record_boundary_score(
            "split.boundary.english.numeric.multiplier_or_unit",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.single_word_completion(
        features,
        hard_max_words=HARD_MAX_WORDS,
    ):
        # A lone lowercase completion is exceptionally strong evidence that a
        # brief speaker-label flip occurred inside one phrase. Keep this above
        # the strong-dependency threshold used by _is_hard_boundary.
        risk += record_boundary_score(
            "split.boundary.english.grammar.single_word_completion",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.short_noun_subject(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.short_noun_subject",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.phrasal_verb(
        features,
        head_is_phrasal_particle=head in _PHRASAL_PARTICLES,
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.phrasal_verb",
            reasons=reasons,
            contributions=contributions,
            reason_values={"left": tail, "right": head},
        )
    if grammar_boundary.take_away(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.take_away",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.subject_complement(
        features,
        tail_is_copula_complement=tail in _COPULA_COMPLEMENT_TAILS,
    ):
        risk += record_boundary_score(
            "split.boundary.english.predicate.subject_complement",
            reasons=reasons,
            contributions=contributions,
            reason_values={"head": head},
        )
    if predicate_boundary.omitted_relative_one(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.omitted_relative_one",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.copula_parenthetical_complement(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.copula_parenthetical_complement",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.new_clause_connective(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.new_clause_connective",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.emphatic_inversion_complement(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.emphatic_inversion_complement",
            reasons=reasons,
            contributions=contributions,
        )
    return risk


def _score_english_boundary_discourse(
    features: EnglishBoundaryFeatures,
    reasons: list[str],
    contributions: list[BoundaryScoreContribution],
) -> int:
    tail = features.tail
    head = features.head
    risk = 0

    if discourse_boundary.filler_demonstrative_noun(features):
        risk += record_boundary_score(
            "split.boundary.english.discourse.filler_demonstrative_noun",
            reasons=reasons,
            contributions=contributions,
        )
    if discourse_boundary.standalone_bridge(features):
        risk += record_boundary_score(
            "split.boundary.english.discourse.standalone_bridge",
            reasons=reasons,
            contributions=contributions,
        )
    if discourse_boundary.filler_only_frame(features):
        risk += record_boundary_score(
            "split.boundary.english.discourse.filler_only_frame",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.standalone_contrast_frame(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.standalone_contrast_frame",
            reasons=reasons,
            contributions=contributions,
        )
    if discourse_boundary.sentence_opening_filler(features):
        risk += record_boundary_score(
            "split.boundary.english.discourse.sentence_opening_filler",
            reasons=reasons,
            contributions=contributions,
        )
    if discourse_boundary.incomplete_predicate_before_filler(features):
        risk += record_boundary_score(
            "split.boundary.english.discourse.incomplete_predicate_before_filler",
            reasons=reasons,
            contributions=contributions,
        )
    if discourse_boundary.predicate_after_modifier_strong(features):
        risk += record_boundary_score(
            "split.boundary.english.discourse.predicate_after_modifier_strong",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.negative_existential_complement(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.negative_existential_complement",
            reasons=reasons,
            contributions=contributions,
        )
    if discourse_boundary.predicate_after_modifier(features):
        risk += record_boundary_score(
            "split.boundary.english.discourse.predicate_after_modifier",
            reasons=reasons,
            contributions=contributions,
        )
    if discourse_boundary.sentence_opening_opinion(features):
        risk += record_boundary_score(
            "split.boundary.english.discourse.sentence_opening_opinion",
            reasons=reasons,
            contributions=contributions,
        )
    if discourse_boundary.parenthetical_opinion(
        features,
        head_is_incomplete_predicate=head in _INCOMPLETE_PREDICATE_TAILS,
    ):
        risk += record_boundary_score(
            "split.boundary.english.discourse.parenthetical_opinion",
            reasons=reasons,
            contributions=contributions,
        )
    if discourse_boundary.standalone_opinion(features):
        risk += record_boundary_score(
            "split.boundary.english.discourse.standalone_opinion",
            reasons=reasons,
            contributions=contributions,
        )
    if coordination_boundary.short_predicate_continuation(features):
        risk += record_boundary_score(
            "split.boundary.english.coordination.short_predicate_continuation",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.sentence_opening_time_marker(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.sentence_opening_time_marker",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.do_not_get_me_wrong(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.do_not_get_me_wrong",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.reason_clause_subject(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.reason_clause_subject",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.sentence_opening_time_adverb(
        features,
        tail_is_sentence_adverb=tail in _SENTENCE_ADVERB_TAILS,
        head_is_subject_or_determiner=head in (_SUBJECT_TAILS | {"our", "the", "a", "an"}),
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.sentence_opening_time_adverb",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.contrastive_beneficiary(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.contrastive_beneficiary",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.lexical_unit(
        tail_head_is_dependency_pair=(tail, head) in _DEPENDENCY_PAIRS
        or (tail == "heart" and features.right_tokens[:2] == ("and", "soul"))
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.lexical_unit",
            reasons=reasons,
            contributions=contributions,
            reason_values={"left": tail, "right": head},
        )
    if grammar_boundary.direction_here(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.direction_here",
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": tail},
        )
    return risk


def _score_english_boundary_clause_ownership(
    features: EnglishBoundaryFeatures,
    reasons: list[str],
    contributions: list[BoundaryScoreContribution],
) -> int:
    head = features.head
    risk = 0

    if numeric_boundary.mixed_measurement(features):
        risk += record_boundary_score(
            "split.boundary.english.numeric.mixed_measurement",
            reasons=reasons,
            contributions=contributions,
        )
    if coordination_boundary.automotive_intake_exhaust(features):
        risk += record_boundary_score(
            "split.boundary.english.coordination.automotive_intake_exhaust",
            reasons=reasons,
            contributions=contributions,
        )
    if entity_boundary.alphanumeric_model_alternative(features):
        risk += record_boundary_score(
            "split.boundary.english.entity.alphanumeric_model_alternative",
            reasons=reasons,
            contributions=contributions,
        )
    if comparison_boundary.same_as(features):
        risk += record_boundary_score(
            "split.boundary.english.comparison.same_as",
            reasons=reasons,
            contributions=contributions,
        )
    if comparison_boundary.frame_counterpart(features):
        risk += record_boundary_score(
            "split.boundary.english.comparison.frame_counterpart",
            reasons=reasons,
            contributions=contributions,
        )
    if comparison_boundary.negated_complement(features):
        risk += record_boundary_score(
            "split.boundary.english.comparison.negated_complement",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.what_is_so_after_demonstrative(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.what_is_so_complement",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.what_is_so_adjective(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.what_is_so_complement",
            reasons=reasons,
            contributions=contributions,
        )
    if comparison_boundary.dynamic_complement(features):
        risk += record_boundary_score(
            "split.boundary.english.comparison.dynamic_complement",
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": head},
        )
    if comparison_boundary.example(features):
        risk += record_boundary_score(
            "split.boundary.english.comparison.example",
            reasons=reasons,
            contributions=contributions,
        )
    if numeric_boundary.range_conjunction(features):
        risk += record_boundary_score(
            "split.boundary.english.numeric.range_conjunction",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.transitive_object_basic(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.transitive_object_basic",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.transitive_object_extended(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.transitive_object_extended",
            reasons=reasons,
            contributions=contributions,
        )
    if discourse_boundary.transitive_predicate_before_filler(features):
        risk += record_boundary_score(
            "split.boundary.english.discourse.transitive_predicate_before_filler",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.perfect_reporting_content(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.perfect_reporting_content",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.transitive_pronoun_object(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.transitive_pronoun_object",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.perfect_reporting_after_adverb(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.perfect_reporting_after_adverb",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.context_dependent_adjective(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.context_dependent_adjective",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.reporting_content(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.reporting_content",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.reporting_copular_content(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.reporting_copular_content",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.embedded_question_complement(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.embedded_question_complement",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.interrogative_complement(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.interrogative_complement",
            reasons=reasons,
            contributions=contributions,
        )
    if coordination_boundary.reported_subject_member(features):
        risk += record_boundary_score(
            "split.boundary.english.coordination.reported_subject_member",
            reasons=reasons,
            contributions=contributions,
        )
    if coordination_boundary.subject_predicate(features):
        risk += record_boundary_score(
            "split.boundary.english.coordination.subject_predicate",
            reasons=reasons,
            contributions=contributions,
        )
    if coordination_boundary.noun_subject_shared_predicate_simple(features):
        risk += record_boundary_score(
            "split.boundary.english.coordination.noun_subject_shared_predicate",
            reasons=reasons,
            contributions=contributions,
        )
    if coordination_boundary.noun_subject_shared_predicate_compound(features):
        risk += record_boundary_score(
            "split.boundary.english.coordination.noun_subject_shared_predicate",
            reasons=reasons,
            contributions=contributions,
        )
    if coordination_boundary.embedded_question_coordinated_subject(features):
        risk += record_boundary_score(
            "split.boundary.english.coordination.noun_subject_shared_predicate",
            reasons=reasons,
            contributions=contributions,
        )
    if coordination_boundary.final_noun_progressive_predicate(features):
        risk += record_boundary_score(
            "split.boundary.english.coordination.final_noun_progressive_predicate",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.coordinated_noun_phrase_contextual(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.coordinated_noun_phrase_contextual",
            reasons=reasons,
            contributions=contributions,
        )
    if coordination_boundary.omitted_subject_predicate(features):
        risk += record_boundary_score(
            "split.boundary.english.coordination.omitted_subject_predicate",
            reasons=reasons,
            contributions=contributions,
        )
    if coordination_boundary.noun_list_progressive_predicate(features):
        risk += record_boundary_score(
            "split.boundary.english.coordination.noun_list_progressive_predicate",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.reported_subject(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.reported_subject",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.transitive_nominal_clause(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.transitive_nominal_clause",
            reasons=reasons,
            contributions=contributions,
        )
    if discourse_boundary.frame_following_clause(features):
        risk += record_boundary_score(
            "split.boundary.english.discourse.frame_following_clause",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.what_use_for_object(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.what_use_for_object",
            reasons=reasons,
            contributions=contributions,
        )
    return risk


def _score_english_boundary_dependencies(
    features: EnglishBoundaryFeatures,
    reasons: list[str],
    contributions: list[BoundaryScoreContribution],
) -> int:
    left_tokens = list(features.left_tokens)
    previous = features.previous
    head = features.head
    risk = 0

    if grammar_boundary.dependent_phrase(
        features,
        head_is_dependent=head in _DEPENDENT_RIGHT_HEADS,
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.dependent_phrase",
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": head},
        )
    if grammar_boundary.standalone_of_phrase(features):
        # Punctuation restored by an ASR/LLM can capitalize a continuation and
        # make it look independent. An ``Of ...`` fragment without a finite
        # clause still belongs to an adjacent noun phrase.
        risk += record_boundary_score(
            "split.boundary.english.grammar.standalone_of_phrase",
            reasons=reasons,
            contributions=contributions,
        )
    that_starts_complement = head == "that" and any(
        len(left_tokens) >= len(phrase) and tuple(left_tokens[-len(phrase) :]) == phrase
        for phrase in _THAT_COMPLEMENT_TAILS
    )
    contracted_relative_head = bool(
        head in _CONTRACTED_RELATIVE_CLAUSE_HEADS
        and features.tail in {"anything", "everything", "nothing", "one", "something"}
        and features.right[:1].islower()
    )
    if grammar_boundary.relative_clause(
        features,
        head_is_relative=head in _RELATIVE_CLAUSE_HEADS or contracted_relative_head,
        that_starts_complement=that_starts_complement,
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.relative_clause",
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": head},
        )
    if grammar_boundary.dependent_adverbial_clause(
        features,
        head_is_translation_sensitive=head in _TRANSLATION_SENSITIVE_HEADS,
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.dependent_adverbial_clause",
            reasons=reasons,
            contributions=contributions,
            reason_values={"token": head},
        )
    if numeric_boundary.model_year_vehicle_name(features):
        risk += record_boundary_score(
            "split.boundary.english.numeric.model_year_vehicle_name",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.dependent_locative_clause(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.dependent_locative_clause",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.way_clause_subject(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.way_clause_subject",
            reasons=reasons,
            contributions=contributions,
        )

    if predicate_boundary.proper_name_subject(
        features,
        previous_is_dependent_head=previous in _DEPENDENT_RIGHT_HEADS,
    ):
        risk += record_boundary_score(
            "split.boundary.english.predicate.proper_name_subject",
            reasons=reasons,
            contributions=contributions,
        )

    # A long cue can contain the end of one clause followed by the complete
    # subject of the next one (often after ASR misses sentence punctuation).
    # Keep that trailing subject with its finite predicate.  Limiting the
    # subject to a determiner/that plus at most three noun-like tokens and the
    # right side to common finite verbs avoids treating ordinary objects as
    # sentence subjects.
    finite_predicate_heads = {
        "are",
        "became",
        "become",
        "can",
        "could",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "is",
        "made",
        "makes",
        "played",
        "plays",
        "should",
        "used",
        "uses",
        "was",
        "were",
        "will",
        "would",
    }
    if predicate_boundary.trailing_noun_subject(
        features,
        head_is_finite_predicate=head in finite_predicate_heads,
    ):
        risk += record_boundary_score(
            "split.boundary.english.predicate.trailing_noun_subject",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.dependent_subject_adverbial_predicate(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.trailing_noun_subject",
            reasons=reasons,
            contributions=contributions,
        )
    if grammar_boundary.clause_final_subject(
        features,
        head_is_finite_predicate=head in finite_predicate_heads,
    ):
        risk += record_boundary_score(
            "split.boundary.english.grammar.clause_final_subject",
            reasons=reasons,
            contributions=contributions,
        )
    if predicate_boundary.gerund_subject(
        features,
        head_is_finite_predicate=head in finite_predicate_heads,
    ):
        risk += record_boundary_score(
            "split.boundary.english.predicate.gerund_subject",
            reasons=reasons,
            contributions=contributions,
        )

    if entity_boundary.proper_name(features):
        risk += record_boundary_score(
            "split.boundary.english.entity.proper_name",
            reasons=reasons,
            contributions=contributions,
        )
    if entity_boundary.attributive_proper_name(features):
        risk += record_boundary_score(
            "split.boundary.english.entity.proper_name",
            reasons=reasons,
            contributions=contributions,
        )

    if predicate_boundary.what_clause_subject(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.what_clause_subject",
            reasons=reasons,
            contributions=contributions,
        )

    if predicate_boundary.degree_complement(features):
        risk += record_boundary_score(
            "split.boundary.english.predicate.degree_complement",
            reasons=reasons,
            contributions=contributions,
        )

    if comparison_boundary.auxiliary_after_already(features):
        risk += record_boundary_score(
            "split.boundary.english.comparison.auxiliary_after_already",
            reasons=reasons,
            contributions=contributions,
        )

    if numeric_boundary.article_model_year_vehicle_name(features):
        risk += record_boundary_score(
            "split.boundary.english.numeric.article_model_year_vehicle_name",
            reasons=reasons,
            contributions=contributions,
        )

    if coordination_boundary.what_clause_predicate(features):
        risk += record_boundary_score(
            "split.boundary.english.coordination.what_clause_predicate",
            reasons=reasons,
            contributions=contributions,
        )

    if entity_boundary.city_state(features):
        risk += record_boundary_score(
            "split.boundary.english.entity.city_state",
            reasons=reasons,
            contributions=contributions,
        )

    if entity_boundary.vehicle_trim_model(features):
        risk += record_boundary_score(
            "split.boundary.english.entity.vehicle_trim_model",
            reasons=reasons,
            contributions=contributions,
        )

    if grammar_boundary.temporal_continuation(features):
        risk += record_boundary_score(
            "split.boundary.english.grammar.temporal_continuation",
            reasons=reasons,
            contributions=contributions,
        )

    return risk


_ENGLISH_BOUNDARY_SCORE_STAGES = (
    _score_english_boundary_foundation,
    _score_english_boundary_relations,
    _score_english_boundary_completions,
    _score_english_boundary_discourse,
    _score_english_boundary_clause_ownership,
    _score_english_boundary_dependencies,
)


def assess_english_boundary(left: str, right: str) -> BoundaryAssessment:
    """Assess whether an English cue boundary splits a dependent phrase."""
    features = extract_english_boundary_features(left, right)
    if not features.eligible:
        return BoundaryAssessment(0, ())

    right = features.right
    risk = 0
    reasons: list[str] = []
    contributions: list[BoundaryScoreContribution] = []

    for score_stage in _ENGLISH_BOUNDARY_SCORE_STAGES:
        risk += score_stage(features, reasons, contributions)
    # A lowercase continuation makes a dangling tail more likely, but is not
    # sufficient by itself: natural subtitle clauses often continue lowercase.
    if risk and right[:1].islower():
        risk += record_boundary_score(
            "split.boundary.english.observation.lowercase_continuation_bonus",
            reasons=reasons,
            contributions=contributions,
            append_reason=False,
        )

    return BoundaryAssessment(
        risk,
        tuple(dict.fromkeys(reasons)),
        tuple(contributions),
    )


def has_unstable_english_boundary(left: str, right: str) -> bool:
    return assess_english_boundary(left, right).unstable


def _join_words(words: Iterable[ASRWord]) -> str:
    result = ""
    no_space_before = set(",.;:!?)]}，。！？；：、）】》")
    no_space_after = set("([{（【《")
    for word in words:
        text = word.text.strip()
        if not text:
            continue
        if not result:
            result = text
        elif (
            text[0] in no_space_before
            or result[-1] in no_space_after
            or re.match(r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u9fff\uac00-\ud7af]", text[0])
            or re.match(r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u9fff\uac00-\ud7af]", result[-1])
        ):
            result += text
        else:
            result += f" {text}"
    return re.sub(r"\s+([,.;:!?，。！？；：、])", r"\1", result).strip()


def _word_count(words: Sequence[ASRWord]) -> int:
    return sum(max(1, len(_tokens(word.text))) for word in words if word.text.strip())


def _dominant_speaker(words: Sequence[ASRWord], fallback: str = "") -> str:
    durations: dict[str, int] = {}
    for word in words:
        if word.speaker_id:
            durations[word.speaker_id] = durations.get(word.speaker_id, 0) + max(
                1, word.end_time - word.start_time
            )
    return max(durations, key=durations.__getitem__) if durations else fallback


def _make_cue(words: Sequence[ASRWord], fallback_speaker: str) -> ASRDataSeg:
    sources = {word.timing_source for word in words if word.timing_source != "unknown"}
    timing_source = cast(
        TimestampSource,
        next(iter(sources)) if len(sources) == 1 else ("mixed" if sources else "unknown"),
    )
    languages = {word.language_code for word in words if word.language_code}
    language_code = next(iter(languages)) if len(languages) == 1 else ("mixed" if languages else "")
    return ASRDataSeg(
        text=_join_words(words),
        start_time=words[0].start_time,
        end_time=words[-1].end_time,
        speaker_id=_dominant_speaker(words, fallback_speaker),
        words=list(words),
        timestamp_granularity="sentence",
        timing_source=timing_source,
        language_code=language_code,
    )


def _is_japanese_cue(segment: ASRDataSeg) -> bool:
    return segment.language_code == "ja" or bool(_JAPANESE_RE.search(segment.text))


def _has_unstable_japanese_boundary(left: str, right: str) -> bool:
    left = str(left or "").strip()
    right = str(right or "").strip()
    if not left or not right:
        return False
    if _KATAKANA_RE.fullmatch(left[-1]) and _KATAKANA_RE.fullmatch(right[0]):
        return True
    if re.fullmatch(r"[\u3400-\u9fff]", left[-1]) and re.match(r"[\u3400-\u9fff]", right[0]):
        return True
    if re.fullmatch(r"[\u3400-\u9fff]", left[-1]) and right.startswith("しい"):
        return True
    if any(
        left.endswith(form[:position]) and right.startswith(form[position:])
        for form in _JAPANESE_BOUND_FORMS
        for position in range(1, len(form))
    ):
        return True
    if _JAPANESE_INFLECTION_HEAD_RE.match(right):
        return True
    if _JAPANESE_PARTICLE_HEAD_RE.match(right):
        return True
    return bool(
        _JAPANESE_FILLER_TAIL_RE.search(left)
        or _JAPANESE_SHORT_TOPIC_TAIL_RE.search(left)
        or _JAPANESE_DANGLING_TAIL_RE.search(left)
    )


def _can_merge_japanese_atomic_pair(
    left: ASRDataSeg,
    right: ASRDataSeg,
    words: Sequence[ASRWord],
    *,
    hard_max_chars: int,
) -> bool:
    """Allow a small overflow only when Japanese grammar leaves no valid split."""
    return bool(
        left.words
        and right.words
        and _is_japanese_cue(left)
        and _is_japanese_cue(right)
        and not _is_hard_boundary(left, right)
        and _has_unstable_japanese_boundary(left.text, right.text)
        and _word_count(words) <= hard_max_chars + MAX_JAPANESE_ATOMIC_OVERFLOW_CHARS
        and words[-1].end_time - words[0].start_time <= MAX_JAPANESE_ATOMIC_DURATION_MS
    )


def _repair_japanese_boundaries(
    segments: Sequence[ASRDataSeg],
    *,
    hard_max_chars: int,
) -> list[ASRDataSeg]:
    """Repair lexical Japanese breaks without changing words or timestamps."""
    result = list(segments)
    index = 0
    while index < len(result) - 1:
        left = result[index]
        right = result[index + 1]
        if (
            not left.words
            or not right.words
            or not _is_japanese_cue(left)
            or not _is_japanese_cue(right)
            or _is_hard_boundary(left, right)
            or not _has_unstable_japanese_boundary(left.text, right.text)
        ):
            index += 1
            continue

        words = [*left.words, *right.words]
        if (
            (
                _JAPANESE_FILLER_TAIL_RE.search(left.text.strip())
                or _JAPANESE_DANGLING_TAIL_RE.search(left.text.strip())
            )
            and _word_count(words) <= hard_max_chars
            and words[-1].end_time - words[0].start_time <= 8000
        ):
            result[index : index + 2] = [_make_cue(words, left.speaker_id or right.speaker_id)]
            logger.info(
                "Merged incomplete Japanese filler cue at subtitles %s-%s",
                index + 1,
                index + 2,
            )
            if index:
                index -= 1
            continue

        original = len(left.words)
        candidates: list[tuple[float, int]] = []
        for position in range(max(1, original - 8), min(len(words), original + 9)):
            left_words = words[:position]
            right_words = words[position:]
            if not left_words or not right_words:
                continue
            if max(_word_count(left_words), _word_count(right_words)) > hard_max_chars:
                continue
            left_text = _join_words(left_words)
            right_text = _join_words(right_words)
            if _has_unstable_japanese_boundary(left_text, right_text):
                continue
            cost = abs(position - original) * 0.8
            cost += abs(_word_count(left_words) - _word_count(right_words)) * 0.08
            if min(_word_count(left_words), _word_count(right_words)) < 3:
                cost += 12.0
            if _JAPANESE_PREFERRED_TAIL_RE.search(left_text):
                cost -= 4.0
            candidates.append((cost, position))

        if not candidates:
            if (
                _word_count(words) <= hard_max_chars
                and words[-1].end_time - words[0].start_time <= 8000
            ):
                result[index : index + 2] = [_make_cue(words, left.speaker_id or right.speaker_id)]
                logger.info(
                    "Merged indivisible Japanese lexical boundary at subtitles %s-%s",
                    index + 1,
                    index + 2,
                )
                if index:
                    index -= 1
                continue
            index += 1
            continue
        _, position = min(candidates)
        if position == original:
            index += 1
            continue
        result[index : index + 2] = [
            _make_cue(words[:position], left.speaker_id),
            _make_cue(words[position:], right.speaker_id),
        ]
        logger.info(
            "Repaired Japanese subtitle boundary %s-%s: %s -> %s",
            index + 1,
            index + 2,
            original,
            position,
        )
        if index:
            index -= 1
        else:
            index += 1
    return result


def _repair_japanese_boundaries_until_stable(
    segments: Sequence[ASRDataSeg],
    *,
    hard_max_chars: int,
) -> list[ASRDataSeg]:
    result = list(segments)
    for _ in range(8):
        signature = [(segment.start_time, segment.end_time, segment.text) for segment in result]
        repaired = _repair_japanese_boundaries(
            result,
            hard_max_chars=hard_max_chars,
        )
        repaired_signature = [
            (segment.start_time, segment.end_time, segment.text) for segment in repaired
        ]
        result = repaired
        if repaired_signature == signature:
            break
    return _merge_japanese_atomic_pairs(result, hard_max_chars=hard_max_chars)


def _merge_japanese_atomic_pairs(
    segments: Sequence[ASRDataSeg],
    *,
    hard_max_chars: int,
) -> list[ASRDataSeg]:
    result = list(segments)
    index = 0
    while index < len(result) - 1:
        left = result[index]
        right = result[index + 1]
        words = [*left.words, *right.words]
        if _can_merge_japanese_atomic_pair(
            left,
            right,
            words,
            hard_max_chars=hard_max_chars,
        ):
            result[index : index + 2] = [_make_cue(words, left.speaker_id or right.speaker_id)]
            logger.info(
                "Finalized indivisible Japanese boundary at subtitles %s-%s",
                index + 1,
                index + 2,
            )
            if index:
                index -= 1
            continue
        index += 1
    return result


def finalize_japanese_boundaries(
    segments: Sequence[ASRDataSeg],
    *,
    hard_max_chars: int = HARD_MAX_CJK_CHARS,
) -> list[ASRDataSeg]:
    """Apply the language-specific final pass after all generic boundary work."""
    result = list(segments)
    for _ in range(8):
        signature = [(segment.start_time, segment.end_time, segment.text) for segment in result]
        repaired = _repair_japanese_boundaries(
            result,
            hard_max_chars=hard_max_chars,
        )
        repaired_signature = [
            (segment.start_time, segment.end_time, segment.text) for segment in repaired
        ]
        result = repaired
        if repaired_signature == signature:
            break
    return _merge_japanese_atomic_pairs(result, hard_max_chars=hard_max_chars)


def _is_singular_correction(left: ASRWord, right: ASRWord) -> bool:
    left_text = left.text.strip()
    left_tokens = _tokens(left_text)
    right_tokens = _tokens(right.text)
    return bool(
        left_text.endswith(",")
        and len(left_tokens) == 1
        and len(right_tokens) == 1
        and len(left_tokens[0]) >= 3
        and right_tokens[0] == f"{left_tokens[0]}s"
        and right.start_time - left.end_time <= MAX_DUPLICATE_CORRECTION_GAP_MS
        and (not left.speaker_id or not right.speaker_id or left.speaker_id == right.speaker_id)
    )


def _remove_singular_corrections(
    segments: Sequence[ASRDataSeg],
) -> list[ASRDataSeg]:
    """Drop a spoken singular false start immediately corrected to its plural."""
    result = list(segments)
    for index, segment in enumerate(result):
        words = list(segment.words)
        keep = [True] * len(words)
        for word_index in range(len(words) - 1):
            if _is_singular_correction(words[word_index], words[word_index + 1]):
                keep[word_index] = False
                logger.info(
                    "Removed in-subtitle singular correction at subtitle %s: %s -> %s",
                    index + 1,
                    _tokens(words[word_index].text)[0],
                    _tokens(words[word_index + 1].text)[0],
                )
        kept_words = [word for word, should_keep in zip(words, keep) if should_keep]
        if len(kept_words) != len(words) and kept_words:
            result[index] = _make_cue(kept_words, segment.speaker_id)

    for index in range(len(result) - 1):
        left = result[index]
        right = result[index + 1]
        if not left.words or not right.words:
            continue
        if left.speaker_id and right.speaker_id and left.speaker_id != right.speaker_id:
            continue
        if right.start_time - left.end_time > MAX_DUPLICATE_CORRECTION_GAP_MS:
            continue
        if not _is_singular_correction(left.words[-1], right.words[0]) or len(left.words) == 1:
            continue
        left_tokens = _tokens(left.words[-1].text)
        right_tokens = _tokens(right.words[0].text)
        result[index] = _make_cue(left.words[:-1], left.speaker_id)
        logger.info(
            "Removed cross-boundary singular correction at subtitles %s-%s: %s -> %s",
            index + 1,
            index + 2,
            left_tokens[0],
            right_tokens[0],
        )
    return result


def _split_internal_terminal_clauses(
    segments: Sequence[ASRDataSeg],
) -> list[ASRDataSeg]:
    """Split a cue that crosses a real sentence end before boundary repair.

    LLM splitting can occasionally place ``Sentence one. Sentence`` in one cue.
    Re-exposing that terminal lets the normal short-cue and dependency passes
    attach the second sentence to its actual continuation. Atomic word timings
    and speaker ownership remain unchanged.
    """
    result: list[ASRDataSeg] = []
    for segment_index, segment in enumerate(segments):
        if not segment.words or segment.translated_text:
            result.append(segment)
            continue

        boundaries: list[int] = []
        for index, word in enumerate(segment.words[:-1], start=1):
            text = word.text.strip()
            if not _TERMINAL_RE.search(text):
                continue
            token = (_tokens(text) or [""])[-1]
            if len(token) == 1 and token.isupper():
                continue
            boundaries.append(index)

        if not boundaries or segment_index + 1 >= len(segments):
            result.append(segment)
            continue

        boundary = boundaries[-1]
        if boundary < 2 or len(segment.words) - boundary < 2:
            result.append(segment)
            continue
        suffix_text = _join_words(segment.words[boundary:])
        following = segments[segment_index + 1]
        if not assess_english_boundary(suffix_text, following.text).unstable:
            result.append(segment)
            continue

        result.append(_make_cue(segment.words[:boundary], segment.speaker_id))
        result.append(_make_cue(segment.words[boundary:], segment.speaker_id))
    return result


def _is_hard_boundary(left: ASRDataSeg, right: ASRDataSeg) -> bool:
    if left.speaker_id and right.speaker_id and left.speaker_id != right.speaker_id:
        gap = max(0, right.start_time - left.end_time)
        assessment = assess_english_boundary(left.text, right.text)
        exact_dependency = bool(
            set(assessment.reasons)
            & {
                "attributive or comparative modifier separated from its head",
                "numeric value separated from its unit or noun",
                "numeric value separated from its multiplier or unit",
                "participle separated from its complement",
                "scalar predicate separated from its comparative complement",
                "determiner and adjective separated from their head noun",
                "determiner and degree modifier separated from their head noun",
            }
        )
        trailing_adverb = (
            "sentence-final temporal adverb separated from its clause" in assessment.reasons
        )
        # Diarization can briefly flip speaker IDs inside one sentence. Give a
        # slightly wider tolerance only to an exceptionally strong dependency,
        # such as a modifier split from its noun.
        allowed_gap = (
            MAX_STRONG_DEPENDENCY_DIARIZATION_GLITCH_GAP_MS
            if assessment.risk >= 50
            else MAX_DIARIZATION_GLITCH_GAP_MS
        )
        if exact_dependency:
            allowed_gap = max(allowed_gap, MAX_EXACT_DEPENDENCY_DIARIZATION_GLITCH_GAP_MS)
        if trailing_adverb:
            allowed_gap = max(allowed_gap, MAX_TRAILING_ADVERB_DIARIZATION_GLITCH_GAP_MS)
        if gap > allowed_gap or (
            assessment.risk < 30
            and not (exact_dependency and gap <= MAX_EXACT_DEPENDENCY_DIARIZATION_GLITCH_GAP_MS)
            and not trailing_adverb
        ):
            return True
    gap = right.start_time - left.end_time
    if (
        gap <= MAX_TRAILING_ADVERB_DIARIZATION_GLITCH_GAP_MS
        and "sentence-final temporal adverb separated from its clause"
        in assess_english_boundary(left.text, right.text).reasons
    ):
        return False
    return gap > MAX_RELOCATABLE_GAP_MS


def _redistribute_cross_speaker_completion(
    left: ASRDataSeg,
    right: ASRDataSeg,
) -> tuple[ASRDataSeg, ASRDataSeg] | None:
    """Keep a short grammatical completion before a genuine new-speaker reply.

    Diarization can switch at the first acoustically ambiguous word of a reply,
    leaving a question such as ``do you still | love her I love this car``.
    Merging the whole pair would erase the real turn. Move only the shortest
    proven completion prefix, and require the remainder to begin a capitalized
    independent subject so ordinary lowercase continuations are untouched.
    """
    if (
        not left.words
        or len(right.words) < 2
        or not left.speaker_id
        or not right.speaker_id
        or left.speaker_id == right.speaker_id
        or right.start_time - left.end_time > MAX_EXACT_DEPENDENCY_DIARIZATION_GLITCH_GAP_MS
        or assess_english_boundary(left.text, right.text).risk < 26
    ):
        return None

    max_prefix = min(MAX_BOUNDARY_SHIFT_WORDS, len(right.words) - 1)
    for prefix_size in range(1, max_prefix + 1):
        remainder = right.words[prefix_size:]
        remainder_tokens = _tokens(_join_words(remainder))
        if not remainder_tokens or remainder_tokens[0] not in _RESPONSE_INTERJECTIONS:
            continue
        if any(token not in _RESPONSE_INTERJECTIONS for token in remainder_tokens):
            continue
        completed_words = [*left.words, *right.words[:prefix_size]]
        repaired_assessment = assess_english_boundary(
            _join_words(completed_words), _join_words(remainder)
        )
        response_only_false_positive = set(repaired_assessment.reasons) <= {
            "single-word completion stranded in the next subtitle"
        }
        if repaired_assessment.unstable and not response_only_false_positive:
            continue
        repaired_left = _make_cue(completed_words, left.speaker_id)
        repaired_right = _make_cue(remainder, right.speaker_id)
        repaired_left.speaker_id = left.speaker_id
        repaired_right.speaker_id = right.speaker_id
        return repaired_left, repaired_right

    if len(right.words) < 3:
        return None

    explicit_subjects = {
        "he",
        "i",
        "it",
        "she",
        "that",
        "they",
        "this",
        "we",
        "you",
    }
    max_prefix = min(MAX_BOUNDARY_SHIFT_WORDS, len(right.words) - 2)
    for prefix_size in range(1, max_prefix + 1):
        remainder = right.words[prefix_size:]
        raw_head = remainder[0].text.strip()
        remainder_tokens = _tokens(_join_words(remainder))
        if (
            not raw_head[:1].isupper()
            or not remainder_tokens
            or remainder_tokens[0] not in explicit_subjects
        ):
            continue

        completed_words = [*left.words, *right.words[:prefix_size]]
        completed_text = _join_words(completed_words)
        remainder_text = _join_words(remainder)
        repaired_assessment = assess_english_boundary(completed_text, remainder_text)
        completed_tail = (_tokens(completed_text) or [""])[-1]
        object_pronoun_false_positive = completed_tail in {"her", "his"} and set(
            repaired_assessment.reasons
        ) <= {f"dangling modifier '{completed_tail}'"}
        if repaired_assessment.unstable and not object_pronoun_false_positive:
            continue

        repaired_left = _make_cue(completed_words, left.speaker_id)
        repaired_right = _make_cue(remainder, right.speaker_id)
        # The grammatical turn boundary is stronger evidence than the few
        # prefix words whose diarization label caused the split.
        repaired_left.speaker_id = left.speaker_id
        repaired_right.speaker_id = right.speaker_id
        return repaired_left, repaired_right
    return None


def _is_short_cross_speaker_reply(left: ASRDataSeg, right: ASRDataSeg) -> bool:
    """Return whether *right* is a genuine brief reply after another speaker."""
    right_tokens = _tokens(right.text)
    return bool(
        left.speaker_id
        and right.speaker_id
        and left.speaker_id != right.speaker_id
        and right_tokens
        and set(right_tokens) <= _RESPONSE_INTERJECTIONS
    )


def _merge_compact_unstable_pairs(
    segments: Sequence[ASRDataSeg],
    *,
    hard_max_words: int,
) -> list[ASRDataSeg]:
    """Merge only compact pairs that still have no stable internal boundary."""
    result = list(segments)
    index = 0
    while index < len(result) - 1:
        left = result[index]
        right = result[index + 1]
        assessment = assess_english_boundary(left.text, right.text)
        redistributed = _redistribute_cross_speaker_completion(left, right)
        if redistributed is not None:
            result[index : index + 2] = list(redistributed)
            logger.info(
                "Redistributed cross-speaker subtitle completion at subtitles %s-%s",
                index + 1,
                index + 2,
            )
            if index:
                index -= 1
            continue
        words = [*left.words, *right.words]
        combined_text = _join_words(words)
        if (
            assessment.unstable
            and "fixed phrase split inside 'do not get me wrong'" not in assessment.reasons
            and not _is_hard_boundary(left, right)
            and left.words
            and right.words
            and _word_count(words) <= hard_max_words
            and words[-1].end_time - words[0].start_time <= 8000
            and not re.search(r"[.!?][\"')\]]*\s+\S", combined_text)
            and not _is_short_cross_speaker_reply(left, right)
        ):
            merged = _make_cue(words, left.speaker_id or right.speaker_id)
            if (
                left.speaker_id
                and right.speaker_id
                and left.speaker_id != right.speaker_id
                and right.text.strip()[:1].islower()
            ):
                # A high-confidence grammatical continuation is stronger than
                # a brief diarization flip on the continuation words.
                merged.speaker_id = left.speaker_id
            result[index : index + 2] = [merged]
            logger.info(
                "Merged compact unstable subtitles %s-%s: %s",
                index + 1,
                index + 2,
                assessment.reasons,
            )
            if index:
                index -= 1
            continue
        index += 1
    return result


def _segment_cost(words: Sequence[ASRWord], soft_max: int, hard_max: int) -> float:
    count = _word_count(words)
    if count <= 0 or count > hard_max:
        return float("inf")
    cost = 0.0
    if count < 3:
        cost += (3 - count) * 7.0
    if count > soft_max:
        cost += (count - soft_max) * 5.0
    duration = words[-1].end_time - words[0].start_time
    if duration > 8000:
        cost += (duration - 8000) / 500.0
    return cost


def _boundary_cost(
    words: Sequence[ASRWord], position: int, original_position: int
) -> tuple[float, int]:
    left = _join_words(words[:position])
    right = _join_words(words[position:])
    assessment = assess_english_boundary(left, right)
    tail = words[position - 1].text.strip()
    right_tokens = _tokens(words[position].text)
    head = right_tokens[0] if right_tokens else ""
    gap = max(0, words[position].start_time - words[position - 1].end_time)

    cost = float(assessment.risk)
    if head in {"how", "if", "when", "where", "whether", "why"} and re.search(
        r"\b(?:decide|figure\s+out|know|remember|see|tell|understand|wonder)$",
        left,
        re.IGNORECASE,
    ):
        # Legal in English, but a poor replacement for an already broken
        # boundary because Chinese normally keeps this complement beside its
        # governing predicate.
        cost += 36.0
    if _TERMINAL_RE.search(tail):
        cost -= 24.0
    elif _CLAUSE_RE.search(tail):
        cost -= 5.0
    if head in _PREFERRED_CLAUSE_HEADS:
        cost -= 10.0
    if head in {"and", "because", "but", "for", "to", "when", "where", "while"} and re.search(
        r"\b(?:at|in|to)\s+[A-Z][A-Za-z0-9'’.+-]*"
        r"(?:\s+[A-Z][A-Za-z0-9'’.+-]*)*"
        r"(?:,\s*[A-Z][A-Za-z0-9'’.+-]*)?$",
        left,
    ):
        # When one long spoken sentence must span two cues, keep a complete
        # destination such as ``in Pebble Beach, California`` together. This
        # avoids choosing a slightly cheaper but unreadable break after the
        # governing preposition.
        cost -= 10.0
    if head == "like" and re.search(r"\bso\s+much$", left, re.IGNORECASE):
        cost -= 12.0
    if head in {
        "become",
        "becomes",
        "became",
        "is",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
    } and re.search(
        r"\b(?:note|mention|observe|report|say|show)\b.*\bthat\b.*"
        r"\band\s+[a-z][a-z'’-]*$",
        left,
        re.IGNORECASE,
    ):
        # A long reported noun-list topic is readable in one cue with its
        # predicate beginning the next cue. Prefer that acoustic boundary to
        # splitting a noun phrase or an auxiliary-participle chain.
        cost -= 24.0
    # Within an already unsafe dependency region, a real acoustic pause is a
    # stronger boundary cue than distance from the model's original split.
    # Keep the reward bounded so syntax risk remains authoritative.
    cost -= min(6.0, gap / 100.0)
    cost += abs(position - original_position) * 0.8
    if position == original_position:
        cost -= 1.5
    return cost, assessment.risk


def _best_region_breaks(
    words: Sequence[ASRWord],
    original_positions: Sequence[int],
    *,
    soft_max: int,
    hard_max: int,
) -> tuple[list[int], float, int] | None:
    """Return same-count local breaks using dynamic programming."""
    candidates: list[list[int]] = []
    total = len(words)
    for original in original_positions:
        start = max(1, original - MAX_BOUNDARY_SHIFT_WORDS)
        end = min(total - 1, original + MAX_BOUNDARY_SHIFT_WORDS)
        candidates.append(list(range(start, end + 1)))

    states: dict[int, tuple[float, list[int], int]] = {0: (0.0, [], 0)}
    for boundary_index, positions in enumerate(candidates):
        next_states: dict[int, tuple[float, list[int], int]] = {}
        original = original_positions[boundary_index]
        for position in positions:
            for previous, (cost, path, risk) in states.items():
                if position <= previous:
                    continue
                segment_cost = _segment_cost(words[previous:position], soft_max, hard_max)
                if segment_cost == float("inf"):
                    continue
                boundary_cost, boundary_risk = _boundary_cost(words, position, original)
                candidate = (
                    cost + segment_cost + boundary_cost,
                    path + [position],
                    risk + boundary_risk,
                )
                existing = next_states.get(position)
                if existing is None or candidate[0] < existing[0]:
                    next_states[position] = candidate
        states = next_states
        if not states:
            return None

    best: tuple[float, list[int], int] | None = None
    for previous, (cost, path, risk) in states.items():
        final_cost = _segment_cost(words[previous:], soft_max, hard_max)
        if final_cost == float("inf"):
            continue
        candidate = (cost + final_cost, path, risk)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        return None
    return best[1], best[0], best[2]


def _region_metrics(
    words: Sequence[ASRWord], positions: Sequence[int], soft_max: int, hard_max: int
) -> tuple[float, int]:
    cost = 0.0
    risk = 0
    previous = 0
    for index, position in enumerate(positions):
        cost += _segment_cost(words[previous:position], soft_max, hard_max)
        boundary_cost, boundary_risk = _boundary_cost(words, position, positions[index])
        cost += boundary_cost
        risk += boundary_risk
        previous = position
    cost += _segment_cost(words[previous:], soft_max, hard_max)
    return cost, risk


def normalize_boundaries(
    segments: Sequence[ASRDataSeg],
    *,
    soft_max_words: int = SOFT_MAX_WORDS,
    hard_max_words: int = HARD_MAX_WORDS,
    hard_max_cjk_chars: int = HARD_MAX_CJK_CHARS,
) -> list[ASRDataSeg]:
    """Move only high-risk boundaries while retaining every atomic word timing."""
    # Every pass below can rebuild cues without translated text. Check before
    # the first pass, including when only part of the input is translated.
    if any(segment.translated_text for segment in segments):
        return list(segments)
    result = _remove_singular_corrections(segments)
    result = _split_internal_terminal_clauses(result)
    result = _repair_japanese_boundaries_until_stable(
        result,
        hard_max_chars=hard_max_cjk_chars,
    )
    result = _merge_compact_unstable_pairs(
        result,
        hard_max_words=hard_max_words,
    )
    if len(result) < 2:
        return result

    for _ in range(3):
        risky = [
            index
            for index in range(len(result) - 1)
            if not _is_hard_boundary(result[index], result[index + 1])
            and not _is_short_cross_speaker_reply(result[index], result[index + 1])
            and has_unstable_english_boundary(result[index].text, result[index + 1].text)
        ]
        if not risky:
            break

        regions: list[tuple[int, int]] = []
        start = previous = risky[0]
        for index in risky[1:]:
            if index == previous + 1:
                previous = index
                continue
            regions.append((start, previous + 1))
            start = previous = index
        regions.append((start, previous + 1))

        changed = False
        for cue_start, cue_end in reversed(regions):
            if cue_end + 1 < len(result):
                short_tail = result[cue_end]
                following = result[cue_end + 1]
                if (
                    _word_count(short_tail.words) <= 4
                    and not _is_hard_boundary(short_tail, following)
                    and not _is_short_cross_speaker_reply(short_tail, following)
                ):
                    # A tiny dependent tail such as ``miles on it?`` needs one
                    # stable cue of look-ahead. Keeping the cue count unchanged
                    # then lets the DP move the number/unit boundary without
                    # creating an overlong subtitle.
                    cue_end += 1
            cues = result[cue_start : cue_end + 1]
            if any(not cue.words for cue in cues) or any(
                _is_hard_boundary(cues[index], cues[index + 1]) for index in range(len(cues) - 1)
            ):
                continue

            words = [word for cue in cues for word in cue.words]
            original_positions: list[int] = []
            cursor = 0
            for cue in cues[:-1]:
                cursor += len(cue.words)
                original_positions.append(cursor)

            baseline_cost, baseline_risk = _region_metrics(
                words, original_positions, soft_max_words, hard_max_words
            )
            best = _best_region_breaks(
                words,
                original_positions,
                soft_max=soft_max_words,
                hard_max=hard_max_words,
            )
            if best is None:
                continue
            positions, candidate_cost, candidate_risk = best
            new_dependency_reasons = {
                "adverb separated from its gerund",
                "progressive predicate separated from its complement",
                "subject and adverb separated from its predicate",
            }
            requires_stable_replacement = any(
                new_dependency_reasons.intersection(
                    assess_english_boundary(cue.text, following.text).reasons
                )
                for cue, following in zip(cues, cues[1:])
            )
            region_reasons = {
                reason
                for cue, following in zip(cues, cues[1:])
                for reason in assess_english_boundary(cue.text, following.text).reasons
            }
            min_improvement = (
                5.0
                if any(
                    reason.startswith("dependent adverbial clause beginning with")
                    for reason in region_reasons
                )
                else MIN_REPAIR_IMPROVEMENT
            )
            if (
                positions == original_positions
                or candidate_risk >= baseline_risk
                or baseline_cost - candidate_cost < min_improvement
                or (requires_stable_replacement and candidate_risk)
            ):
                continue

            rebuilt: list[ASRDataSeg] = []
            previous_position = 0
            fallback_speaker = cues[0].speaker_id
            for position in [*positions, len(words)]:
                rebuilt.append(_make_cue(words[previous_position:position], fallback_speaker))
                previous_position = position
            result[cue_start : cue_end + 1] = rebuilt
            changed = True
            logger.info(
                "Repaired subtitle boundaries %s-%s: %s -> %s (risk %s -> %s)",
                cue_start + 1,
                cue_end + 1,
                original_positions,
                positions,
                baseline_risk,
                candidate_risk,
            )

        if not changed:
            break

    result = _merge_compact_unstable_pairs(
        result,
        hard_max_words=hard_max_words,
    )
    return _repair_japanese_boundaries_until_stable(
        result,
        hard_max_chars=hard_max_cjk_chars,
    )
