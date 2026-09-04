"""Pure English grammar-boundary detectors."""

from __future__ import annotations

import re
from collections.abc import Set

from subforge.core.split.boundary_features import (
    CLAUSE_RE,
    TERMINAL_RE,
    EnglishBoundaryFeatures,
)


def incomplete_multiword(*, left_ends_with_dangling_phrase: bool) -> bool:
    return left_ends_with_dangling_phrase


def dangling_function_word(
    features: EnglishBoundaryFeatures,
    *,
    tail_is_hard_dangling: bool,
) -> bool:
    return tail_is_hard_dangling and not features.complete_visibility_preposition


def dangling_function_word_before_filler(
    features: EnglishBoundaryFeatures,
    *,
    semantic_tail_is_hard_dangling: bool,
) -> bool:
    return bool(features.semantic_tail != features.tail and semantic_tail_is_hard_dangling)


def dangling_subject(
    features: EnglishBoundaryFeatures,
    *,
    tail_is_subject: bool,
) -> bool:
    return bool(
        tail_is_subject
        and features.right[:1].islower()
        and not CLAUSE_RE.search(features.left)
        and not features.complete_demonstrative_object
    )


def standalone_subject(
    features: EnglishBoundaryFeatures,
    *,
    tail_is_subject: bool,
) -> bool:
    return bool(
        len(features.left_tokens) == 1
        and tail_is_subject
        and features.head not in {"and", "but", "or", "so"}
    )


def sentence_final_subject(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"(?:^|[.!?]\s+)(?:i|you|he|she|it|we|they)(?:\s+(?:all|both|guys))?$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and features.head not in {"and", "but", "or", "so"}
    )


def subject_before_adverbial_predicate(
    features: EnglishBoundaryFeatures,
    *,
    tail_is_subject: bool,
) -> bool:
    return tail_is_subject and bool(
        re.match(
            r"^(?:really|actually|definitely|certainly|probably|usually|often)\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def incomplete_predicate(
    features: EnglishBoundaryFeatures,
    *,
    tail_is_incomplete_predicate: bool,
) -> bool:
    return tail_is_incomplete_predicate and not features.complete_does_clause


def contracted_negative_auxiliary(features: EnglishBoundaryFeatures) -> bool:
    return features.tail in {"haven't", "hasn't", "hadn't"}


def subject_auxiliary(
    features: EnglishBoundaryFeatures,
    *,
    tail_is_subject_auxiliary: bool,
) -> bool:
    return tail_is_subject_auxiliary and features.right[:1].islower()


def adverb_gerund(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        features.previous in {"about", "by", "for", "of", "with", "without"}
        and features.tail in {"actually", "basically", "just", "literally", "really", "simply"}
        and re.match(
            r"^[a-z][a-z'’-]*ing\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def dangling_modifier(
    features: EnglishBoundaryFeatures,
    *,
    tail_is_modifier: bool,
) -> bool:
    return bool(
        tail_is_modifier
        and not (
            features.complete_degree_adverb
            or features.complete_superlative
            or features.complete_predicative_adjective
            or features.complete_scalar_complement
        )
    )


def dangling_attributive(
    features: EnglishBoundaryFeatures,
    *,
    tail_is_attributive: bool,
) -> bool:
    return bool(
        tail_is_attributive and features.right[:1].islower() and not features.complete_own_idiom
    )


def filler_noun_modifier(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:a|an|the|such\s+a)\s+(?:kind\s+of|sort\s+of|like)$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and features.semantic_right[:1].islower()
    )


def determiner_head(
    features: EnglishBoundaryFeatures,
    *,
    tail_is_attributive: bool,
) -> bool:
    return bool(
        tail_is_attributive
        and len(features.left_tokens) >= 2
        and features.left_tokens[-2]
        in {"a", "an", "the", "his", "her", "its", "my", "our", "their", "your"}
        and not features.complete_own_idiom
    )


def time_frame_participle(features: EnglishBoundaryFeatures) -> bool:
    return features.tail == "year" and bool(
        re.match(
            r"^leading\s+up\s+to\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def frequency_quantified_statement(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.fullmatch(
            r"(?:each|every|per)\s+(?:day|week|month|year|annum)[.!?]?",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def distance_location_noun(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"(?:^|[.!?]\s+)(?:about|around|roughly|approximately)\s+"
            r"\d[\d,.]*\s+(?:kilometres?|kilometers?|metres?|meters?|miles?)\s+away[,]?$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(
            r"^(?:a|an|the|this|that|these|those|new)\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def morphological_attributive_modifier(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        features.right[:1].islower()
        and features.tail != "other"
        and re.fullmatch(
            r"[a-z][a-z'’-]*(?:al|ble|ful|ic|ive|less|ous)",
            features.tail,
        )
        and not features.complete_predicative_adjective
    )


def determiner_adjective_head(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:a|an|some|any)\s+"
            r"[a-z][a-z'’-]*(?:al|ant|ary|ent|ful|ic|ive|less|ory|ous)$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(
            r"^[A-Za-z][A-Za-z'’-]*\b",
            features.semantic_right,
        )
    )


def determiner_degree_modifier_head(
    features: EnglishBoundaryFeatures,
    *,
    tail_is_attributive_degree_adverb: bool,
) -> bool:
    return bool(
        tail_is_attributive_degree_adverb
        and len(features.left_tokens) >= 2
        and features.left_tokens[-2]
        in {
            "a",
            "an",
            "another",
            "any",
            "his",
            "her",
            "its",
            "my",
            "our",
            "some",
            "the",
            "their",
            "this",
            "that",
            "these",
            "those",
            "your",
        }
        and features.right[:1].islower()
    )


def lexical_comparative_modifier(
    features: EnglishBoundaryFeatures,
    *,
    tail_is_comparative_modifier: bool,
) -> bool:
    return bool(
        features.right[:1].islower()
        and tail_is_comparative_modifier
        and not features.complete_predicative_adjective
    )


def participle_complement(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        features.right[:1].islower()
        and re.fullmatch(r"[a-z][a-z'’-]*(?:ing|ed)", features.tail)
        and features.head
        in {
            "a",
            "all",
            "an",
            "another",
            "any",
            "by",
            "each",
            "every",
            "for",
            "from",
            "his",
            "her",
            "its",
            "my",
            "of",
            "other",
            "our",
            "some",
            "the",
            "their",
            "this",
            "those",
            "to",
            "with",
            "your",
        }
        and not re.search(
            rf"\b(?:a|an|the|this|that|my|your|his|her|its|our|their)\s+"
            rf"{re.escape(features.tail)}$",
            features.semantic_left,
            flags=re.IGNORECASE,
        )
    )


def postpositive_participle_modifier(features: EnglishBoundaryFeatures) -> bool:
    """Keep a noun with a following reduced-relative participial phrase."""
    return bool(
        not TERMINAL_RE.search(features.left)
        and features.right[:1].islower()
        and re.match(
            r"^(?:built|connected|constructed|covered|designed|equipped|fitted|"
            r"installed|intended|located|made|mounted|placed|powered|used)\s+"
            r"(?:at|by|for|from|in|inside|into|of|on|onto|to|under|up|with)\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def coordinated_noun_phrase(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        features.head in {"and", "or"}
        and len(features.right_tokens) > 1
        and re.search(
            r"\b(?:of|with|between)\s+[a-z][a-z'’-]*$",
            features.semantic_left,
            re.IGNORECASE,
        )
    )


def preposition_gerund(features: EnglishBoundaryFeatures) -> bool:
    return features.tail in {
        "after",
        "before",
        "by",
        "despite",
        "during",
        "through",
        "while",
        "without",
    } and features.head.endswith("ing")


def hyphenated_attributive_tail(
    features: EnglishBoundaryFeatures,
    *,
    allowed_tails: Set[str],
) -> str:
    raw_tail = re.search(r"([A-Za-z]+(?:-[A-Za-z]+)+)$", features.left)
    if not raw_tail:
        return ""
    normalized = raw_tail.group(1).lower()
    if normalized in allowed_tails and features.right[:1].islower():
        return normalized
    return ""


def open_complement(
    features: EnglishBoundaryFeatures,
    *,
    tail_is_open_complement: bool,
) -> bool:
    return tail_is_open_complement and features.right[:1].islower()


def expect_to_always(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\bexpect\s+to\s+always$",
            features.semantic_left,
            re.IGNORECASE,
        )
    )


def quantifying_phrase_noun(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:number|amount|share|range|kind|sort)\s+of(?:\s+like)?$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and features.right[:1].islower()
    )


def degree_so(features: EnglishBoundaryFeatures) -> bool:
    return features.tail == "so" and features.head in {"many", "much", "few", "little"}


def up_and_running(features: EnglishBoundaryFeatures) -> bool:
    return features.tail == "up" and features.right_tokens[:2] == ("and", "running")


def up_and_running_aspect(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        features.tail == "finally"
        and features.head == "up"
        and features.right_tokens[1:3] == ("and", "running")
    )


def single_word_completion(
    features: EnglishBoundaryFeatures,
    *,
    hard_max_words: int,
) -> bool:
    return bool(
        len(features.right_tokens) == 1
        and features.right[:1].islower()
        and re.fullmatch(r"[A-Za-z][A-Za-z'’-]*[.!?]?", features.right)
        and len(features.left_tokens) + len(features.right_tokens) <= hard_max_words
    )


def phrasal_verb(
    features: EnglishBoundaryFeatures,
    *,
    head_is_phrasal_particle: bool,
) -> bool:
    return head_is_phrasal_particle and features.tail in {
        "fall",
        "get",
        "gets",
        "go",
        "look",
        "take",
    }


def take_away(features: EnglishBoundaryFeatures) -> bool:
    return features.head == "away" and bool(
        re.search(
            r"\b(?:take|takes|took|taken|taking)\b(?:\s+\S+){1,6}$",
            features.left,
            flags=re.IGNORECASE,
        )
    )


def new_clause_connective(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"(?:^|[.!?]\s+)(?:and|but|now|so),?$",
            features.left,
            re.IGNORECASE,
        )
    )


def sentence_opening_time_marker(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"[.!?]\s+(?:today|tomorrow|tonight|meanwhile|instead),?$",
            features.left,
            re.IGNORECASE,
        )
    )


def do_not_get_me_wrong(features: EnglishBoundaryFeatures) -> bool:
    return features.head == "wrong" and bool(
        re.search(
            r"\b(?:don['’]t|do\s+not)\s+get\s+me$",
            features.left,
            re.IGNORECASE,
        )
    )


def sentence_opening_time_adverb(
    features: EnglishBoundaryFeatures,
    *,
    tail_is_sentence_adverb: bool,
    head_is_subject_or_determiner: bool,
) -> bool:
    return tail_is_sentence_adverb and bool(
        head_is_subject_or_determiner
        or re.fullmatch(
            r"(?:i|you|he|she|it|we|they)['’](?:d|ll|m|re|s|ve)",
            features.head,
            re.IGNORECASE,
        )
    )


def contrastive_beneficiary(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\bbut\s+for\s+(?:them|him|her|us|you),?$",
            features.left,
            re.IGNORECASE,
        )
    )


def lexical_unit(*, tail_head_is_dependency_pair: bool) -> bool:
    return tail_head_is_dependency_pair


def direction_here(features: EnglishBoundaryFeatures) -> bool:
    return features.tail in {"down", "over", "up"} and features.head == "here"


def context_dependent_adjective(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\bwell\s+suited$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(
            r"^and\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def reporting_copular_content(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:point|thing)\s+that\s+(?:i|we|you|they|he|she)\s+"
            r"(?:was\s+)?(?:trying\s+to\s+)?(?:make|say|show),?$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and features.head in {"is", "was"}
    )


def interrogative_complement(features: EnglishBoundaryFeatures) -> bool:
    return features.semantic_tail == "what" and features.right[:1].islower()


def coordinated_noun_phrase_contextual(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.match(
            r"^and\s+(?:a|an|the|our|your|their|his|her|its|this|that|these|those)\b",
            features.semantic_right,
            re.IGNORECASE,
        )
        and re.search(
            r"\b(?:the|our|your|their|his|her|its|this|that|these|those)\s+"
            r"[a-z][a-z'’-]*(?:\s+age)?$",
            features.semantic_left,
            re.IGNORECASE,
        )
    )


def dependent_phrase(
    features: EnglishBoundaryFeatures,
    *,
    head_is_dependent: bool,
) -> bool:
    return bool(
        head_is_dependent
        and features.right[:1].islower()
        and not CLAUSE_RE.search(features.left)
        and not features.complete_fixed_phrase
        and not features.complete_ease_of_use_phrase
    )


def standalone_of_phrase(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        features.head == "of"
        and re.match(
            r"^Of\s+(?!course\b)",
            features.right,
            re.IGNORECASE,
        )
        and not CLAUSE_RE.search(features.right)
    )


def relative_clause(
    features: EnglishBoundaryFeatures,
    *,
    head_is_relative: bool,
    that_starts_complement: bool,
) -> bool:
    return bool(
        head_is_relative and not that_starts_complement and not CLAUSE_RE.search(features.left)
    )


def dependent_adverbial_clause(
    features: EnglishBoundaryFeatures,
    *,
    head_is_translation_sensitive: bool,
) -> bool:
    return head_is_translation_sensitive and not TERMINAL_RE.search(features.left)


def clause_final_subject(
    features: EnglishBoundaryFeatures,
    *,
    head_is_finite_predicate: bool,
) -> bool:
    return head_is_finite_predicate and bool(
        re.search(
            r"[,;:]\s*(?:and|but|or|so|while|whereas)\s+"
            r"[a-z][a-z'’-]*(?:\s+[a-z][a-z'’-]*){0,2}$",
            features.semantic_left,
            flags=re.IGNORECASE,
        )
    )


def temporal_continuation(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        features.head == "and"
        and len(features.right_tokens) >= 2
        and features.right_tokens[1] in {"go", "purchase", "buy"}
        and re.search(
            r"\b(?:after|before)\b[^,;.!?]*$",
            features.left,
            flags=re.IGNORECASE,
        )
    )
