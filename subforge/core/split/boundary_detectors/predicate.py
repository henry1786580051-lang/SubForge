"""Pure English predicate-boundary detectors."""

from __future__ import annotations

import re

from subforge.core.split.boundary_features import TERMINAL_RE, EnglishBoundaryFeatures


def subject_adverb(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:i|you|he|she|it|we|they|this|that|who|which)"
            r"(?:['’](?:d|ll|m|re|s|ve))?\s+"
            r"(?:probably|possibly|maybe|never|always|still|already|currently|"
            r"actually|really|definitely|usually|often)"
            r"(?:\s+(?:probably|possibly|maybe|never|always|still|already|currently|"
            r"actually|really|definitely|usually|often))*$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and features.semantic_right[:1].islower()
    )


def negative_auxiliary_complement(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:can|could|did|do|does|had|has|have|will|would|"
            r"can['’]t|couldn['’]t|didn['’]t|doesn['’]t|don['’]t|"
            r"won['’]t|wouldn['’]t)\s+(?:actually\s+|really\s+)?(?:never|no)$",
            features.semantic_left,
            flags=re.IGNORECASE,
        )
    )


def linking_verb_complement(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        features.tail
        in {
            "become",
            "became",
            "becomes",
            "remain",
            "remained",
            "remains",
        }
        and not re.match(
            r"^(?:what|whatever|whoever)\b",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(
            r"^(?:a|an|the|one|another|any|some|this|that|these|those|"
            r"[A-Za-z][A-Za-z'’-]*(?:ly|al|ble|ful|ic|ive|less|ous)\b)",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def progressive_complement(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        features.semantic_right[:1].islower()
        and re.search(
            r"\b(?:am|is|are|was|were|be|been|being|"
            r"(?:i|you|he|she|it|we|they|this|that|who|which)"
            r"['’](?:m|re|s))\s+"
            r"(?:(?:probably|possibly|maybe|never|always|still|already|currently|"
            r"actually|really|definitely|usually|often)\s+)*"
            r"[a-z][a-z'’-]*ing$",
            features.semantic_left,
            re.IGNORECASE,
        )
    )


def auxiliary_participle(features: EnglishBoundaryFeatures) -> bool:
    return features.head in {"been", "being"} and bool(
        re.search(
            r"\b(?:have|has|had)\s+(?:always|already|also|never|not|often|still|"
            r"typically|traditionally|usually|[a-z]+ly)$",
            features.semantic_left,
            re.IGNORECASE,
        )
    )


def sentence_adverb_finite(
    features: EnglishBoundaryFeatures,
    *,
    tail_is_sentence_adverb: bool,
) -> bool:
    return tail_is_sentence_adverb and bool(
        re.match(
            r"^(?:(?:i|you|he|she|it|we|they)(?:['’](?:d|ll|m|re|s|ve))?|"
            r"am|are|can|could|did|do|does|had|has|have|is|looks?|may|might|must|"
            r"seems?|shall|should|was|were|will|would)\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def auxiliary_sentence_adverb(
    features: EnglishBoundaryFeatures,
    *,
    tail_is_sentence_adverb: bool,
    previous_is_subject_auxiliary: bool,
) -> bool:
    return bool(
        tail_is_sentence_adverb
        and (
            previous_is_subject_auxiliary
            or re.fullmatch(
                r"(?:am|are|is|was|were|be|been|can|could|may|might|must|"
                r"shall|should|will|would)",
                features.previous,
                re.IGNORECASE,
            )
        )
        and re.match(
            r"^[a-z][a-z'’-]*ing\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def infinitive_adverb(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:needs?|needed)\s+to\s+[a-z][a-z'’-]*ly$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(
            r"^[a-z][a-z'’-]*(?:\s+|$)",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def participle_quantified_object(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:welcoming|serving|handling|accommodating)$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(
            r"^(?:about|around|between|more\s+than|over|up\s+to|\d)",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def modal_adverb(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:can|could|may|might|must|shall|should|will|would)\s+"
            r"[a-z][a-z'’-]*ly$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(
            r"^[a-z][a-z'’-]*(?:\s+|$)",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def contrastive_prepositional_frame(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(r"\bbut\s+for\s+[^.!?]+[,]$", features.left, re.IGNORECASE)
        and re.match(
            r"^(?:a|an|the|this|that|these|those)\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def sentence_final_temporal_adverb(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        not TERMINAL_RE.search(features.left)
        and re.fullmatch(
            r"(?:currently|now|today|tonight|yesterday)[.!?]?",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def reporting_quoted_object(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:call(?:ed|s)?|name(?:d|s)?|read(?:s)?|say(?:s)?)$",
            features.semantic_left,
            re.IGNORECASE,
        )
    )


def prepositional_gerund_complement(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:after|before|by|despite|during|through|while|without)\s+"
            r"[a-z][a-z'’-]*ing$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and features.right[:1].islower()
    )


def condition_qualified_predicate(features: EnglishBoundaryFeatures) -> bool:
    return features.head in {"if", "unless", "without"} and bool(
        re.search(
            r"\b(?:am|is|are|was|were|seem(?:s|ed)?|become(?:s|ing)?)\s+"
            r"(?:too\s+|quite\s+|rather\s+|really\s+|very\s+)?"
            r"[a-z]+(?:able|al|ful|ible|ic|ive|less|ous)$",
            features.semantic_left,
            re.IGNORECASE,
        )
    )


def relative_clause_subject(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:who|which|that)(?:\s+(?:you\s+know|i\s+mean))?\s+"
            r"(?:previously|currently|also|still|often|usually|"
            r"generally|actually|really|just|never|always)$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and features.right[:1].islower()
    )


def progressive_object(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:may|might|can|could|will|would|should|must)\s+be\s+"
            r"(?:reading|using|watching|doing|making|seeing|getting)$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and features.right[:1].islower()
    )


def topic_frame(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\bone\s+thing\b.*\binteresting\s+is,?\s+(?:you\s+know,?\s+)?"
            r"with\s+[a-z][a-z'’-]*s?,?$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(
            r"^(?:i|we|you|they|he|she|it)\b",
            features.right,
            re.IGNORECASE,
        )
    )


def up_and_running_subject(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.match(
            r"^(?:finally\s+)?up\s+and\s+running\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def short_noun_subject(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        len(features.left_tokens) <= 5
        and features.left_tokens[0] in {"a", "an", "the", "this", "that", "these", "those"}
        and features.head
        in {
            "became",
            "becomes",
            "caused",
            "created",
            "did",
            "does",
            "had",
            "has",
            "made",
            "makes",
            "played",
            "plays",
            "provided",
            "provides",
            "showed",
            "shows",
            "used",
            "uses",
            "was",
            "were",
        }
    )


def subject_complement(
    features: EnglishBoundaryFeatures,
    *,
    tail_is_copula_complement: bool,
) -> bool:
    return features.head in {"is", "are", "was", "were"} and tail_is_copula_complement


def omitted_relative_one(features: EnglishBoundaryFeatures) -> bool:
    return features.tail == "one" and features.right[:1].islower()


def copula_parenthetical_complement(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:this|that|it)\s+(?:is|was),?\s+i\s+(?:think|guess),?$",
            features.left,
            re.IGNORECASE,
        )
    )


def emphatic_inversion_complement(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\bdid\s+(?:i|you|he|she|we|they)\s+(?:think|feel|believe)$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(
            r"^(?:i|you|he|she|it|we|they)\s+(?:am|are|is|was|were|would|could)\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def standalone_contrast_frame(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.fullmatch(
            r"(?:but\s+)?at\s+the\s+same\s+time"
            r"(?:,?\s+(?:you\s+know|i\s+mean))?[,]?",
            features.left,
            re.IGNORECASE,
        )
    )


def negative_existential_complement(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\bthere(?:['’]s|\s+is)\s+no,?$",
            features.left,
            re.IGNORECASE,
        )
    )


def reason_clause_subject(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\bbecause\s+at\s+the\s+time,?$",
            features.left,
            re.IGNORECASE,
        )
    )


def what_is_so_after_demonstrative(features: EnglishBoundaryFeatures) -> bool:
    return features.head == "so" and bool(
        re.search(r"\bthis\s+is\s+what['’]s$", features.left, re.IGNORECASE)
    )


def what_is_so_adjective(features: EnglishBoundaryFeatures) -> bool:
    return features.head in {"good", "great", "special"} and bool(
        re.search(r"\bwhat['’]s\s+so$", features.left, re.IGNORECASE)
    )


def transitive_object_basic(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:i|we|you|they)\s+(?:(?:can|could|do|did)\s+)?"
            r"(?:see|saw|notice|noticed|find|found),?$",
            features.left,
            re.IGNORECASE,
        )
        and re.match(
            r"^(?:a|an|the|some|many|several|changes?)\b",
            features.right,
            re.IGNORECASE,
        )
    )


def transitive_object_extended(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:i|we|you|they|he|she)\s+"
            r"(?:(?:can|could|did|do|does|don['’]t|doesn['’]t|didn['’]t|"
            r"may|might|must|should|will|would)\s+)?"
            r"(?:build|choose|create|find|get|give|make|seek|take|use),?$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(
            r"^(?:a|an|any|it|methods?|something|the|them|these|those|ways?)\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def perfect_reporting_content(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:i|we|you|they|he|she)(?:['’]ve|\s+have)\s+seen,?$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and features.semantic_right[:1].islower()
    )


def transitive_pronoun_object(features: EnglishBoundaryFeatures) -> bool:
    if features.head not in {"her", "him", "it", "me", "them", "us", "you"}:
        return False
    return bool(
        re.search(
            r"\b(?:can|could|did|do|does|may|might|must|should|to|will|would)\s+"
            r"[a-z][a-z'’-]*$",
            features.semantic_left,
            re.IGNORECASE,
        )
        or re.search(
            r"\b(?:i|you|he|she|it|we|they|this|that)\s+"
            r"(?:(?:also|always|already|currently|just|never|now|often|really|"
            r"still|today|usually)\s+)*"
            r"(?:ask(?:ed|s)?|bring(?:s|ing)?|call(?:ed|s)?|follow(?:ed|s)?|"
            r"help(?:ed|s)?|invite(?:d|s)?|join(?:ed|s)?|meet(?:s|ing)?|"
            r"remind(?:ed|s)?|show(?:ed|n|s)?|teach(?:es|ing|t)?|tell(?:s|ing|t)?|"
            r"watch(?:ed|es|ing)?)$",
            features.semantic_left,
            re.IGNORECASE,
        )
    )


def perfect_reporting_after_adverb(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:i|we|you|they|he|she)(?:['’]ve|\s+have)\s+"
            r"(?:clearly|definitely|really|certainly|already)\s+seen,?$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and features.semantic_right[:1].islower()
    )


def reporting_content(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\b(?:(?:start(?:ed|ing)?|continue(?:d|s|ing)?)\s+to\s+see|"
            r"(?:i|we|you|they|he|she)\s+(?:think\s+that\s+)?"
            r"(?:i|we|you|they|he|she)?\s*(?:see|seen)),?$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and features.right[:1].islower()
    )


def embedded_question_complement(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\bwhether\s+(?:i|we|you|they|he|she)\s+(?:think|believe|know),?$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(
            r"^(?:how|what|where|which|who|why)\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def reported_subject(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.match(
            r"^and\s+(?:is|are|was|were|became|becomes?)\b",
            features.semantic_right,
            re.IGNORECASE,
        )
        and re.search(
            r"\b(?:think|believe|consider|say|said|thought)\s+that\b",
            features.semantic_left,
            re.IGNORECASE,
        )
    )


def transitive_nominal_clause(features: EnglishBoundaryFeatures) -> bool:
    return features.head == "what" and bool(
        re.search(
            r"\b(?:access|choose|find|know|read|remember|see|select|understand|use|value)$",
            features.semantic_left,
            re.IGNORECASE,
        )
    )


def what_use_for_object(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        re.search(
            r"\bwhat\s+(?:i|you|we|they|he|she|it)(?:['’][a-z]+)?"
            r"(?:\s+[a-z]+){0,6}\s+use$",
            features.left,
            flags=re.IGNORECASE,
        )
        and re.match(
            r"^(?:a|an|the|this|that|these|those|it|them|him|her|us)\b"
            r"(?:\s+\S+){0,4}\s+for\b",
            features.right,
            flags=re.IGNORECASE,
        )
    )


def dependent_locative_clause(features: EnglishBoundaryFeatures) -> bool:
    return bool(
        features.head == "here"
        and len(features.right_tokens) >= 2
        and features.right_tokens[1] in {"in", "on", "with"}
        and features.right[:1].islower()
    )


def way_clause_subject(features: EnglishBoundaryFeatures) -> bool:
    return features.head in {"is", "was"} and bool(
        re.match(
            r"^the\s+(?:easiest|best|only|main)\s+way\b",
            features.left,
            flags=re.IGNORECASE,
        )
    )


def proper_name_subject(
    features: EnglishBoundaryFeatures,
    *,
    previous_is_dependent_head: bool,
) -> bool:
    explicit_named_subject = bool(
        re.search(
            r"\b(?:a|an|my|our|his|her|their|the|this|that)\s+"
            r"(?:[a-z][a-z'’-]*\s+){0,5}(?:of|in|from)\s+"
            r"[A-Z][A-Za-z0-9'’.+-]*,?$",
            features.left,
        )
    )
    return bool(
        features.head in {"is", "are", "was", "were"}
        and re.search(r"\b[A-Z][A-Za-z0-9'’.+-]*,?$", features.left)
        and (
            explicit_named_subject
            or len(features.left_tokens) < 2
            or not previous_is_dependent_head
        )
    )


def trailing_noun_subject(
    features: EnglishBoundaryFeatures,
    *,
    head_is_finite_predicate: bool,
) -> bool:
    short_subject = re.search(
        r"\b(?:the|these|those|our|their|his|her|that)\s+"
        r"[a-z][a-z'’-]*(?:\s+[a-z][a-z'’-]*){0,2}$",
        features.semantic_left,
        flags=re.IGNORECASE,
    )
    reported_technical_subject = re.search(
        r"\b(?:say|think|know)\s+that\s+"
        r"(?:the|these|those|our|their|his|her|that)\s+"
        r"[a-z0-9][a-z0-9'’-]*(?:\s+[a-z0-9][a-z0-9'’-]*){0,5}$",
        features.semantic_left,
        flags=re.IGNORECASE,
    )
    return head_is_finite_predicate and bool(short_subject or reported_technical_subject)


def dependent_subject_adverbial_predicate(features: EnglishBoundaryFeatures) -> bool:
    """Keep a dependent-clause subject with an adverb-led finite predicate."""
    return bool(
        re.search(
            r"\b(?:as|because|if|that|when|whereas|while)\s+"
            r"(?:the|these|those|our|their|his|her|that)\s+"
            r"[a-z][a-z'’-]*(?:\s+[a-z][a-z'’-]*){0,3}$",
            features.semantic_left,
            re.IGNORECASE,
        )
        and re.match(
            r"^(?:also|already|currently|eventually|just|now|still|then|usually)\s+"
            r"(?!(?:a|an|the|this|that|these|those|of|to|with)\b)"
            r"[a-z][a-z'’-]*\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )


def gerund_subject(
    features: EnglishBoundaryFeatures,
    *,
    head_is_finite_predicate: bool,
) -> bool:
    return bool(
        head_is_finite_predicate
        and features.tail.endswith("ing")
        and (
            len(features.left_tokens) <= 3
            or re.search(
                r"\bi\s+(?:think|guess)(?:\s+it(?:['’]s|\s+is))?\s+[a-z]+ing$",
                features.semantic_left,
                re.IGNORECASE,
            )
        )
    )


def what_clause_subject(features: EnglishBoundaryFeatures) -> bool:
    return features.head in {"is", "are", "was", "were"} and bool(
        re.match(
            r"^(?:(?:and|but)\s+)?what\b",
            features.left,
            flags=re.IGNORECASE,
        )
    )


def degree_complement(features: EnglishBoundaryFeatures) -> bool:
    right_starts_comparative = bool(
        re.match(
            r"^(?:better|bigger|broader|cheaper|closer|faster|fewer|greater|higher|"
            r"larger|less|longer|lower|more|narrower|older|shorter|slower|smaller|"
            r"stronger|taller|thicker|thinner|wider|worse)\b",
            features.semantic_right,
            re.IGNORECASE,
        )
    )
    scalar_modifier_tail = bool(
        right_starts_comparative
        and re.search(
            r"\b(?:am|are|be|been|being|became|become|is|remain(?:ed|s)?|"
            r"seem(?:ed|s)?|was|were|['’]re|['’]s)\s+"
            r"(?:both\s+)?(?:considerably|even|far|much|significantly|substantially)$",
            features.semantic_left,
            re.IGNORECASE,
        )
    )
    return bool(
        features.head == "so"
        and features.right[:1].islower()
        and len(features.right_tokens) >= 2
        and (
            features.right_tokens[1].endswith("ly")
            or features.right_tokens[1]
            in {"far", "long", "many", "much", "strong", "well", "widespread"}
        )
        or scalar_modifier_tail
    )
