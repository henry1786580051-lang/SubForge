import json
from types import SimpleNamespace

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.translate import context as context_module
from subforge.core.translate.context import (
    MAX_TERMINOLOGY_CHARS,
    _compact_transcript,
    _document_entity_alias_groups,
    _document_entity_contexts,
    _document_entity_corrections,
    _document_entity_mentions,
    _document_entity_variant_candidates,
    _document_lexical_context_hints,
    _document_lexical_variant_candidates,
    _document_numeric_contexts,
    _document_numeric_corrections,
    _extend_confirmed_alias_corrections,
    _format_terms,
    build_translation_context,
)
from subforge.core.translate.types import TargetLanguage


def test_compact_transcript_samples_the_middle_of_long_transcript():
    segments = [f"section-{index} " + (chr(65 + index) * 80) for index in range(9)]

    compact = _compact_transcript(segments, limit=500)

    assert len(compact) <= 500
    assert "section-0" in compact
    assert "section-4" in compact
    assert "section-8" in compact
    assert compact.count("\n...\n") == 4


def test_compact_transcript_keeps_short_transcript_unchanged():
    assert _compact_transcript([" first ", "second"], limit=100) == "first second"


def test_format_terms_bounds_long_context_payload():
    terms = [
        {"source": f"candidate-{index}", "target": f"候选人{index}", "note": "n" * 300}
        for index in range(80)
    ]

    rendered = _format_terms(terms)

    assert len(rendered) <= MAX_TERMINOLOGY_CHARS
    assert "candidate-0" in rendered
    assert "candidate-79" not in rendered


def test_format_terms_promotes_asr_corrections_and_recovers_canonical_note_target():
    terms = [
        {"source": f"basic-{index}", "target": f"基础{index}", "note": "common term"}
        for index in range(60)
    ]
    terms.append(
        {
            "source": "Grimina GR Corolla",
            "target": "Grimina GR Corolla",
            "note": "Likely ASR error for 'GRMN' (a Toyota performance variant).",
        }
    )

    rendered = _format_terms(terms)

    assert "Grimina -> GRMN (probable ASR correction derived from context)" not in rendered
    assert "Grimina GR Corolla -> GRMN GR Corolla" in rendered
    assert "basic-59" not in rendered


def test_format_terms_prefers_explicit_asr_spelling_over_translated_target():
    rendered = _format_terms(
        [
            {
                "source": "Infinity",
                "target": "英菲尼迪",
                "note": (
                    "Probable ASR correction: 'Infinity' should be 'Infiniti' "
                    "(brand name)."
                ),
            }
        ]
    )

    assert "Infinity -> Infiniti" in rendered
    assert "Infinity -> 英菲尼迪" not in rendered


def test_format_terms_recovers_unquoted_canonical_name_from_asr_note():
    rendered = _format_terms(
        [
            {
                "source": "rubber veil",
                "target": "马拉巴谷",
                "note": "probable ASR correction; phonetic candidate for Marabba Vale",
            }
        ]
    )

    assert "rubber veil -> Marabba Vale" in rendered


def test_format_terms_does_not_parse_canonical_explanation_as_a_name():
    rendered = _format_terms(
        [
            {
                "source": "Marabba Vale",
                "target": "马拉巴谷",
                "note": (
                    "probable ASR correction; recurring entity name, canonical form "
                    "confirmed by multiple mentions and context"
                ),
            }
        ]
    )

    assert "Marabba Vale -> 马拉巴谷" in rendered
    assert "-> confirmed by" not in rendered


def test_format_terms_does_not_parse_generic_variant_description_as_a_name():
    rendered = _format_terms(
        [
            {
                "source": "Maraba Vale",
                "target": "马拉巴谷",
                "note": "probable ASR correction; variant of the same tower name",
            }
        ]
    )

    assert "Maraba Vale -> 马拉巴谷" in rendered
    assert "-> the" not in rendered


def test_format_terms_keeps_nonliteral_phrases_ahead_of_basic_vocabulary():
    terms = [
        {"source": f"basic-{index}", "target": f"基础{index}", "note": "common term"}
        for index in range(60)
    ]
    terms.append(
        {
            "source": "take your hands off your pearls",
            "target": "别大惊小怪",
            "note": "ironic figurative wording",
        }
    )

    rendered = _format_terms(terms)

    assert "take your hands off your pearls -> 别大惊小怪" in rendered
    assert "basic-59" not in rendered


def test_document_entity_mentions_cover_model_evidence_outside_sampled_windows():
    mentions = _document_entity_mentions(
        [
            "Today we are driving the Toyota GR Corolla.",
            "The Lexus LBX Morizo RR uses the G16E-GTS engine.",
        ]
    )

    assert "Toyota GR Corolla" in mentions
    assert "Lexus LBX Morizo RR uses" in mentions
    assert "G16E-GTS" in mentions


def test_document_entity_mentions_surface_internal_phonetic_name_candidates():
    mentions = _document_entity_mentions(
        ["So if you want to make your Chiarco roll a little bit louder."]
    )

    assert "Chiarco" in mentions


def test_document_entity_contexts_keep_neighbors_for_uncertain_model_names():
    contexts = _document_entity_contexts(
        [
            "This is the same G16E engine used by Toyota.",
            "The Lexus LMXX Grimina or something has this engine as well.",
            "But under here there are no hood struts.",
            "You pay extra for the big Marizzo-style spoiler.",
        ]
    )

    rendered = "\n".join(contexts)
    assert "G16E engine" in rendered
    assert "Lexus LMXX Grimina or something" in rendered
    assert "Marizzo-style spoiler" in rendered


def test_document_entity_contexts_are_bounded():
    contexts = _document_entity_contexts(
        [f"Toyota Model{index} has technical detail {index}." for index in range(200)]
    )

    assert len(contexts) <= context_module.MAX_ENTITY_CONTEXTS
    assert sum(len(item) for item in contexts) <= context_module.MAX_ENTITY_CONTEXT_CHARS


def test_document_entity_variant_candidates_require_recurring_canonical_evidence():
    candidates = _document_entity_variant_candidates(
        [
            "The RHIC tunnel is underground.",
            "RHIC accelerates heavy ions.",
            "Rick is the older machine on the site.",
            "The OHIC ring is nearly four kilometres around.",
            "Alice works in the control room.",
        ]
    )

    pairs = {
        (item["heard"], item["possible_canonical"])
        for item in candidates
    }
    assert ("Rick", "RHIC") in pairs
    assert ("OHIC", "RHIC") in pairs
    assert not any(item["heard"] == "Alice" for item in candidates)


def test_document_entity_variant_candidates_do_not_guess_from_one_off_acronym():
    assert _document_entity_variant_candidates(
        ["The RHIC tunnel is underground.", "Rick is nearby."]
    ) == []


def test_document_entity_candidates_surface_ampersand_acronym_with_one_canonical_use():
    candidates = _document_entity_variant_candidates(
        ["The detector is what B&L named it.", "BNL is leading the project."]
    )

    assert any(
        item["heard"] == "B&L" and item["possible_canonical"] == "BNL"
        for item in candidates
    )
    assert _document_entity_corrections(candidates) == [
        {
            "source": "B&L",
            "target": "BNL",
            "note": (
                "probable ASR correction; the heard form closely matches a recurring "
                "document-attested acronym"
            ),
        }
    ]


def test_document_entity_alias_groups_surface_recurring_phonetic_name_variants():
    groups = _document_entity_alias_groups(
        [
            "This is Marabba Vale, a very slender tower.",
            "The apartments in Moorabbah Vale are expensive.",
            "Design your own Maraba Vale apartment.",
            "Mirabevail also uses hydraulic dampers.",
            "This is what the concrete core looks like on a rubber veil.",
        ]
    )

    variants = {
        item["text"]
        for group in groups
        for item in group["variants"]
    }
    assert {"Marabba Vale", "Moorabbah Vale", "Maraba Vale", "Mirabevail"} <= variants
    candidates = {
        item["text"].casefold()
        for group in groups
        for item in group["phonetic_candidates"]
    }
    assert "rubber veil" in candidates
    assert "moorabbah vale's" not in candidates
    assert "a moorabbah" not in candidates


def test_document_entity_alias_groups_do_not_merge_related_distinct_names():
    groups = _document_entity_alias_groups(
        [
            "New York has many towers.",
            "New York zoning shaped the skyline.",
            "A New Yorker discussed the project.",
        ]
    )

    assert groups == []


def test_document_lexical_candidates_surface_rare_asr_variant_of_recurring_term():
    candidates = _document_lexical_variant_candidates(
        [
            "We are in a new era of the supertall.",
            "Most supertall buildings are residential.",
            "But these supertools aren't designed for the average New Yorker.",
            "The structure is extremely thin.",
            "The extreme ratio changes the engineering.",
            "The apartment is incredible.",
            "The views are incredibly broad.",
            "The result is incredibly expensive.",
        ]
    )

    pairs = {
        (item["heard"].casefold(), item["possible_canonical"].casefold())
        for item in candidates
    }
    assert ("supertools", "supertall") in pairs
    assert ("extremely", "extreme") not in pairs
    assert ("incredible", "incredibly") not in pairs


def test_document_lexical_hints_keep_close_domain_candidate_but_not_inflection():
    hints = _document_lexical_context_hints(
        [
            {
                "heard": "supertools",
                "possible_canonical": "supertall",
                "canonical_count": 2,
                "similarity": 0.823,
            },
            {
                "heard": "incredible",
                "possible_canonical": "incredibly",
                "canonical_count": 2,
                "similarity": 0.95,
            },
            {
                "heard": "stiffen",
                "possible_canonical": "stiffness",
                "canonical_count": 2,
                "similarity": 0.819,
            },
        ]
    )

    assert hints == [
        {
            "source": "supertools",
            "target": "supertalls",
            "note": (
                "unconfirmed lexical similarity hint; use only when the local sentence "
                "and recurring document subject prove this reading"
            ),
        }
    ]


def test_confirmed_alias_variants_extend_to_lowercase_phonetic_candidate():
    groups = _document_entity_alias_groups(
        [
            "This is Marabba Vale.",
            "Moorabbah Vale has a concrete core.",
            "Maraba Vale is in Dubai.",
            "Mirabevail has dampers.",
            "This is what the core looks like on a rubber veil.",
        ]
    )
    corrections = _extend_confirmed_alias_corrections(
        groups,
        [
            {
                "source": "Marabba Vale",
                "target": "Muraba Veil",
                "note": "probable ASR correction",
            },
            {
                "source": "Moorabbah Vale",
                "target": "Muraba Veil",
                "note": "phonetic ASR variant",
            },
        ],
    )

    assert {item["source"] for item in corrections} == {"rubber veil"}
    assert {item["target"] for item in corrections} == {"Muraba Veil"}


def test_alias_candidate_is_not_extended_from_one_or_conflicting_mapping():
    groups = _document_entity_alias_groups(
        [
            "This is Marabba Vale.",
            "Moorabbah Vale has a concrete core.",
            "Marabba Vale is in Dubai.",
            "This is what the core looks like on a rubber veil.",
        ]
    )
    one_mapping = [
        {
            "source": "Marabba Vale",
            "target": "Muraba Veil",
            "note": "probable ASR correction",
        }
    ]
    conflicting = [
        *one_mapping,
        {
            "source": "Moorabbah Vale",
            "target": "Murabba Veil",
            "note": "probable ASR correction",
        },
    ]

    assert _extend_confirmed_alias_corrections(groups, one_mapping) == []
    assert _extend_confirmed_alias_corrections(groups, conflicting) == []


def test_document_entity_corrections_promote_recurring_acronym_but_keep_wordplay():
    candidates = _document_entity_variant_candidates(
        [
            "RHIC studies heavy ions.",
            "RHIC operated for decades.",
            "RHIC is being converted into a new collider.",
            "We call RIC a relativistic collider.",
            "As for OHIC itself, construction started decades ago.",
            "The EPIC detector is at the collision point.",
            "That is not just me calling it an epic detector.",
            "It is really what BNL named it.",
            "The EIC will replace RHIC.",
            "EIC construction is under way.",
        ]
    )

    corrections = _document_entity_corrections(candidates)

    assert any(item["source"] == "RIC" and item["target"] == "RHIC" for item in corrections)
    assert any(item["source"] == "OHIC" and item["target"] == "RHIC" for item in corrections)
    assert not any(item["source"] == "EPIC" for item in corrections)


def test_document_numeric_contexts_keep_only_cross_cue_evidence_windows():
    contexts = _document_numeric_contexts(
        [
            "The known particles explain 2% or 5% of the universe.",
            "The remaining 95% is still unknown.",
            "The tunnel covers 34,000 square kilometres.",
            "That is about six football fields.",
            "A standalone number is 12.",
        ]
    )

    rendered = "\n".join(contexts)
    assert "2% or 5%" in rendered
    assert "remaining 95%" in rendered
    assert "34,000 square kilometres" in rendered
    assert "football fields" in rendered
    assert _document_numeric_contexts(["A standalone number is 12."]) == []


def test_document_numeric_corrections_require_explicit_arithmetic_evidence():
    corrections = _document_numeric_corrections(
        [
            "The known particles explain 2% or 5% of the universe.",
            "The remaining 95% is still unknown.",
            "The site covers 34,000 square kilometres.",
            "In American football fields, that is approximately six and a half of them.",
        ]
    )

    mappings = {(item["source"], item["target"]) for item in corrections}
    assert ("2% or 5%", "5%") in mappings
    assert ("34,000 square kilometres", "34,000 square metres") in mappings


def test_document_numeric_corrections_do_not_guess_without_proof():
    assert _document_numeric_corrections(
        ["The value might be 2% or 5%.", "The site covers 34,000 square kilometres."]
    ) == []


def test_context_disables_native_reasoning_for_deepseek_v4(monkeypatch):
    calls = []
    response_text = json.dumps(
        {
            "summary": "Retirement policy",
            "terminology": [],
            "style": "Conversational",
        }
    )

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=response_text))
            ]
        )

    monkeypatch.setattr(context_module, "call_llm", fake_call_llm)
    data = ASRData([ASRDataSeg("Social Security is changing.", 0, 1000)])

    result = build_translation_context(
        data,
        model="deepseek-v4-flash",
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        use_cache=False,
    )

    assert result.summary == "Retirement policy"
    assert [call["reasoning_mode"] for call in calls] == ["disabled"]
    assert all(call["max_output_tokens"] == 4096 for call in calls)
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert "document_entity_mentions" in payload
    assert "document_entity_contexts" in payload
    assert "document_entity_variant_candidates" in payload
    assert "document_entity_alias_groups" in payload
    assert "document_lexical_variant_candidates" in payload
    assert "document_numeric_contexts" in payload
    assert "high-confidence idioms" in calls[0]["messages"][0]["content"]


def test_context_keeps_deterministic_numeric_corrections_when_model_omits_them(
    monkeypatch,
):
    response_text = json.dumps(
        {"summary": "Particle collider", "terminology": [], "style": "Technical"}
    )
    monkeypatch.setattr(
        context_module,
        "call_llm",
        lambda **_kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_text))]
        ),
    )
    data = ASRData(
        [
            ASRDataSeg("The known particles explain 2% or 5%.", 0, 1000),
            ASRDataSeg("The remaining 95% is unknown.", 1000, 2000),
            ASRDataSeg("It covers 34,000 square kilometres.", 2000, 3000),
            ASRDataSeg(
                "In football fields, that is approximately six and a half of them.",
                3000,
                4000,
            ),
        ]
    )

    result = build_translation_context(
        data,
        model="deepseek-v4-flash",
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        use_cache=False,
    )

    assert "2% or 5% -> 5%" in result.terminology
    assert "34,000 square kilometres -> 34,000 square metres" in result.terminology


def test_context_retry_remains_without_reasoning_when_answer_has_no_json(monkeypatch):
    calls = []
    responses = iter(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "summary": "Automotive review",
                                    "terminology": [],
                                    "style": "Conversational",
                                }
                            )
                        )
                    )
                ]
            ),
        ]
    )

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(context_module, "call_llm", fake_call_llm)
    data = ASRData([ASRDataSeg("A GR Corolla review.", 0, 1000)])

    result = build_translation_context(
        data,
        model="deepseek-v4-flash",
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        use_cache=False,
    )

    assert result.summary == "Automotive review"
    assert [call["reasoning_mode"] for call in calls] == ["disabled", "disabled"]


def test_context_rejects_acronym_collapse_when_transcript_confirms_wordplay(monkeypatch):
    response_text = json.dumps(
        {
            "summary": "Particle collider",
            "terminology": [
                {
                    "source": "EPIC",
                    "target": "EIC",
                    "note": "probable ASR correction",
                },
                {
                    "source": "RIC",
                    "target": "RHIC",
                    "note": "probable ASR correction",
                },
            ],
            "style": "Technical",
        }
    )
    monkeypatch.setattr(
        context_module,
        "call_llm",
        lambda **_kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_text))]
        ),
    )
    data = ASRData(
        [
            ASRDataSeg("The collision happens at the EPIC detector.", 0, 1000),
            ASRDataSeg("That is not just me calling it an epic detector.", 1000, 2000),
            ASRDataSeg("It is really what BNL named it.", 2000, 3000),
            ASRDataSeg("We call RIC a heavy-ion collider.", 3000, 4000),
            ASRDataSeg("RHIC ran for decades.", 4000, 5000),
        ]
    )

    result = build_translation_context(
        data,
        model="deepseek-v4-flash",
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        use_cache=False,
    )

    assert "EPIC -> EIC" not in result.terminology
    assert "RIC -> RHIC" in result.terminology
