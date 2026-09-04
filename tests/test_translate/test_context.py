import json
from types import SimpleNamespace

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.translate import context as context_module
from subforge.core.translate.context import (
    MAX_TERMINOLOGY_CHARS,
    _compact_transcript,
    _confirm_call_to_action_entity_corrections,
    _document_audio_homophone_corrections,
    _document_branded_common_noun_terms,
    _document_call_to_action_entity_candidates,
    _document_entity_alias_groups,
    _document_entity_contexts,
    _document_entity_corrections,
    _document_entity_mentions,
    _document_entity_variant_candidates,
    _document_lexical_context_hints,
    _document_lexical_variant_candidates,
    _document_manufacturer_identifiers,
    _document_numeric_contexts,
    _document_numeric_corrections,
    _document_rhetorical_name_candidates,
    _extend_confirmed_alias_corrections,
    _format_terms,
    _refine_rhetorical_name_terms,
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


def test_document_branded_common_noun_terms_require_repeated_naming_frames():
    terms = _document_branded_common_noun_terms(
        [
            "This is the Sphere.",
            "This is the MSG Sphere.",
            "Sphere Entertainment licenses the design.",
            "The Sphere is now expanding.",
            "A sphere is an efficient geometric structure.",
            "Other spheres may be built as venues.",
        ]
    )

    assert terms == [
        {
            "source": "Sphere",
            "target": "Sphere",
            "note": (
                "branded common-noun family confirmed by repeated naming frames; preserve "
                "this Latin name when it denotes the recurring venue, product, or project "
                "family, including lowercase ASR occurrences, but translate clearly generic "
                "shape or category uses by meaning"
            ),
        }
    ]


def test_document_branded_common_noun_terms_do_not_promote_generic_repetition():
    assert (
        _document_branded_common_noun_terms(
            [
                "This is the bridge.",
                "The Bridge carries traffic.",
                "A bridge crosses the river.",
                "Another bridge is planned.",
                "The bridge needs repairs.",
            ]
        )
        == []
    )


def test_document_rhetorical_name_candidates_keep_local_evidence_without_deciding():
    candidates = _document_rhetorical_name_candidates(
        [
            "Toronto is growing quickly.",
            "Why has the Great White North chosen to build so high?",
            "The answer starts with housing demand.",
        ]
    )

    assert candidates == [
        {
            "phrase": "Great White North",
            "context": (
                "Toronto is growing quickly. Why has the Great White North chosen to build "
                "so high? The answer starts with housing demand."
            ),
        }
    ]


def test_document_rhetorical_name_candidates_rebuild_word_timestamp_sequence():
    candidates = _document_rhetorical_name_candidates(
        [
            "<S1> Why",
            "<S1> has",
            "<S1> the",
            "<S1> Great",
            "<S1> White",
            "<S1> North",
            "<S1> decided",
            "<S1> now?",
        ]
    )

    assert candidates[0]["phrase"] == "Great White North"
    assert "the Great White North decided" in candidates[0]["context"]


def test_document_rhetorical_name_candidates_exclude_clear_facility_names():
    candidates = _document_rhetorical_name_candidates(
        [
            "The Chrysler Building is famous.",
            "The Pinnacle Sky Tower is under construction.",
            "The Great White North is the actual nickname here.",
        ]
    )

    assert [item["phrase"] for item in candidates] == ["Great White North"]


def test_refine_rhetorical_name_terms_uses_selective_deepseek_reasoning(monkeypatch):
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "terms": [
                                    {
                                        "source": "Great White North",
                                        "is_epithet": True,
                                        "target": "北方雪国",
                                        "note": "加拿大的地理文化别称",
                                    }
                                ]
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(context_module, "call_llm", fake_call_llm)
    result = _refine_rhetorical_name_terms(
        [
            {
                "source": "Great White North",
                "target": "大白北",
                "note": "Epithet for Canada",
            }
        ],
        [
            {
                "phrase": "Great White North",
                "context": "Why has the Great White North chosen to build so high?",
            }
        ],
        model="deepseek-v4-flash",
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        use_cache=False,
        llm_client=object(),
    )

    assert result[0]["target"] == "北方雪国"
    assert calls[0]["reasoning_mode"] == "enabled"
    assert calls[0]["max_output_tokens"] == 4096
    assert "substituted verbatim at the source" in calls[0]["messages"][0]["content"]


def test_refine_rhetorical_name_terms_ignores_unconfirmed_official_name(monkeypatch):
    response_text = json.dumps(
        {
            "terms": [
                {
                    "source": "United States",
                    "is_epithet": False,
                    "target": "",
                    "note": "official country name",
                }
            ]
        }
    )
    monkeypatch.setattr(
        context_module,
        "call_llm",
        lambda **_kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_text))]
        ),
    )

    result = _refine_rhetorical_name_terms(
        [{"source": "United States", "target": "美国", "note": "official name"}],
        [{"phrase": "United States", "context": "Across the United States."}],
        model="deepseek-v4-flash",
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        use_cache=False,
        llm_client=object(),
    )

    assert result[0]["target"] == "美国"


def test_refine_rhetorical_name_terms_falls_back_when_reasoning_keeps_literal_draft(
    monkeypatch,
):
    calls = []
    responses = iter(
        [
            {
                "source": "Great White North",
                "is_epithet": True,
                "target": "大白北",
            },
            {
                "source": "Great White North",
                "is_epithet": True,
                "target": "加拿大北境",
            },
        ]
    )

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        item = next(responses)
        content = json.dumps({"terms": [item]}, ensure_ascii=False)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    monkeypatch.setattr(context_module, "call_llm", fake_call_llm)
    result = _refine_rhetorical_name_terms(
        [{"source": "Great White North", "target": "大白北", "note": "epithet"}],
        [{"phrase": "Great White North", "context": "A Canadian nickname."}],
        model="deepseek-v4-flash",
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        use_cache=False,
        llm_client=object(),
    )

    assert result[0]["target"] == "加拿大北境"
    assert [call["reasoning_mode"] for call in calls] == ["enabled", "disabled"]


def test_refine_rhetorical_name_terms_prohibits_repeated_literal_draft(monkeypatch):
    calls = []
    responses = iter(
        [
            {"source": "Great White North", "is_epithet": True, "target": "大白北"},
            {"source": "Great White North", "is_epithet": True, "target": "大白北"},
            {"source": "Great White North", "is_epithet": True, "target": "加拿大北境"},
        ]
    )

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        content = json.dumps({"terms": [next(responses)]}, ensure_ascii=False)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    monkeypatch.setattr(context_module, "call_llm", fake_call_llm)
    result = _refine_rhetorical_name_terms(
        [{"source": "Great White North", "target": "大白北", "note": "epithet"}],
        [{"phrase": "Great White North", "context": "A Canadian nickname."}],
        model="deepseek-v4-flash",
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        use_cache=False,
        llm_client=object(),
    )

    assert result[0]["target"] == "加拿大北境"
    assert "prohibited literal calque" in calls[2]["messages"][0]["content"]


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
                "note": ("Probable ASR correction: 'Infinity' should be 'Infiniti' (brand name)."),
            }
        ]
    )

    assert "Infinity -> Infiniti" in rendered
    assert "Infinity -> 英菲尼迪" not in rendered


def test_format_terms_rejects_pronoun_to_vehicle_global_mapping():
    rendered = _format_terms(
        [
            {
                "source": "me",
                "target": "F-150",
                "note": "probable ASR correction",
            },
            {
                "source": "me's",
                "target": "F-150's",
                "note": "probable ASR correction",
            },
            {
                "source": "bear with me",
                "target": "请多包涵",
                "note": "idiom",
            },
        ]
    )

    assert "me -> F-150" not in rendered
    assert "me's -> F-150's" not in rendered
    assert "bear with me -> 请多包涵" in rendered


def test_format_terms_keeps_uppercase_and_alphanumeric_identifiers():
    rendered = _format_terms(
        [
            {"source": "US", "target": "美国", "note": "geographic abbreviation"},
            {"source": "I-95", "target": "I-95", "note": "highway identifier"},
        ]
    )

    assert "US -> 美国" in rendered
    assert "I-95 -> I-95" in rendered


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


def test_document_call_to_action_candidates_use_recurring_channel_evidence():
    candidates = _document_call_to_action_entity_candidates(
        [
            "B1M covers major infrastructure projects.",
            "This B1M documentary follows the airport expansion.",
            "Thanks for watching.",
            "Make sure you subscribe to Night for more videos.",
        ]
    )

    assert candidates == [
        {
            "heard": "Night",
            "recurring_candidates": [{"canonical": "B1M", "count": 2}],
            "context": ("Thanks for watching. | Make sure you subscribe to Night for more videos."),
        }
    ]


def test_document_call_to_action_candidates_require_recurring_identity():
    assert (
        _document_call_to_action_entity_candidates(
            ["Thanks for watching.", "Make sure you subscribe to Night."]
        )
        == []
    )


def test_call_to_action_correction_requires_one_independently_identified_channel():
    candidates = [
        {
            "heard": "Night",
            "recurring_candidates": [
                {"canonical": "B1M", "count": 3},
                {"canonical": "ZHA", "count": 2},
            ],
        }
    ]
    terminology = [
        {"source": "B1M", "target": "B1M", "note": "Channel name; keep as is."},
        {"source": "ZHA", "target": "扎哈·哈迪德建筑事务所", "note": "Architecture firm."},
    ]

    assert _confirm_call_to_action_entity_corrections(candidates, terminology) == [
        {
            "source": "Night",
            "target": "B1M",
            "note": (
                "probable ASR correction; the closing call to action and independently "
                "identified document channel name confirm the intended identity"
            ),
        }
    ]


def test_call_to_action_correction_accepts_unique_self_media_evidence():
    candidates = _document_call_to_action_entity_candidates(
        [
            "This B1M documentary covers a major airport.",
            "This is the highest I have ever filmed for B1M.",
            "Make sure you subscribe to Night.",
        ]
    )

    assert _confirm_call_to_action_entity_corrections(candidates, []) == [
        {
            "source": "Night",
            "target": "B1M",
            "note": (
                "probable ASR correction; the closing call to action and independently "
                "identified document channel name confirm the intended identity"
            ),
        }
    ]


def test_call_to_action_correction_rejects_ambiguous_channel_identity():
    candidates = [
        {
            "heard": "Night",
            "recurring_candidates": [
                {"canonical": "B1M", "count": 3},
                {"canonical": "ABC", "count": 3},
            ],
        }
    ]
    terminology = [
        {"source": "B1M", "target": "B1M", "note": "Channel name."},
        {"source": "ABC", "target": "ABC", "note": "Publication name."},
    ]

    assert _confirm_call_to_action_entity_corrections(candidates, terminology) == []


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

    pairs = {(item["heard"], item["possible_canonical"]) for item in candidates}
    assert ("Rick", "RHIC") in pairs
    assert ("OHIC", "RHIC") in pairs
    assert not any(item["heard"] == "Alice" for item in candidates)


def test_document_entity_variant_candidates_do_not_guess_from_one_off_acronym():
    assert (
        _document_entity_variant_candidates(["The RHIC tunnel is underground.", "Rick is nearby."])
        == []
    )


def test_document_entity_candidates_surface_ampersand_acronym_with_one_canonical_use():
    candidates = _document_entity_variant_candidates(
        ["The detector is what B&L named it.", "BNL is leading the project."]
    )

    assert any(
        item["heard"] == "B&L" and item["possible_canonical"] == "BNL" for item in candidates
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

    variants = {item["text"] for group in groups for item in group["variants"]}
    assert {"Marabba Vale", "Moorabbah Vale", "Maraba Vale", "Mirabevail"} <= variants
    candidates = {
        item["text"].casefold() for group in groups for item in group["phonetic_candidates"]
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
        (item["heard"].casefold(), item["possible_canonical"].casefold()) for item in candidates
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
    assert (
        _document_numeric_corrections(
            ["The value might be 2% or 5%.", "The site covers 34,000 square kilometres."]
        )
        == []
    )


def test_document_audio_homophone_correction_requires_complete_equipment_evidence():
    corrections = _document_audio_homophone_corrections(
        [
            "You get a six-speaker sound system as standard.",
            "If you go for the XSE, there is an upgraded JBL sound system.",
            "That adds tweeters and a subwoofer.",
            "None of the bass systems in these cars are exceptional.",
            "A lot of bass sound systems are getting better.",
        ]
    )

    assert {(item["source"], item["target"]) for item in corrections} == {
        ("bass systems", "base systems"),
        ("bass sound systems", "base sound systems"),
    }


def test_document_audio_homophone_correction_preserves_real_bass_discussion():
    assert (
        _document_audio_homophone_corrections(
            [
                "The bass system controls low-frequency effects.",
                "Turn up the bass and listen to the subwoofer.",
            ]
        )
        == []
    )


def test_document_manufacturer_identifier_preserves_canonical_feature_name():
    identifiers = _document_manufacturer_identifiers(
        ["These are what Toyota calls the sport touring seats."]
    )

    assert identifiers == [
        {
            "source": "sport touring seats",
            "target": "Sport Touring",
            "note": (
                "official manufacturer identifier introduced by Toyota; preserve this "
                "canonical Latin identifier and translate only its generic head noun"
            ),
        }
    ]


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
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_text))]
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
    assert "document_rhetorical_name_candidates" in payload
    assert "document_entity_variant_candidates" in payload
    assert "document_call_to_action_entity_candidates" in payload
    assert "document_entity_alias_groups" in payload
    assert "document_lexical_variant_candidates" in payload
    assert "document_numeric_contexts" in payload
    assert "high-confidence idioms" in calls[0]["messages"][0]["content"]
    assert "established cultural or geographic epithet" in calls[0]["messages"][0]["content"]
    assert "do not merely concatenate dictionary translations" in calls[0]["messages"][0]["content"]
    assert "Repeated transcript spelling is not proof" in calls[0]["messages"][0]["content"]
    assert "what a manufacturer calls a feature" in calls[0]["messages"][0]["content"]


def test_kimi_k3_context_uses_compact_model_specific_prompt(monkeypatch):
    calls = []
    response_text = json.dumps(
        {"summary": "Nile infrastructure", "terminology": [], "style": "Documentary"}
    )

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_text))]
        )

    monkeypatch.setattr(context_module, "call_llm", fake_call_llm)
    data = ASRData([ASRDataSeg("Egypt is building a new delta.", 0, 1000)])

    result = build_translation_context(
        data,
        model="moonshotai/kimi-k3",
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        use_cache=False,
    )

    assert result.summary == "Nile infrastructure"
    assert calls[0]["messages"][0]["content"] == context_module.KIMI_K3_CONTEXT_PROMPT
    assert calls[0]["reasoning_mode"] == "disabled"


def test_lmstudio_qwen_38_context_uses_bounded_model_specific_payload(monkeypatch):
    calls = []
    response_text = json.dumps(
        {"summary": "Airport megaproject", "terminology": [], "style": "Documentary"}
    )

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_text))]
        )

    monkeypatch.setattr(context_module, "call_llm", fake_call_llm)
    data = ASRData(
        [
            ASRDataSeg(
                f"Section {index} discusses Al Maktoum International Airport and capacity.",
                index * 1000,
                (index + 1) * 1000,
            )
            for index in range(180)
        ]
    )
    client = SimpleNamespace(_subforge_base_url="http://127.0.0.1:1234/v1")

    result = build_translation_context(
        data,
        model="qwen/qwen3.8-27b",
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        use_cache=False,
        llm_client=client,
    )

    assert result.summary == "Airport megaproject"
    assert calls[0]["messages"][0]["content"] == context_module.QWEN_LOCAL_CONTEXT_PROMPT
    assert calls[0]["reasoning_mode"] == "disabled"
    payload_text = calls[0]["messages"][1]["content"]
    payload = json.loads(payload_text)
    assert len(payload["transcript_excerpt"]) <= context_module.QWEN_LOCAL_CONTEXT_CHARS
    assert len(payload_text) < 8_500
    assert "document_entity_alias_groups" not in payload
    assert set(payload) == {
        "target_language",
        "user_requirements",
        "transcript_excerpt",
        "document_entity_mentions",
        "document_entity_contexts",
        "document_entity_variant_candidates",
        "document_numeric_contexts",
    }


def test_kimi_k2_context_keeps_shared_prompt(monkeypatch):
    calls = []
    response_text = json.dumps(
        {"summary": "Nile infrastructure", "terminology": [], "style": "Documentary"}
    )
    monkeypatch.setattr(
        context_module,
        "call_llm",
        lambda **kwargs: calls.append(kwargs)
        or SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_text))]
        ),
    )
    data = ASRData([ASRDataSeg("Egypt is building a new delta.", 0, 1000)])

    build_translation_context(
        data,
        model="moonshotai/kimi-k2.6",
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        use_cache=False,
    )

    assert calls[0]["messages"][0]["content"] != context_module.KIMI_K3_CONTEXT_PROMPT
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
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))]),
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
