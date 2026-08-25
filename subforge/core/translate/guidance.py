"""Compact, source-aware guidance for subtitle translation and repair."""

from __future__ import annotations

import re
from collections.abc import Iterable

_CHINESE_TARGETS = {"简体中文", "繁体中文", "粤语"}


def _source_text(source_texts: Iterable[str]) -> str:
    return " ".join(str(text or "").strip() for text in source_texts).lower()


def _contains(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def target_language_style_rules(
    target_language: str,
    source_texts: Iterable[str],
) -> str:
    """Return concise Chinese guidance selected by the current source batch.

    Structural validators remain authoritative. These hints help the first pass
    avoid predictable errors without attaching every historical case to every
    request.
    """
    if target_language not in _CHINESE_TARGETS:
        return ""

    source = _source_text(source_texts)
    rules = [
        "Reconstruct idiomatic Chinese syntax instead of mirroring English word order.",
        "Preserve imagery, contrast, irony, and rhetorical force already present in the source "
        "with concise idiomatic Chinese. When a coherent concrete metaphor also works in Chinese, "
        "keep one consistent image instead of flattening it into generic success, impact, or domain "
        "wording; never add decorative language, emphasis, or facts absent from the source.",
        "Keep every material subject, predicate, object, modifier, negation, comparison, "
        "number, name, and conclusion exactly once under its owning key.",
        "Make adjacent cues read naturally in sequence without completing one cue with "
        "meaning borrowed from another.",
        "Map the complete source clause before translating fragments: a conjunction, subject, "
        "linking predicate, complement, number, and unit must each appear under exactly one key; "
        "never restart the same clause with a second Chinese subject or connective.",
        "Resolve explicit spoken self-corrections to the final intended value when nearby source "
        "makes that choice certain; do not turn alternatives such as 'X or Y' into a range.",
        "Resolve contrastive references from the full local construction, especially 'A rather "
        "than B, which ...'; do not attach the following belief or comment to the wrong option.",
        "Avoid repeating the same Chinese head noun twice inside one cue when one coherent noun "
        "phrase can express the source exactly.",
        "Resolve a bare pronoun, demonstrative, or omitted conversational object from the nearest "
        "explicit source noun only when the local reference is unique. Repeat only that concise "
        "head noun when Chinese would otherwise be vague; never infer an unseen part, property, "
        "or action from general knowledge or the video image.",
        "Choose the Chinese verb that matches the explicitly established interaction and medium. "
        "For example, an audio demonstration is something the audience hears, while a displayed "
        "control or physical fit is something they see or test. Replace a vague literal 'show/do "
        "this' only when the current source and read-only local context prove one concrete action.",
        "Choose technical meanings from the object, operation, and local domain rather than the "
        "most literal dictionary sense; reject a rendering that is grammatically possible but "
        "physically or professionally implausible in context.",
        "Map semantic roles before choosing Chinese word order. Preserve who causes, experiences, "
        "receives, accompanies, funds, or performs an action; never turn a family, audience, partner, "
        "or other participant into the instrument of that action merely because English uses 'with'.",
        "Render figurative actions and idioms by their function in the sentence rather than by "
        "an impossible literal action. Preserve a one-off unfamiliar proper noun as written unless "
        "document evidence gives one unambiguous canonical form.",
        "Omit semantically empty oral fillers, but preserve discourse markers that carry "
        "contrast, correction, uncertainty, or turn-taking intent.",
        "After fidelity and cue ownership are secure, preserve the speaker's voice, emphasis, "
        "imagery, and rhythm. Prefer compact, vivid, idiomatic Chinese over flat explanatory "
        "paraphrase, but never embellish, intensify, or invent an image absent from the source.",
        "Avoid translationese and bureaucratic nominalization. Recast abstract English noun chains "
        "as precise Chinese verbs, states, or modifier-head phrases when that preserves the meaning; "
        "do not mechanically write 具有意义, 作为一种, 进行交付, or 成本高达. Use an implicit Chinese "
        "subject when its reference is unmistakable, but retain or minimally recover any subject "
        "needed to prevent ambiguity across subtitle boundaries.",
    ]

    conditional_rules = (
        (
            r"\bhow\s+(?:quiet|loud|fast|slow|good|bad|hard|soft)\b|\bhow\s+\w+\s+.*\bgets?\b",
            "Render English degree constructions as natural Chinese results or states; do "
            "not mechanically translate the surface 'how ...' structure.",
        ),
        (
            r"\b\d{1,3}%\b|\bpercent\b|\buse cases?\b",
            "For percentages and use cases, identify the actual evaluated feature and make "
            "it the Chinese subject; do not mistake an example for the use case itself.",
        ),
        (
            r"\b(?:rpm|horsepower|torque|mpg|gear|clutch|steering|suspension|trim|"
            r"reverse|cargo|wheel|tire|tyre|engine|vehicle|truck|sedan|suv|"
            r"sound system|speaker|subwoofer|tweeter|jbl|proxy key|parking brake|"
            r"auto stop.start|auto down window|vent)\b",
            "Use established automotive Chinese, preserve trims and model identifiers, and "
            "recover an elliptical unit only when the local vehicle context makes it unique. "
            "Translate controls and components by their demonstrated function rather than a "
            "generic dictionary gloss. In audio-option comparisons, distinguish base/standard "
            "equipment from bass/low-frequency sound only when speaker count, an optional branded "
            "upgrade, or a subwoofer contrast makes the intended homophone unambiguous. Describe "
            "awkward mechanical operation as jerky or abrupt rather than calling the component "
            "clumsy; describe a quiet vent or fan by its low operating or wind noise; and use the "
            "established one-touch-down term for an auto-down window.",
        ),
        (
            r"\bwhat\s+(?:[A-Z][A-Za-z0-9&.'’-]*|the\s+(?:maker|manufacturer|company|brand))"
            r"\s+calls?\b|\b(?:named|marketed|sold)\s+as\b",
            "When the source explicitly introduces what a manufacturer calls a feature, trim, "
            "seat, system, or product, treat the following phrase as an official identifier. "
            "Preserve its canonical form unless global terminology supplies one established "
            "localized name; do not replace it with an improvised literal label.",
        ),
        (
            r"\bhot\s+(?:left|right)?[- ]?(?:hander|corner|turn|lap)\b",
            "In performance-driving context, 'hot' describes a fast or aggressively driven corner "
            "or lap, not temperature. Render the actual driving sense without adding a speed or "
            "maneuver that the source does not support.",
        ),
        (
            r"\bbiblically\s+accurate\b",
            "Treat the contemporary figurative expression 'biblically accurate' as unusually "
            "faithful to the archetype or to what the category ought to be. Do not introduce "
            "the Bible or religion unless the surrounding topic is actually religious.",
        ),
        (
            r"\bon\s+par\s+with\b",
            "Render 'on par with' as being at the same level or broadly comparable. Do not turn "
            "a neutral equality comparison into praise such as 'excellent' or 'very good'.",
        ),
        (
            r"\bon\s+paper\b|\b(?:economic|financial|commercial)\s+sense\b|"
            r"\bcosts?\s+of\b.{0,100}\bruns?\s+into\b",
            "Use natural Chinese financial and feasibility phrasing. Distinguish 账面上, 理论上, "
            "and financially viable from a literal 纸面上 or 具有经济意义, and make recurring cost "
            "statements predicate-led rather than leaving a noun phrase for the next cue.",
        ),
        (
            r"\b(?:spark|ignite|inspire|awaken)\b.{0,100}"
            r"\b(?:family|children|kids|audience|community|team)\b",
            "For causative emotion or interest, keep the animate participant as the experiencer or "
            "beneficiary. Express that they become interested, excited, or inspired; do not make "
            "them the tool used to create the emotion.",
        ),
        (
            r"\b(?:approach|architecture|design|concept|strategy|technology)\b.{0,100}"
            r"\b(?:product|delivered|delivery|implementation|implemented)\b",
            "When English packages a concept as a product, delivery, approach, or implementation, "
            "state the actual Chinese relation directly, such as productization, deployment, or how "
            "the idea is realized; avoid stacked abstract nouns and literal 作为一种/交付方式 calques.",
        ),
        (
            r"\b(?:home\s+run|ball\s+game|with\s+a\s+vengeance|roaring\s+success|"
            r"enter(?:ed|s|ing)?\s+the\s+chat)\b",
            "Preserve the source's recognizable rhetorical image, escalation, or punchline when it "
            "remains natural in Chinese. Do not reduce a coherent image to generic 成功, 领域, 加入, "
            "or 取得 unless a literal image would genuinely mislead the audience.",
        ),
        (
            r"\btraffic\s+situation\b.{0,100}\bget\s+(?:get\s+)?through\s+it\b",
            "When a vehicle gets through a traffic situation, describe maneuvering or threading "
            "through congestion naturally; do not translate it as literally passing a vehicle "
            "or merely 'getting through it'.",
        ),
        (
            r"\b(?:airport|terminal|concourse|pier|aircraft|runway|foundation|column|"
            r"structural|load|excavation|elevation|varying heights?)\b",
            "Use established civil-aviation and construction Chinese. Distinguish a terminal "
            "concourse, passenger gate, and aircraft stand; express visible structural members "
            "as exposed when that is the physical meaning. In an airport route-network context, "
            "render possible passenger connections as concise transfer combinations or route "
            "connections rather than a literal generic possibility; reserve physical connection "
            "wording for infrastructure. "
            "Follow the actual structural load "
            "path; and use elevation or level terminology instead of a generic height when the "
            "source describes designed vertical levels. Express forces transferred 'down the "
            "height' as travelling downward along the building, and render forces redistributed "
            "back into a structure without the calque '重新分布回'. Translate functional "
            "architectural metaphors such as a spine or backbone with the established domain "
            "term for its actual role, rather than an anatomical body part.",
        ),
        (
            r"\ban\s+exercise\s+in\b",
            "In an abstract or engineering phrase such as 'an exercise in controlling X', "
            "'exercise' means a concentrated undertaking, demonstration, or governing task, "
            "not physical practice; choose the concise Chinese form supported by its object.",
        ),
        (
            r"\b(?:create|creating|created)\s+(?:even\s+)?more\s+land\b.{0,100}\bsea\b|"
            r"\b(?:drill|drilling|dredge|dredging)\b.{0,80}\b(?:land|sea)\b",
            "When a coastal city creates land from the sea, use the established land-reclamation "
            "meaning supported by context; do not preserve a literally impossible drilling verb.",
        ),
        (
            r"\b(?:nuclear|reactor|power plant|turbine|tokamak|fusion|fission|smr)\b",
            "Use established nuclear-engineering Chinese. Distinguish a power-plant site, "
            "physical reactor, generating unit, turbine, and tokamak from context; keep recurring "
            "reactor names and acronyms consistent and do not expand an acronym early when the "
            "following source cue explicitly supplies its full name.",
        ),
        (
            r"\b(?:reading|writing|literate|literacy|post[- ]?literacy)\b",
            "In reading and literacy discussions, distinguish literacy from general culture "
            "or education and keep recurring academic terms consistent.",
        ),
        (
            r"\b(?:email|e-mail|drop us a line|at\s+\w+\s+dot|dot\s+(?:com|org|net))\b",
            "When speech explicitly introduces an email address, normalize an unambiguous "
            "spoken at/dot form as an email rather than a website.",
        ),
        (
            r"\b(?:not|never|isn't|aren't|wasn't|weren't|don't|doesn't|didn't|"
            r"can't|couldn't|won't|wouldn't)\b",
            "Preserve the complete logical scope of negation and comparison across adjacent "
            "fragments; avoid meaning-reversing Chinese double negatives.",
        ),
        (
            r"\b(?:sarcasm|sarcastic|ironically|not actually|yeah,? right)\b",
            "Express clearly supported irony through natural wording without adding editorial "
            "labels or turning it into a sincere statement.",
        ),
        (
            r"\bi\s+don['’]t\s+know\s+if\s+i\s+(?:think|believe)\b",
            "Render nested uncertainty by its actual stance and polarity in idiomatic Chinese; "
            "avoid literal frames such as '我不确定我认为'.",
        ),
        (
            r"\bthanks\s+(?:in\s+no\s+small\s+part\s+)?to\b",
            "Interpret 'thanks to' from the consequence rather than its positive surface form. "
            "For a harmful or unwanted result, use a neutral or adverse cause such as '由于' or "
            "'归咎于', not the complimentary '归功于'.",
        ),
        (
            r"\b\d+(?:st|nd|rd|th)\s+(?:avenue|boulevard|drive|road|street)\b",
            "An ASR token may collapse a building number and an ordinal street name. Restore a "
            "canonical address only when repeated document evidence unambiguously supplies both "
            "parts; otherwise preserve the uncertain source rather than inventing an address.",
        ),
        (
            r"\btalk\s+about\s+(?:a|an|the)\s+(?:case|example|illustration|lesson)\b",
            "Treat 'talk about' as an emphatic example marker when the context supports it, "
            "not automatically as an instruction to discuss something.",
        ),
        (
            r"\bnot\b.{0,80}\bcalling\s+it\b.{0,120}\b(?:named|called)\s+it\b",
            "When a common word is also an official all-caps name and the speaker explicitly "
            "jokes that it is not merely their description but the real name, preserve both "
            "layers: translate the ordinary meaning at the joke and introduce the canonical "
            "name at the naming clause. Terminology must not erase the wordplay.",
        ),
        (
            r"\bsynonymous\s+with\b",
            "Render metalinguistic emotional associations as natural Chinese such as "
            "'让人联想到' or '几乎意味着'; avoid tautologies built from '等同于' and '同义词'.",
        ),
        (
            r"\bbeam(?:ed|ing|s)?\b.*\b(?:content|information|stuff)\b|"
            r"\b(?:content|information|stuff)\b.*\bbeam(?:ed|ing|s)?\b",
            "When digital content is figuratively beamed toward a face or eyes, express the "
            "effect as content being pushed or flooding into view, not as a literal ray.",
        ),
    )
    rules.extend(rule for pattern, rule in conditional_rules if _contains(source, pattern))

    return (
        "\n\n<target_language_style>\n"
        + "\n".join(f"{index}. {rule}" for index, rule in enumerate(rules, 1))
        + "\n</target_language_style>"
    )


def repair_mode_guidance(multispeaker: bool) -> str:
    """Return distinct repair constraints for monologue and dialogue content."""
    shared = (
        "Treat cue-level readability as a hard requirement. Reconstruct the complete local "
        "idea first, then render each key as a concise Chinese reading unit. Do not strand a "
        "subject, predicate, object, modifier, measure word, complement, or connective across "
        "the boundary. Preserve the combined meaning exactly once and prefer rephrasing within "
        "each key over moving content. When a coordinated subject list spans several keys before "
        "one shared predicate, keep each noun under its source key, make the pre-predicate keys "
        "readable nominal units, and begin the predicate key with only the minimum collective "
        "reference needed in Chinese, such as 'these groups'; do not duplicate the predicate or "
        "any fact. Map the complete source clause before repairing individual fragments. A "
        "conjunction, subject, linking predicate, complement, number, and unit must each appear "
        "under exactly one key; never restart the same clause with a second Chinese subject or "
        "connective. Minimal grammatical scaffolding is allowed when a cue would otherwise be "
        "unreadable: a pronoun, demonstrative, classifier, or already established head noun may "
        "be restated only when it adds no new fact. A name, number, distinct action, opinion, or "
        "conclusion is material meaning rather than scaffolding and must still appear exactly "
        "once. Resolve an explicit spoken self-correction to its final value only when the "
        "nearby source proves that value, and resolve contrastive references from the complete "
        "local construction rather than the nearest noun. Avoid repeating the same Chinese head "
        "noun twice inside one cue when one coherent noun phrase conveys the source exactly."
    )
    if not multispeaker:
        return (
            shared
            + " This is a continuous single-speaker passage: preserve the speaker's argument "
            "and register across cues. Use only the minimum non-material grammatical scaffolding "
            "needed for an individual cue to read naturally; never repeat a material subject, "
            "action, or conclusion merely to make an isolated fragment grammatical."
        )
    return (
        shared + " Speaker values are read-only metadata. Preserve every turn, question, answer, "
        "interruption, qualification, tone, and speaker-specific viewpoint. A speaker change "
        "is normally a hard semantic boundary. At a tightly edited handoff that cuts one "
        "grammatical phrase, restate only the minimum shared grammatical frame needed for both "
        "cues to read naturally; never move or duplicate a name, number, fact, opinion, answer, "
        "or conclusion between speakers. Never emit speaker labels."
    )
