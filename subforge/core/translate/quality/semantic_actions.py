"""Pure detectors for high-confidence semantic action translation defects."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class SemanticRepairSignal:
    """One source-anchored semantic defect that is safe to repair selectively."""

    rule_id: str
    hint: str


def detect_semantic_action_mismatch(
    source: str,
    target: str,
    *,
    previous_source: str = "",
    next_source: str = "",
) -> SemanticRepairSignal | None:
    """Detect literal control actions or unresolved references, never style preferences."""
    source_text = str(source or "").strip()
    target_text = str(target or "").strip()
    if not source_text or not target_text:
        return None

    context = " ".join((previous_source, source_text, next_source))
    automotive_checks = (
        (
            bool(re.search(r"\blimited\s+(?:trim|Ioniq)\b", source_text, re.I)
                 and not re.search(r"\blimited[- ]edition\b", source_text, re.I)
                 and "限量" in target_text),
            "trim_name",
            "Limited is a trim identifier here, not limited production. Preserve the trim name; do not add rarity.",
        ),
        (
            bool(re.search(r"\bnot placeful\b", source_text, re.I)
                 and re.search(r"(?:不够|不太|不|没那么)宽敞", target_text)),
            "vehicle_placement",
            "The source's unusual 'not placeful' is ambiguous ASR. Check its vehicle-position meaning using supplied context; do not assert poor cabin space without evidence.",
        ),
        (
            bool(re.search(r"\bget[- ]out[- ]of[- ]the[- ]way[- ]able\b", source_text, re.I)
                 and re.search(r"让(?:别人|其他车|别的车).{0,4}让路", target_text)),
            "movement_agent",
            "The vehicle can get out of the way quickly. Do not reverse this into forcing other road users to yield.",
        ),
        (
            bool(re.search(r"\bsaves? you from\b.*\bpeople\b", source_text, re.I)
                 and re.search(r"(?:救的就是|救的是|拯救|救了)(?:那些|这些|那群)", target_text)),
            "protection_roles",
            "You are protected from the other people's behavior; those people are not the beneficiaries being rescued.",
        ),
        (
            bool(re.search(r"(?:^|[.!])\s*No leak(?:age)?s?[.!]?\s*$", source_text, re.I)
                 and re.search(r"(?:看看|是否|漏不漏|会不会漏)", target_text)
                 and not re.search(r"(?:没(?:有)?|未)漏", target_text)),
            "observed_no_leak",
            "No leakages states an observed negative result. Preserve no leakage, rather than changing it into a question or a future inspection.",
        ),
        (
            bool(re.search(r"\bmiles of range\b", previous_source, re.I)
                 and re.search(r"\bhad\s+\d+\s+on it still\b", source_text, re.I)
                 and not re.search(r"\b(?:fuel|gas|petrol|diesel|tank)\b", source_text, re.I)
                 and re.search(r"(?:剩|还).{0,8}(?:油|汽油|燃油)", target_text)),
            "remaining_range",
            "The adjacent source establishes remaining driving range in miles. Resolve the omitted unit as range, without adding fuel or a tank to this cue.",
        ),
        (
            bool(re.search(r"\blevel\s*2\s+charg", context, re.I)
                 and re.search(r"\b(?:level\s*2|level\s*$)", source_text, re.I)
                 and not re.search(r"\blevel\s*3\b", source_text, re.I)
                 and re.search(r"(?:三级|Level\s*3)", target_text, re.I)),
            "charging_level",
            "The source establishes Level 2, not Level 3. Correct the invented level; respect each cue's ownership if the label spans two source keys.",
        ),
    )
    for matches, name, hint in automotive_checks:
        if matches:
            return SemanticRepairSignal("translation.semantic.automotive." + name, hint)

    if (
        re.search(r"\b(?:parking|hand)\s*brake\b", source_text, re.IGNORECASE)
        and re.search(r"\b(?:down|off|released|disengaged)\b", source_text, re.IGNORECASE)
        and re.search(r"(?:放下|降下|往下|向下).{0,5}(?:手刹|驻车制动)|"
                      r"(?:手刹|驻车制动).{0,5}(?:放下|降下|往下|向下)", target_text)
    ):
        return SemanticRepairSignal(
            rule_id="translation.semantic.control.parking_brake_state",
            hint=(
                "The named parking brake is in its released/disengaged state. Translate the "
                "mechanical state naturally; do not render 'down' as the spatial action 放下 or "
                "降下. Preserve only the current source meaning."
            ),
        )

    mode_match = re.search(
        r"\bput\b.{0,30}\bback\s+into\s+([A-Z][A-Za-z0-9-]*)[.!?]?\s*$",
        source_text,
    )
    mode_context = " ".join((previous_source, next_source))
    if (
        mode_match
        and re.search(
            r"\b(?:drive|driving|gear|gearbox|manual|mode|sport|touring|transmission)\b",
            mode_context,
            re.IGNORECASE,
        )
        and (
            re.search(r"(?:样子|风格|气质|该有的状态)", target_text)
            or not re.search(r"(?:模式|挡|档)", target_text)
        )
    ):
        return SemanticRepairSignal(
            rule_id="translation.semantic.control.named_mode",
            hint=(
                f"Nearby source explicitly establishes a vehicle control or drive-mode operation. "
                f"In current_source, '{mode_match.group(1)}' is the selected named mode, not a "
                "general vehicle quality or appearance. Express the mode-selection action "
                "naturally without importing any neighboring clause."
            ),
        )

    additive_reference = bool(
        re.search(r"\b(?:it|this|that)\b.{0,24}\balso\b|\balso\b.{0,24}\b(?:it|this|that)\b",
                  source_text, re.IGNORECASE)
    )
    vague_additive_target = bool(
        re.search(r"(?:上|下|里|内|中)?(?:也有|也在|也是|也一样|也如此)[\s，。！？,.!?]*$", target_text)
    )
    if additive_reference and previous_source.strip() and vague_additive_target:
        return SemanticRepairSignal(
            rule_id="translation.semantic.reference.additive_object",
            hint=(
                "The current source uses an additive pronoun whose antecedent is explicitly "
                "established in previous_source. Resolve only the concise head noun or property "
                "needed to make this cue intelligible. If more than one antecedent is plausible, "
                "keep the wording conservative; never copy a neighboring action or fact."
            ),
        )

    take_down = re.search(
        r"\b(?:take|takes|taking|took)\s+down\b",
        source_text,
        re.IGNORECASE,
    )
    natural_event = re.compile(
        r"\b(?:bad\s+weather|earthquake|flood(?:ing)?|hurricane|landslide|storm|"
        r"tornado|typhoon|wildfire|wind)\b",
        re.IGNORECASE,
    )
    # A coordinated predicate is commonly split after its subject: "bad weather struck" /
    # "and took down ...". Only consult the immediately preceding cue when the current
    # cue owns the destructive action, so unrelated document context cannot trigger repair.
    non_agentive_cause = bool(
        take_down
        and (
            natural_event.search(source_text)
            or (
                re.match(r"^(?:and|then)\b", source_text, re.IGNORECASE)
                and natural_event.search(str(previous_source or ""))
            )
        )
    )
    intentional_attack_target = re.search(r"(?:击落|击毁|打下|打倒|击败)", target_text)
    if non_agentive_cause and intentional_attack_target:
        return SemanticRepairSignal(
            rule_id="translation.semantic.causation.non_agentive_take_down",
            hint=(
                "The grammatical subject is a natural event, not an intentional attacker. "
                "Render 'take down' as the event causing damage, collapse, loss, or a crash "
                "supported by current_source and local context; do not use an intentional "
                "attack verb such as 击落, 击毁, or 击败. Preserve only the current source fact."
            ),
        )

    return None


def detect_document_shortened_place(
    source: str,
    document_sources: Iterable[str],
) -> tuple[str, str] | None:
    """Resolve a shortened numbered road only from unique repeated document evidence."""
    source_text = str(source or "").strip()
    if not re.search(
        r"\b(?:back|return|head|heading|go|going|get|take|drive|driving)\b.{0,24}\bto\b",
        source_text,
        re.IGNORECASE,
    ):
        return None
    shortened = re.search(r"\b(\d{1,3})\s+miles\b(?!\s+per\b)", source_text, re.IGNORECASE)
    if not shortened:
        return None

    number = shortened.group(1)
    canonical_pattern = re.compile(
        rf"\b{re.escape(number)}\s+Mile\s+"
        r"(?:Drive|Road|Highway|Route|Trail|Boulevard|Avenue|Parkway)\b",
        re.IGNORECASE,
    )
    canonical_names = {
        match.group(0)
        for text in document_sources
        for match in canonical_pattern.finditer(str(text or ""))
    }
    if len({name.casefold() for name in canonical_names}) != 1:
        return None
    canonical = sorted(canonical_names, key=lambda value: (value.casefold(), value))[0]
    normalized = source_text[: shortened.start()] + canonical + source_text[shortened.end() :]
    return canonical, normalized
