"""Source-backed phrase atoms; absence of a match is not proof of completeness."""

from __future__ import annotations

import re

_ATOMS = re.compile(
    r"\b(?:hybrid\s+(?:setup|system|powertrain|technology|version)|"
    r"flat[- ](?:four|six|sixth|eight|twelve)|door\s+pockets?|"
    r"wind\s+deflector|front[- ]end\s+lift|feel\s+like|(?:tire|tyre)\s+technology|"
    r"(?:something|anything|nothing)\s+like|"
    r"grand\s+(?:old\s+)?time|entry\s+point|"
    r"as\s+(?:far|well|soon)\s+as|at\s+the\s+same\s+time)\b",
    re.I,
)
_DESIRE = re.compile(
    r"\b(?:if|when|unless)\s+(?:i|you|we|they|he|she)\s+"
    r"(?:(?:really|also|just|ever)\s+)?(?:want|need|expect|wish)$",
    re.I,
)
_MODAL_ADVERB = re.compile(
    r"\b(?:will|would|can|could|should|may|might|must|shall)"
    r"(?:\s+(?:already|still|just|not|never|[a-z]+ly)){0,3}$",
    re.I,
)
_PREDICATE = re.compile(
    r"^(?:be|have|do|go|come|make|keep|take|get|give|put|find|work|"
    r"start|begin|stop|continue|lift|lower|raise|move|turn|open|close|"
    r"accelerate|brake|drive|change|adjust|allow|help|return|respond|"
    r"feel|look|sound|use|show|tell|need|want|[a-z]+ing)\b",
    re.I,
)


def unit_dependency(left: str, right: str) -> tuple[str, str, str] | None:
    """Describe only evidenced dependencies at this precise source seam."""
    left, right = left.strip(), right.strip()
    if not left or not right or re.search(r"[.!?][\"')\]]*$", left):
        return None
    text = left + " " + right
    for match in _ATOMS.finditer(text):
        if match.start() < len(left) < match.end():
            return "lexical_atom", match.group(), match.group()
    powertrain = re.search(
        r"\b\d+(?:\.\d+)?\s+lit(?:er|re)\s+(?:(?:twin|twin[- ]turbo|turbo|supercharged|naturally[- ]aspirated)\s+){0,3}(?:flat[- ](?:four|six|sixth|eight)|[vViI][- ]?\d+)\b",
        text,
        re.I,
    )
    if powertrain and powertrain.start() < len(left) < powertrain.end():
        return "lexical_atom", powertrain.group(), powertrain.group()
    identifier = re.match(r"^(?:[A-Z]{1,6}[- ]?\d+[A-Za-z]*|\d{3,4}s?)\b", right)
    if identifier and re.search(r"\b(?:base|modern|new|latest|older|classic)$", left, re.I):
        numeric = identifier.group().rstrip("s")
        if not numeric.isdigit() or not 1900 <= int(numeric) <= 2100:
            return "identifier_modifier", left.split()[-1], identifier.group()
    if re.search(r"\bthe\s+(?:most\s+[a-z-]+|[a-z]+est)\s+[^.!?]{1,60}$", left, re.I) and re.match(
        r"^(?:i(?:['’]ve|\s+have)|we(?:['’]ve|\s+have))\s+(?:ever\s+)?(?:seen|driven|used|heard|tested|experienced|had)\b",
        right,
        re.I,
    ):
        return "superlative_relative", left.split()[-1], right.split()[0]
    if _DESIRE.search(left) and re.match(
        r"^(?:this|that|it|them|him|her|us|you|these|those)\s+to\b", right, re.I
    ):
        return "conditional_complement", _DESIRE.search(left).group(), right.split(",")[0]
    if re.search(
        r"\b(?:i|you|we|they|he|she)\s+(?:wish|think|believe|hope)$", left, re.I
    ) and re.match(
        r"^(?:i|you|we|they|he|she|it|this|that)\s+(?:would|could|should|had|have|has|was|were|is|are|can|will)\b",
        right,
        re.I,
    ):
        return "governing_clause", left.split()[-1], right.split()[0]
    predicate = re.sub(
        r"^(?:(?:already|still|just|not|never|[a-z]+ly)\s+){1,3}", "", right, flags=re.I
    )
    if _MODAL_ADVERB.search(left) and _PREDICATE.match(predicate):
        return "modal_adverb_predicate", _MODAL_ADVERB.search(left).group(), right.split()[0]
    if re.search(r"\b(?:and|but)\s+(?:then|so)$", left, re.I) and re.match(
        r"^(?:i|you|we|they|he|she|it|the|this|that)\b", right, re.I
    ):
        return "new_clause_lead", " ".join(left.split()[-2:]), right.split()[0]
    if re.search(
        r"(?:\b(?:is|are|was|were)|['’]s)\s+(?:especially|particularly|very|quite|really|extremely|too)$",
        left,
        re.I,
    ) and re.match(
        r"^(?:good|bad|nice|fast|slow|[a-z]+(?:able|ible|ful|less|ive|ant|ent|ous|al))\b",
        right,
        re.I,
    ):
        return "copular_degree", left.split()[-1], right.split()[0]
    if re.search(r"\b(?:start|begin|continue|stop|keep)$", left, re.I) and re.match(
        r"^[a-z]+ing\b", right, re.I
    ):
        return "aspect_predicate", left.split()[-1], right.split()[0]
    if re.search(
        r"\b(?:fit|stop|lift|lifting|raise|raising|lower|lowering|put|give|giving|take|taking)$",
        left,
        re.I,
    ) and re.match(r"^(?:the|this|that|these|those|my|your|our|its|many|more)\b", right, re.I):
        return "governing_object", left.split()[-1], right.split()[0]
    if re.search(r"\b(?:make|makes|made|making)\s+[^,.;!?]{1,45}$", left, re.I) and re.match(
        r"^(?:(?:any|even|much|slightly|far)\s+)?(?:faster|slower|better|worse)\b", right, re.I
    ):
        return "result_complement", left.split()[-1], right.split()[0]
    if re.match(r"^or\s+(?:something|anything)(?:\s+like\s+that)?[,;]\s+", right, re.I):
        return "trailing_qualification", left.split()[-1], right.split(",")[0]
    if (
        re.search(
            r"\b(?:the|this|that|these|those|a|an)\s+(?:[A-Za-z][A-Za-z-]*\s+){0,3}[A-Za-z][A-Za-z-]*$",
            left,
            re.I,
        )
        and re.match(
            r"^(?:thing|setup|system|design|feature|approach|pattern|rattle|noise|sound|engine|road|pocket|car|vehicle|wheel|brake|seat|technology|button|switch|component|interior|generation|version|model)\b",
            right,
            re.I,
        )
        and not re.match(r"^[a-z-]+\s+(?:is|are|was|were|can|will|has|have)\b", right, re.I)
    ):
        return "nominal_head", left.split()[-1], right.split()[0]
    return None


def is_atomic_seam(left: str, right: str) -> bool:
    return unit_dependency(left, right) is not None


def descriptive_clause_seam(left: str, right: str) -> bool:
    return bool(
        re.search(r"\b(?:thing|setup|system|design|feature|approach)[,]$", left, re.I)
        and re.search(r"\b(?:do|does|is|has|have|uses|use)\b", left, re.I)
        and re.match(r"^where\s+(?:as\s+soon\s+as|when|you|it|the|this)\b", right, re.I)
    )
