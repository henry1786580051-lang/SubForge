"""Repartition cleaned English through conservative, lossless raw-word provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Sequence

from subforge.core.asr.asr_data import ASRDataSeg


@dataclass(frozen=True)
class CueWordMapping:
    # Each display token points to one unchanged raw word. Unmapped deletions
    # remain in the cue's raw words, so no cleanup destroys the ASR evidence.
    display: tuple[str, ...]
    raw_indices: tuple[int, ...]
    deleted_indices: tuple[int, ...]
    error: str = ""


def _key(text: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+(?:['’][a-z]+)?", text.lower())).replace("’", "'")


def map_cleaned_words(cue: ASRDataSeg) -> CueWordMapping:
    display = tuple(cue.text.split())
    raw = cue.words

    def fail(reason: str) -> CueWordMapping:
        return CueWordMapping(display, (), (), reason)

    if not raw or not display:
        return fail("missing words")
    if any(
        not _key(w.text) or w.end_time <= w.start_time or w.timing_source == "estimated"
        for w in raw
    ):
        return fail("unreliable word timing")
    if any(a.end_time > b.start_time for a, b in zip(raw, raw[1:])):
        return fail("overlapping words")
    speakers = {w.speaker_id for w in raw if w.speaker_id}
    if len(speakers) > 1 or (speakers and cue.speaker_id and cue.speaker_id not in speakers):
        return fail("mixed speaker evidence")
    before, after = [_key(w.text) for w in raw], [_key(t) for t in display]
    if any(not token for token in after):
        return fail("unmapped punctuation token")
    indices, deleted = [], []
    for op, a, b, _c, _d in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if op == "equal":
            indices.extend(range(a, b))
        elif op == "delete":
            removed = before[a:b]
            filler = " ".join(removed) in {"uh", "um", "uh um", "you know", "i mean"}
            duplicate = len(removed) == 1 and (
                (a > 0 and before[a - 1] == removed[0])
                or (b < len(before) and before[b] == removed[0])
            )
            if not filler and not duplicate:
                return fail("unsupported deletion")
            deleted.extend(range(a, b))
        else:
            return fail("unsupported source replacement")
    if len(indices) != len(display):
        return fail("incomplete alignment")
    return CueWordMapping(display, tuple(indices), tuple(deleted))


def normalize_mapped_boundaries(
    segments: Sequence[ASRDataSeg],
    *,
    soft_max_words: int = 18,
    hard_max_words: int = 22,
    diagnostics: list[dict] | None = None,
) -> list[ASRDataSeg]:
    """Keep bilingual input intact; callers explicitly stage source-only repair copies."""
    from subforge.core.split.boundary import normalize_boundaries, repeated_identifier_phrases

    if any(c.translated_text for c in segments):
        return list(segments)
    document_phrases = repeated_identifier_phrases(segments)
    result: list[ASRDataSeg] = []
    region: list[tuple[ASRDataSeg, CueWordMapping]] = []

    def speaker(cue: ASRDataSeg) -> str:
        known = {w.speaker_id for w in cue.words if w.speaker_id}
        return cue.speaker_id or (next(iter(known)) if len(known) == 1 else "")

    def flush() -> None:
        if not region:
            return
        originals = [c for c, _ in region]
        raw = [w for c in originals for w in c.words]
        projected, origins, offset = [], {}, 0
        for cue, mapping in region:
            words = []
            for text, index in zip(mapping.display, mapping.raw_indices):
                word = replace(cue.words[index], text=text)
                words.append(word)
                origins[id(word)] = offset + index
            projected.append(
                ASRDataSeg(
                    cue.text,
                    words[0].start_time,
                    words[-1].end_time,
                    speaker_id=speaker(cue),
                    words=words,
                    timing_source=cue.timing_source,
                    language_code=cue.language_code,
                    timestamp_granularity=cue.timestamp_granularity,
                )
            )
            offset += len(cue.words)
        normalized = normalize_boundaries(
            projected,
            soft_max_words=soft_max_words,
            hard_max_words=hard_max_words,
            document_phrases=document_phrases,
        )
        old_words = [w for c in projected for w in c.words]
        new_words = [w for c in normalized for w in c.words]
        if [id(w) for w in new_words] != [id(w) for w in old_words]:
            if diagnostics is not None:
                diagnostics.append(
                    {"start": originals[0].start_time, "reason": "normalizer changed word sequence"}
                )
            result.extend(originals)
            region.clear()
            return
        if [len(c.words) for c in normalized] == [len(c.words) for c in projected]:
            result.extend(originals)
            region.clear()
            return
        starts = [0] + [origins[id(c.words[0])] for c in normalized[1:]] + [len(raw)]
        original_by_words = {tuple(id(w) for w in p.words): c for p, c in zip(projected, originals)}
        rebuilt = []
        for i, cue in enumerate(normalized):
            unchanged = original_by_words.get(tuple(id(w) for w in cue.words))
            words = raw[starts[i] : starts[i + 1]]
            if unchanged is not None and unchanged.words == words:
                rebuilt.append(unchanged)
                continue
            rebuilt.append(
                ASRDataSeg(
                    cue.text,
                    words[0].start_time,
                    words[-1].end_time,
                    speaker_id=cue.speaker_id,
                    words=words,
                    timing_source=cue.timing_source,
                    language_code=cue.language_code,
                    timestamp_granularity=cue.timestamp_granularity,
                )
            )
        if [w for c in rebuilt for w in c.words] != raw:
            raise AssertionError("Mapped boundary repair lost raw word provenance")
        # Deleted fillers can lengthen the reconstructed display interval.
        if any(
            c.end_time - c.start_time > 8000 or c.end_time - c.start_time < 600
            for c in rebuilt
            if c not in originals
        ):
            result.extend(originals)
            if diagnostics is not None:
                diagnostics.append(
                    {"start": originals[0].start_time, "reason": "reconstructed duration limit"}
                )
        else:
            result.extend(rebuilt)
        region.clear()

    for i, cue in enumerate(segments):
        mapping = map_cleaned_words(cue)
        previous = segments[i - 1] if i else None
        if mapping.error or (
            previous
            and (
                speaker(previous) != speaker(cue)
                or cue.start_time - previous.end_time > 1800
                or cue.start_time < previous.end_time
            )
        ):
            flush()
        if mapping.error:
            result.append(cue)
            if diagnostics is not None:
                diagnostics.append({"index": i, "start": cue.start_time, "reason": mapping.error})
        else:
            region.append((cue, mapping))
    flush()
    if diagnostics is not None:
        from subforge.core.split.structural import structural_dependency

        for left, right in zip(result, result[1:]):
            dependency = structural_dependency(left.text, right.text)
            if dependency:
                diagnostics.append(
                    {
                        "start": left.start_time,
                        "reason": "unresolved dependency",
                        "kind": dependency.kind,
                    }
                )
    return result
