"""Atomic repair of existing bilingual cues; no file writes or provider calls here."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Sequence

from subforge.core.asr.asr_data import ASRDataSeg
from subforge.core.split.mapped_boundary import normalize_mapped_boundaries


def repair_bilingual_boundaries(
    cues: Sequence[ASRDataSeg],
    translate: Callable[[list[ASRDataSeg]], list[str]],
    validate: Callable[[list[ASRDataSeg], list[str]], bool],
    *,
    diagnostics: list[dict],
    max_repair_windows: int = 32,
) -> list[ASRDataSeg]:
    """Commit only validated, disjoint windows; failed windows retain original objects.

    Callbacks receive detached inputs. The caller owns cancellation/provider policy
    and semantic validation. Original SRT ids may change after a successful merge.
    """
    staged = deepcopy(list(cues))
    for cue in staged:
        cue.translated_text = ""
    planned = normalize_mapped_boundaries(staged, diagnostics=diagnostics)
    if not all(c.words for c in staged):
        # Cues without raw evidence cannot define a reliable global partition.
        diagnostics.append({"reason": "incomplete historical word evidence"})
        return list(cues)
    old_ends, new_ends = {}, {}
    total = 0
    for i, cue in enumerate(staged):
        total += len(cue.words)
        old_ends[total] = i + 1
    new_total = 0
    for i, cue in enumerate(planned):
        new_total += len(cue.words)
        new_ends[new_total] = i + 1
    old_words = [w for c in staged for w in c.words]
    new_words = [w for c in planned for w in c.words]
    if old_words != new_words or total != new_total:
        diagnostics.append({"reason": "raw word preservation failure"})
        return list(cues)
    result, a, b, attempted = [], 0, 0, 0
    for end in sorted(set(old_ends) & set(new_ends)):
        ai, bi = old_ends[end], new_ends[end]
        before, proposal = list(cues[a:ai]), planned[b:bi]
        a, b = ai, bi

        def signature(group):
            return [(c.text, c.start_time, c.end_time) for c in group]

        if signature(before) == signature(proposal):
            result.extend(before)
            continue
        if len(before) > 3 or len({c.speaker_id for c in before}) != 1:
            diagnostics.append(
                {"start": before[0].start_time, "reason": "window exceeds safe repair scope"}
            )
            result.extend(before)
            continue
        if attempted >= max(0, max_repair_windows):
            diagnostics.append(
                {"start": before[0].start_time, "reason": "repair window budget exhausted"}
            )
            result.extend(before)
            continue
        attempted += 1
        try:
            translations = translate(deepcopy(proposal))
            if len(translations) != len(proposal) or any(
                not isinstance(t, str) or not t.strip() for t in translations
            ):
                raise ValueError("incomplete translations")
            if not validate(deepcopy(proposal), list(translations)):
                raise ValueError("semantic validation rejected repair")
            for cue, target in zip(proposal, translations):
                cue.translated_text = target.strip()
            result.extend(proposal)
            diagnostics.append(
                {
                    "start": before[0].start_time,
                    "reason": "accepted",
                    "old_count": len(before),
                    "new_count": len(proposal),
                }
            )
        except Exception as error:
            # Cancellation is never converted into a successful partial repair.
            if (
                isinstance(error, (InterruptedError, KeyboardInterrupt))
                or "cancel" in type(error).__name__.lower()
            ):
                raise
            diagnostics.append(
                {
                    "start": before[0].start_time,
                    "reason": "rolled back",
                    "error_type": type(error).__name__,
                }
            )
            result.extend(before)
    return result
