"""Audit the exported word timeline and conservatively replace damaged neighborhoods."""

from __future__ import annotations

import copy
import json
import logging
import re
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path

from .asr_data import ASRData, reasonable_word_duration_ms
from .speech_gap_repair import corroborates, word_key

logger = logging.getLogger(__name__)
MAX_WINDOWS = 3
MAX_DECODE_SECONDS = 180
CONTEXT_MS = 10_000


def _words(data: ASRData) -> list[dict]:
    # Segment times are the final export contract, not possibly stale nested words.
    return [
        {"word": s.text, "start": s.start_time / 1000, "end": s.end_time / 1000}
        for s in data.segments
    ]


def _audit(data: ASRData, samples):
    from .whisperx_asr import (
        _critical_aligned_speech_gaps,
        _detect_speech_in_mlx_gaps,
        _find_uncovered_mlx_gaps,
        _gap_is_covered_by_foreign_range,
        _SpeechBackedGap,
    )

    words = _words(data)
    # Inspect a shadow timeline: a stretched lexical word is not evidence that
    # its entire interval was transcribed. Never truncate the user's data here.
    # Number compounds and non-Latin tokens need different duration models.
    for word in words:
        token = word["word"].strip().strip(".,;:!?()[]{}")
        if not re.fullmatch(r"[A-Za-z]+(?:['’][A-Za-z]+)?", token):
            continue
        allowance = max(900, reasonable_word_duration_ms(token)) / 1000
        if word["end"] - word["start"] > max(2.4, allowance * 2):
            word["end"] = word["start"] + allowance
    ranges = _find_uncovered_mlx_gaps(
        [{"text": w["word"], "start": w["start"], "end": w["end"]} for w in words],
        len(samples) / 16000,
        min_gap_seconds=0.8,
    )
    speech = _detect_speech_in_mlx_gaps(samples, 16000, [(a, b) for a, b in ranges if b - a >= 6])
    speech += _detect_speech_in_mlx_gaps(
        samples, 16000, [(a, b) for a, b in ranges if b - a < 6], threshold=0.75
    )
    # Subtract confirmed non-speech from VAD evidence, not entire uncovered gaps:
    # another real omitted utterance in the same gap must still be detected.
    for excluded in data.confirmed_non_speech_ranges:
        lo, hi = round(excluded["start"] * 1000), round(excluded["end"] * 1000)
        speech = [
            piece
            for a, b in speech
            for piece in ((a, min(b, lo)), (max(a, hi), b))
            if piece[1] > piece[0]
        ]
    gaps = _critical_aligned_speech_gaps(
        {"segments": [{"words": words}]}, speech, len(samples) / 16000
    )
    # Candidate discovery is deliberately weaker than automatic replacement.
    # The old helper has a hidden 2.5-second floor and a 75% short-gap gate;
    # ordinary pauses within a missing sentence made both Vietnam examples pass.
    # Strict VAD nominates these regions; two anchored decodes still must agree.
    for start, end in ranges:
        if not (0.05 < start and end < len(samples) / 16000 - 0.05 and end - start < 6):
            continue
        voiced = sum(max(0, min(b / 1000, end) - max(a / 1000, start)) for a, b in speech)
        if voiced >= 2.0 and voiced / (end - start) >= 0.5:
            if not any(g.start < end and g.end > start for g in gaps):
                gaps.append(_SpeechBackedGap(start, end, voiced, voiced / (end - start), True))
    return sorted(
        [g for g in gaps if not _gap_is_covered_by_foreign_range(g, data.excluded_speech_ranges)],
        key=lambda g: g.start,
    )


def _quantity_evidence(words):
    """Compare equivalent number/unit spellings without rewriting source text.

    A number and its unit are one timed acoustic group, even when one decoder
    emits `1,541km` and the other emits `1541 kilometer`. Values remain exact.
    """
    units = {"km", "kilometer", "kilometers", "kilometre", "kilometres"}
    result = []
    index = 0
    while index < len(words):
        word = words[index]
        text = str(word["word"]).strip().rstrip(".;:!?").casefold()
        match = re.fullmatch(r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)([a-z]*)", text)
        if match:
            number, unit = match.groups()
            end = word["end"]
            if not unit and index + 1 < len(words):
                following = words[index + 1]
                next_unit = str(following["word"]).strip().rstrip(".,;:!?").casefold()
                if next_unit in units and 0 <= following["start"] - end <= 1.0:
                    unit, end = next_unit, following["end"]
                    index += 1
            if unit in units:
                word = {**word, "word": number.replace(",", "") + "km", "end": end}
        result.append(word)
        index += 1
    return result


def _corroborates_repair(first, second):
    def numbers(words):
        text = " ".join(str(w["word"]) for w in words)
        values = re.findall(r"[+-]?\d+(?:[,.]\d+)*", text)
        return [
            value.replace(",", "")
            if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", value)
            else value
            for value in values
        ]

    # Generic word keys discard punctuation; decimals and signs are semantic.
    if numbers(first) != numbers(second):
        return False
    return corroborates(first, second) or corroborates(
        _quantity_evidence(first), _quantity_evidence(second)
    )


def _checkpoint(config):
    event = getattr(config, "cancel_event", None)
    if event is not None and event.is_set():
        raise RuntimeError("Transcription cancelled")


def _matching_anchors(original: ASRData, decoded: ASRData):
    """Find unique three-word anchors outside the suspect neighborhood.

    Keep a two-second margin around the suspect interval; choose the nearest
    common safe anchors so unrelated context variations cannot block a repair.
    """
    old, new = _words(original), _words(decoded)
    old_keys, new_keys = [word_key(w) for w in old], [word_key(w) for w in new]
    matches = []
    for i in range(len(old) - 2):
        key = old_keys[i : i + 3]
        if not all(key):
            continue
        hits = [
            j
            for j in range(len(new) - 2)
            if new_keys[j : j + 3] == key
            and abs(new[j]["start"] - old[i]["start"]) <= 1.5
            and abs(new[j + 2]["end"] - old[i + 2]["end"]) <= 1.5
        ]
        if len(hits) == 1:
            matches.append((i, hits[0]))
    return matches


def _anchors(original, decoded, start, end, allowed=None):
    old, new = _words(original), _words(decoded)
    matches = _matching_anchors(original, decoded)
    if allowed is not None:
        matches = [m for m in matches if m[0] in allowed]
    left = [
        m
        for m in matches
        if old[m[0] + 2]["end"] <= start - 2
        and m[1] + 3 < len(new)
        and new[m[1] + 3]["start"] >= old[m[0] + 2]["end"]
    ]
    right = [
        m
        for m in matches
        if old[m[0]]["start"] >= end + 2 and m[1] > 0 and new[m[1] - 1]["end"] <= old[m[0]]["start"]
    ]
    if not left or not right:
        return None
    left_match, right_match = left[-1], right[0]
    if left_match[0] + 3 > right_match[0] or left_match[1] + 3 > right_match[1]:
        return None
    return left_match[0] + 3, right_match[0], left_match[1] + 3, right_match[1]


def _proposal(original, decoded, start, end, allowed=None):
    match = _anchors(original, decoded, start, end, allowed)
    if match is None:
        return None
    lo, hi, dl, dr = match
    replacement = decoded.segments[dl:dr]
    if len(replacement) < 3:
        return None
    # Preserve outside anchors exactly; never clamp a mismatched splice to fit.
    if replacement[0].start_time < original.segments[lo - 1].end_time:
        return None
    if replacement[-1].end_time > original.segments[hi].start_time:
        return None
    return lo, hi, replacement


def _decode(context, left, right, config, factory, callback):
    _checkpoint(config)
    with tempfile.TemporaryDirectory(prefix="subforge-gap-") as temp:
        path = Path(temp) / "context.wav"
        context.audio_segment()[left:right].export(str(path), format="wav").close()
        decoded = factory(str(path), config).run(callback=lambda _p, _m: _checkpoint(config))
    _checkpoint(config)
    if decoded.coverage_issues or not decoded.is_word_timestamp():
        raise ValueError("Local recognition still needs review")
    decoded.clip_to_media_duration(right - left)
    # Audit after transformations: native overlong words cannot hide the hole.
    decoded.cap_abnormal_word_durations()
    decoded.fix_boundary_overlaps()
    for seg in decoded.segments:
        seg.start_time += left
        seg.end_time += left
        for word in seg.words:
            word.start_time += left
            word.end_time += left
    return decoded


def _save_evidence(records):
    """Retain at most twenty small reports; never retain audio or credentials."""
    from subforge.config import VERSION
    from subforge.core.utils.logger import get_active_log_file

    directory = get_active_log_file().parent / "speech-coverage"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps(
            {"schema": 1, "app_version": VERSION, "records": records}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    for old in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[20:]:
        old.unlink(missing_ok=True)
    logger.info("Final speech coverage evidence: %s", path)


def repair_final_coverage(data, audio_path, config, callback, factory, analysis_context=None):
    """Bounded local recovery, with unresolved regions propagated to existing UI."""
    if not data.is_word_timestamp():
        return data
    from .audio_analysis import AudioAnalysisContext

    _checkpoint(config)
    records = []
    try:
        # Always check original audio, even if earlier ASR used enhanced audio.
        context = analysis_context
        if context is None or context.audio_path != audio_path:
            context = AudioAnalysisContext(audio_path)
        samples = context.samples()
        gaps = _audit(data, samples)
    except Exception:
        _checkpoint(config)
        logger.warning("Final speech coverage audit unavailable", exc_info=True)
        duration = (getattr(data, "media_duration_ms", None) or 0) / 1000
        data.coverage_issues.append(
            {"start": 0, "end": duration, "reason": "final_audit_unavailable"}
        )
        return data
    if not gaps:
        return data
    duration_ms = round(len(samples) / 16)
    decoded_seconds = 0
    attempts = 0
    current = gaps
    for gap in gaps:
        _checkpoint(config)
        # Previous accepted repair may have covered more than one original gap.
        if not any(g.start < gap.end and g.end > gap.start for g in current):
            continue
        left = max(0, round(gap.start * 1000) - CONTEXT_MS)
        right = min(duration_ms, round(gap.end * 1000) + CONTEXT_MS)
        confirm_left, confirm_right = max(0, left - 2000), min(duration_ms, right + 2000)
        if (confirm_left, confirm_right) == (left, right):
            # A whole-file window cannot be confirmed by decoding identical input.
            confirm_left = min(left + 1000, max(left, round(gap.start * 1000) - 3000))
            if confirm_left == left:
                continue
        cost = (right - left + confirm_right - confirm_left) / 1000
        record = {
            "gap": asdict(gap),
            "window": [left / 1000, right / 1000],
            "model": str(getattr(config, "whisperx_model", "")),
            "status": "budget_exhausted",
        }
        records.append(record)
        if attempts >= MAX_WINDOWS or decoded_seconds + cost > MAX_DECODE_SECONDS:
            continue
        attempts += 1
        decoded_seconds += cost
        callback(95, "Recovering missing speech with surrounding context...")
        original = ASRData(
            copy.deepcopy(
                [s for s in data.segments if s.start_time >= left and s.end_time <= right]
            ),
            granularity="word",
        )
        record["before"] = _words(original)
        try:
            first = _decode(context, left, right, config, factory, callback)
            record["decoded"] = _words(first)
            proposal = _proposal(original, first, gap.start, gap.end)
            if proposal is None:
                raise ValueError("No safe surrounding anchors")
            second = _decode(context, confirm_left, confirm_right, config, factory, callback)
            record["confirmation"] = _words(second)
            # Choose the same anchors from the intersection, not independently
            # selected matches whose availability changes with clip boundaries.
            common = {i for i, _ in _matching_anchors(original, first)} & {
                i for i, _ in _matching_anchors(original, second)
            }
            proposal = _proposal(original, first, gap.start, gap.end, common)
            confirmed = _proposal(original, second, gap.start, gap.end, common)
            if proposal is None or confirmed is None or proposal[:2] != confirmed[:2]:
                raise ValueError("Context decodes disagree on replacement boundaries")
            lo, hi, replacement = proposal
            if any(
                seg.start_time / 1000 < span["end"] and seg.end_time / 1000 > span["start"]
                for seg in replacement
                for span in data.confirmed_non_speech_ranges
            ):
                raise ValueError("Recovery conflicts with confirmed non-speech evidence")
            if not _corroborates_repair(
                _words(ASRData(replacement)), _words(ASRData(confirmed[2]))
            ):
                raise ValueError("Context decodes disagree on recovered words")
            begin, finish = original.segments[lo].start_time, original.segments[hi].start_time
            candidate = copy.deepcopy(data)
            candidate.segments = [
                s for s in candidate.segments if not begin <= s.start_time < finish
            ]
            candidate.segments.extend(copy.deepcopy(replacement))
            candidate.segments.sort(key=lambda s: s.start_time)
            after = _audit(candidate, samples)
            if any(g.start < gap.end and g.end > gap.start for g in after):
                raise ValueError("Recovered timeline still has missing speech")
            # Reject any newly introduced hole, including either splice edge.
            if any(
                not any(
                    g.start >= prev.start - 0.05 and g.end <= prev.end + 0.05 for prev in current
                )
                for g in after
            ):
                raise ValueError("Recovery introduced a new speech gap")
            data.segments = candidate.segments
            current = after
            data.coverage_issues = [
                i
                for i in data.coverage_issues
                if not (begin / 1000 <= i["start"] and i["end"] <= finish / 1000)
            ]
            record.update(
                status="recovered",
                replacement=[begin / 1000, finish / 1000],
                after=_words(ASRData(replacement)),
            )
        except Exception as exc:
            _checkpoint(config)
            record.update(status="needs_review", error=str(exc))
            logger.warning("Final speech recovery %.3f-%.3fs: %s", gap.start, gap.end, exc)
    # No timeline mutation may follow this check (speaker labels are safe).
    try:
        remaining = _audit(data, samples)
    except Exception:
        _checkpoint(config)
        logger.warning("Final recovery verification unavailable", exc_info=True)
        data.coverage_issues.append(
            {"start": 0, "end": duration_ms / 1000, "reason": "final_audit_unavailable"}
        )
        remaining = current
    for gap in remaining:
        if not any(i["start"] <= gap.start and i["end"] >= gap.end for i in data.coverage_issues):
            data.coverage_issues.append({**asdict(gap), "reason": "final_export_gap"})
    try:
        _save_evidence(records)
    except Exception:
        logger.warning("Unable to save speech coverage evidence", exc_info=True)
    return data


def record_alignment_evidence(native, aligned, gaps, model):
    """Keep bounded before/after evidence for suspicious alignment windows."""
    from .whisperx_asr import _alignment_word_coverage

    before = _alignment_word_coverage(native)
    after = _alignment_word_coverage(aligned)
    windows = [(g.start, g.end) for g in gaps]
    windows.extend((w["start"], w["end"]) for w in after if w["end"] - w["start"] >= 3)
    if not windows:
        return
    records = []
    for start, end in windows[:3]:
        left, right = max(0, start - 10), min(end + 10, start + 50)
        records.append(
            {
                "stage": "alignment_before_native_fallback",
                "model": model,
                "window": [left, right],
                "native": [w for w in before if left <= w["start"] < right][:300],
                "aligned": [w for w in after if left <= w["start"] < right][:300],
            }
        )
    try:
        _save_evidence(records)
    except Exception:
        logger.warning("Unable to retain suspicious alignment evidence", exc_info=True)
