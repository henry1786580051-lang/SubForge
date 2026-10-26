"""Review decoder loops against unmodified audio before exporting word timelines."""

from __future__ import annotations

import copy
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .asr_data import ASRData
from .final_coverage import _checkpoint, _corroborates_repair, _save_evidence, _words
from .speech_gap_repair import keys, word_key

logger = logging.getLogger(__name__)
MAX_REVIEWS = 3
MAX_DECODE_SECONDS = 180
MAX_LOOP_MS = 45_000


@dataclass(frozen=True)
class Repetition:
    first: int
    stop: int
    phrase: tuple[str, ...]
    repeats: int


def candidates(data: ASRData) -> list[Repetition]:
    """Nominate multiword loops, including loops split into atomic word cues.

    Text alone never authorizes removal. Short emphatic repeats and stutters do
    not spend the acoustic review budget. No particular phrase is blacklisted.
    """
    if not data.is_word_timestamp():
        return []
    segments = data.segments
    tokens = [word_key({"text": s.text}) for s in segments]
    found = []
    index = 0
    while index < len(tokens):
        best = None
        for width in range(1, 13):
            phrase = tokens[index : index + width]
            if len(phrase) < width or not all(phrase):
                break
            if any(
                not 0 <= segments[j].start_time - segments[j - 1].end_time <= 10_000
                for j in range(index + 1, index + width)
            ):
                break
            stop = index + width
            while (
                tokens[stop : stop + width] == phrase
                and all(
                    0 <= segments[j].start_time - segments[j - 1].end_time <= 10_000
                    for j in range(max(index + 1, stop), stop + width)
                )
            ):
                stop += width
            repeats = (stop - index) // width
            if (
                repeats >= (8 if width == 1 else 4)
                and stop - index >= 12
                and segments[stop - 1].end_time - segments[index].start_time >= 5000
                and (best is None or stop > best.stop)
            ):
                best = Repetition(index, stop, tuple(phrase), repeats)
        if best is None:
            index += 1
        else:
            found.append(best)
            index = best.stop
    return found


def _speech_ranges(context, start, end):
    from . import silero_vad, ten_vad

    if not ten_vad.is_available() or not silero_vad.is_available():
        raise RuntimeError("Independent speech detectors unavailable")
    left, right = max(0, start - 1000), min(len(context.samples()) // 16, end + 1000)
    samples = context.samples()[left * 16 : right * 16]
    results = {}
    for name, backend, threshold in (
        ("strict", ten_vad, 0.75),
        ("independent", silero_vad, 0.35),
    ):
        results[name] = [
            (a + left, b + left)
            for a, b in backend.run_vad_inference(
                samples,
                sample_rate=16000,
                threshold=threshold,
                min_speech_ms=100,
                min_silence_ms=180,
                speech_pad_ms=0,
            )
        ]
    return results


def _decode(context, left, right, config, factory):
    """Decode original audio in an isolated worker; never denoise or reuse ASR cache."""
    _checkpoint(config)
    with tempfile.TemporaryDirectory(prefix="subforge-repetition-") as temp:
        path = Path(temp) / "original.wav"
        context.audio_segment()[left:right].export(str(path), format="wav").close()
        asr = factory(str(path), config)
        asr.use_cache = False
        if getattr(asr, "uses_mlx", False):
            from .whisperx_asr import _prepare_mlx_model_path

            model = _prepare_mlx_model_path(asr.mlx_model, Path(temp))
            result = asr._transcribe_mlx_in_worker(str(path), model, lambda *a: _checkpoint(config))
            if result.get("coverage_issues"):
                raise ValueError("Original-audio recognition still needs review")
            decoded = ASRData(asr._make_segments(result))
        else:
            decoded = asr.run(callback=lambda *a: _checkpoint(config))
    _checkpoint(config)
    if decoded.coverage_issues or (decoded.segments and not decoded.is_word_timestamp()):
        raise ValueError("Original-audio recognition did not return a verified word timeline")
    decoded.clip_to_media_duration(right - left)
    decoded.fix_boundary_overlaps()
    for segment in decoded.segments:
        segment.start_time += left
        segment.end_time += left
        for word in segment.words:
            word.start_time += left
            word.end_time += left
    return decoded


def _uncovered(start, end, covered):
    pieces = [(start, end)]
    for left, right in sorted(covered):
        pieces = [
            (a, b)
            for lo, hi in pieces
            for a, b in ((lo, min(hi, left)), (max(lo, right), hi))
            if b > a
        ]
    return pieces


def _validate_replacement(old, first, second, speech, neighbors):
    if first.segments or second.segments:
        if not _corroborates_repair(_words(first), _words(second)):
            raise ValueError("Original-audio context decodes disagree")
    old_keys, new_keys = keys(_words(old)), keys(_words(first))
    if old_keys != new_keys and len(new_keys) > len(old_keys) // 2:
        raise ValueError("Context decodes did not resolve the repetition loop")

    # A speech detector cannot tell which words were spoken. Conversely, two
    # identical empty decodes cannot authorize deleting a quieter real voice.
    # Every detected voice must remain covered, and every retained word group
    # needs acoustic support. The short margin tolerates VAD onset/offset drift.
    voiced = [r for ranges in speech.values() for r in ranges]
    groups = []
    for segment in first.segments:
        if not groups or segment.start_time - groups[-1][-1].end_time > 900:
            groups.append([])
        groups[-1].append(segment)
    for group in groups:
        left, right = group[0].start_time - 350, group[-1].end_time + 350
        if not any(min(right, b) - max(left, a) >= 80 for a, b in voiced):
            raise ValueError("Replacement words have no independent speech support")
    covered = [(s.start_time - 350, s.end_time + 350) for s in [*first.segments, *neighbors]]
    start, end = old.segments[0].start_time, old.segments[-1].end_time
    for a, b in voiced:
        left, right = max(start, a), min(end, b)
        if right > left and sum(y - x for x, y in _uncovered(left, right, covered)) >= 100:
            raise ValueError("Replacement would discard detected speech")
    return old_keys != new_keys


def review_repetitions(data, audio_path, config, callback, factory, analysis_context=None):
    """Repair corroborated loops; retain and flag anything acoustically uncertain."""
    from .audio_analysis import AudioAnalysisContext

    suspects = candidates(data)
    if not suspects:
        return data
    records, decoded_seconds = [], 0
    original = list(data.segments)
    context = analysis_context
    for index, suspect in enumerate(suspects):
        _checkpoint(config)
        group = original[suspect.first : suspect.stop]
        start, end = group[0].start_time, group[-1].end_time
        record = {
            "stage": "repetition_review",
            "start": start / 1000,
            "end": end / 1000,
            "phrase": list(suspect.phrase),
            "repeats": suspect.repeats,
            "status": "needs_review",
        }
        records.append(record)
        try:
            if index >= MAX_REVIEWS or end - start > MAX_LOOP_MS:
                raise ValueError("Repetition review budget exhausted")
            if context is None or context.audio_path != audio_path:
                context = AudioAnalysisContext(audio_path)
            duration = len(context.samples()) // 16
            # Millisecond differences in forced alignment should not shift the
            # decoder's entire audio window. Use stable whole-second boundaries
            # while keeping the complete suspect interval in both contexts.
            anchor_start, anchor_end = ((value + 999) // 1000 * 1000 for value in (start, end))
            windows = [
                (max(0, anchor_start - pad), min(duration, anchor_end + pad))
                for pad in (6000, 10000)
            ]
            cost = sum(right - left for left, right in windows) / 1000
            if windows[0] == windows[1] or decoded_seconds + cost > MAX_DECODE_SECONDS:
                raise ValueError("Independent repetition review windows unavailable")
            decoded_seconds += cost
            speech = _speech_ranges(context, start, end)
            _checkpoint(config)
            callback(93, "Checking repeated subtitles against original audio...")
            decodes = []
            for left, right in windows:
                decoded = _decode(context, left, right, config, factory)
                _checkpoint(config)
                if any(
                    s.start_time < start < s.end_time or s.start_time < end < s.end_time
                    for s in decoded.segments
                ):
                    raise ValueError("A decoded word crosses the repetition repair boundary")
                # Do not rewrite neighboring phrases merely because a local
                # decoder chose a different pronoun or punctuation there.
                decoded.segments = [
                    s for s in decoded.segments if s.start_time >= start and s.end_time <= end
                ]
                decodes.append(decoded)
            record.update(
                windows=[[a / 1000, b / 1000] for a, b in windows],
                speech=speech,
                decodes=[_words(d) for d in decodes],
            )
            old = ASRData(group)
            neighbors = original[max(0, suspect.first - 2) : suspect.first] + original[
                suspect.stop : suspect.stop + 2
            ]
            changed = _validate_replacement(old, *decodes, speech, neighbors)
            if changed:
                removed_ids = {id(s) for s in group}
                data.segments = [s for s in data.segments if id(s) not in removed_ids]
                replacement = copy.deepcopy(decodes[0].segments)
                data.segments.extend(replacement)
                data.segments.sort(key=lambda s: s.start_time)
                # Protect only non-speech holes. Resumed speech and genuine short
                # replies must remain eligible for the final coverage repair.
                protected = [(a - 500, b + 500) for ranges in speech.values() for a, b in ranges]
                protected += [(s.start_time - 500, s.end_time + 500) for s in [*replacement, *neighbors]]
                data.confirmed_non_speech_ranges.extend(
                    {"start": a / 1000, "end": b / 1000}
                    for a, b in _uncovered(start, end, protected)
                )
                logger.warning("Repaired ASR repetition %.3f-%.3fs (%d repeats)", start / 1000, end / 1000, suspect.repeats)
            record["status"] = "repaired" if changed else "confirmed_speech"
        except Exception as exc:
            _checkpoint(config)
            record["error"] = str(exc)
            data.coverage_issues.append(
                {"start": start / 1000, "end": end / 1000, "reason": "suspected_repetition"}
            )
            logger.warning("ASR repetition needs review %.3f-%.3fs: %s", start / 1000, end / 1000, exc)
    try:
        _save_evidence(records)
    except Exception:
        logger.warning("Unable to save repetition review evidence", exc_info=True)
    return data
