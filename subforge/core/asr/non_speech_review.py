"""Conservative acoustic review of isolated short ASR outputs in non-speech."""

from __future__ import annotations

import logging
import math
import re
import tempfile
from pathlib import Path

from .final_coverage import _checkpoint, _save_evidence

logger = logging.getLogger(__name__)
MAX_REVIEWS = 3
MAX_WINDOW_MS = 30_000


def _overlap(start, end, ranges):
    intervals = sorted((max(start, a), min(end, b)) for a, b in ranges if a < end and b > start)
    total, last = 0, start
    for a, b in intervals:
        total += max(0, b - max(a, last))
        last = max(last, b)
    return total


def _candidates(data, duration_ms):
    groups = []
    for segment in data.segments:
        if not groups or segment.start_time - groups[-1][-1].end_time > 1200:
            groups.append([])
        groups[-1].append(segment)
    for index, group in enumerate(groups):
        start, end = group[0].start_time, group[-1].end_time
        previous = groups[index - 1][-1].end_time if index else 0
        following = groups[index + 1][0].start_time if index + 1 < len(groups) else duration_ms
        units = re.findall(r"\w+", " ".join(s.text for s in group), re.UNICODE)
        if 1 <= len(units) <= 2 and start - previous >= 10000 and following - end >= 10000:
            yield group


def _acoustic_evidence(context, start, end):
    from . import silero_vad, ten_vad

    if not ten_vad.is_available() or not silero_vad.is_available():
        raise RuntimeError("Independent speech detectors unavailable")
    left, right = max(0, start - 10000), min(len(context.samples()) // 16, end + 10000)
    samples = context.samples()[left * 16 : right * 16]
    results = {}
    for name, backend, threshold in (
        ("regular", ten_vad, 0.5),
        ("strict", ten_vad, 0.75),
        ("independent", silero_vad, 0.35),
    ):
        ranges = backend.run_vad_inference(
            samples,
            sample_rate=16000,
            threshold=threshold,
            min_speech_ms={"regular": 160, "strict": 120, "independent": 100}[name],
            min_silence_ms=200 if name == "strict" else 180,
            speech_pad_ms=0,
        )
        ranges = [(a + left, b + left) for a, b in ranges]
        results[name] = {
            "word_overlap_ms": _overlap(start, end, ranges),
            "context_overlap_ms": _overlap(max(left, start - 5000), min(right, end + 5000), ranges),
        }
    return results


def _probe(context, group, config, factory):
    """Decode unmodified context; independently align the disputed words.

    Repeated decoded text is deliberately NOT evidence that a word was spoken.
    Missing native probability or alignment evidence prevents automatic deletion.
    """
    from .whisperx_asr import _prepare_mlx_model_path, install_whisperx_runtime_stubs

    _checkpoint(config)
    start, end = group[0].start_time, group[-1].end_time
    duration = len(context.samples()) // 16
    left = max(0, (start + end) // 2 - MAX_WINDOW_MS // 2)
    right = min(duration, left + MAX_WINDOW_MS)
    with tempfile.TemporaryDirectory(prefix="subforge-nonspeech-") as temp:
        path = Path(temp) / "context.wav"
        context.audio_segment()[left:right].export(str(path), format="wav").close()
        asr = factory(str(path), config)
        if not getattr(asr, "uses_mlx", False):
            raise RuntimeError("Native non-speech evidence unavailable for this backend")
        model = _prepare_mlx_model_path(asr.mlx_model, Path(temp))
        native = asr._transcribe_mlx_in_worker(str(path), model, lambda *a: _checkpoint(config))
        _checkpoint(config)
        # The local input is at most one 30s decoder window. Score applies to
        # that context; it is not presented as a per-word acoustic probability.
        probabilities = [
            float(s["no_speech_prob"])
            for s in native.get("segments", [])
            if isinstance(s.get("no_speech_prob"), (int, float))
            and math.isfinite(s["no_speech_prob"])
        ]
        install_whisperx_runtime_stubs()
        import whisperx.alignment as alignment

        align_left, align_right = max(0, start - 500), min(duration, end + 500)
        text = " ".join(s.text for s in group)
        aligned = asr._align_result(
            {"segments": [{"text": text, "start": 0, "end": (align_right - align_left) / 1000}]},
            context.samples()[align_left * 16 : align_right * 16],
            group[0].language_code or asr.language or "en",
            lambda *a: _checkpoint(config),
            alignment,
        )
        words = aligned.get("word_segments") or [
            w for s in aligned.get("segments", []) for w in s.get("words", [])
        ]
        scores = [
            float(w["score"])
            for w in words
            if isinstance(w.get("score"), (int, float)) and math.isfinite(w["score"])
        ]
        expected = len(re.findall(r"\w+", text))
        return {
            "decoded_text": str(native.get("text", ""))[:1000],
            "no_speech_prob": min(probabilities) if probabilities else None,
            "alignment_scores": scores,
            "alignment_complete": len(scores) >= expected,
            "window": [left / 1000, right / 1000],
        }


def _decision(acoustic, probe):
    # Keep genuine short replies supported by either stronger acoustic check.
    if any(acoustic[k]["word_overlap_ms"] >= 80 for k in ("strict", "independent")):
        return "keep"
    quiet_context = all(acoustic[k]["context_overlap_ms"] == 0 for k in ("strict", "independent"))
    scores = probe.get("alignment_scores", [])
    probability = probe.get("no_speech_prob")
    weak_alignment = (
        probe.get("alignment_complete")
        and scores
        and sum(scores) / len(scores) < 0.35
        and max(scores) < 0.6
    )
    if quiet_context and weak_alignment and probability is not None and probability >= 0.6:
        return "remove"
    return "review"


def review_non_speech(data, audio_path, config, callback, factory, analysis_context=None):
    if not data.is_word_timestamp():
        return data
    from .audio_analysis import AudioAnalysisContext

    duration = data.media_duration_ms or max((s.end_time for s in data.segments), default=0)
    groups = list(_candidates(data, duration))
    if not groups:
        return data
    records, removed = [], set()
    context = analysis_context
    attempts = 0
    for group in groups:
        _checkpoint(config)
        start, end = group[0].start_time, group[-1].end_time
        record = {
            "stage": "isolated_short_speech_review",
            "start": start / 1000,
            "end": end / 1000,
            "text": " ".join(s.text for s in group),
            "status": "review",
        }
        records.append(record)
        try:
            if context is None or context.audio_path != audio_path:
                context = AudioAnalysisContext(audio_path)
            acoustic = _acoustic_evidence(context, start, end)
            record["acoustic"] = acoustic
            if _decision(acoustic, {}) == "keep":
                record["status"] = "keep"
                continue
            if attempts >= MAX_REVIEWS:
                raise RuntimeError("Local non-speech review budget exhausted")
            attempts += 1
            callback(93, "Checking an isolated subtitle against original audio...")
            probe = _probe(context, group, config, factory)
            _checkpoint(config)
            record["probe"] = probe
            record["status"] = _decision(acoustic, probe)
            if record["status"] == "remove":
                removed.update(id(s) for s in group)
                # Exclude only the independently checked neighborhood, not an
                # entire music passage that may contain resumed dialogue.
                data.confirmed_non_speech_ranges.append(
                    {"start": max(0, start - 5000) / 1000, "end": (end + 5000) / 1000}
                )
                logger.warning(
                    "Removed acoustically rejected short subtitle %.3f-%.3f: %s",
                    start / 1000,
                    end / 1000,
                    record["text"],
                )
        except Exception as exc:
            _checkpoint(config)
            record["error"] = str(exc)
            logger.warning("Isolated subtitle needs review: %s", exc)
        if record["status"] == "review":
            data.coverage_issues.append(
                {"start": start / 1000, "end": end / 1000, "reason": "suspected_non_speech_text"}
            )
    data.segments = [s for s in data.segments if id(s) not in removed]
    try:
        _save_evidence(records)
    except Exception:
        logger.warning("Unable to save non-speech review evidence", exc_info=True)
    return data
