#!/usr/bin/env python3
"""VAD Parameter Benchmark — iterative optimization with content integrity checks.

Metrics:
    - Content Integrity Score (CIS): recall relative to threshold=0.2 baseline
    - Gap Density: gaps per minute (cross-video comparable)
    - Segment Length Distribution: P5/P50/P95 percentiles
    - Coverage: health reference only, NOT an optimization target

Usage:
    uv run python scripts/vad_benchmark.py --video /path/to/video
    uv run python scripts/vad_benchmark.py --grid
    uv run python scripts/vad_benchmark.py --report
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_FILE = Path(__file__).parent / "vad_benchmark_results.jsonl"

# Content type presets for gap density thresholds
CONTENT_PRESETS = {
    "dialogue":  {"gap_density_range": (3, 8),   "coverage_range": (60, 80)},
    "lecture":   {"gap_density_range": (0.5, 3), "coverage_range": (80, 95)},
    "general":   {"gap_density_range": (1, 5),   "coverage_range": (70, 85)},
}


@dataclass
class VADParams:
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 300
    speech_pad_ms: int = 300


@dataclass
class SegmentStats:
    count: int = 0
    p5_ms: float = 0
    p50_ms: float = 0
    p95_ms: float = 0
    mean_ms: float = 0


@dataclass
class BenchmarkResult:
    video_name: str
    video_duration_s: float
    params: VADParams
    # Health metrics (constraint layer)
    coverage_pct: float
    gap_density: float          # gaps per minute
    max_gap_s: float
    segment_stats: SegmentStats
    content_integrity: float    # recall vs threshold=0.2 baseline
    # Raw counts (for debugging)
    vad_segments: int
    gaps_gt1s: int
    vad_time_s: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def _run_vad(audio_path: str, params: VADParams) -> List[Tuple[int, int]]:
    """Core VAD logic, shared by current and baseline runs."""
    import numpy as np
    import torch
    from pydub import AudioSegment

    if not hasattr(_run_vad, "_model"):
        print("  Loading Silero VAD model...")
        _run_vad._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
    model = _run_vad._model

    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    audio_len_ms = len(audio)

    window_size = 512
    speech_probs = []
    model.reset_states()

    for i in range(0, len(samples), window_size):
        chunk = samples[i:i + window_size]
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))
        prob = model(torch.from_numpy(chunk), 16000).item()
        speech_probs.append(prob)

    frame_ms = window_size / 16000 * 1000
    segments = []
    in_speech = False
    seg_start = 0.0

    for i, prob in enumerate(speech_probs):
        if prob >= params.threshold and not in_speech:
            in_speech = True
            seg_start = i * frame_ms
        elif prob < params.threshold and in_speech:
            in_speech = False
            seg_end = i * frame_ms
            if seg_end - seg_start >= params.min_speech_ms:
                segments.append((seg_start, seg_end))

    if in_speech:
        seg_end = len(speech_probs) * frame_ms
        if seg_end - seg_start >= params.min_speech_ms:
            segments.append((seg_start, seg_end))

    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < params.min_silence_ms:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    padded = []
    for start, end in merged:
        p_start = max(0, start - params.speech_pad_ms)
        p_end = min(audio_len_ms, end + params.speech_pad_ms)
        padded.append((int(p_start), int(p_end)))

    return padded


def compute_segment_stats(segments: List[Tuple[int, int]]) -> SegmentStats:
    """Compute segment length distribution."""
    if not segments:
        return SegmentStats()

    durations = sorted(e - s for s, e in segments)
    n = len(durations)

    def percentile(data, p):
        k = (len(data) - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        return data[f] * (c - k) + data[c] * (k - f)

    return SegmentStats(
        count=n,
        p5_ms=round(percentile(durations, 5)),
        p50_ms=round(percentile(durations, 50)),
        p95_ms=round(percentile(durations, 95)),
        mean_ms=round(statistics.mean(durations)),
    )


def run_benchmark(video_path: str, params: VADParams, content_type: str = "general") -> BenchmarkResult:
    """Run benchmark with content integrity check."""
    video_name = Path(video_path).stem
    audio_path = f"/tmp/vad_bench_{os.getpid()}.wav"

    print(f"\n{'='*70}")
    print(f"Video: {video_name}")
    print(f"Params: t={params.threshold} pad={params.speech_pad_ms} "
          f"sil={params.min_silence_ms} min={params.min_speech_ms}")

    # Extract audio
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", audio_path],
        capture_output=True, check=True,
    )

    from pydub import AudioSegment
    audio_len_s = len(AudioSegment.from_file(audio_path)) / 1000
    audio_len_min = audio_len_s / 60

    # --- Content Integrity Score ---
    # Compare current params against threshold=0.2 (near-total recall baseline)
    print("  Computing content integrity score...")
    baseline_params = VADParams(threshold=0.2, min_speech_ms=100, min_silence_ms=100, speech_pad_ms=200)
    baseline_segs = _run_vad(audio_path, baseline_params)
    baseline_duration = sum(e - s for s, e in baseline_segs) / 1000

    # --- Current params ---
    t0 = time.time()
    current_segs = _run_vad(audio_path, params)
    vad_time = time.time() - t0
    current_duration = sum(e - s for s, e in current_segs) / 1000

    content_integrity = current_duration / baseline_duration if baseline_duration > 0 else 0

    # --- Gap analysis ---
    gaps = [current_segs[i][0] - current_segs[i - 1][1]
            for i in range(1, len(current_segs))]
    gaps_gt1s = sum(1 for g in gaps if g > 1000)
    max_gap = max(gaps) / 1000 if gaps else 0
    gap_density = gaps_gt1s / audio_len_min if audio_len_min > 0 else 0

    # --- Coverage ---
    coverage = current_duration / audio_len_s * 100 if audio_len_s > 0 else 0

    # --- Segment stats ---
    seg_stats = compute_segment_stats(current_segs)

    print(f"  Segments: {len(current_segs)}, Coverage: {coverage:.1f}%")
    print(f"  Content Integrity: {content_integrity:.3f} (baseline {baseline_duration:.0f}s vs current {current_duration:.0f}s)")
    print(f"  Gap Density: {gap_density:.1f}/min ({gaps_gt1s} gaps in {audio_len_min:.1f}min), Max Gap: {max_gap:.1f}s")
    print(f"  Segment Length: P5={seg_stats.p5_ms}ms P50={seg_stats.p50_ms}ms P95={seg_stats.p95_ms}ms")
    print(f"  VAD time: {vad_time:.1f}s")

    # --- Health check warnings ---
    preset = CONTENT_PRESETS.get(content_type, CONTENT_PRESETS["general"])
    warnings = []

    if content_integrity < 0.85:
        warnings.append(f"CRITICAL: Content integrity {content_integrity:.3f} < 0.85 — losing >15% speech content!")
    elif content_integrity < 0.90:
        warnings.append(f"WARNING: Content integrity {content_integrity:.3f} < 0.90 — monitor closely")

    gd_min, gd_max = preset["gap_density_range"]
    if gap_density < gd_min:
        warnings.append(f"LOW gap density {gap_density:.1f}/min < {gd_min} — VAD may be under-detecting silence")
    elif gap_density > gd_max:
        warnings.append(f"HIGH gap density {gap_density:.1f}/min > {gd_max} — VAD may be over-splitting")

    cov_min, cov_max = preset["coverage_range"]
    if coverage < cov_min:
        warnings.append(f"LOW coverage {coverage:.1f}% < {cov_min}% — check for content loss")
    elif coverage > cov_max:
        warnings.append(f"HIGH coverage {coverage:.1f}% > {cov_max}% — silence may not be detected")

    if seg_stats.p5_ms < 500:
        warnings.append(f"Many very short segments (P5={seg_stats.p5_ms}ms) — possible fragmentation")

    if warnings:
        print(f"\n  WARNINGS:")
        for w in warnings:
            print(f"    {w}")

    Path(audio_path).unlink(missing_ok=True)

    return BenchmarkResult(
        video_name=video_name,
        video_duration_s=round(audio_len_s, 1),
        params=params,
        coverage_pct=round(coverage, 1),
        gap_density=round(gap_density, 2),
        max_gap_s=round(max_gap, 1),
        segment_stats=seg_stats,
        content_integrity=round(content_integrity, 3),
        vad_segments=len(current_segs),
        gaps_gt1s=gaps_gt1s,
        vad_time_s=round(vad_time, 1),
    )


def save_result(result: BenchmarkResult):
    """Append result to benchmark log."""
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def load_results() -> List[BenchmarkResult]:
    """Load all historical results."""
    if not RESULTS_FILE.exists():
        return []
    results = []
    for line in RESULTS_FILE.read_text().strip().split("\n"):
        if line:
            d = json.loads(line)
            d["params"] = VADParams(**d["params"])
            d["segment_stats"] = SegmentStats(**d.get("segment_stats", {}))
            results.append(BenchmarkResult(**d))
    return results


def print_report():
    """Print summary with content integrity as primary metric."""
    results = load_results()
    if not results:
        print("No benchmark results found.")
        return

    videos: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        videos.setdefault(r.video_name, []).append(r)

    print(f"\n{'='*90}")
    print("VAD Benchmark Report")
    print(f"{'='*90}")
    print(f"Total: {len(results)} runs across {len(videos)} videos\n")

    for video_name, video_results in videos.items():
        dur = video_results[0].video_duration_s
        print(f"\n--- {video_name} ({dur:.0f}s / {dur/60:.1f}min) ---")
        header = f"{'Params':<38} {'CIS':>6} {'Cov%':>6} {'GapD':>6} {'P5':>6} {'P50':>6} {'P95':>6} {'Status'}"
        print(header)
        print("-" * len(header) + "-" * 15)

        for r in sorted(video_results, key=lambda x: x.content_integrity, reverse=True):
            p = r.params
            param_str = f"t={p.threshold} pad={p.speech_pad_ms} sil={p.min_silence_ms} min={p.min_speech_ms}"
            s = r.segment_stats

            # Status indicator
            if r.content_integrity < 0.85:
                status = "FAIL"
            elif r.content_integrity < 0.90:
                status = "WARN"
            else:
                status = "OK"

            print(f"{param_str:<38} {r.content_integrity:>5.3f} {r.coverage_pct:>5.1f}% "
                  f"{r.gap_density:>5.1f} {s.p5_ms:>5}ms {s.p50_ms:>5}ms {s.p95_ms:>5}ms  {status}")

    # Best params per video
    print(f"\n{'='*90}")
    print("Recommended parameters (CIS > 0.90, highest gap density):")
    for video_name, video_results in videos.items():
        safe = [r for r in video_results if r.content_integrity >= 0.90]
        if safe:
            best = max(safe, key=lambda r: r.gap_density)
            p = best.params
            print(f"  {video_name}:")
            print(f"    t={p.threshold} pad={p.speech_pad_ms} sil={p.min_silence_ms} min={p.min_speech_ms}")
            print(f"    CIS={best.content_integrity:.3f} Cov={best.coverage_pct}% GapD={best.gap_density}/min")
        else:
            print(f"  {video_name}: NO SAFE PARAMETERS (all CIS < 0.90)")


def sensitivity_analysis(video_paths: List[str]):
    """Analyze parameter sensitivity across videos."""
    print(f"\n{'='*70}")
    print("Parameter Sensitivity Analysis")
    print(f"{'='*70}")

    base = VADParams()
    variations = {
        "threshold":    [0.3, 0.4, 0.5, 0.6, 0.7],
        "min_silence_ms": [100, 200, 300, 400, 500],
        "speech_pad_ms":  [100, 200, 300, 400, 500],
        "min_speech_ms":  [100, 250, 500, 1000],
    }

    for param_name, values in variations.items():
        print(f"\n  Sensitivity: {param_name}")
        print(f"  {'Value':<10} {'Avg CIS':>10} {'Avg Cov%':>10} {'Avg GapD':>10}")
        print(f"  {'-'*45}")

        for val in values:
            params = VADParams(**{param_name: val})
            cis_list, cov_list, gd_list = [], [], []

            for vp in video_paths:
                r = run_benchmark(vp, params)
                cis_list.append(r.content_integrity)
                cov_list.append(r.coverage_pct)
                gd_list.append(r.gap_density)

            avg_cis = statistics.mean(cis_list)
            avg_cov = statistics.mean(cov_list)
            avg_gd = statistics.mean(gd_list)
            marker = " <-- current" if getattr(base, param_name) == val else ""
            print(f"  {val:<10} {avg_cis:>9.3f} {avg_cov:>9.1f}% {avg_gd:>9.1f}{marker}")


def grid_search(video_paths: List[str]):
    """Grid search with content integrity guard."""
    param_grid = [
        VADParams(threshold=t, speech_pad_ms=p, min_silence_ms=s, min_speech_ms=m)
        for t in [0.3, 0.4, 0.5, 0.6]
        for p in [200, 300, 400]
        for s in [200, 300, 400, 500]
        for m in [250, 500]
    ]

    total = len(param_grid) * len(video_paths)
    print(f"Grid search: {len(param_grid)} param combos x {len(video_paths)} videos = {total} runs\n")

    for video_path in video_paths:
        for i, params in enumerate(param_grid):
            print(f"[{i+1}/{len(param_grid)}]", end="")
            result = run_benchmark(video_path, params)
            save_result(result)

    print_report()


def main():
    parser = argparse.ArgumentParser(description="VAD Parameter Benchmark")
    parser.add_argument("--video", type=str, help="Single video to test")
    parser.add_argument("--grid", action="store_true", help="Run grid search")
    parser.add_argument("--sensitivity", action="store_true", help="Parameter sensitivity analysis")
    parser.add_argument("--report", action="store_true", help="Show historical results")
    parser.add_argument("--content-type", choices=["dialogue", "lecture", "general"], default="general")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pad", type=int, default=300)
    parser.add_argument("--silence", type=int, default=300)
    parser.add_argument("--min-speech", type=int, default=250)
    args = parser.parse_args()

    default_videos = [
        "/Users/guwenhan/Desktop/YouTube/2026 Lexus ES 350h Hybrid Premium FWD - POV First Driving Impressions.mp4",
    ]

    if args.report:
        print_report()
        return

    if args.video:
        params = VADParams(
            threshold=args.threshold, speech_pad_ms=args.pad,
            min_silence_ms=args.silence, min_speech_ms=args.min_speech,
        )
        result = run_benchmark(args.video, params, content_type=args.content_type)
        save_result(result)
        print_report()
        return

    if args.sensitivity:
        videos = [v for v in default_videos if Path(v).exists()]
        if not videos:
            print("No benchmark videos found.")
            return
        sensitivity_analysis(videos)
        return

    if args.grid:
        videos = [v for v in default_videos if Path(v).exists()]
        if not videos:
            print("No benchmark videos found.")
            return
        grid_search(videos)
        return

    # Default: run current params on all videos
    videos = [v for v in default_videos if Path(v).exists()]
    if not videos:
        print("No benchmark videos found. Use --video to specify.")
        return

    params = VADParams()
    for vp in videos:
        result = run_benchmark(vp, params, content_type=args.content_type)
        save_result(result)

    print_report()


if __name__ == "__main__":
    main()
