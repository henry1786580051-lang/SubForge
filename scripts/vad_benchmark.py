#!/usr/bin/env python3
"""VAD Parameter Benchmark — iterative optimization across multiple videos.

Usage:
    python scripts/vad_benchmark.py                           # Run all benchmark videos
    python scripts/vad_benchmark.py --video /path/to/video    # Test single video
    python scripts/vad_benchmark.py --grid                    # Full grid search
    python scripts/vad_benchmark.py --report                  # Show historical results
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_FILE = Path(__file__).parent / "vad_benchmark_results.jsonl"


@dataclass
class VADParams:
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 300
    speech_pad_ms: int = 300


@dataclass
class BenchmarkResult:
    video_name: str
    video_duration_s: float
    params: VADParams
    vad_segments: int
    coverage_pct: float
    gaps_gt1s: int
    gaps_gt3s: int
    max_gap_s: float
    vad_time_s: float
    asr_segments: int = 0
    asr_time_s: float = 0.0
    hallucination_score: float = 0.0  # 0=none, 1=heavy
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def detect_speech_segments(audio_path: str, params: VADParams) -> Tuple[List[Tuple[int, int]], float]:
    """Run Silero VAD and return segments + time taken."""
    import numpy as np
    import torch
    from pydub import AudioSegment

    # Load model once
    if not hasattr(detect_speech_segments, "_model"):
        print("  Loading Silero VAD model...")
        detect_speech_segments._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
    model = detect_speech_segments._model

    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    audio_len_ms = len(audio)

    t0 = time.time()
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

    # Merge nearby segments
    if segments:
        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end < params.min_silence_ms:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
    else:
        merged = []

    # Apply padding
    padded = []
    for start, end in merged:
        p_start = max(0, start - params.speech_pad_ms)
        p_end = min(audio_len_ms, end + params.speech_pad_ms)
        padded.append((int(p_start), int(p_end)))

    vad_time = time.time() - t0
    return padded, vad_time


def evaluate_srt_quality(srt_path: str) -> dict:
    """Evaluate SRT output quality metrics."""
    import re

    with open(srt_path) as f:
        content = f.read()

    times = re.findall(
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})", content
    )
    if not times:
        return {"segments": 0, "coverage_pct": 0, "gaps_gt1s": 0, "gaps_gt3s": 0, "max_gap_s": 0}

    segments = []
    for t in times:
        start = int(t[0]) * 3600000 + int(t[1]) * 60000 + int(t[2]) * 1000 + int(t[3])
        end = int(t[4]) * 3600000 + int(t[5]) * 60000 + int(t[6]) * 1000 + int(t[7])
        segments.append((start, end))

    total_duration = segments[-1][1] - segments[0][0]
    total_speech = sum(e - s for s, e in segments)
    gaps = [segments[i][0] - segments[i - 1][1] for i in range(1, len(segments))]

    return {
        "segments": len(segments),
        "coverage_pct": total_speech / total_duration * 100 if total_duration > 0 else 0,
        "gaps_gt1s": sum(1 for g in gaps if g > 1000),
        "gaps_gt3s": sum(1 for g in gaps if g > 3000),
        "max_gap_s": max(gaps) / 1000 if gaps else 0,
    }


def run_benchmark(video_path: str, params: VADParams, run_asr: bool = False) -> BenchmarkResult:
    """Run full benchmark on a single video with given parameters."""
    import subprocess

    video_name = Path(video_path).stem
    print(f"\n{'='*60}")
    print(f"Video: {video_name}")
    print(f"Params: threshold={params.threshold}, pad={params.speech_pad_ms}, "
          f"silence={params.min_silence_ms}, min_speech={params.min_speech_ms}")

    # Extract audio
    audio_path = f"/tmp/vad_bench_{os.getpid()}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", audio_path],
        capture_output=True, check=True,
    )

    from pydub import AudioSegment
    audio = AudioSegment.from_file(audio_path)
    video_duration = len(audio) / 1000

    # VAD detection
    segments, vad_time = detect_speech_segments(audio_path, params)
    total_speech = sum(e - s for s, e in segments) / 1000
    coverage = total_speech / video_duration * 100

    gaps = [segments[i][0] - segments[i - 1][1] for i in range(1, len(segments))]
    gaps_gt1s = sum(1 for g in gaps if g > 1000)
    gaps_gt3s = sum(1 for g in gaps if g > 3000)
    max_gap = max(gaps) / 1000 if gaps else 0

    print(f"  VAD: {len(segments)} segments, {coverage:.1f}% coverage, "
          f"{gaps_gt1s} gaps >1s, max={max_gap:.1f}s, {vad_time:.1f}s")

    result = BenchmarkResult(
        video_name=video_name,
        video_duration_s=video_duration,
        params=params,
        vad_segments=len(segments),
        coverage_pct=round(coverage, 1),
        gaps_gt1s=gaps_gt1s,
        gaps_gt3s=gaps_gt3s,
        max_gap_s=round(max_gap, 1),
        vad_time_s=round(vad_time, 1),
    )

    # Optional: run actual ASR transcription
    if run_asr:
        from subforge.core.asr.whisper_cpp import WhisperCppASR
        import tempfile

        print("  Running ASR on VAD segments...")
        all_asr_segs = []
        t_start = time.time()

        with tempfile.TemporaryDirectory() as tmp:
            for i, (s, e) in enumerate(segments):
                chunk_path = str(Path(tmp) / f"s{i}.wav")
                audio[s:e].export(chunk_path, format="wav")
                asr = WhisperCppASR(chunk_path, language="en", whisper_model="large",
                                    use_vad=False, use_cache=False)
                r = asr.run()
                for seg in r.segments:
                    seg.start_time += s
                    seg.end_time += s
                all_asr_segs.extend(r.segments)

        asr_time = time.time() - t_start
        result.asr_segments = len(all_asr_segs)
        result.asr_time_s = round(asr_time, 1)
        print(f"  ASR: {len(all_asr_segs)} segments, {asr_time:.1f}s")

    # Clean up
    Path(audio_path).unlink(missing_ok=True)

    return result


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
            results.append(BenchmarkResult(**d))
    return results


def print_report():
    """Print summary of all benchmark results."""
    results = load_results()
    if not results:
        print("No benchmark results found.")
        return

    # Group by video
    videos = {}
    for r in results:
        videos.setdefault(r.video_name, []).append(r)

    print(f"\n{'='*80}")
    print("VAD Benchmark Report")
    print(f"{'='*80}")
    print(f"Total results: {len(results)} across {len(videos)} videos\n")

    for video_name, video_results in videos.items():
        print(f"\n--- {video_name} ({video_results[0].video_duration_s:.0f}s) ---")
        print(f"{'Params':<40} {'Cov%':>6} {'Gaps>1s':>8} {'Gaps>3s':>8} {'MaxGap':>8}")
        print("-" * 75)
        for r in sorted(video_results, key=lambda x: x.coverage_pct):
            p = r.params
            param_str = f"t={p.threshold} pad={p.speech_pad_ms} sil={p.min_silence_ms} min={p.min_speech_ms}"
            print(f"{param_str:<40} {r.coverage_pct:>5.1f}% {r.gaps_gt1s:>7} {r.gaps_gt3s:>7} {r.max_gap_s:>7.1f}s")

    # Find best overall parameters
    print(f"\n{'='*80}")
    print("Best parameters per video (lowest coverage with gaps > 0):")
    for video_name, video_results in videos.items():
        # Filter to results that have gaps (i.e., silence detected)
        with_gaps = [r for r in video_results if r.gaps_gt1s > 0]
        if with_gaps:
            best = min(with_gaps, key=lambda r: r.coverage_pct)
            p = best.params
            print(f"  {video_name}: t={p.threshold} pad={p.speech_pad_ms} "
                  f"sil={p.min_silence_ms} min={p.min_speech_ms} "
                  f"-> {best.coverage_pct}% coverage, {best.gaps_gt1s} gaps")


def grid_search(video_paths: List[str]):
    """Run grid search over parameter space."""
    param_grid = [
        VADParams(threshold=t, speech_pad_ms=p, min_silence_ms=s, min_speech_ms=m)
        for t in [0.4, 0.5, 0.6]
        for p in [200, 300, 400]
        for s in [200, 300, 400]
        for m in [250, 500]
    ]

    print(f"Grid search: {len(param_grid)} parameter combinations x {len(video_paths)} videos")
    print(f"Total runs: {len(param_grid) * len(video_paths)}")

    for video_path in video_paths:
        for params in param_grid:
            result = run_benchmark(video_path, params, run_asr=False)
            save_result(result)

    print_report()


def main():
    parser = argparse.ArgumentParser(description="VAD Parameter Benchmark")
    parser.add_argument("--video", type=str, help="Single video to test")
    parser.add_argument("--grid", action="store_true", help="Run grid search")
    parser.add_argument("--report", action="store_true", help="Show historical results")
    parser.add_argument("--asr", action="store_true", help="Also run ASR (slow)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pad", type=int, default=300)
    parser.add_argument("--silence", type=int, default=300)
    parser.add_argument("--min-speech", type=int, default=250)
    args = parser.parse_args()

    if args.report:
        print_report()
        return

    if args.video:
        params = VADParams(
            threshold=args.threshold,
            speech_pad_ms=args.pad,
            min_silence_ms=args.silence,
            min_speech_ms=args.min_speech,
        )
        result = run_benchmark(args.video, params, run_asr=args.asr)
        save_result(result)
        print_report()
        return

    if args.grid:
        # Default benchmark videos
        default_videos = [
            "/Users/guwenhan/Desktop/YouTube/2026 Lexus ES 350h Hybrid Premium FWD - POV First Driving Impressions.mp4",
        ]
        videos = [v for v in default_videos if Path(v).exists()]
        if not videos:
            print("No benchmark videos found. Use --video to specify.")
            return
        grid_search(videos)
        return

    # Default: run current params on all benchmark videos
    default_videos = [
        "/Users/guwenhan/Desktop/YouTube/2026 Lexus ES 350h Hybrid Premium FWD - POV First Driving Impressions.mp4",
    ]
    videos = [v for v in default_videos if Path(v).exists()]
    if not videos:
        print("No benchmark videos found. Use --video to specify.")
        return

    params = VADParams()
    for video_path in videos:
        result = run_benchmark(video_path, params, run_asr=args.asr)
        save_result(result)

    print_report()


if __name__ == "__main__":
    main()
