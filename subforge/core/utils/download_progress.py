"""Byte-based download estimates; callers supply monotonic time and real counters."""

from __future__ import annotations

import math
from collections import deque


class DownloadEstimator:
    def __init__(self):
        self.samples: deque[tuple[float, int]] = deque()
        self.last_change: float | None = None
        self.last_bytes = 0
        self.last_downloaded = 0
        self.rate = 0.0
        self.last_time: float | None = None

    def update(self, downloaded: int, total: int | None, transferred: int, now: float) -> dict:
        downloaded, transferred = max(0, downloaded), max(0, transferred)
        total = total if total is not None and total > 0 else None
        # Resume/cache bytes count towards completion, never towards throughput.
        reset = bool(
            self.samples
            and (transferred < self.samples[-1][1] or downloaded < self.last_downloaded)
        )
        self.last_downloaded = downloaded
        if reset:
            self.samples.clear()
            self.rate = 0.0
            self.last_change = now
        if self.last_change is None or transferred != self.last_bytes:
            # A recovered connection needs fresh samples, not a stale fast ETA.
            if self.last_change is not None and now - self.last_change >= 10:
                self.samples.clear()
                self.rate = 0.0
            self.last_change = now
        self.last_bytes = transferred
        self.samples.append((now, transferred))
        while len(self.samples) > 2 and now - self.samples[1][0] > 15:
            self.samples.popleft()
        span = now - self.samples[0][0]
        raw = (transferred - self.samples[0][1]) / span if span > 0 else 0
        dt = max(0, now - self.last_time) if self.last_time is not None else 0
        alpha = 1 - math.exp(-dt / 4)
        self.rate = raw if self.rate == 0 else self.rate + alpha * (raw - self.rate)
        self.last_time = now
        stalled = now - self.last_change >= 10
        ready = span >= 4 and self.rate > 0 and not stalled and not reset
        remaining = max(0, total - downloaded) if total else None
        eta = remaining / self.rate if ready and remaining else None
        rates = [
            (b[1] - a[1]) / (b[0] - a[0])
            for a, b in zip(self.samples, list(self.samples)[1:])
            if b[0] > a[0]
        ]
        mean = sum(rates) / len(rates) if rates else 0
        deviation = math.sqrt(sum((r - mean) ** 2 for r in rates) / len(rates)) if rates else 0
        uncertain = len(rates) >= 3 and mean > 0 and deviation / mean > 0.5
        return {
            "phase": "retrying" if reset else "waiting" if stalled else "downloading",
            "downloaded_bytes": downloaded,
            "total_bytes": total,
            "speed_bps": round(self.rate) if ready else None,
            "eta_seconds": max(1, round(eta)) if eta is not None else None,
            "eta_range_seconds": [max(1, round(eta / 1.5)), max(1, round(eta * 2))]
            if eta is not None and uncertain
            else None,
            "progress": min(99, int(downloaded / total * 100)) if total else None,
        }
