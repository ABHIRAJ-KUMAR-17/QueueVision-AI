"""
analytics.py – Time-series analytics and report generation for QueueVision AI.

Responsibilities:
  - Accumulate per-frame QueueSnapshots
  - Compute aggregate statistics (peak, average, etc.)
  - Generate time-series CSV
  - Generate per-person tracking CSV
  - Write a human-readable text report

Application logic is entirely original.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.queue_analyzer import QueueSnapshot
from src.tracker import Track
from src.utils import ensure_dir, setup_logger, seconds_to_hms

logger = setup_logger("Analytics")


class QueueAnalytics:
    """
    Collects snapshots over time and produces aggregate metrics and reports.

    Parameters
    ----------
    fps : float
        Source video FPS – used to convert frame counts to real time.
    results_dir : str
        Directory where CSV files are written.
    reports_dir : str
        Directory where the text report is written.
    time_series_interval : int
        Seconds between sampled time-series data points for the CSV.
    """

    def __init__(
        self,
        fps: float = 30.0,
        results_dir: str = "results",
        reports_dir: str = "reports",
        time_series_interval: int = 30,
    ) -> None:
        self.fps = max(fps, 1e-6)
        self.results_dir = Path(results_dir)
        self.reports_dir = Path(reports_dir)
        self.time_series_interval = time_series_interval

        self._snapshots: List[QueueSnapshot] = []
        self._interval_frames = max(1, int(time_series_interval * fps))
        self._last_sample_frame: int = -self._interval_frames  # force first sample

    # ------------------------------------------------------------------ #
    # Data ingestion                                                       #
    # ------------------------------------------------------------------ #

    def record(self, snapshot: QueueSnapshot) -> None:
        """Add one snapshot.  Called every processed frame."""
        self._snapshots.append(snapshot)

    # ------------------------------------------------------------------ #
    # Aggregate statistics                                                 #
    # ------------------------------------------------------------------ #

    @property
    def total_snapshots(self) -> int:
        return len(self._snapshots)

    @property
    def peak_queue_size(self) -> int:
        if not self._snapshots:
            return 0
        return max(s.queue_count for s in self._snapshots)

    @property
    def average_queue_size(self) -> float:
        if not self._snapshots:
            return 0.0
        return sum(s.queue_count for s in self._snapshots) / len(self._snapshots)

    @property
    def average_queue_density(self) -> float:
        if not self._snapshots:
            return 0.0
        return sum(s.queue_density for s in self._snapshots) / len(self._snapshots)

    @property
    def average_wait_min(self) -> float:
        waits = [s.estimated_wait_min for s in self._snapshots if s.estimated_wait_min > 0]
        return sum(waits) / len(waits) if waits else 0.0

    @property
    def max_wait_min(self) -> float:
        if not self._snapshots:
            return 0.0
        return max(s.estimated_wait_min for s in self._snapshots)

    @property
    def final_snapshot(self) -> Optional[QueueSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    def time_series(self) -> List[QueueSnapshot]:
        """Return one snapshot per time-series interval."""
        sampled: List[QueueSnapshot] = []
        for snap in self._snapshots:
            if snap.frame_number >= self._last_sample_frame + self._interval_frames or not sampled:
                sampled.append(snap)
                self._last_sample_frame = snap.frame_number
        # Reset internal counter so future calls still work
        self._last_sample_frame = -self._interval_frames
        return sampled

    # ------------------------------------------------------------------ #
    # CSV export                                                           #
    # ------------------------------------------------------------------ #

    def save_queue_csv(self, filename: str = "queue_analysis.csv") -> Path:
        """Write a time-series queue CSV."""
        ensure_dir(self.results_dir)
        path = self.results_dir / filename
        series = self.time_series()

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "timestamp",
                "frame_number",
                "queue_count",
                "total_detected",
                "queue_density",
                "queue_status",
                "estimated_wait_time_min",
                "average_wait_time_min",
                "people_served",
                "service_rate",
                "queue_length_px",
            ])
            for snap in series:
                ts = _frame_to_timestamp(snap.frame_number, self.fps)
                writer.writerow([
                    ts,
                    snap.frame_number,
                    snap.queue_count,
                    snap.total_detected,
                    f"{snap.queue_density:.3f}",
                    snap.queue_status,
                    f"{snap.estimated_wait_min:.2f}",
                    f"{snap.average_wait_min:.2f}",
                    snap.service_rate,   # used as people_served proxy here
                    f"{snap.service_rate:.2f}",
                    f"{snap.queue_length_px:.1f}",
                ])

        logger.info("Queue analysis CSV saved → %s", path)
        return path

    def save_tracking_csv(
        self,
        all_tracks: List[Track],
        filename: str = "person_tracking.csv",
    ) -> Path:
        """Write per-person tracking CSV."""
        ensure_dir(self.results_dir)
        path = self.results_dir / filename

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "track_id",
                "entry_frame",
                "exit_frame",
                "frames_seen",
                "time_in_queue_frames",
                "time_in_queue_sec",
                "was_in_queue",
            ])
            for track in all_tracks:
                exit_frame = track.last_seen_frame
                frames_seen = track.age
                tiq_frames = track.time_in_queue_frames
                tiq_sec = tiq_frames / self.fps
                writer.writerow([
                    track.track_id,
                    track.first_seen_frame,
                    exit_frame,
                    frames_seen,
                    tiq_frames,
                    f"{tiq_sec:.2f}",
                    int(track.queue_entry_frame is not None),
                ])

        logger.info("Person tracking CSV saved → %s", path)
        return path

    # ------------------------------------------------------------------ #
    # Text report                                                          #
    # ------------------------------------------------------------------ #

    def save_report(
        self,
        all_tracks: List[Track],
        people_served: int,
        avg_service_time: float,
        filename: str = "analysis_report.txt",
    ) -> Path:
        """Write a human-readable analysis report."""
        ensure_dir(self.reports_dir)
        path = self.reports_dir / filename
        final = self.final_snapshot

        lines = [
            "=" * 60,
            "        QueueVision AI - Queue Analysis Report",
            "=" * 60,
            f"  Generated at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Frames processed: {self.total_snapshots}",
            "",
            "-" * 60,
            "  DETECTION & TRACKING SUMMARY",
            "-" * 60,
            f"  Total unique persons tracked : {len(all_tracks)}",
            f"  People served (exited queue) : {people_served}",
            "",
            "-" * 60,
            "  QUEUE STATISTICS",
            "-" * 60,
            f"  Peak queue size              : {self.peak_queue_size} people",
            f"  Average queue size           : {self.average_queue_size:.1f} people",
            f"  Average queue density        : {self.average_queue_density:.1%}",
            "",
            "-" * 60,
            "  WAITING TIME ESTIMATES  (NOTE: these are estimates only)",
            "-" * 60,
            f"  Average estimated wait time  : {self.average_wait_min:.1f} min",
            f"  Maximum estimated wait time  : {self.max_wait_min:.1f} min",
            f"  Avg service time per person  : {avg_service_time:.2f} min",
            f"  Estimated service rate       : {(1/avg_service_time if avg_service_time else 0):.2f} people/min",
            "",
        ]

        if final:
            lines += [
                "-" * 60,
                "  FINAL FRAME STATUS",
                "-" * 60,
                f"  Queue count    : {final.queue_count}",
                f"  Queue status   : {final.queue_status}",
                f"  Est. wait time : {final.estimated_wait_min:.1f} min",
            ]

        lines += [
            "",
            "-" * 60,
            "  TIME-SERIES SNAPSHOT",
            "-" * 60,
            f"  {'Time':>8}   {'Queue Size':>10}   {'Status':>10}   {'Est. Wait (min)':>16}",
        ]
        for snap in self.time_series():
            ts = _frame_to_timestamp(snap.frame_number, self.fps)
            lines.append(
                f"  {ts:>8}   {snap.queue_count:>10}   {snap.queue_status:>10}"
                f"   {snap.estimated_wait_min:>16.1f}"
            )

        lines += ["", "=" * 60, "  End of Report", "=" * 60, ""]

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        logger.info("Analysis report saved → %s", path)
        return path


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _frame_to_timestamp(frame: int, fps: float) -> str:
    """Convert a frame number to a MM:SS timestamp string."""
    total_sec = int(frame / fps)
    m = total_sec // 60
    s = total_sec % 60
    return f"{m:02d}:{s:02d}"
