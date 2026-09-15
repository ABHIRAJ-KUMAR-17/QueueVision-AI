"""
queue_analyzer.py – Queue region analysis for QueueVision AI.

Responsibilities:
  - Determine which tracked persons are inside the queue ROI
  - Compute queue count, density, length, and status classification
  - Update per-track queue membership state
  - Emit a QueueSnapshot for every frame

Application logic is entirely original; only standard Python/NumPy used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.tracker import Track
from src.utils import setup_logger

logger = setup_logger("QueueAnalyzer")


# ─────────────────────────────────────────────
# Queue status levels
# ─────────────────────────────────────────────

class QueueStatus:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    # Colours used in visualisation (BGR)
    COLORS: Dict[str, Tuple[int, int, int]] = {
        LOW:      (0, 200, 0),      # Green
        MEDIUM:   (0, 165, 255),    # Orange
        HIGH:     (0, 0, 255),      # Red
        CRITICAL: (0, 0, 180),      # Dark Red
    }


@dataclass
class QueueSnapshot:
    """Per-frame summary produced by the QueueAnalyzer."""
    frame_number: int
    timestamp_sec: float            # Seconds from start of video

    queue_count: int                # People currently inside ROI
    total_detected: int             # All detected persons in frame
    outside_count: int              # People outside ROI

    queue_density: float            # Normalised 0–1
    queue_length_px: float          # Estimated queue length (pixels)
    queue_status: str               # LOW / MEDIUM / HIGH / CRITICAL

    tracks_in_queue: List[int]      # track_ids inside ROI
    roi: Optional[Tuple[int, int, int, int]]  # Current ROI

    # Populated by WaitingTimeEstimator
    estimated_wait_min: float = 0.0
    average_wait_min: float = 0.0
    service_rate: float = 0.0       # People served per minute


# ─────────────────────────────────────────────
# Queue Analyzer
# ─────────────────────────────────────────────

class QueueAnalyzer:
    """
    Classifies each track as in-queue or not and emits a QueueSnapshot.

    Parameters
    ----------
    roi : tuple (x1, y1, x2, y2) or None
        Queue region.  When None, the *entire* frame is treated as queue.
    density_thresholds : dict
        Keys 'low', 'medium', 'high' map to fractional density cut-offs.
    person_area_estimate : int
        Approximate pixel area of one person bounding box (for density calc).
    frame_width : int
        Width of video frame in pixels.
    frame_height : int
        Height of video frame in pixels.
    """

    def __init__(
        self,
        roi: Optional[Tuple[int, int, int, int]],
        density_thresholds: Dict[str, float],
        person_area_estimate: int = 5000,
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> None:
        self.roi = roi
        self.thresholds = density_thresholds
        self.person_area_estimate = max(person_area_estimate, 1)
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Precompute ROI area
        if roi:
            x1, y1, x2, y2 = roi
            self._roi_area = max(1, (x2 - x1) * (y2 - y1))
        else:
            self._roi_area = max(1, frame_width * frame_height)

        logger.info(
            "QueueAnalyzer initialised. ROI=%s  ROI area=%d px²",
            roi, self._roi_area,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def analyse(
        self,
        tracks: List[Track],
        frame_number: int,
        timestamp_sec: float,
    ) -> QueueSnapshot:
        """
        Analyse the current list of active *tracks* and return a snapshot.

        Also mutates each track's ``in_queue``, ``queue_entry_frame``, and
        ``queue_exit_frame`` fields to maintain longitudinal state.
        """
        in_queue: List[Track] = []
        out_of_queue: List[Track] = []

        for track in tracks:
            if self._is_in_roi(track.center):
                in_queue.append(track)
                # Update track queue state
                if not track.in_queue:
                    track.in_queue = True
                    track.queue_entry_frame = frame_number
                    track.queue_exit_frame = None
            else:
                out_of_queue.append(track)
                if track.in_queue:
                    track.in_queue = False
                    track.queue_exit_frame = frame_number

        queue_count = len(in_queue)
        density = self._compute_density(queue_count)
        status = self._classify_status(density)
        length_px = self._estimate_queue_length(in_queue)

        return QueueSnapshot(
            frame_number=frame_number,
            timestamp_sec=timestamp_sec,
            queue_count=queue_count,
            total_detected=len(tracks),
            outside_count=len(out_of_queue),
            queue_density=density,
            queue_length_px=length_px,
            queue_status=status,
            tracks_in_queue=[t.track_id for t in in_queue],
            roi=self.roi,
        )

    def update_roi(self, roi: Tuple[int, int, int, int]) -> None:
        """Dynamically change the queue ROI."""
        self.roi = roi
        x1, y1, x2, y2 = roi
        self._roi_area = max(1, (x2 - x1) * (y2 - y1))
        logger.info("ROI updated to %s", roi)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _is_in_roi(self, center: Tuple[int, int]) -> bool:
        """Return True if *center* is inside the configured ROI."""
        if self.roi is None:
            return True  # No ROI → entire frame is the queue
        x1, y1, x2, y2 = self.roi
        cx, cy = center
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _compute_density(self, queue_count: int) -> float:
        """
        Compute a normalised queue density [0, 1].

        Density = (queue_count × avg_person_area) / roi_area
        Clamped to [0, 1].
        """
        occupied = queue_count * self.person_area_estimate
        density = occupied / self._roi_area
        return min(1.0, density)

    def _classify_status(self, density: float) -> str:
        """Map a density fraction to a queue status label."""
        low = self.thresholds.get("low", 0.30)
        medium = self.thresholds.get("medium", 0.60)
        high = self.thresholds.get("high", 0.80)

        if density < low:
            return QueueStatus.LOW
        elif density < medium:
            return QueueStatus.MEDIUM
        elif density < high:
            return QueueStatus.HIGH
        else:
            return QueueStatus.CRITICAL

    def _estimate_queue_length(self, in_queue: List[Track]) -> float:
        """
        Estimate physical queue length as the max pairwise centroid distance
        (pixels) among queue members.  Returns 0 when queue is empty.
        """
        if len(in_queue) < 2:
            return 0.0

        centers = np.array([t.center for t in in_queue], dtype=np.float32)
        # Use bounding-box of centroids to estimate spread
        span_x = centers[:, 0].max() - centers[:, 0].min()
        span_y = centers[:, 1].max() - centers[:, 1].min()
        return float(np.hypot(span_x, span_y))
