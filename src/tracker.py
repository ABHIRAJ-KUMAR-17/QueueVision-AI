"""
tracker.py – IoU-based centroid tracker for QueueVision AI.

Responsibilities:
  - Assign persistent IDs to detected persons across frames
  - Maintain per-track state: position, age, first/last seen timestamps
  - Handle occlusion via a configurable max_age (keep-alive window)
  - Provide a clean Track dataclass consumed by downstream modules

Design notes:
  This is an *original implementation* of a centroid + IoU tracker.
  It is inspired by the classical SORT algorithm but written from
  scratch for clarity and educational value.  No external tracking
  library is required.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.detector import Detection
from src.utils import setup_logger

logger = setup_logger("Tracker")


# ─────────────────────────────────────────────
# Track dataclass
# ─────────────────────────────────────────────

@dataclass
class Track:
    """State associated with a single tracked person."""
    track_id: int
    bbox: Tuple[int, int, int, int]     # Most recent bounding box
    center: Tuple[int, int]             # Most recent centroid
    confidence: float

    # Lifecycle counters
    age: int = 0                        # Total frames since creation
    hits: int = 1                       # Frames with a matching detection
    time_since_update: int = 0          # Frames since last successful match

    # Timing
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    entry_time: float = field(default_factory=time.time)
    exit_time: Optional[float] = None

    # Queue membership
    in_queue: bool = False
    queue_entry_frame: Optional[int] = None
    queue_exit_frame: Optional[int] = None

    @property
    def is_confirmed(self) -> bool:
        """A track is 'confirmed' once it has been matched enough times."""
        return self.hits >= 2  # configurable via min_hits in the tracker

    @property
    def time_in_queue_frames(self) -> int:
        """Number of frames this track was inside the queue ROI."""
        if self.queue_entry_frame is None:
            return 0
        last = self.queue_exit_frame if self.queue_exit_frame else self.last_seen_frame
        return max(0, last - self.queue_entry_frame)

    def mark_exited(self) -> None:
        self.exit_time = time.time()


# ─────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────

class CentroidTracker:
    """
    Lightweight IoU + centroid tracker.

    Parameters
    ----------
    max_age : int
        How many consecutive frames a track may go unmatched before deletion.
    min_hits : int
        Minimum matches before a track is considered confirmed.
    iou_threshold : float
        Minimum IoU to associate a detection with an existing track.
    max_distance : int
        Maximum centroid distance (pixels) allowed for matching (secondary guard).
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 2,
        iou_threshold: float = 0.3,
        max_distance: int = 100,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance

        self._tracks: Dict[int, Track] = {}
        self._next_id: int = 1
        self._frame_count: int = 0

        # Exited tracks (for analytics)
        self.exited_tracks: List[Track] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Consume a list of detections from the current frame and return the
        list of *active, confirmed* tracks.

        Call this once per frame, in order.
        """
        self._frame_count += 1

        # ── Step 1: match detections to existing tracks ──────────────── #
        matched, unmatched_dets, unmatched_tracks = self._match(detections)

        # ── Step 2: update matched tracks ────────────────────────────── #
        for track_id, det_idx in matched:
            det = detections[det_idx]
            trk = self._tracks[track_id]
            trk.bbox = det.bbox
            trk.center = det.center
            trk.confidence = det.confidence
            trk.hits += 1
            trk.age += 1
            trk.time_since_update = 0
            trk.last_seen_frame = self._frame_count

        # ── Step 3: create new tracks for unmatched detections ───────── #
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_track = Track(
                track_id=self._next_id,
                bbox=det.bbox,
                center=det.center,
                confidence=det.confidence,
                first_seen_frame=self._frame_count,
                last_seen_frame=self._frame_count,
            )
            self._tracks[self._next_id] = new_track
            self._next_id += 1

        # ── Step 4: age out unmatched tracks ─────────────────────────── #
        to_delete = []
        for track_id in unmatched_tracks:
            trk = self._tracks[track_id]
            trk.time_since_update += 1
            trk.age += 1
            if trk.time_since_update > self.max_age:
                to_delete.append(track_id)

        for tid in to_delete:
            trk = self._tracks.pop(tid)
            trk.mark_exited()
            self.exited_tracks.append(trk)
            logger.debug("Track #%d removed after %d frames.", tid, trk.age)

        # ── Step 5: return only confirmed, active tracks ──────────────── #
        active = [
            t for t in self._tracks.values()
            if t.hits >= self.min_hits
        ]
        return active

    def get_all_tracks(self) -> List[Track]:
        """Return all active tracks (confirmed + tentative)."""
        return list(self._tracks.values())

    def reset(self) -> None:
        """Clear all state (used between independent runs)."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0
        self.exited_tracks.clear()

    # ------------------------------------------------------------------ #
    # Matching                                                             #
    # ------------------------------------------------------------------ #

    def _match(
        self, detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match *detections* to existing tracks using IoU + centroid distance.

        Returns
        -------
        matched       : list of (track_id, detection_index) pairs
        unmatched_dets: detection indices with no matching track
        unmatched_trks: track IDs with no matching detection
        """
        track_ids = list(self._tracks.keys())

        if not track_ids or not detections:
            return [], list(range(len(detections))), track_ids

        # Build IoU cost matrix [tracks × detections]
        iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        for ti, tid in enumerate(track_ids):
            trk_box = self._tracks[tid].bbox
            for di, det in enumerate(detections):
                iou_matrix[ti, di] = _iou(trk_box, det.bbox)

        # Greedy matching: pick highest IoU pairs first
        matched: List[Tuple[int, int]] = []
        used_tracks: set = set()
        used_dets: set = set()

        # Sort all (ti, di) pairs by descending IoU
        pairs = sorted(
            [(ti, di) for ti in range(len(track_ids)) for di in range(len(detections))],
            key=lambda p: iou_matrix[p[0], p[1]],
            reverse=True,
        )

        for ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            iou_val = iou_matrix[ti, di]
            if iou_val < self.iou_threshold:
                # Below IoU threshold — try centroid distance as fallback
                trk_center = self._tracks[track_ids[ti]].center
                det_center = detections[di].center
                dist = _euclidean(trk_center, det_center)
                if dist > self.max_distance:
                    continue  # no match
            matched.append((track_ids[ti], di))
            used_tracks.add(ti)
            used_dets.add(di)

        unmatched_dets = [di for di in range(len(detections)) if di not in used_dets]
        unmatched_trks = [track_ids[ti] for ti in range(len(track_ids)) if ti not in used_tracks]

        return matched, unmatched_dets, unmatched_trks


# ─────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────

def _iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union between two (x1,y1,x2,y2) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _euclidean(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Euclidean distance between two 2-D points."""
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
