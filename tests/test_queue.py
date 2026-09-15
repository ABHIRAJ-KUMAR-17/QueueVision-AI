"""
tests/test_queue.py – Unit tests for QueueAnalyzer and related helpers.

Tests cover:
  - ROI membership (inside / outside / edge cases)
  - Queue status classification at threshold boundaries
  - Density calculation
  - Queue length estimation
  - parse_roi validation
  - Zero-people edge case
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.queue_analyzer import QueueAnalyzer, QueueStatus
from src.tracker import Track
from src.utils import parse_roi


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

_THRESHOLDS = {"low": 0.30, "medium": 0.60, "high": 0.80}


def _make_track(track_id: int, cx: int, cy: int) -> Track:
    """Create a minimal Track at the given centroid."""
    t = Track(
        track_id=track_id,
        bbox=(cx - 30, cy - 60, cx + 30, cy + 60),
        center=(cx, cy),
        confidence=0.9,
    )
    return t


def _make_analyzer(roi=None, person_area=5000) -> QueueAnalyzer:
    return QueueAnalyzer(
        roi=roi,
        density_thresholds=_THRESHOLDS,
        person_area_estimate=person_area,
        frame_width=1280,
        frame_height=720,
    )


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

class TestROIMembership(unittest.TestCase):

    def setUp(self):
        self.analyzer = _make_analyzer(roi=(100, 100, 900, 600))

    def test_center_inside_roi(self):
        track = _make_track(1, 500, 350)
        snap = self.analyzer.analyse([track], frame_number=1, timestamp_sec=0.0)
        self.assertEqual(snap.queue_count, 1)
        self.assertIn(1, snap.tracks_in_queue)

    def test_center_outside_roi(self):
        track = _make_track(1, 50, 50)   # outside ROI
        snap = self.analyzer.analyse([track], frame_number=1, timestamp_sec=0.0)
        self.assertEqual(snap.queue_count, 0)
        self.assertNotIn(1, snap.tracks_in_queue)

    def test_center_on_roi_edge_included(self):
        track = _make_track(1, 100, 100)  # exactly on boundary
        snap = self.analyzer.analyse([track], frame_number=1, timestamp_sec=0.0)
        self.assertEqual(snap.queue_count, 1)

    def test_no_roi_full_frame(self):
        """With no ROI, every person is in the queue."""
        analyzer = _make_analyzer(roi=None)
        tracks = [_make_track(i, i * 100, 300) for i in range(5)]
        snap = analyzer.analyse(tracks, frame_number=1, timestamp_sec=0.0)
        self.assertEqual(snap.queue_count, 5)

    def test_empty_track_list(self):
        snap = self.analyzer.analyse([], frame_number=1, timestamp_sec=0.0)
        self.assertEqual(snap.queue_count, 0)
        self.assertEqual(snap.total_detected, 0)


class TestQueueStatusClassification(unittest.TestCase):

    def setUp(self):
        # Use large ROI so density is purely controlled by person_area_estimate
        # ROI area = 1000×1000 = 1_000_000
        # person_area = 100_000
        # 1 person → density 0.10 → LOW
        # 4 persons → density 0.40 → MEDIUM
        # 7 persons → density 0.70 → HIGH
        # 9 persons → density 0.90 → CRITICAL
        self.analyzer = _make_analyzer(roi=(0, 0, 1000, 1000), person_area=100_000)

    def _status_for_n(self, n: int) -> str:
        tracks = [_make_track(i, 100 + i * 80, 500) for i in range(n)]
        snap = self.analyzer.analyse(tracks, frame_number=1, timestamp_sec=0.0)
        return snap.queue_status

    def test_low_status(self):
        self.assertEqual(self._status_for_n(1), QueueStatus.LOW)

    def test_medium_status(self):
        self.assertEqual(self._status_for_n(4), QueueStatus.MEDIUM)

    def test_high_status(self):
        self.assertEqual(self._status_for_n(7), QueueStatus.HIGH)

    def test_critical_status(self):
        self.assertEqual(self._status_for_n(9), QueueStatus.CRITICAL)

    def test_zero_people_is_low(self):
        self.assertEqual(self._status_for_n(0), QueueStatus.LOW)


class TestDensityCalculation(unittest.TestCase):

    def test_density_clamped_at_one(self):
        # Tiny ROI, many people → density > 1 → clamped to 1.0
        analyzer = _make_analyzer(roi=(0, 0, 100, 100), person_area=5000)
        tracks = [_make_track(i, 50, 50) for i in range(50)]
        snap = analyzer.analyse(tracks, frame_number=1, timestamp_sec=0.0)
        self.assertLessEqual(snap.queue_density, 1.0)

    def test_density_zero_when_empty(self):
        analyzer = _make_analyzer(roi=(0, 0, 1000, 1000))
        snap = analyzer.analyse([], frame_number=1, timestamp_sec=0.0)
        self.assertAlmostEqual(snap.queue_density, 0.0)


class TestQueueLength(unittest.TestCase):

    def test_single_person_length_zero(self):
        analyzer = _make_analyzer(roi=(0, 0, 1000, 1000))
        tracks = [_make_track(1, 500, 500)]
        snap = analyzer.analyse(tracks, frame_number=1, timestamp_sec=0.0)
        self.assertEqual(snap.queue_length_px, 0.0)

    def test_two_people_length_positive(self):
        analyzer = _make_analyzer(roi=(0, 0, 1000, 1000))
        tracks = [_make_track(1, 100, 500), _make_track(2, 900, 500)]
        snap = analyzer.analyse(tracks, frame_number=1, timestamp_sec=0.0)
        self.assertGreater(snap.queue_length_px, 0)


class TestParseROI(unittest.TestCase):

    def test_valid_roi(self):
        roi = parse_roi("100,100,800,600")
        self.assertEqual(roi, (100, 100, 800, 600))

    def test_none_returns_none(self):
        self.assertIsNone(parse_roi(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(parse_roi(""))

    def test_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            parse_roi("100,200,300")  # only 3 values

    def test_inverted_coords_raises(self):
        with self.assertRaises(ValueError):
            parse_roi("800,600,100,100")  # x1 > x2


class TestTrackQueueStateTransition(unittest.TestCase):
    """Verify that track queue membership state is updated correctly."""

    def test_track_enters_queue(self):
        analyzer = _make_analyzer(roi=(0, 0, 500, 500))
        track = _make_track(1, 250, 250)
        self.assertFalse(track.in_queue)
        analyzer.analyse([track], frame_number=10, timestamp_sec=0.5)
        self.assertTrue(track.in_queue)
        self.assertEqual(track.queue_entry_frame, 10)

    def test_track_exits_queue(self):
        analyzer = _make_analyzer(roi=(0, 0, 500, 500))
        track = _make_track(1, 250, 250)
        analyzer.analyse([track], frame_number=10, timestamp_sec=0.5)
        # Move outside ROI
        track.center = (600, 600)
        analyzer.analyse([track], frame_number=20, timestamp_sec=1.0)
        self.assertFalse(track.in_queue)
        self.assertEqual(track.queue_exit_frame, 20)


if __name__ == "__main__":
    unittest.main()
