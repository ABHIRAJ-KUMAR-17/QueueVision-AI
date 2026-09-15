"""
tests/test_waiting_time.py – Unit tests for WaitingTimeEstimator.

Tests cover:
  - Default service time fallback
  - Correct wait estimation formula
  - Service time learning from exited tracks
  - Service rate calculation
  - Clamping of observed service times
  - Zero-queue case
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tracker import Track
from src.waiting_time import WaitingTimeEstimator


def _make_estimator(**kwargs) -> WaitingTimeEstimator:
    defaults = dict(
        default_service_time=2.5,
        min_service_time=0.5,
        max_service_time=10.0,
        fps=30.0,
    )
    defaults.update(kwargs)
    return WaitingTimeEstimator(**defaults)


def _make_exited_track(
    track_id: int,
    entry_frame: int,
    exit_frame: int,
) -> Track:
    """Create a track that has been in the queue and then left."""
    t = Track(
        track_id=track_id,
        bbox=(100, 100, 160, 220),
        center=(130, 160),
        confidence=0.9,
    )
    t.queue_entry_frame = entry_frame
    t.queue_exit_frame  = exit_frame
    t.in_queue = False  # has exited
    t.first_seen_frame = entry_frame
    t.last_seen_frame  = exit_frame
    return t


class TestDefaultServiceTime(unittest.TestCase):

    def test_default_used_when_no_observations(self):
        est = _make_estimator(default_service_time=3.0)
        self.assertAlmostEqual(est.average_service_time(), 3.0)

    def test_zero_queue_gives_zero_wait(self):
        est = _make_estimator()
        self.assertAlmostEqual(est.estimate_wait(0), 0.0)


class TestWaitEstimationFormula(unittest.TestCase):

    def test_wait_equals_count_times_service_time(self):
        est = _make_estimator(default_service_time=2.0)
        # 5 people × 2.0 min = 10.0 min
        self.assertAlmostEqual(est.estimate_wait(5), 10.0)

    def test_wait_scales_with_queue_count(self):
        est = _make_estimator(default_service_time=1.5)
        self.assertLess(est.estimate_wait(3), est.estimate_wait(6))


class TestServiceTimeLearning(unittest.TestCase):

    def test_service_time_learned_from_exits(self):
        """Observed service time should replace the default."""
        # 7200 frames at 60 fps → 120 seconds → 2.0 minutes
        est = _make_estimator(default_service_time=5.0, fps=60.0,
                              min_service_time=0.1, max_service_time=10.0)

        track = _make_exited_track(1, entry_frame=0, exit_frame=7200)
        est.observe_exits([track], current_frame=7200)

        observed = est.average_service_time()
        self.assertAlmostEqual(observed, 2.0, places=5)

    def test_no_double_counting(self):
        """Same track should be counted only once."""
        est = _make_estimator(fps=60.0)
        track = _make_exited_track(1, entry_frame=0, exit_frame=60)

        est.observe_exits([track], current_frame=60)
        est.observe_exits([track], current_frame=60)  # duplicate call

        self.assertEqual(est.people_served, 1)

    def test_track_never_in_queue_is_ignored(self):
        """Tracks that never entered the queue should not affect service time."""
        est = _make_estimator(default_service_time=2.5)
        track = Track(
            track_id=99,
            bbox=(0, 0, 60, 120),
            center=(30, 60),
            confidence=0.8,
        )
        # queue_entry_frame is None → never entered queue
        est.observe_exits([track], current_frame=100)
        self.assertEqual(est.people_served, 0)
        self.assertAlmostEqual(est.average_service_time(), 2.5)


class TestServiceTimeClamping(unittest.TestCase):

    def test_very_short_time_clamped(self):
        """Abnormally short service times should be clamped up."""
        est = _make_estimator(min_service_time=0.5, fps=60.0)
        # 1 frame at 60 fps = 1/60 min ≈ 0.017 min → clamped to 0.5
        track = _make_exited_track(1, entry_frame=0, exit_frame=1)
        est.observe_exits([track], current_frame=1)
        self.assertGreaterEqual(est.average_service_time(), 0.5)

    def test_very_long_time_clamped(self):
        """Abnormally long service times should be clamped down."""
        est = _make_estimator(max_service_time=10.0, fps=30.0)
        # 36000 frames at 30 fps = 1200 min → clamped to 10.0
        track = _make_exited_track(1, entry_frame=0, exit_frame=36000)
        est.observe_exits([track], current_frame=36000)
        self.assertLessEqual(est.average_service_time(), 10.0)


class TestServiceRate(unittest.TestCase):

    def test_service_rate_reciprocal_of_service_time(self):
        est = _make_estimator(default_service_time=2.0)
        # rate = 1 / 2.0 = 0.5 people/min
        self.assertAlmostEqual(est.service_rate(), 0.5, places=2)

    def test_people_served_counter_increments(self):
        est = _make_estimator(fps=30.0)
        tracks = [
            _make_exited_track(i, entry_frame=0, exit_frame=300)
            for i in range(1, 4)
        ]
        for t in tracks:
            est.observe_exits([t], current_frame=300)
        self.assertEqual(est.people_served, 3)


if __name__ == "__main__":
    unittest.main()
