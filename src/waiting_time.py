"""
waiting_time.py – Waiting-time estimator for QueueVision AI.

Methodology
-----------
Estimated Waiting Time = (people ahead) × (average service time per person)

"People ahead" = current queue count (the person asking is at the back).
"Average service time" is learned dynamically from observed departures:
  - When a person's track leaves the ROI it is treated as a "service event".
  - The interval between entering and leaving the queue is their service time.
  - We maintain a rolling window of the last N service times.
  - If no observations exist, a configurable default is used.

All outputs are clearly labelled as ESTIMATES.

Application logic is entirely original.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

from src.tracker import Track
from src.utils import clamp, setup_logger

logger = setup_logger("WaitingTime")


class WaitingTimeEstimator:
    """
    Estimates queue waiting time from observation of people exiting the queue.

    Parameters
    ----------
    default_service_time : float
        Fallback service time (minutes/person) used when no observations exist.
    min_service_time : float
        Minimum clamp for observed service time (minutes).
    max_service_time : float
        Maximum clamp for observed service time (minutes).
    fps : float
        Video frames per second – used to convert frame counts to seconds.
    window_size : int
        Number of most-recent service events kept in the rolling average.
    """

    def __init__(
        self,
        default_service_time: float = 2.5,
        min_service_time: float = 0.5,
        max_service_time: float = 10.0,
        fps: float = 30.0,
        window_size: int = 20,
    ) -> None:
        self.default_service_time = default_service_time
        self.min_service_time = min_service_time
        self.max_service_time = max_service_time
        self.fps = max(fps, 1e-6)
        self._observations: Deque[float] = deque(maxlen=window_size)
        self._people_served: int = 0

        # Track IDs we've already processed so we don't double-count
        self._processed_exits: set = set()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def observe_exits(
        self,
        exited_tracks: List[Track],
        current_frame: int,
    ) -> None:
        """
        Inspect recently exited tracks and learn service time from those
        that were in the queue.

        Call this each frame with the tracker's ``exited_tracks`` list.
        """
        for track in exited_tracks:
            if track.track_id in self._processed_exits:
                continue
            if track.queue_entry_frame is None:
                continue
            if not track.in_queue and track.queue_exit_frame is not None:
                # Frames spent in queue → convert to minutes
                frames_in_queue = max(
                    0, track.queue_exit_frame - track.queue_entry_frame
                )
                if frames_in_queue <= 0:
                    continue
                service_time_min = (frames_in_queue / self.fps) / 60.0
                service_time_min = clamp(
                    service_time_min,
                    self.min_service_time,
                    self.max_service_time,
                )
                self._observations.append(service_time_min)
                self._people_served += 1
                self._processed_exits.add(track.track_id)
                logger.debug(
                    "Service event: track #%d, %.2f min in queue.",
                    track.track_id, service_time_min,
                )

    def average_service_time(self) -> float:
        """
        Return the current average service time (minutes/person).
        Falls back to the configured default when no observations exist.
        """
        if self._observations:
            return sum(self._observations) / len(self._observations)
        return self.default_service_time

    def estimate_wait(self, queue_count: int) -> float:
        """
        Estimate waiting time for a new arrival.

        Returns
        -------
        float
            Estimated wait in minutes.  This is an ESTIMATE only.
        """
        if queue_count <= 0:
            return 0.0
        avg_svc = self.average_service_time()
        return queue_count * avg_svc

    def service_rate(self) -> float:
        """
        Estimated throughput in people per minute.
        Returns 0 when average service time is unknown.
        """
        avg = self.average_service_time()
        if avg <= 0:
            return 0.0
        return round(1.0 / avg, 2)

    @property
    def people_served(self) -> int:
        return self._people_served

    # ------------------------------------------------------------------ #
    # Aggregate stats                                                       #
    # ------------------------------------------------------------------ #

    def max_observed_service_time(self) -> float:
        return max(self._observations, default=self.default_service_time)

    def min_observed_service_time(self) -> float:
        return min(self._observations, default=self.default_service_time)
