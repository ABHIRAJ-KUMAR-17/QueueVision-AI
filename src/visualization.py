"""
visualization.py – Drawing and overlay helpers for QueueVision AI.

Responsibilities:
  - Draw bounding boxes with track IDs and confidence
  - Draw the queue ROI rectangle
  - Render the stats overlay panel
  - Generate a heatmap from accumulated density data
  - Provide colour helpers

Uses only OpenCV and NumPy (no extra deps).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.queue_analyzer import QueueSnapshot, QueueStatus
from src.tracker import Track
from src.utils import ensure_dir, setup_logger

logger = setup_logger("Visualizer")


# ─────────────────────────────────────────────
# Colours (BGR)
# ─────────────────────────────────────────────

COLOR_WHITE   = (255, 255, 255)
COLOR_BLACK   = (0,   0,   0)
COLOR_CYAN    = (255, 200, 0)
COLOR_GREEN   = (0,   220, 0)
COLOR_ORANGE  = (0,   165, 255)
COLOR_RED     = (0,   0,   255)
COLOR_DARK_RED = (0,  0,   160)
COLOR_GREY    = (180, 180, 180)

_STATUS_COLOR: Dict[str, Tuple[int, int, int]] = {
    QueueStatus.LOW:      COLOR_GREEN,
    QueueStatus.MEDIUM:   COLOR_ORANGE,
    QueueStatus.HIGH:     COLOR_RED,
    QueueStatus.CRITICAL: COLOR_DARK_RED,
}


# ─────────────────────────────────────────────
# Visualizer
# ─────────────────────────────────────────────

class Visualizer:
    """
    Draws all visual elements onto frames and generates the heatmap.

    Parameters
    ----------
    frame_width, frame_height : int
        Dimensions of the video frame.
    panel_alpha : float
        Opacity of the stats panel background (0=transparent, 1=opaque).
    """

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        panel_alpha: float = 0.6,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.panel_alpha = panel_alpha

        # Accumulator for heatmap (float32, single channel)
        self._heatmap_acc = np.zeros((frame_height, frame_width), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Main annotation entry point                                          #
    # ------------------------------------------------------------------ #

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        snapshot: QueueSnapshot,
    ) -> np.ndarray:
        """
        Return a copy of *frame* with all annotations applied.

        Annotations:
          1. Queue ROI rectangle
          2. Bounding boxes + track IDs
          3. Stats panel
        """
        out = frame.copy()

        # 1. Draw ROI
        if snapshot.roi:
            self._draw_roi(out, snapshot.roi, snapshot.queue_status)

        # 2. Draw all person tracks
        for track in tracks:
            in_q = track.in_queue
            self._draw_track(out, track, in_q)

        # 3. Stats panel
        self._draw_stats_panel(out, snapshot)

        # 4. Accumulate heatmap data for queue members
        for track in tracks:
            if track.in_queue:
                self._accumulate_heatmap(track)

        return out

    # ------------------------------------------------------------------ #
    # Drawing helpers                                                      #
    # ------------------------------------------------------------------ #

    def _draw_roi(
        self,
        frame: np.ndarray,
        roi: Tuple[int, int, int, int],
        status: str,
    ) -> None:
        """Draw the queue boundary rectangle."""
        x1, y1, x2, y2 = roi
        color = _STATUS_COLOR.get(status, COLOR_CYAN)

        # Filled semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        # Border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"Queue Region [{status}]"
        _draw_label(frame, label, (x1, y1 - 8), color, font_scale=0.55)

    def _draw_track(
        self,
        frame: np.ndarray,
        track: Track,
        in_queue: bool,
    ) -> None:
        """Draw a bounding box with track ID and confidence."""
        x1, y1, x2, y2 = track.bbox
        color = COLOR_GREEN if in_queue else COLOR_GREY
        thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Track label: ID + confidence
        label = f"#{track.track_id}  {track.confidence:.2f}"
        _draw_label(frame, label, (x1, y1 - 6), color, font_scale=0.5)

        # Dot at centroid
        cv2.circle(frame, track.center, 4, color, -1)

    def _draw_stats_panel(
        self,
        frame: np.ndarray,
        snap: QueueSnapshot,
    ) -> None:
        """Render the semi-transparent stats panel in the top-left corner."""
        panel_x, panel_y = 10, 10
        panel_w, panel_h = 310, 190

        status_color = _STATUS_COLOR.get(snap.queue_status, COLOR_WHITE)

        # Background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            COLOR_BLACK,
            -1,
        )
        cv2.addWeighted(overlay, self.panel_alpha, frame, 1 - self.panel_alpha, 0, frame)

        # Border
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            status_color,
            1,
        )

        # Text rows
        x = panel_x + 12
        rows = [
            ("QueueVision AI", 0.58, COLOR_CYAN, True),
            ("─" * 34, 0.4,  COLOR_GREY, False),
            (f"People in Queue : {snap.queue_count}", 0.52, COLOR_WHITE, False),
            (f"Total Detected  : {snap.total_detected}", 0.52, COLOR_WHITE, False),
            (f"Queue Density   : {snap.queue_density:.1%}", 0.52, COLOR_WHITE, False),
            (f"Queue Status    : {snap.queue_status}", 0.52, status_color, True),
            (f"Est. Wait       : {snap.estimated_wait_min:.1f} min (EST)", 0.50, COLOR_WHITE, False),
            (f"Avg  Wait       : {snap.average_wait_min:.1f} min (EST)", 0.50, COLOR_WHITE, False),
            (f"Service Rate    : {snap.service_rate:.2f} p/min", 0.50, COLOR_WHITE, False),
        ]

        y = panel_y + 22
        for text, scale, color, bold in rows:
            thickness = 2 if bold else 1
            cv2.putText(
                frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA,
            )
            y += 20

    def _accumulate_heatmap(self, track: Track) -> None:
        """Add a Gaussian blob at the track's centroid to the heatmap."""
        cx, cy = track.center
        if 0 <= cx < self.frame_width and 0 <= cy < self.frame_height:
            # Use a simple radius-based accumulation
            r = 30
            x1 = max(0, cx - r)
            y1 = max(0, cy - r)
            x2 = min(self.frame_width - 1, cx + r)
            y2 = min(self.frame_height - 1, cy + r)
            self._heatmap_acc[y1:y2, x1:x2] += 1.0

    # ------------------------------------------------------------------ #
    # Heatmap generation                                                   #
    # ------------------------------------------------------------------ #

    def save_heatmap(self, output_path: str | Path) -> Path:
        """
        Normalise the accumulated heatmap and save it as a colour PNG.

        Returns the path to the saved image.
        """
        path = Path(output_path)
        ensure_dir(path.parent)

        acc = self._heatmap_acc.copy()
        if acc.max() == 0:
            logger.warning("Heatmap accumulator is empty – saving blank image.")
            blank = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            cv2.imwrite(str(path), blank)
            return path

        # Normalise to 0-255 and apply colour map
        norm = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        # Add a title
        cv2.putText(
            heatmap_color,
            "QueueVision AI – Queue Density Heatmap",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_WHITE,
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(str(path), heatmap_color)
        logger.info("Heatmap saved → %s", path)
        return path


# ─────────────────────────────────────────────
# Standalone helpers
# ─────────────────────────────────────────────

def _draw_label(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Draw a text label with a dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    y = max(y, th + 4)  # keep inside frame top

    # Background rectangle
    cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + baseline), COLOR_BLACK, -1)
    # Text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_demo_text(frame: np.ndarray, message: str) -> np.ndarray:
    """Utility: overlay a large centred message (used in demo/error frames)."""
    out = frame.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(message, font, scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.putText(out, message, (x, y), font, scale, COLOR_WHITE, thickness, cv2.LINE_AA)
    return out
