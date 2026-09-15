"""
setup_roi.py – Interactive ROI selector for QueueVision AI.

Usage:
    python setup_roi.py --video data/sample_videos/queue.mp4
    python setup_roi.py --image data/sample_images/queue.jpg
    python setup_roi.py --video data/sample_videos/queue.mp4 --config config.yaml

Instructions (on-screen):
    - Drag to draw the queue rectangle.
    - Press ENTER or SPACE to confirm and save.
    - Press R to redraw.
    - Press ESC to cancel without saving.

The selected ROI is written back to config.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from src.utils import load_config, save_roi_to_config, setup_logger, validate_input_file

logger = setup_logger("SetupROI")


# ─────────────────────────────────────────────
# Mouse callback state
# ─────────────────────────────────────────────

class _ROIDrawer:
    """Tracks mouse events for interactive ROI selection."""

    def __init__(self, frame: np.ndarray) -> None:
        self._base = frame.copy()
        self.drawing = False
        self.start   = (0, 0)
        self.end     = (0, 0)
        self.roi     = None  # (x1, y1, x2, y2) when confirmed

    def callback(self, event, x, y, flags, param) -> None:  # noqa: ARG002
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start   = (x, y)
            self.end     = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end = (x, y)
            x1, y1 = min(self.start[0], self.end[0]), min(self.start[1], self.end[1])
            x2, y2 = max(self.start[0], self.end[0]), max(self.start[1], self.end[1])
            if x2 - x1 > 5 and y2 - y1 > 5:
                self.roi = (x1, y1, x2, y2)

    def current_frame(self) -> np.ndarray:
        """Return the base frame with the in-progress rectangle drawn."""
        frame = self._base.copy()
        if self.start != self.end:
            x1 = min(self.start[0], self.end[0])
            y1 = min(self.start[1], self.end[1])
            x2 = max(self.start[0], self.end[0])
            y2 = max(self.start[1], self.end[1])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = f"ROI: {x1},{y1}  →  {x2},{y2}"
            cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
        return frame


# ─────────────────────────────────────────────
# Main interactive loop
# ─────────────────────────────────────────────

def select_roi(frame: np.ndarray, config_path: str = "config.yaml") -> None:
    """Open an interactive window and let the user draw the queue ROI."""
    WIN = "QueueVision AI – Draw Queue ROI  (ENTER=save  R=reset  ESC=cancel)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    drawer = _ROIDrawer(frame)
    cv2.setMouseCallback(WIN, drawer.callback)

    # Overlay usage instructions
    instructions = frame.copy()
    lines = [
        "Draw the QUEUE REGION with your mouse.",
        "ENTER / SPACE  →  Save ROI",
        "R              →  Reset",
        "ESC            →  Cancel",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            instructions, line, (15, 30 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
        )
    drawer._base = instructions

    logger.info("Interactive ROI window open. Follow on-screen instructions.")

    while True:
        display = drawer.current_frame()

        # Show confirmed ROI in bright colour
        if drawer.roi:
            x1, y1, x2, y2 = drawer.roi
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
            msg = f"Selected: {x1},{y1},{x2},{y2}   Press ENTER to save"
            cv2.putText(display, msg, (15, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):  # ENTER or SPACE
            if drawer.roi is None:
                logger.warning("No ROI drawn yet. Draw a rectangle first.")
            else:
                save_roi_to_config(drawer.roi, config_path)
                logger.info("ROI saved: %s", drawer.roi)
                break
        elif key == ord("r"):
            drawer.roi  = None
            drawer.start = (0, 0)
            drawer.end   = (0, 0)
            logger.info("ROI reset.")
        elif key == 27:  # ESC
            logger.info("ROI selection cancelled.")
            break

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive queue ROI selector – QueueVision AI",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="Path to video file.")
    src.add_argument("--image", type=str, help="Path to image file.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file to update.")
    args = parser.parse_args()

    if args.video:
        path = validate_input_file(args.video, "Video")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.error("Cannot open video '%s'.", path)
            sys.exit(1)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            logger.error("Could not read first frame from video.")
            sys.exit(1)
    else:
        path = validate_input_file(args.image, "Image")
        frame = cv2.imread(str(path))
        if frame is None:
            logger.error("Could not read image '%s'.", path)
            sys.exit(1)

    select_roi(frame, args.config)


if __name__ == "__main__":
    main()
