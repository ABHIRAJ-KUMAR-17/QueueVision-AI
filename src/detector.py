"""
detector.py – YOLOv8 person detector for QueueVision AI.

Responsibilities:
  - Load YOLOv8 model (auto-download if missing)
  - Run inference on a single BGR frame
  - Return only "person" class detections
  - Expose a clean Detection dataclass

Third-party model: YOLOv8 from Ultralytics (Apache-2.0 licence).
Application logic (filtering, result packaging) is our own.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.utils import setup_logger

logger = setup_logger("Detector")

# COCO class index for "person"
_PERSON_CLASS_ID = 0


@dataclass
class Detection:
    """A single person detection result."""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixels
    confidence: float                  # Detection score 0–1
    center: Tuple[int, int]           # (cx, cy) centre of bounding box

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height


class PersonDetector:
    """
    Wraps a YOLOv8 model to detect people in BGR frames.

    Parameters
    ----------
    model_path : str
        Path to a YOLO .pt weights file.  If the file does not exist,
        ultralytics will attempt to auto-download ``yolov8n.pt``.
    confidence : float
        Minimum confidence threshold (0–1).
    iou_threshold : float
        Non-maximum suppression IoU threshold.
    input_size : int
        Inference image size (pixels, square).
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        input_size: int = 640,
    ) -> None:
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self._model = None
        self._model_path = model_path
        self._load_model(model_path)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single BGR NumPy frame.

        Returns a list of :class:`Detection` objects (persons only).
        Returns an empty list when no people are found or if the model
        failed to load.
        """
        if self._model is None:
            logger.warning("Model not loaded – returning empty detections.")
            return []

        try:
            results = self._model(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                imgsz=self.input_size,
                classes=[_PERSON_CLASS_ID],
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Inference error: %s", exc)
            return []

        return self._parse_results(results)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_model(self, model_path: str) -> None:
        """Load YOLO weights; auto-downloads yolov8n.pt if file is absent."""
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            logger.error(
                "ultralytics is not installed. Run: pip install ultralytics"
            )
            self._model = None
            return

        path = Path(model_path)
        if not path.exists():
            # Derive just the filename so ultralytics can download from its CDN
            filename = path.name
            logger.info(
                "Model file '%s' not found locally. "
                "Ultralytics will auto-download '%s' on first run.",
                path, filename,
            )
            # Pass only the filename; ultralytics resolves it from its hub
            effective_path: str | Path = filename
        else:
            effective_path = path

        try:
            self._model = YOLO(str(effective_path))
            logger.info("YOLO model loaded: %s", effective_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load YOLO model: %s", exc)
            self._model = None

    def _parse_results(self, results) -> List[Detection]:
        """Convert ultralytics Results into a list of Detection objects."""
        detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id != _PERSON_CLASS_ID:
                    continue  # safety check – should already be filtered

                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())

                # Clamp to frame boundary (0-indexed)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        center=(cx, cy),
                    )
                )

        return detections
