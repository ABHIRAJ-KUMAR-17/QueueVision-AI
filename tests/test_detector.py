"""
tests/test_detector.py – Unit tests for the PersonDetector.

Tests cover:
  - Detection dataclass properties
  - Graceful handling of missing model
  - Correct filtering of non-person classes (mocked)
"""

import sys
import os
import unittest
from unittest.mock import MagicMock

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detector import Detection, PersonDetector


class TestDetectionDataclass(unittest.TestCase):
    """Test the Detection dataclass."""

    def _make_det(self, x1=10, y1=20, x2=110, y2=220, conf=0.9):
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return Detection(bbox=(x1, y1, x2, y2), confidence=conf, center=(cx, cy))

    def test_width(self):
        det = self._make_det(0, 0, 100, 200)
        self.assertEqual(det.width, 100)

    def test_height(self):
        det = self._make_det(0, 0, 100, 200)
        self.assertEqual(det.height, 200)

    def test_area(self):
        det = self._make_det(0, 0, 50, 100)
        self.assertEqual(det.area, 5000)

    def test_center_calculated(self):
        det = self._make_det(0, 0, 100, 100)
        self.assertEqual(det.center, (50, 50))

    def test_confidence_stored(self):
        det = self._make_det(conf=0.77)
        self.assertAlmostEqual(det.confidence, 0.77)


class TestPersonDetectorMissingModel(unittest.TestCase):
    """PersonDetector should not crash when YOLO model is absent."""

    def test_missing_model_returns_empty(self):
        """detect() should return [] when _model is None (simulates load failure)."""
        import numpy as np
        detector = PersonDetector.__new__(PersonDetector)
        detector._model = None
        detector.confidence = 0.5
        detector.iou_threshold = 0.45
        detector.input_size = 640
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(blank)
        self.assertEqual(result, [])

    def test_detect_with_none_model(self):
        """detect() must return [] silently when _model is None."""
        import numpy as np
        det = PersonDetector.__new__(PersonDetector)
        det._model = None
        det.confidence = 0.5
        det.iou_threshold = 0.45
        det.input_size = 640
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(det.detect(blank), [])


class TestDetectorParsesResults(unittest.TestCase):
    """Test _parse_results filtering logic (mocked ultralytics output)."""

    def _make_detector(self):
        det = PersonDetector.__new__(PersonDetector)
        det._model = None
        det.confidence = 0.5
        det.iou_threshold = 0.45
        det.input_size = 640
        return det

    def _fake_box(self, cls_id, x1, y1, x2, y2, conf):
        import torch
        box = MagicMock()
        box.cls = [torch.tensor(float(cls_id))]
        box.conf = [torch.tensor(conf)]
        box.xyxy = [torch.tensor([float(x1), float(y1), float(x2), float(y2)])]
        return box

    def test_only_person_class_returned(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        detector = self._make_detector()

        person_box = self._fake_box(0, 10, 20, 110, 220, 0.9)
        car_box    = self._fake_box(2, 10, 20, 110, 220, 0.8)

        result_mock = MagicMock()
        result_mock.boxes = [person_box, car_box]

        detections = detector._parse_results([result_mock])
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.9)

    def test_empty_result(self):
        detector = self._make_detector()
        result_mock = MagicMock()
        result_mock.boxes = []
        detections = detector._parse_results([result_mock])
        self.assertEqual(detections, [])


if __name__ == "__main__":
    unittest.main()
