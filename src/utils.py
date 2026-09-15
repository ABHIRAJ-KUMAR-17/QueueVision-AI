"""
utils.py – Utility helpers for QueueVision AI.

Responsibilities:
  - Load and merge YAML configuration
  - Parse / validate ROI from CLI string
  - Logging helpers
  - Graceful error messages
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import yaml


# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────

def setup_logger(name: str = "QueueVision", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with a human-readable format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logger()


# ─────────────────────────────────────────────
# Configuration loading
# ─────────────────────────────────────────────

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(config_path: Optional[str | Path] = None) -> dict:
    """
    Load YAML configuration from *config_path*.

    Falls back to the project-level ``config.yaml`` when *config_path* is None.
    Returns a Python dict. Raises a helpful error if the file is missing or malformed.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.warning(
            "Config file not found at '%s'. Using built-in defaults.", path
        )
        return _default_config()

    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        if cfg is None:
            cfg = {}
        logger.info("Loaded configuration from '%s'", path)
        return _deep_merge(_default_config(), cfg)
    except yaml.YAMLError as exc:
        logger.error("Failed to parse config file '%s': %s", path, exc)
        logger.warning("Falling back to built-in defaults.")
        return _default_config()


def _default_config() -> dict:
    """Return a safe built-in default configuration."""
    return {
        "detection": {
            "model_path": "models/yolov8n.pt",
            "confidence": 0.5,
            "iou_threshold": 0.45,
            "input_size": 640,
        },
        "tracking": {
            "max_age": 30,
            "min_hits": 2,
            "iou_threshold": 0.3,
            "max_distance": 100,
        },
        "queue": {
            "roi": None,
            "density_thresholds": {"low": 0.30, "medium": 0.60, "high": 0.80},
            "person_area_estimate": 5000,
        },
        "waiting_time": {
            "default_service_time": 2.5,
            "min_service_time": 0.5,
            "max_service_time": 10.0,
        },
        "analytics": {"time_series_interval": 30},
        "video": {"target_fps": None, "max_frames": None},
        "output": {
            "save_annotated_video": True,
            "save_heatmap": True,
            "results_dir": "results",
            "reports_dir": "reports",
            "video_codec": "mp4v",
        },
        "visualization": {
            "bbox_color_in_queue": [0, 255, 0],
            "bbox_color_out_queue": [200, 200, 200],
            "roi_color": [0, 200, 255],
            "text_color": [255, 255, 255],
            "panel_alpha": 0.6,
        },
    }


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


# ─────────────────────────────────────────────
# ROI helpers
# ─────────────────────────────────────────────

def parse_roi(roi_string: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """
    Parse an ROI string ``"x1,y1,x2,y2"`` into a tuple of ints.

    Returns None when *roi_string* is None or empty.
    Raises ValueError with a helpful message on bad input.
    """
    if not roi_string:
        return None
    try:
        parts = [int(v.strip()) for v in roi_string.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Expected 4 values, got {len(parts)}")
        x1, y1, x2, y2 = parts
        if x1 >= x2 or y1 >= y2:
            raise ValueError(
                f"ROI must satisfy x1 < x2 and y1 < y2 (got {parts})"
            )
        return (x1, y1, x2, y2)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid ROI '{roi_string}'. Expected format: x1,y1,x2,y2 "
            f"(e.g. 100,100,800,600). Error: {exc}"
        ) from exc


def roi_from_config(cfg: dict) -> Optional[Tuple[int, int, int, int]]:
    """Extract ROI from a loaded config dict. Returns None if not set."""
    raw = cfg.get("queue", {}).get("roi")
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return tuple(int(v) for v in raw)  # type: ignore[return-value]
    if isinstance(raw, str):
        return parse_roi(raw)
    return None


def save_roi_to_config(roi: Tuple[int, int, int, int],
                       config_path: Optional[str | Path] = None) -> None:
    """Persist an ROI back to the YAML config file."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    cfg = load_config(path)
    cfg.setdefault("queue", {})["roi"] = list(roi)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False)
    logger.info("ROI %s saved to '%s'", roi, path)


# ─────────────────────────────────────────────
# General helpers
# ─────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if it does not already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_input_file(path: str, label: str = "Input file") -> Path:
    """
    Check that *path* exists and is a file.
    Exits with a clear error message if not.
    """
    p = Path(path)
    if not p.exists():
        logger.error("%s '%s' does not exist.", label, path)
        sys.exit(1)
    if not p.is_file():
        logger.error("%s '%s' is not a file.", label, path)
        sys.exit(1)
    return p


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def seconds_to_hms(seconds: float) -> str:
    """Convert seconds to a human-readable H:MM:SS string."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"
