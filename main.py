"""
main.py – QueueVision AI command-line entry point.

Usage examples:
    python main.py --image data/sample_images/queue.jpg
    python main.py --video data/sample_videos/queue.mp4
    python main.py --video data/sample_videos/queue.mp4 --roi 100,100,800,600
    python main.py --image data/sample_images/queue.jpg --roi 50,50,900,650 --output results/
    python main.py --demo   (generates a synthetic test frame – no input file needed)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.analytics import QueueAnalytics
from src.detector import PersonDetector
from src.queue_analyzer import QueueAnalyzer
from src.tracker import CentroidTracker
from src.utils import (
    ensure_dir,
    load_config,
    parse_roi,
    roi_from_config,
    setup_logger,
    validate_input_file,
)
from src.visualization import Visualizer
from src.waiting_time import WaitingTimeEstimator

logger = setup_logger("Main")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="QueueVision AI",
        description=(
            "Intelligent Queue and Waiting-Time Analyzer\n"
            "  Detects and tracks people in queue regions from images or video.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --demo\n"
            "  python main.py --image data/sample_images/queue.jpg\n"
            "  python main.py --video data/sample_videos/queue.mp4\n"
            "  python main.py --video data/sample_videos/queue.mp4 --roi 100,100,800,600\n"
            "  python main.py --image data/sample_images/queue.jpg --confidence 0.4\n"
        ),
    )

    # ── Input ─────────────────────────────────────────────────────────── #
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--video",  type=str, help="Path to input video file.")
    input_group.add_argument("--image",  type=str, help="Path to input image file.")
    input_group.add_argument(
        "--demo",
        action="store_true",
        help="Run a synthetic demo without any real input file.",
    )

    # ── Queue region ──────────────────────────────────────────────────── #
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        metavar="x1,y1,x2,y2",
        help="Queue region-of-interest in pixels, e.g. 100,100,800,600.",
    )

    # ── Model / detection ─────────────────────────────────────────────── #
    parser.add_argument("--model",      type=str,   default=None, help="Path to YOLO .pt weights file.")
    parser.add_argument("--confidence", type=float, default=None, help="Detection confidence threshold (0–1).")

    # ── Config & output ───────────────────────────────────────────────── #
    parser.add_argument("--config",  type=str, default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--output",  type=str, default=None, help="Directory for output files.")

    return parser


# ══════════════════════════════════════════════════════════════════════════
# Image pipeline
# ══════════════════════════════════════════════════════════════════════════

def run_image(args: argparse.Namespace, cfg: dict) -> None:
    """Process a single image."""
    frame = cv2.imread(str(validate_input_file(args.image, "Image")))
    if frame is None:
        logger.error("Could not read image '%s'. Is it a valid image file?", args.image)
        sys.exit(1)

    _process_frame_pipeline(frame, args, cfg, source_name=Path(args.image).stem)


def run_demo(args: argparse.Namespace, cfg: dict) -> None:
    """Generate and process a synthetic demo frame."""
    logger.info("Running in DEMO mode – generating synthetic queue scene.")
    frame = _create_demo_frame(width=1280, height=720, n_people=8)
    _process_frame_pipeline(frame, args, cfg, source_name="demo")


def _process_frame_pipeline(
    frame: np.ndarray,
    args: argparse.Namespace,
    cfg: dict,
    source_name: str = "output",
) -> None:
    """Shared single-frame processing pipeline."""
    h, w = frame.shape[:2]

    # Resolve ROI
    roi = _resolve_roi(args, cfg, w, h)

    # Build components
    det_cfg = cfg["detection"]
    detector = PersonDetector(
        model_path=args.model or det_cfg["model_path"],
        confidence=args.confidence or det_cfg["confidence"],
        iou_threshold=det_cfg["iou_threshold"],
        input_size=det_cfg["input_size"],
    )
    # For single images, min_hits=1 so every detection is immediately confirmed
    tracking_cfg = {**cfg["tracking"], "min_hits": 1}
    tracker = CentroidTracker(**tracking_cfg)
    analyzer = QueueAnalyzer(
        roi=roi,
        density_thresholds=cfg["queue"]["density_thresholds"],
        person_area_estimate=cfg["queue"]["person_area_estimate"],
        frame_width=w,
        frame_height=h,
    )
    wait_est = WaitingTimeEstimator(**cfg["waiting_time"], fps=25.0)
    analytics = QueueAnalytics(fps=25.0, **_analytics_dirs(args, cfg))
    viz = Visualizer(frame_width=w, frame_height=h, panel_alpha=cfg["visualization"]["panel_alpha"])

    # ── Detect & Track ────────────────────────────────────────────────── #
    detections = detector.detect(frame)
    logger.info("Detected %d person(s).", len(detections))

    tracks = tracker.update(detections)

    # ── Analyse queue ─────────────────────────────────────────────────── #
    snapshot = analyzer.analyse(tracks, frame_number=1, timestamp_sec=0.0)
    snapshot.estimated_wait_min = wait_est.estimate_wait(snapshot.queue_count)
    snapshot.average_wait_min   = snapshot.estimated_wait_min
    snapshot.service_rate       = wait_est.service_rate()

    analytics.record(snapshot)

    # ── Annotate ──────────────────────────────────────────────────────── #
    annotated = viz.annotate_frame(frame, tracks, snapshot)

    # ── Save output ───────────────────────────────────────────────────── #
    results_dir = ensure_dir(args.output or cfg["output"]["results_dir"])
    out_image_path = results_dir / f"{source_name}_annotated.jpg"
    cv2.imwrite(str(out_image_path), annotated)
    logger.info("Annotated image saved → %s", out_image_path)

    # Heatmap
    if cfg["output"]["save_heatmap"]:
        viz.save_heatmap(results_dir / "queue_heatmap.png")

    # CSVs
    all_tracks = tracker.get_all_tracks() + tracker.exited_tracks
    analytics.save_queue_csv()
    analytics.save_tracking_csv(all_tracks)

    # Report
    reports_dir = ensure_dir(cfg["output"]["reports_dir"])
    analytics.results_dir = results_dir
    analytics.reports_dir = reports_dir
    analytics.save_report(
        all_tracks,
        people_served=wait_est.people_served,
        avg_service_time=wait_est.average_service_time(),
    )

    # ── Print summary ─────────────────────────────────────────────────── #
    _print_summary(snapshot, wait_est, analytics, len(all_tracks))


# ══════════════════════════════════════════════════════════════════════════
# Video pipeline
# ══════════════════════════════════════════════════════════════════════════

def run_video(args: argparse.Namespace, cfg: dict) -> None:
    """Process a video file frame by frame."""
    video_path = validate_input_file(args.video, "Video")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.error(
            "Cannot open video '%s'. Ensure the file is a valid video format.", video_path
        )
        sys.exit(1)

    # Video properties
    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_fps = cfg["video"]["target_fps"] or src_fps
    max_frames = cfg["video"]["max_frames"]

    logger.info(
        "Video: %s  |  %dx%d  |  %.1f fps  |  %d frames",
        video_path.name, w, h, src_fps, total_frm,
    )

    # Resolve ROI
    roi = _resolve_roi(args, cfg, w, h)

    # Build components
    det_cfg = cfg["detection"]
    detector = PersonDetector(
        model_path=args.model or det_cfg["model_path"],
        confidence=args.confidence or det_cfg["confidence"],
        iou_threshold=det_cfg["iou_threshold"],
        input_size=det_cfg["input_size"],
    )
    tracker    = CentroidTracker(**cfg["tracking"])
    analyzer   = QueueAnalyzer(
        roi=roi,
        density_thresholds=cfg["queue"]["density_thresholds"],
        person_area_estimate=cfg["queue"]["person_area_estimate"],
        frame_width=w,
        frame_height=h,
    )
    wait_est   = WaitingTimeEstimator(**cfg["waiting_time"], fps=src_fps)
    analytics  = QueueAnalytics(
        fps=src_fps,
        time_series_interval=cfg["analytics"]["time_series_interval"],
        **_analytics_dirs(args, cfg),
    )
    viz        = Visualizer(frame_width=w, frame_height=h, panel_alpha=cfg["visualization"]["panel_alpha"])

    # Output video writer
    results_dir = ensure_dir(args.output or cfg["output"]["results_dir"])
    out_video_path = results_dir / f"{video_path.stem}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*cfg["output"]["video_codec"])
    writer = cv2.VideoWriter(str(out_video_path), fourcc, target_fps, (w, h))

    # ── Frame loop ────────────────────────────────────────────────────── #
    frame_idx = 0
    processed = 0
    skip_ratio = max(1, int(round(src_fps / target_fps)))

    logger.info("Processing video…  (press Ctrl+C to abort)")
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if max_frames and processed >= max_frames:
                logger.info("Reached max_frames limit (%d).", max_frames)
                break

            # Optionally skip frames to hit target_fps
            if (frame_idx - 1) % skip_ratio != 0:
                continue

            processed += 1
            timestamp_sec = frame_idx / src_fps

            # Detect
            detections = detector.detect(frame)

            # Track
            tracks = tracker.update(detections)

            # Observe exits for service time learning
            wait_est.observe_exits(tracker.exited_tracks, frame_idx)

            # Analyse
            snapshot = analyzer.analyse(tracks, frame_number=frame_idx, timestamp_sec=timestamp_sec)
            snapshot.estimated_wait_min = wait_est.estimate_wait(snapshot.queue_count)
            snapshot.average_wait_min   = analytics.average_wait_min or snapshot.estimated_wait_min
            snapshot.service_rate       = wait_est.service_rate()

            analytics.record(snapshot)

            # Annotate & write
            annotated = viz.annotate_frame(frame, tracks, snapshot)
            writer.write(annotated)

            # Progress log every 50 frames
            if processed % 50 == 0:
                elapsed = time.time() - start_time
                pct = (frame_idx / total_frm * 100) if total_frm > 0 else 0
                logger.info(
                    "Frame %d/%d (%.0f%%) | Queue: %d | Status: %s | %.1fs elapsed",
                    frame_idx, total_frm, pct,
                    snapshot.queue_count, snapshot.queue_status, elapsed,
                )

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user.")
    finally:
        cap.release()
        writer.release()

    elapsed_total = time.time() - start_time
    logger.info(
        "Done. %d frames processed in %.1fs. Output: %s",
        processed, elapsed_total, out_video_path,
    )

    # ── Post-processing outputs ───────────────────────────────────────── #
    if cfg["output"]["save_heatmap"]:
        viz.save_heatmap(results_dir / "queue_heatmap.png")

    all_tracks = tracker.get_all_tracks() + tracker.exited_tracks
    analytics.results_dir = results_dir
    analytics.reports_dir = ensure_dir(cfg["output"]["reports_dir"])
    analytics.save_queue_csv()
    analytics.save_tracking_csv(all_tracks)
    analytics.save_report(
        all_tracks,
        people_served=wait_est.people_served,
        avg_service_time=wait_est.average_service_time(),
    )

    _print_summary(analytics.final_snapshot, wait_est, analytics, len(all_tracks))


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _resolve_roi(args, cfg, frame_w, frame_h):
    """Return ROI tuple or None, merging CLI arg > config > None."""
    if args.roi:
        try:
            roi = parse_roi(args.roi)
            logger.info("Using CLI ROI: %s", roi)
            return roi
        except ValueError as exc:
            logger.error("%s", exc)
            sys.exit(1)
    roi = roi_from_config(cfg)
    if roi:
        logger.info("Using config ROI: %s", roi)
    else:
        logger.info("No ROI configured – treating full frame as queue region.")
    return roi


def _analytics_dirs(args, cfg) -> dict:
    base = args.output or None
    return {
        "results_dir": base or cfg["output"]["results_dir"],
        "reports_dir": cfg["output"]["reports_dir"],
    }


def _print_summary(snapshot, wait_est, analytics, n_tracks):
    """Print a rich terminal summary."""
    lines = [
        "",
        "╔══════════════════════════════════════════════╗",
        "║         QueueVision AI – Run Summary          ║",
        "╠══════════════════════════════════════════════╣",
    ]
    if snapshot:
        density_str = f"{snapshot.queue_density:.1%}"
        wait_str    = f"{snapshot.estimated_wait_min:.1f} min (ESTIMATE)"
        lines += [
            f"║  Queue count      : {snapshot.queue_count:<26}║",
            f"║  Queue status     : {snapshot.queue_status:<26}║",
            f"║  Queue density    : {density_str:<26}║",
            f"║  Est. wait time   : {wait_str:<26}║",
        ]
    avg_q_str  = f"{analytics.average_queue_size:.1f}"
    avg_w_str  = f"{analytics.average_wait_min:.1f} min"
    avg_s_str  = f"{wait_est.average_service_time():.2f} min/person"
    lines += [
        f"║  Peak queue size  : {analytics.peak_queue_size:<26}║",
        f"║  Avg queue size   : {avg_q_str:<26}║",
        f"║  Avg wait time    : {avg_w_str:<26}║",
        f"║  People tracked   : {n_tracks:<26}║",
        f"║  People served    : {wait_est.people_served:<26}║",
        f"║  Avg service time : {avg_s_str:<26}║",
        "╚══════════════════════════════════════════════╝",
        "",
        "  Outputs → results/   reports/",
        "",
    ]
    print("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════
# Demo frame generator
# ══════════════════════════════════════════════════════════════════════════

def _create_demo_frame(width: int = 1280, height: int = 720, n_people: int = 8) -> np.ndarray:
    """
    Synthesise a simple demo frame with coloured rectangles representing people.
    Used when --demo flag is set; no real input file is required.
    """
    # Background: a simple corridor / queue hall
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)  # dark grey background

    # Draw floor lines
    for y in range(0, height, 60):
        cv2.line(frame, (0, y), (width, y), (50, 50, 50), 1)

    # Draw people (coloured rectangles standing in a line)
    colours = [
        (180, 100, 60), (100, 180, 60), (60, 100, 180),
        (200, 60, 100), (60, 200, 200), (180, 180, 60),
        (140, 60, 200), (60, 180, 140),
    ]
    spacing = width // (n_people + 1)
    p_w, p_h = 60, 120
    queue_y_center = height // 2

    for i in range(n_people):
        cx = spacing * (i + 1)
        cy = queue_y_center
        x1 = cx - p_w // 2
        y1 = cy - p_h // 2
        x2 = cx + p_w // 2
        y2 = cy + p_h // 2
        color = colours[i % len(colours)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        # Head
        cv2.circle(frame, (cx, y1 - 20), 20, color, -1)

    # Watermark
    cv2.putText(
        frame,
        "QueueVision AI – Synthetic Demo Frame",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA,
    )
    return frame


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not (args.video or args.image or args.demo):
        parser.print_help()
        print("\nError: please supply --video, --image, or --demo.\n")
        sys.exit(1)

    # Load config
    cfg = load_config(args.config)

    # Dispatch
    if args.demo:
        run_demo(args, cfg)
    elif args.image:
        run_image(args, cfg)
    else:
        run_video(args, cfg)


if __name__ == "__main__":
    main()
