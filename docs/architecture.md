# QueueVision AI – System Architecture

## Overview

```
                        ┌─────────────────────────────────┐
                        │         CLI (main.py)            │
                        │  argparse  ·  config loader      │
                        └──────────────┬──────────────────┘
                                       │
                        ┌──────────────▼──────────────────┐
                        │     Frame Source                  │
                        │  cv2.VideoCapture  ·  imread     │
                        └──────────────┬──────────────────┘
                                       │ raw BGR frame
                        ┌──────────────▼──────────────────┐
                        │     PersonDetector               │
                        │  YOLOv8n (ultralytics)           │
                        │  class filter → person only      │
                        └──────────────┬──────────────────┘
                                       │ List[Detection]
                        ┌──────────────▼──────────────────┐
                        │     CentroidTracker              │
                        │  IoU matching  ·  persistent IDs │
                        │  age-out  ·  exited_tracks       │
                        └──────────────┬──────────────────┘
                                       │ List[Track]
               ┌───────────────────────┼───────────────────────┐
               │                       │                       │
  ┌────────────▼──────────┐ ┌──────────▼──────────┐ ┌────────▼────────────┐
  │   QueueAnalyzer       │ │ WaitingTimeEstimator │ │    QueueAnalytics   │
  │  ROI membership       │ │  service time learn  │ │  time-series        │
  │  density & status     │ │  wait estimation     │ │  peak/avg stats     │
  └────────────┬──────────┘ └──────────┬──────────┘ └────────┬────────────┘
               │                       │                       │
               └────────────┬──────────┘                       │
                             │ QueueSnapshot                    │
                  ┌──────────▼──────────┐                       │
                  │    Visualizer       │                       │
                  │  bbox draw          │                       │
                  │  ROI draw           │                       │
                  │  stats panel        │                       │
                  │  heatmap accum.     │                       │
                  └──────────┬──────────┘                       │
                             │ annotated frame                  │
                  ┌──────────▼──────────┐          ┌───────────▼──────────┐
                  │  VideoWriter /      │          │  CSV + Report Writer  │
                  │  imwrite            │          │  queue_analysis.csv   │
                  └─────────────────────┘          │  person_tracking.csv  │
                                                   │  analysis_report.txt  │
                                                   │  queue_heatmap.png    │
                                                   └──────────────────────┘
```

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `main.py` | CLI entry point, pipeline orchestration |
| `src/detector.py` | YOLOv8 inference, Detection dataclass |
| `src/tracker.py` | IoU+centroid tracking, Track dataclass |
| `src/queue_analyzer.py` | ROI filtering, density, status classification |
| `src/waiting_time.py` | Service time learning, wait estimation |
| `src/analytics.py` | Aggregate stats, CSV/report export |
| `src/visualization.py` | Bounding boxes, panels, heatmap |
| `src/utils.py` | Config loading, ROI parsing, logging |
| `setup_roi.py` | Interactive OpenCV ROI selector |
| `config.yaml` | All configurable parameters |

## Computer Vision Concepts Used

| Concept | Where Used |
|---|---|
| Object Detection (YOLO) | `detector.py` |
| Non-Maximum Suppression | Inside YOLOv8 |
| Multi-Object Tracking | `tracker.py` (IoU + centroid) |
| Region-of-Interest (ROI) | `queue_analyzer.py` |
| Intersection-over-Union (IoU) | `tracker.py` – `_iou()` |
| Heatmap / density map | `visualization.py` – `save_heatmap()` |
| Video frame extraction | `main.py` – `cv2.VideoCapture` |
| Image annotation | `visualization.py` |

## Data Flow (Video Mode)

```
Frame N
  → YOLOv8 → detections[]
  → Tracker.update() → tracks[] + exited_tracks[]
  → WaitEst.observe_exits(exited_tracks)
  → QueueAnalyzer.analyse(tracks) → QueueSnapshot
  → Snapshot enriched with wait/service data
  → Analytics.record(snapshot)
  → Visualizer.annotate_frame(frame, tracks, snapshot)
  → VideoWriter.write(annotated_frame)

End of video:
  → Visualizer.save_heatmap()
  → Analytics.save_queue_csv()
  → Analytics.save_tracking_csv()
  → Analytics.save_report()
```

## Tracking Algorithm (IoU + Centroid)

```
Given: existing tracks T, new detections D

1. Build cost matrix C[|T|×|D|] using IoU(T_i.bbox, D_j.bbox)
2. Sort all (i,j) pairs by descending IoU
3. Greedily assign pairs with IoU ≥ threshold
4. For unmatched pairs, fallback: match by Euclidean centroid distance ≤ max_distance
5. Unmatched detections → new tracks
6. Unmatched tracks → increment time_since_update
7. Tracks with time_since_update > max_age → remove → exited_tracks
```

## Waiting Time Model

```
Estimated Wait = N_queue × T_service

Where:
  N_queue   = current people in ROI
  T_service = rolling average of observed service times
            = fallback to config.yaml default_service_time when
              no observations are available yet

Observed service time per person:
  T_service_i = (exit_frame - entry_frame) / FPS / 60
  Clamped to [min_service_time, max_service_time]
```
