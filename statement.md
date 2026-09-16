# QueueVision AI – Project Statement

---

## 1. Problem Statement

Long queues are a persistent source of inefficiency and frustration in public services such as
banks, hospitals, railway stations, supermarkets, and government offices. Traditional queue
management relies on manual staff counting, token systems, or fixed-capacity estimates — none
of which provide **real-time, data-driven insight** into how many people are waiting, how
long they have been waiting, or how severe the congestion is.

This creates three core problems:

1. **No visibility**: Service managers cannot see queue severity in real time without
   physically walking to the queue.
2. **No proactivity**: Staff cannot open extra counters or alert supervisors before the
   queue reaches a critical state.
3. **No analytics**: There is no historical data to optimise staffing schedules or predict
   peak hours.

**QueueVision AI** addresses all three problems by applying Computer Vision to automatically
detect, track, and analyse queues from a camera feed — with zero manual intervention.

---

## 2. Scope of the Project

### In Scope

| Area | What is Covered |
|---|---|
| Input sources | Static images (JPG/PNG) and video files (MP4/AVI/MOV) |
| Detection | Person detection using a pre-trained YOLOv8n model |
| Tracking | Custom IoU + centroid multi-object tracker with persistent IDs |
| Queue analysis | ROI-based filtering, density estimation, status classification |
| Waiting time | Service-time learning from observed exits + rolling average estimation |
| Reporting | CSV time-series export, per-person tracking log, heatmap, text report |
| Configuration | Fully config-driven via config.yaml |
| Testing | Unit tests for detector, queue analyser, and waiting-time estimator |
| Documentation | README, architecture doc, UML diagrams, project statement |

### Out of Scope

- **Live RTSP / webcam streams** (a future enhancement)
- **GPU acceleration / TensorRT optimisation**
- **Person re-identification (ReID)** after prolonged occlusion
- **Web dashboard or REST API**
- **Multiple simultaneous queue zones**
- **Real-world metric calibration** (queue length in metres)

---

## 3. Target Users

| User Group | How They Benefit |
|---|---|
| **Service managers** (banks, hospitals, cafeterias) | Real-time queue status and waiting-time estimates to take proactive action |
| **Operations analysts** | Historical CSV data for staffing optimisation and peak-hour forecasting |
| **Security / surveillance teams** | Annotated video output for crowd-density monitoring |
| **Researchers / developers** | Modular, well-documented codebase as a starting point for advanced queue analysis |
| **College evaluators / academics** | A complete Computer Vision pipeline demonstrating all key CV concepts |

---

## 4. High-Level Features

### Module 1 – Person Detection (src/detector.py)
- Uses the **YOLOv8n** model (Ultralytics) to detect bounding boxes for the person class only.
- Applies configurable confidence and IoU thresholds to suppress false positives.
- Returns a typed Detection dataclass (bounding box, confidence, class, centroid).

### Module 2 – Multi-Object Tracking (src/tracker.py)
- Custom **IoU + Euclidean centroid** tracker — no external dependency (no SORT, DeepSORT).
- Assigns a **persistent unique ID** to each person across frames.
- Handles track birth (new person), track age-out (person left), and reports exited_tracks
  for downstream service-time learning.

### Module 3 – Queue Region Analysis (src/queue_analyzer.py)
- Filters tracked persons to only those whose bounding-box centre lies within a
  user-defined **Region of Interest (ROI)**.
- Computes **queue density** (normalised count per unit area) and classifies status as
  LOW, MEDIUM, HIGH, or CRITICAL.
- Produces a QueueSnapshot dataclass consumed by all downstream modules.

### Module 4 – Waiting-Time Estimation (src/waiting_time.py)
- Learns **service time** from persons observed exiting the ROI (data-driven, not hardcoded).
- Maintains a rolling window of the last 20 service observations.
- Estimates: wait = queue_count x avg_service_time, with clear labelling as an estimate.

### Module 5 – Analytics & Reporting (src/analytics.py)
- Records every QueueSnapshot as a time-series row.
- Exports queue_analysis.csv (frame-by-frame metrics) and person_tracking.csv
  (per-person dwell time).
- Generates a human-readable analysis_report.txt with peak queue, average wait, etc.

### Module 6 – Visualisation (src/visualization.py)
- Draws bounding boxes colour-coded by queue membership.
- Overlays a semi-transparent stats panel (count, density, status, wait time).
- Accumulates person positions into a **density heatmap** saved at the end of processing.

### Module 7 – Utilities & Configuration (src/utils.py, config.yaml)
- Loads config.yaml with sensible defaults and environment overrides.
- Provides ROI parsing, structured logging, and directory creation helpers.
- Keeps all magic numbers out of business logic.

---

## 5. Subject Relevance

This project is submitted for the **Computer Vision** course. The following core course concepts
are directly implemented:

| Course Concept | Implementation |
|---|---|
| Object Detection (YOLO / CNN) | src/detector.py – YOLOv8n inference |
| Non-Maximum Suppression | Built into YOLOv8 pipeline |
| Intersection-over-Union (IoU) | src/tracker.py – _iou() |
| Multi-Object Tracking | src/tracker.py – IoU + centroid matching |
| Region of Interest (ROI) | src/queue_analyzer.py |
| Density / Heatmap | src/visualization.py – save_heatmap() |
| Video frame processing | main.py – cv2.VideoCapture |
| Image annotation | src/visualization.py – OpenCV drawing |
