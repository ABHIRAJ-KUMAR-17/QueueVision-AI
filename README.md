# QueueVision AI “ Intelligent Queue and Waiting-Time Analyzer

> **Computer Vision course evaluation project**
> Analyzes images and videos of real-world queues using YOLOv8 person detection,
> custom multi-object tracking, ROI-based queue filtering, and waiting-time estimation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Features](#4-features)
5. [Non-Functional Requirements](#5-non-functional-requirements)
6. [Computer Vision Concepts Used](#6-computer-vision-concepts-used)
7. [System Architecture](#7-system-architecture)
8. [Installation](#8-installation)
9. [Python Version](#9-python-version)
10. [Virtual Environment Setup](#10-virtual-environment-setup)
11. [Dependency Installation](#11-dependency-installation)
12. [Model Setup](#12-model-setup)
13. [How to Run](#13-how-to-run)
14. [CLI Arguments](#14-cli-arguments)
15. [Example Commands](#15-example-commands)
16. [Input Format](#16-input-format)
17. [Output Format](#17-output-format)
18. [Queue Detection Methodology](#18-queue-detection-methodology)
19. [Waiting-Time Estimation Methodology](#19-waiting-time-estimation-methodology)
20. [Testing](#20-testing)
21. [Documentation & Design Diagrams](#21-documentation--design-diagrams)
22. [Limitations](#22-limitations)
23. [Future Improvements](#23-future-improvements)
24. [Project Structure](#24-project-structure)

---

## 1. Project Overview

**QueueVision AI** is a complete, end-to-end Computer Vision pipeline that ingests
a video or image of a real-world queue (bank, hospital, cafeteria, etc.) and outputs:

- Number of people in the queue
- Queue density and status (LOW / MEDIUM / HIGH / CRITICAL)
- Estimated and average waiting time *(clearly labelled as estimates)*
- Annotated output video with bounding boxes and live stats overlay
- CSV reports and a human-readable analysis report
- Queue density heatmap

Everything runs from the command line â€” no GUI IDE is required.

---

## 2. Problem Statement

Long queues are a daily source of frustration and inefficiency in public services.
Traditional queue management relies on manual counting or ticket systems that provide
no real-time insight.

**QueueVision AI solves this by:**
- Automatically detecting and counting people from a camera feed
- Tracking individuals persistently so they are never double-counted
- Estimating waiting time from observed service rates
- Classifying queue severity so operators can take proactive action

---

## 3. Objectives

- Detect every person in a video/image using a modern object-detection model
- Track each person across frames using a persistent unique ID
- Restrict counting to a user-defined **queue region** (ROI)
- Estimate waiting time using a data-driven, explainable algorithm
- Classify queue status (LOW / MEDIUM / HIGH / CRITICAL)
- Generate structured reports (CSV + text) for further analysis
- Produce an annotated output video for visual verification

---

## 4. Features

| Feature | Detail |
|---|---|
| Person detection | YOLOv8n, person class only |
| Multi-object tracking | Custom IoU + centroid tracker (persistent IDs) |
| ROI-based queue filtering | CLI argument or interactive selector |
| Queue density calculation | Normalised, configurable thresholds |
| Waiting-time estimation | Learned from observed service events |
| Live stats overlay | Semi-transparent panel on annotated video |
| CSV output | `queue_analysis.csv`, `person_tracking.csv` |
| Text report | `reports/analysis_report.txt` |
| Heatmap | `results/queue_heatmap.png` |
| Demo mode | `--demo` flag runs without any input file |
| Config-driven | All parameters in `config.yaml` |
| Unit tests | 20+ tests covering all major modules |

---

## 5. Non-Functional Requirements

| NFR | Description |
|---|---|
| **Performance** | â‰¥5 FPS on CPU for 720p input with YOLOv8n; frame skipping configurable |
| **Reliability** | All errors logged; no silent crashes; invalid inputs reported clearly |
| **Usability** | Full CLI interface; `--demo` for zero-setup run; `--help` on all commands |
| **Maintainability** | 7 decoupled modules with typed interfaces; docstrings on all public APIs |
| **Scalability** | Drop-in support for larger YOLO models (yolov8s/m/l/x) via `config.yaml` |
| **Security/Privacy** | 100% local processing; no cloud API calls; no images stored |
| **Testability** | 20+ unit tests; no real video required; `pytest` with shared fixtures |
| **Resource Efficiency** | Bounded rolling window; fixed heatmap buffer; tracks cleared on age-out |

See [`docs/requirements.md`](docs/requirements.md) for full NFR specifications.

---

## 6. Computer Vision Concepts Used

| Concept | Description | Module |
|---|---|---|
| **Object Detection** | YOLOv8 detects bounding boxes + confidence scores | `detector.py` |
| **Non-Maximum Suppression** | Removes duplicate detections | Built into YOLOv8 |
| **Multi-Object Tracking** | Associates detections across frames by IoU + centroid distance | `tracker.py` |
| **Region of Interest (ROI)** | Rectangular mask that defines the queue area | `queue_analyzer.py` |
| **Intersection-over-Union** | Geometric overlap metric used for track matching | `tracker.py` |
| **Density Estimation** | Normalises people count by ROI area | `queue_analyzer.py` |
| **Temporal Analysis** | Time-series of queue metrics over video duration | `analytics.py` |
| **Heatmap Generation** | Accumulates person positions into a colour density map | `visualization.py` |
| **Frame Extraction** | `cv2.VideoCapture` reads frames at configurable FPS | `main.py` |
| **Image Annotation** | Draws boxes, text, panels using OpenCV primitives | `visualization.py` |

---

## 7. System Architecture

```
Input (Image/Video)
      â”‚
      â–¼
Person Detection (YOLOv8n)
      â”‚  List[Detection]
      â–¼
Multi-Object Tracker (IoU + Centroid)
      â”‚  List[Track]
      â–¼
Queue Region Analyzer (ROI filter)
      â”‚  QueueSnapshot
      â–¼
Waiting-Time Estimator
      â”‚  estimated_wait_min
      â–¼
Analytics Recorder
      â”‚
      â”œâ”€â”€â†’ Visualizer â†’ Annotated frame â†’ output video
      â”‚
      â””â”€â”€â†’ CSV + Report Generator
                 â”‚
                 â”œâ”€â”€ results/queue_analysis.csv
                 â”œâ”€â”€ results/person_tracking.csv
                 â”œâ”€â”€ results/queue_heatmap.png
                 â””â”€â”€ reports/analysis_report.txt
```

See [`docs/architecture.md`](docs/architecture.md) for a detailed diagram.

---

## 8. Installation

### Prerequisites

- Python 3.9 or higher
- pip
- (Optional) A CUDA-capable GPU for faster inference

### Clone the repository

```bash
git clone https://github.com/ABHIRAJ-KUMAR-17/QueueVision-AI.git
cd QueueVision-AI
```

---

## 9. Python Version

This project requires **Python 3.9+**.

Check your version:

```bash
python --version
```

---

## 10. Virtual Environment Setup

It is strongly recommended to use a virtual environment.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 11. Dependency Installation

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---|---|
| `opencv-python` | Video/image I/O, drawing, heatmap |
| `ultralytics` | YOLOv8 model and inference |
| `numpy` | Array operations |
| `pandas` | (used internally by ultralytics) |
| `PyYAML` | Configuration file parsing |
| `matplotlib` | (optional, heatmap colour maps via OpenCV) |
| `scipy` | (available for signal processing extensions) |

---

## 12. Model Setup

**No manual download is required.**

YOLOv8n weights (`yolov8n.pt`) are **automatically downloaded** by the
`ultralytics` library on the first run. They are cached locally
(~6 MB, CPU-compatible).

If you want to pre-download the model:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

The downloaded file will be placed in `~/.cache/ultralytics/` or the local
`models/` directory depending on your environment.

You can use a larger, more accurate model by changing `config.yaml`:
```yaml
detection:
  model_path: "yolov8s.pt"   # or yolov8m.pt, yolov8l.pt, yolov8x.pt
```

---

## 13. How to Run

### Quick demo (no input file needed)

```bash
python main.py --demo
```

### Process a single image

```bash
python main.py --image data/sample_images/queue.jpg
```

### Process a video

```bash
python main.py --video data/sample_videos/queue.mp4
```

### Process with an explicit queue ROI

```bash
python main.py --video data/sample_videos/queue.mp4 --roi 100,100,800,600
```

### Select ROI interactively (before analysis)

```bash
python setup_roi.py --video data/sample_videos/queue.mp4
# â†’ drag to draw rectangle â†’ press ENTER â†’ ROI saved to config.yaml
python main.py --video data/sample_videos/queue.mp4
```

### Run unit tests

```bash
python -m pytest tests/ -v
```

---

## 14. CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--video` | str | â€” | Path to input video file |
| `--image` | str | â€” | Path to input image file |
| `--demo` | flag | â€” | Run synthetic demo without input |
| `--roi` | str | config | Queue ROI: `x1,y1,x2,y2` in pixels |
| `--model` | str | config | Path to YOLO `.pt` weights |
| `--confidence` | float | config | Detection confidence threshold (0â€“1) |
| `--config` | str | `config.yaml` | Path to configuration file |
| `--output` | str | `results/` | Directory for output files |

`--video`, `--image`, and `--demo` are mutually exclusive. You must provide exactly one.

---

## 15. Example Commands

```bash
# Demo mode
python main.py --demo

# Image analysis with default ROI
python main.py --image data/sample_images/queue.jpg

# Image with custom ROI
python main.py --image data/sample_images/queue.jpg --roi 50,50,900,650

# Video analysis
python main.py --video data/sample_videos/queue.mp4

# Video with custom ROI and lower confidence threshold
python main.py --video data/sample_videos/queue.mp4 --roi 100,100,800,600 --confidence 0.4

# Custom output directory
python main.py --video data/sample_videos/queue.mp4 --output my_results/

# Use a different YOLO model
python main.py --video data/sample_videos/queue.mp4 --model models/yolov8s.pt

# Custom config
python main.py --video data/sample_videos/queue.mp4 --config my_config.yaml
```

---

## 16. Input Format

| Type | Supported Formats |
|---|---|
| Image | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff` |
| Video | `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` |

### Recommended input characteristics

- Resolution: 720p or 1080p
- Queue region clearly visible in frame
- Camera angle: overhead or 45Â° diagonal preferred
- Adequate lighting (avoid heavy shadows or overexposure)

---

## 17. Output Format

All outputs are written to `results/` and `reports/` (configurable via `--output`).

| File | Description |
|---|---|
| `results/<name>_annotated.jpg/.mp4` | Annotated image or video |
| `results/queue_analysis.csv` | Time-series queue metrics |
| `results/person_tracking.csv` | Per-person tracking data |
| `results/queue_heatmap.png` | Queue density heatmap |
| `reports/analysis_report.txt` | Human-readable summary report |

### queue_analysis.csv columns

```
timestamp, frame_number, queue_count, total_detected,
queue_density, queue_status, estimated_wait_time_min,
average_wait_time_min, people_served, service_rate, queue_length_px
```

### person_tracking.csv columns

```
track_id, entry_frame, exit_frame, frames_seen,
time_in_queue_frames, time_in_queue_sec, was_in_queue
```

---

## 18. Queue Detection Methodology

### Step 1 â€“ Person Detection
YOLOv8n runs on each frame and returns bounding boxes for the `person` class only.
Confidence and IoU thresholds filter out low-quality detections.

### Step 2 â€“ Person Tracking
Each detection is matched to an existing track using:
1. **IoU** (Intersection-over-Union) between bounding boxes
2. **Euclidean centroid distance** as a fallback when IoU is too low

Unmatched detections create new tracks with new IDs.
Tracks that go unmatched for `max_age` frames are removed.

### Step 3 â€“ ROI Filtering
A person is counted as **in the queue** only if their bounding-box centre point
lies within the configured ROI rectangle.
People outside the ROI (passers-by, staff) are detected and tracked but **not counted**.

### Step 4 â€“ Density & Status
```
queue_density = (queue_count Ã— person_area_estimate) / roi_area
```
Clamped to [0, 1].

| Density Range | Status |
|---|---|
| 0â€“30% | LOW |
| 30â€“60% | MEDIUM |
| 60â€“80% | HIGH |
| 80â€“100% | CRITICAL |

Thresholds are configurable in `config.yaml`.

---

## 19. Waiting-Time Estimation Methodology

> **IMPORTANT**: All waiting-time values are **ESTIMATES**. They are based on
> observations of people leaving the queue and statistical averaging.
> They do not represent exact real-world waiting times.

### Algorithm

```
Estimated Wait (min) = N_queue Ã— T_service_avg

N_queue        = current number of people in ROI
T_service_avg  = rolling average of observed service times
```

### Service Time Learning

When a tracked person is observed to **leave the queue ROI**, their time spent
inside is calculated:

```
T_service_i = (exit_frame - entry_frame) / FPS / 60   [minutes]
```

This value is clamped to `[min_service_time, max_service_time]` to reject outliers,
then added to a rolling window (last 20 observations).

When no observations are available yet, the system falls back to
`default_service_time` from `config.yaml` (default: 2.5 min/person).

### Example

```
Queue: 10 people
Avg observed service time: 2.5 min/person
Estimated wait: 10 Ã— 2.5 = 25 minutes  [ESTIMATE]
```

---

## 20. Testing

### Run all unit tests

```bash
python -m pytest tests/ -v
```

### Run a specific test file

```bash
python -m pytest tests/test_detector.py -v
python -m pytest tests/test_queue.py -v
python -m pytest tests/test_waiting_time.py -v
```

### Test coverage summary

| Test File | What is Tested |
|---|---|
| `tests/test_detector.py` | Detection dataclass, confidence filtering, centroid calculation, edge cases |
| `tests/test_queue.py` | ROI membership, density computation, status classification, `parse_roi` |
| `tests/test_waiting_time.py` | Wait estimation, service-time learning, rolling window, clamping |

> Tests use synthetic data only â€” no real video or image files required.

---

## 21. Documentation & Design Diagrams

| Document | Path | Contents |
|---|---|---|
| Project Statement | [`statement.md`](statement.md) | Problem statement, scope, target users, high-level features |
| System Architecture | [`docs/architecture.md`](docs/architecture.md) | Detailed architecture diagram, module responsibilities, data flow |
| UML Diagrams | [`docs/uml_diagrams.md`](docs/uml_diagrams.md) | Use Case, Class, Sequence, Component, ER diagrams |
| Requirements Spec | [`docs/requirements.md`](docs/requirements.md) | Full FR and NFR specifications |

---

## 22. Limitations

- **Overhead camera recommended**: The system assumes people are visible from
  above or at a strong diagonal angle. Front-facing cameras cause occlusion issues.
- **Waiting time is an estimate**: Real service times depend on transaction complexity,
  which cannot be known from video alone.
- **No real-world scale**: Queue length is reported in pixels, not metres, unless
  a calibration step is added.
- **Tracking gaps**: Severe occlusion may cause ID switches. The max_age parameter
  helps but cannot fully eliminate this.
- **CPU inference**: YOLOv8n is optimised for CPU but real-time processing of HD video
  may be slow without a GPU.
- **Fixed ROI**: The ROI does not auto-adapt if the camera moves.

---

## 23. Future Improvements

- **Perspective correction** for accurate metre-scale queue length
- **GPU acceleration** with CUDA / TensorRT
- **Re-identification (ReID)** for robust tracking after long occlusions
- **WebSocket live stream** input from RTSP cameras
- **REST API** for integration with dashboards
- **Alert system** (email/SMS) when queue exceeds CRITICAL threshold
- **Multiple queue zones** in a single frame
- **Dynamic ROI** that adapts when people form the queue boundary

---

## 24. Project Structure

```
QueueVision-AI/
│
├── README.md                    <- Project overview, install, run, test
├── statement.md                 <- Problem statement, scope, target users, features
├── requirements.txt             <- Python dependencies
├── config.yaml                  <- All tunable parameters
├── main.py                      <- CLI entry point and pipeline orchestrator
├── setup_roi.py                 <- Interactive ROI selector
│
├── src/
│   ├── __init__.py
│   ├── detector.py              <- YOLOv8 person detector
│   ├── tracker.py               <- IoU + centroid multi-object tracker
│   ├── queue_analyzer.py        <- ROI filter, density, status classification
│   ├── waiting_time.py          <- Service-time learning, wait estimation
│   ├── analytics.py             <- Time-series, CSV, report generation
│   ├── visualization.py         <- Bounding boxes, panels, heatmap
│   └── utils.py                 <- Config, logging, helpers
│
├── models/
│   ├── README.txt
│   └── yolov8n.pt               <- YOLO weights (auto-downloaded)
│
├── data/
│   ├── sample_images/           <- Put test images here
│   └── sample_videos/           <- Put test videos here
│
├── results/                     <- Annotated output, CSVs, heatmap
│
├── reports/
│   └── analysis_report.txt
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py              <- Shared pytest fixtures
│   ├── test_detector.py         <- Detection dataclass + edge cases
│   ├── test_queue.py            <- ROI, density, status, parse_roi
│   └── test_waiting_time.py     <- Wait estimation, learning, clamping
│
└── docs/
    ├── architecture.md          <- Detailed system architecture diagram and data flow
    ├── uml_diagrams.md          <- Use Case, Class, Sequence, Component, ER diagrams
    └── requirements.md          <- Full FR (FR-01 to FR-10) and NFR (NFR-01 to NFR-08)
```

---

## Acknowledgements

- **YOLOv8** by [Ultralytics](https://github.com/ultralytics/ultralytics) (Apache-2.0)
- **OpenCV** (Apache-2.0)
- Application logic, tracking algorithm, queue analysis, waiting-time model,
  and all report/CSV generation are original implementations.

---

*QueueVision AI â€“ College Computer Vision Project*

