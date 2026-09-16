# QueueVision AI – Requirements Specification

---

## 1. Functional Requirements

### FR-01 – Person Detection
The system shall detect all persons visible in a given video frame or image using a
YOLOv8n model with configurable confidence and IoU thresholds.

### FR-02 – Multi-Object Tracking
The system shall assign a unique, persistent ID to each detected person and track them
across consecutive video frames using an IoU + centroid matching strategy.

### FR-03 – Queue Region Filtering (ROI)
The system shall allow the user to define a rectangular Region of Interest (ROI).
Only persons whose bounding-box centroid falls within the ROI shall be counted as
queue members.

### FR-04 – Queue Density & Status Classification
The system shall compute a normalised queue density value and map it to one of four
statuses: LOW, MEDIUM, HIGH, or CRITICAL.

### FR-05 – Waiting-Time Estimation
The system shall estimate the waiting time for a new person joining the queue based
on the current queue count and a rolling average of observed service times.

### FR-06 – Report Generation
The system shall export:
  (a) queue_analysis.csv — frame-by-frame queue metrics
  (b) person_tracking.csv — per-person dwell time
  (c) analysis_report.txt — human-readable summary
  (d) queue_heatmap.png — density heatmap image

### FR-07 – Annotated Output
The system shall produce an annotated output video (or image) with bounding boxes,
ROI overlay, and a live stats panel showing queue count, status, and waiting time.

### FR-08 – ROI Setup Tool
The system shall provide a standalone script (setup_roi.py) that allows the user to
interactively draw a queue ROI on a sample frame and save it to config.yaml.

### FR-09 – Demo Mode
The system shall support a demo mode (--demo flag) that generates synthetic queue
data and runs the full pipeline without requiring a real input file.

### FR-10 – Configuration-Driven Behaviour
All tunable parameters (model path, confidence threshold, density thresholds, service
time bounds, output paths) shall be externally configurable via config.yaml.

---

## 2. Non-Functional Requirements

### NFR-01 – Performance
- **Description**: The system shall process video frames at an acceptable rate on
  consumer-grade hardware without a dedicated GPU.
- **Metric**: At least 5 frames per second (FPS) on a modern CPU for 720p input
  using the YOLOv8n model.
- **Implementation**: YOLOv8n is the smallest YOLO variant (~6 MB), optimised for
  CPU inference. Frame skipping is configurable via `processing.frame_skip` in
  `config.yaml`.

### NFR-02 – Reliability & Error Handling
- **Description**: The system shall not crash silently on invalid inputs or missing
  resources. All errors shall be logged with a clear message.
- **Implementation**:
  - All file paths are validated before processing begins.
  - Missing YOLO weights trigger a clear error with download instructions.
  - Invalid ROI coordinates are caught and reported with the expected format.
  - OpenCV VideoCapture failures are detected and reported immediately.
  - All exceptions are caught at the top-level pipeline and logged.

### NFR-03 – Usability
- **Description**: The system shall be operable entirely from the command line with
  no GUI IDE required.
- **Implementation**:
  - Full CLI interface via `argparse` with `--help` support.
  - `--demo` flag allows zero-setup first run.
  - `setup_roi.py` provides an interactive ROI selector with on-screen instructions.
  - `config.yaml` provides human-readable, commented configuration.
  - All estimated values are clearly labelled as ESTIMATES in output.

### NFR-04 – Maintainability & Modularity
- **Description**: The codebase shall be structured so that individual modules can
  be replaced or extended without affecting other modules.
- **Implementation**:
  - Seven independent modules in `src/` with well-defined interfaces.
  - Each module communicates via typed dataclasses (Detection, Track, QueueSnapshot).
  - All magic numbers are externalised to `config.yaml`.
  - Docstrings on all public functions and classes.
  - `utils.py` centralises cross-cutting concerns (logging, config loading, path helpers).

### NFR-05 – Scalability
- **Description**: The system shall support larger YOLO models without code changes
  to improve accuracy as hardware permits.
- **Implementation**:
  - `model_path` in `config.yaml` accepts any Ultralytics-compatible `.pt` file
    (yolov8n/s/m/l/x).
  - CUDA GPU acceleration is automatically used if available via `device: auto`.
  - Future support for live RTSP streams is architecturally possible by replacing
    the `cv2.VideoCapture` source in `main.py`.

### NFR-06 – Security & Privacy
- **Description**: The system shall not transmit any video data or personally
  identifiable information to external services.
- **Implementation**:
  - All processing is fully local — no cloud API calls.
  - No person images or bounding boxes are stored, only numerical statistics.
  - YOLO model weights are downloaded once from the official Ultralytics CDN
    over HTTPS and cached locally.

### NFR-07 – Testability
- **Description**: All core business logic modules shall have automated unit tests.
- **Implementation**:
  - 20+ unit tests across `tests/test_detector.py`, `tests/test_queue.py`,
    and `tests/test_waiting_time.py`.
  - Tests run with `pytest` and require no real video/image input (synthetic data).
  - `conftest.py` provides shared fixtures.
  - `pytest.ini` configures test discovery and output format.

### NFR-08 – Resource Efficiency
- **Description**: The system shall not consume unbounded memory during long video
  processing runs.
- **Implementation**:
  - The rolling service-time window is bounded to 20 observations (configurable).
  - Analytics records are buffered and written to CSV in a single pass at the end.
  - The heatmap accumulator uses a fixed-size float32 numpy array (frame-size).
  - Tracker clears aged-out tracks immediately, preventing memory growth.
