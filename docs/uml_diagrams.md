# QueueVision AI – UML Diagrams

---

## 1. Use Case Diagram

```
+----------------------------------------------------------+
|                  QueueVision AI System                   |
|                                                          |
|   +---------------------+    +------------------------+ |
|   |  Service Manager    |    |   Operations Analyst   | |
|   +----------+----------+    +-----------+------------+ |
|              |                           |               |
|   +----------v---------------------------v-----------+   |
|   |              Use Cases                           |   |
|   |                                                  |   |
|   |  [UC1] Provide Input (video / image / demo)      |   |
|   |  [UC2] Configure Queue ROI                       |   |
|   |  [UC3] View Real-time Queue Count                |   |
|   |  [UC4] View Queue Status (LOW/MED/HIGH/CRIT)     |   |
|   |  [UC5] View Estimated Waiting Time               |   |
|   |  [UC6] View Annotated Output Video               |   |
|   |  [UC7] Export CSV Reports                        |   |
|   |  [UC8] View Queue Density Heatmap                |   |
|   |  [UC9] Run Unit Tests                            |   |
|   +--------------------------------------------------+   |
|                                                          |
+----------------------------------------------------------+

Actors:
  Service Manager  -> UC1, UC2, UC3, UC4, UC5, UC6
  Operations Analyst -> UC7, UC8
  Developer / Tester -> UC9
  System (automated) -> UC3, UC4, UC5 (background processing)
```

---

## 2. Class Diagram

```
+--------------------+          +--------------------+
|   PersonDetector   |          |  CentroidTracker   |
|--------------------|          |--------------------|
| - model: YOLO      |          | - tracks: list     |
| - conf_threshold   |          | - next_id: int     |
| - iou_threshold    |          | - max_age: int     |
| - device: str      |          | - iou_threshold    |
|--------------------|          | - max_distance     |
| + detect(frame)    |          |--------------------|
| + _filter_persons()|          | + update(dets)     |
+--------------------+          | + _iou(a,b)        |
         |                      | + _centroid_dist() |
         | List[Detection]      +--------------------+
         v                               |
+--------------------+                  | List[Track]
|    Detection       |                  v
|--------------------|       +--------------------+
| + bbox: tuple      |       |       Track        |
| + confidence: float|       |--------------------|
| + class_id: int    |       | + track_id: int    |
| + centroid: tuple  |       | + bbox: tuple      |
+--------------------+       | + centroid: tuple  |
                             | + age: int         |
                             | + time_since_update|
                             | + entry_frame: int |
                             +--------------------+
                                      |
              +-----------+-----------+-----------+
              |           |           |           |
              v           v           v           v
  +-----------+  +--------+---+ +----+------+ +--+---------+
  |QueueAnalyzer| |WaitingTime | |QueueAnalyt| |Visualizer  |
  |-------------| |Estimator   | |ics        | |------------|
  |- roi: tuple | |------------| |-----------| |- heat_accum|
  |- thresholds | |- window:[] | |- records[]| |- config    |
  |-------------| |- fallback  | |-----------| |------------|
  |+analyse(trks)| |+observe_  | |+record()  | |+annotate_  |
  |+in_roi(trk) | | exits()    | |+save_csv()| | frame()    |
  |+get_density()| |+estimate() | |+save_rpt()| |+save_heatm |
  +-------------+ +------------+ +-----------+ |  ap()      |
        |                                       +------------+
        v
  +-------------------+
  |  QueueSnapshot    |
  |-------------------|
  | + frame_number    |
  | + timestamp       |
  | + queue_count     |
  | + total_detected  |
  | + queue_density   |
  | + queue_status    |
  | + estimated_wait  |
  | + avg_wait        |
  | + people_served   |
  | + service_rate    |
  +-------------------+
```

---

## 3. Sequence Diagram – Video Processing Flow

```
User       main.py     Detector    Tracker    QueueAnalyzer   WaitEst   Analytics   Visualizer
 |            |            |           |             |            |          |            |
 |--run cmd-->|            |           |             |            |          |            |
 |            |--load cfg->|           |             |            |          |            |
 |            |--init all->|           |             |            |          |            |
 |            |            |           |             |            |          |            |
 |            |====== For each Frame ==============================================|      |
 |            |            |           |             |            |          |            |
 |            |--detect(f)->|           |             |            |          |            |
 |            |<-detects[]--|           |             |            |          |            |
 |            |            |           |             |            |          |            |
 |            |--update(detects)------->|             |            |          |            |
 |            |<-tracks[], exited[]-----|             |            |          |            |
 |            |            |           |             |            |          |            |
 |            |--observe_exits(exited)-------------->|            |          |            |
 |            |            |           |             |            |          |            |
 |            |--analyse(tracks)-------------------->|            |          |            |
 |            |<-QueueSnapshot-----------------------|            |          |            |
 |            |            |           |             |            |          |            |
 |            |--estimate(snapshot)----------------------------------->|      |            |
 |            |<-estimated_wait_min------------------------------------|      |            |
 |            |            |           |             |            |          |            |
 |            |--record(snapshot)---------------------------------------------->|         |
 |            |            |           |             |            |          |            |
 |            |--annotate_frame(frame, tracks, snapshot)------------------------------>|  |
 |            |<-annotated_frame------------------------------------------------------|  |
 |            |            |           |             |            |          |            |
 |            |--write(annotated_frame)|             |            |          |            |
 |            |====== End Frame Loop ==============================================|      |
 |            |            |           |             |            |          |            |
 |            |--save_heatmap()---------------------------------------------------------->|
 |            |--save_queue_csv()---------------------------------------------->|         |
 |            |--save_tracking_csv()------------------------------------------->|         |
 |            |--save_report()------------------------------------------------->|         |
 |            |            |           |             |            |          |            |
 |<--outputs--|            |           |             |            |          |            |
```

---

## 4. Component Diagram

```
+-------------------------------------------------------------+
|                    QueueVision AI Application               |
|                                                             |
|  +------------------+    +--------------------------------+ |
|  |   CLI Interface   |    |      Configuration Layer       | |
|  |   (main.py)       |    |      (config.yaml + utils.py)  | |
|  |   argparse        |    |      YAML parsing, logging     | |
|  +--------+---------+    +--------------------------------+ |
|           |                                                  |
|           v                                                  |
|  +--------+---------+                                        |
|  |  Processing       |                                       |
|  |  Pipeline         |                                       |
|  |                   |                                       |
|  |  +-----------+    |    +------------------------------+  |
|  |  | Detector  |<---+--->|  YOLOv8n Model (ultralytics) |  |
|  |  | (YOLOv8)  |    |    |  yolov8n.pt                  |  |
|  |  +-----------+    |    +------------------------------+  |
|  |       |           |                                       |
|  |  +-----------+    |                                       |
|  |  | Tracker   |    |                                       |
|  |  | (IoU+     |    |                                       |
|  |  |  centroid)|    |                                       |
|  |  +-----------+    |                                       |
|  |       |           |                                       |
|  |  +-----------+  +-----------+                            |
|  |  | Queue     |  | WaitTime  |                            |
|  |  | Analyzer  |  | Estimator |                            |
|  |  +-----------+  +-----------+                            |
|  +---+-------+---+---+-------+---+                         |
|      |       |       |       |                               |
|  +---v---+ +-v-----+ +v-----++                              |
|  |Visual-| |Analyt-| |Output|                               |
|  |izer   | |ics    | |Files |                               |
|  +-------+ +-------+ +------+                               |
|                          |                                   |
|  +-----------------------v---------------------------------+ |
|  |                   Storage Layer                          | |
|  |  results/queue_analysis.csv  results/heatmap.png        | |
|  |  results/person_tracking.csv reports/analysis_report.txt| |
|  +----------------------------------------------------------+ |
+-------------------------------------------------------------+
```

---

## 5. ER Diagram (CSV Storage Schema)

```
+-----------------------------+       +------------------------------+
|     queue_analysis.csv      |       |    person_tracking.csv       |
|-----------------------------|       |------------------------------|
| PK  frame_number   INTEGER  |       | PK  track_id       INTEGER   |
|     timestamp      REAL     |       |     entry_frame    INTEGER   |
|     queue_count    INTEGER  |       |     exit_frame     INTEGER   |
|     total_detected INTEGER  |       |     frames_seen    INTEGER   |
|     queue_density  REAL     |       |     time_in_queue_ INTEGER   |
|     queue_status   TEXT     |       |       frames                 |
|     estimated_wait REAL     |       |     time_in_queue_ REAL      |
|     average_wait   REAL     |       |       sec                    |
|     people_served  INTEGER  |       |     was_in_queue   BOOLEAN   |
|     service_rate   REAL     |       +------------------------------+
|     queue_length_px INTEGER  |
+-----------------------------+

Relationship:
  queue_analysis.csv records one row per frame.
  person_tracking.csv records one row per unique tracked person.
  Both are keyed independently (frame_number, track_id respectively).
  They share a temporal link: a track active at frame N appears in both tables.
```
