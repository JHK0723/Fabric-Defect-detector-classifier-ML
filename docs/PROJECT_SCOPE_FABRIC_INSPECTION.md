# Real-Time AI-Based Textile Fabric Defect Detection
## Master Project Specification for an AI Coding Agent

**Document purpose:** This is the single source of truth for implementing the complete project. It consolidates the responsibilities and interfaces from the three member guides into one end-to-end specification. Use it to understand the complete system, build the repository, implement modules, and verify that all parts work together.

---

## 1. Project Overview

Build a software-first prototype that inspects textile fabric as an operator manually moves it beneath a smartphone camera. The phone streams video to a Windows laptop. A computer-vision model detects and classifies visible fabric defects, draws bounding boxes and labels on the live feed, and triggers an alert when a defect is confirmed. A Streamlit dashboard shows the live feed, system performance, defect counts, event history, snapshots, and session analytics.

This is a **prototype, not an industrial machine-control system**. Fabric movement is manual. Do not add motors, Arduino, ESP32, PLCs, relays, automated rollers, rotary encoders, or any hardware-control feature. The only physical setup is a fixed phone mount, lighting, a flat support board, and fabric moved by hand.

### Main user workflow

1. The operator lays fabric on the inspection board and moves it by hand through the camera's field of view.
2. The smartphone captures the fabric continuously and streams frames to the laptop.
3. OpenCV captures frames asynchronously so the camera does not block while the model performs inference.
4. A YOLO detector identifies defects and returns class, confidence, and bounding-box coordinates.
5. A temporal tracker associates detections across frames so the same physical defect is counted once rather than once per frame.
6. Once a defect is confirmed, the system logs an event and snapshot, shows a visible warning, and optionally beeps: **“DEFECT DETECTED — STOP MOVING FABRIC.”**
7. The operator can review the session, defect counts by class, confidence, snapshots, timestamps, and CSV exports.

### Project goals

- Demonstrate real-time or near-real-time fabric defect detection from a smartphone feed.
- Detect and classify four target defect types.
- Display bounding boxes, class labels, and confidence values on the video.
- Count each physical defect once under normal forward fabric movement.
- Record inspection sessions and defect events persistently.
- Report model quality and system performance using measured experiments, not fabricated metrics.
- Keep modules independent so the model, camera source, tracker, and dashboard can be changed separately.

### Explicit non-goals

- No automated fabric transport or motor control.
- No claims of production-grade quality assurance or guaranteed detection.
- No cloud service or hosted backend is required; the prototype should run locally.
- No requirement to classify “normal” as a model class. A frame with no confirmed detections is treated as normal.
- No automatic diagnosis of defect types outside the four supported classes. Unsupported defects may be missed or misclassified.

---

## 2. Target Defect Classes

The class IDs and names are a shared contract across training, inference, tracking, logging, UI, and reports. Keep the order exactly as follows:

| Class ID | Internal class | Examples / intended meaning |
|---:|---|---|
| 0 | `hole` | Holes, cuts, tears |
| 1 | `stain` | Stains, oil marks, dye spots |
| 2 | `broken_thread` | Broken, loose, or protruding yarn ends |
| 3 | `irregular_yarn` | Missing yarn lines, thick/thin areas, weaving irregularities |

There is **no `normal` class**. A frame with no confirmed detections is considered normal. Include defect-free training images with empty YOLO label files so the model learns background fabric textures and lighting variation.

If `broken_thread` and `irregular_yarn` prove consistently difficult to distinguish, evaluate the confusion matrix and consider merging them into `yarn_defect` only as a documented, deliberate dataset/model change. Do not silently rename classes because the other modules depend on these names.

---

## 3. Intended Runtime Environment

- OS: Windows 10/11.
- Python: 3.11, 64-bit.
- Laptop hardware target: NVIDIA RTX 3050 6 GB GPU, Ryzen 7 7840HS, 16 GB RAM.
- ML framework: PyTorch with a compatible CUDA build, installed before the other dependencies.
- Object detection: Ultralytics YOLO; begin with a nano model for integration and baseline training, then test a small model if performance permits.
- Video/image processing: OpenCV and NumPy.
- Dashboard: Streamlit.
- Persistence: SQLite, using Python's built-in `sqlite3` module.
- Analytics/export: pandas and CSV.
- Configuration: YAML.

### Dependencies

The initial `requirements.txt` should include at least:

```text
ultralytics>=8.4.0
opencv-python
numpy
pandas
pyyaml
psutil
streamlit>=1.37
```

Install a CUDA-enabled PyTorch build using the official PyTorch installation selector before installing `requirements.txt`. Verify GPU availability with:

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The application must still have a CPU fallback where practical, but the expected demo configuration is GPU inference. Clearly report which device is being used.

Ultralytics licensing must be reviewed and acknowledged in the project report. Check the licenses of every downloaded dataset, pretrained weight, repository, and external model before redistribution. Publicly visible source code or weights do not automatically grant reuse rights.

---

## 4. High-Level Architecture

```text
Smartphone camera
       │ USB webcam mode preferred; Wi-Fi stream URL is fallback
       ▼
streaming.camera.OpenCVCamera
       │ camera thread continuously captures frames
       ▼
streaming.frame_queue.LatestFrameBuffer
       │ size-1 latest-frame buffer; stale frames may be dropped
       ▼
vision.pipeline.InspectionPipeline
       ├── optional region-of-interest crop
       ├── vision.detector.FabricDetector
       │       └── YOLO inference → per-frame detections
       ├── vision.tracker.DefectEventTracker
       │       └── temporal matching / confirmation / deduplication
       ├── vision.postprocessing
       │       └── draw boxes, labels, confidence, and status overlays
       └── publish latest annotated frame + stats
                         │
                         ├── on confirmed event → DefectLogger (SQLite + snapshot)
                         └── InspectionRuntime
                                    │
                                    ▼
                              Streamlit dashboard
                              ├── Live inspection
                              ├── Analytics and event history
                              └── Past inspection sessions
```

### Key architectural rules

1. Camera capture and model inference must not run inside Streamlit's main script rerun path.
2. Use a camera thread and inference/pipeline thread. The dashboard polls the latest result.
3. Use a size-1 latest-frame buffer rather than an unbounded FIFO queue. Prefer low latency over processing every old frame.
4. Keep the detector unaware of cameras, event tracking, SQLite, and Streamlit.
5. Keep the dashboard unaware of raw YOLO APIs and camera capture details. It talks to `InspectionRuntime` only.
6. Use the same `config/config.yaml` for the standalone runner and dashboard.
7. Use mock mode so dashboard work and integration tests can proceed before the real camera and trained model are ready.
8. Running the application must not require motors or any physical automation.

---

## 5. Repository Structure

Create the following structure from the repository root. Python package folders must include `__init__.py` files.

```text
fabric-inspection/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/                         # downloaded datasets and original captures; never edit in place
│   │   ├── own_capture/
│   │   ├── rf_sqc/
│   │   ├── rf_defect_clothes/
│   │   └── fd_figshare/
│   ├── yolo/                        # generated merged YOLO dataset
│   │   ├── images/{train,val,test}/
│   │   ├── labels/{train,val,test}/
│   │   └── data.yaml
│   └── external/                    # external evaluation images/tiles
├── models/
│   └── fabric_defect.pt             # trained project model; generated by training
├── training/
│   ├── sources.yaml
│   ├── extract_frames.py
│   ├── prepare_dataset.py
│   ├── masks_to_yolo.py
│   ├── check_labels.py
│   ├── train.py
│   ├── evaluate.py
│   ├── threshold_sweep.py
│   ├── tile_external.py
│   └── benchmark_detector.py
├── vision/
│   ├── __init__.py
│   ├── detector.py                  # Member 1
│   ├── preprocessing.py             # Member 2
│   ├── postprocessing.py             # Member 2
│   ├── tracker.py                   # Member 2
│   └── pipeline.py                  # Member 2
├── streaming/
│   ├── __init__.py
│   ├── frame_queue.py
│   ├── camera.py
│   └── test_camera.py
├── defect_logging/
│   ├── __init__.py
│   └── defect_logger.py
├── dashboard/
│   ├── __init__.py
│   ├── mocks.py
│   ├── runtime.py
│   └── app.py
├── tests/
│   └── test_tracker.py
├── logs/                            # runtime DB, snapshots, benchmark CSVs
├── run_pipeline.py                  # standalone OpenCV-window runner
├── benchmark_pipeline.py
├── experiments.md
├── requirements.txt
└── README.md
```

Do not create a top-level package named `logging/`; that can shadow Python's built-in `logging` module. Run commands from the repository root so imports and relative paths are consistent. Generated data, weights, logs, and large datasets should not be committed to Git unless the team deliberately decides otherwise.

---

## 6. Shared Module Contracts

These interfaces are mandatory because the three workstreams depend on them.

### 6.1 Per-frame detection

`FabricDetector.detect(frame_bgr)` accepts a BGR `uint8` NumPy image and returns a list of dictionaries. Bounding boxes use full-frame pixel coordinates in `[x1, y1, x2, y2]` order.

```python
{
    "class": "hole",
    "class_id": 0,
    "confidence": 0.947,
    "bbox": [120, 80, 210, 165]
}
```

Return an empty list if no detections are produced. Confidence is a float from 0 to 1. `class` and `class_id` must agree with the four-class contract.

### 6.2 Confirmed defect event

A confirmed event represents one physical defect, deduplicated across processed frames:

```python
{
    "event_id": 12,
    "track_id": 12,
    "class": "hole",
    "confidence": 0.95,
    "bbox": [120, 80, 210, 165],
    "first_frame": 100,
    "last_frame": 104,
    "first_seen": "2026-10-20T14:32:17",
    "last_seen": "2026-10-20T14:32:18",
    "frames_seen": 5,
    "snapshot": None
}
```

The `snapshot` field is either a BGR NumPy image or `None`. Timestamps should use local ISO 8601 strings with seconds precision unless the team documents a different consistent policy.

### 6.3 Pipeline latest snapshot

`InspectionPipeline.latest()` must return a thread-safe snapshot containing the most recent annotated frame, frame ID, detections, active tracks, and stats. Use a stable schema such as:

```python
{
    "frame": annotated_bgr_ndarray_or_none,
    "frame_id": 1234,
    "detections": [],
    "active_tracks": [],
    "stats": {
        "running": True,
        "camera_connected": True,
        "model_loaded": True,
        "camera_fps": 29.8,
        "inference_fps": 20.4,
        "inference_ms": 35.0,
        "latency_ms": 52.0,
        "frames_processed": 1234,
        "frames_dropped": 21,
        "events_total": 8,
        "defect_in_view": False
    }
}
```

The exact values above are examples only. Populate stats from measurements. Do not hardcode example metrics. If no frame is available yet, return `None` for the frame and sensible zero/false defaults for stats.

### 6.4 Event callbacks

When a track is confirmed, the pipeline invokes an `on_event(event)` callback exactly once for that event. When a confirmed event/track leaves the view and closes, invoke `on_event_closed(event)` if the logger/runtime needs to update its final timestamps and frame count. Callback errors must be handled without silently killing the camera or inference loop.

---

## 7. ML / Computer Vision Workstream

### 7.1 Model choice

- Start with `yolo26n.pt` as a small baseline/integration model where available in the installed Ultralytics version.
- Train the four-class project model and save it as `models/fabric_defect.pt`.
- Compare a nano and small model if time and GPU memory allow. A small model may improve small-defect detection but must be benchmarked on the target laptop.
- If YOLO26 causes a compatibility issue, use a supported YOLO11 nano/small model as a documented fallback.
- Do not use a generic COCO model as evidence of fabric-defect detection; it is only a plumbing test until the custom model is trained.

The model inference image size in `config/config.yaml` must match the size used for training/evaluation.

### 7.2 Candidate datasets

Treat these as candidate sources, not as guaranteed compatible datasets. Before use, verify that the link is accessible, inspect its annotations and class definitions, and read its license.

| Dataset | Intended role | Notes |
|---|---|---|
| Roboflow Fabric Defect Detection (`sqc`) — https://universe.roboflow.com/sqc/fabric-defect-detection-ehhyx | Training | Reported classes include hole, stain, gray stitch, selvet, tear; inspect and map classes manually. |
| Roboflow Defect-Clothes (`hawadz`) — https://universe.roboflow.com/hawadz/defect-clothes-ukrks | Training | Reported classes include hole, stain, clothes, cut; drop garment-level boxes rather than treating `clothes` as a fabric defect. |
| Figshare FD_Dataset — https://figshare.com/articles/dataset/FD_Dataset_7z/25546465 | Training | Reported classes: oil, hole, cutting, crack; verify annotation format and license. |
| AITEX / AFID — https://www.aitex.es/afid/ and https://datasetninja.com/afid | External evaluation | Image/mask dataset with defect-free and defective fabric; originally industrial line-scan images, so expect domain shift. |
| University of Moratuwa fabric defect/stain datasets referenced at https://data.mendeley.com/datasets/663j22s43c/3 | Optional training | Inspect source and license; convert masks to bounding boxes only if the masks are suitable. |
| Team-collected phone-camera images | Essential train/validation/test data | Captured using the exact rig and lighting planned for the demo. |

The model must not depend exclusively on public data. Public datasets may use different lighting, scales, fabric textures, and cameras. The team’s own captures are necessary to measure and improve performance on the actual demo setup.

### 7.3 Class mapping rules

Use the shared four-class mapping only after inspecting representative images and annotations.

- `hole`, `tear`, `cut`, `cutting` → `hole` when the appearance is genuinely a hole/cut/tear.
- `stain`, `oil` → `stain` when they represent a fabric mark.
- Loose/broken yarn ends → `broken_thread`.
- Missing yarn lines and clear weave irregularities → `irregular_yarn`.
- Drop irrelevant boxes (e.g. a whole garment box) and exclude images when an ambiguous, unlabelled defect would teach the model incorrect background labels.
- Do not guess the meaning of source-specific labels. Review sample images first.

### 7.4 Dataset capture and annotation

Capture on the same rig used for the demo:

- 6–7 fabric pieces, identified as F01–F07; use a variety of plain cotton, coloured cotton, denim/twill, polyester, and checked/striped material.
- Create controlled examples of holes, stains, broken threads, and irregular/missing yarn. Include clean fabric areas.
- Aim for at least 30 physical examples per class distributed across fabrics, and roughly 400–800 captured/selected images overall.
- Target about 15–20% clean/background images initially; increase if clean fabric produces too many false alarms.
- Record short videos while moving fabric at the expected demo speed, then extract frames at an interval (for example every 0.5 seconds). Skip heavily blurred frames.
- Use Label Studio, CVAT, or Roboflow Annotate to label bounding boxes and export YOLO annotations.

Annotation rules:
1. Label every visible target defect in each selected image.
2. Use tight boxes around the full visible defect.
3. Separate physical defects get separate boxes.
4. Long thread/yarn defects can use long, narrow boxes.
5. If the defect type cannot be determined confidently, exclude the image or review it; do not invent a label.
6. Clean images have empty YOLO label files.
7. Each team member reviews a sample of another member's annotations.

### 7.5 Leak-free data splitting

- Split by fabric piece, capture session, and original source image groups—not randomly by adjacent video frames.
- Keep near-duplicate frames and augmented copies in the same split.
- A suggested own-data plan is F01–F04 for training, F05 for validation, F06 for final test, and F07 held out for an external/never-trained-on check.
- Public datasets may have their own train/validation groups; avoid relying on public validation alone.
- Do not tune model or alert thresholds on the final test set.
- Keep AITEX or another independent source as an external evaluation set when feasible; explain differences in image acquisition.

### 7.6 Required training utilities

Implement the following scripts as reproducible, command-line tools with clear help and errors:

- `training/extract_frames.py`: sample frames from videos, name files with fabric/session IDs, optionally skip blurry frames.
- `training/prepare_dataset.py`: merge source datasets, map classes, deduplicate/group related images, split without leakage, and generate YOLO `data.yaml`.
- `training/masks_to_yolo.py`: optional conversion from valid segmentation masks to bounding boxes.
- `training/check_labels.py`: verify class IDs, boxes, image-label pairs, empty-label backgrounds, and draw a visual sample of labels.
- `training/train.py`: train and save the best model to `models/fabric_defect.pt` (or copy the best run checkpoint there), record settings and metrics.
- `training/evaluate.py`: compute precision, recall, F1, mAP@50, mAP@50–95, confusion matrix, and PR curves as supported by the framework.
- `training/threshold_sweep.py`: evaluate confidence thresholds on validation data and recommend per-class/alert thresholds.
- `training/tile_external.py`: optionally tile very wide external images such as line-scan fabric images for evaluation; document how image-level metrics are derived.
- `training/benchmark_detector.py`: benchmark inference latency/FPS and device/model/image-size settings.

Save each training run’s configuration, metrics, and plots. Keep a run table in the README or a training log so model improvements can be explained.

### 7.7 Detector API

Implement `vision/detector.py` with a `FabricDetector` class:

```python
from vision.detector import FabricDetector

detector = FabricDetector(
    weights="models/fabric_defect.pt",
    conf=0.25,
    imgsz=640,
)
detector.warmup()
detections = detector.detect(frame_bgr)
```

Requirements:
- Load `.pt` weights with Ultralytics; allow ONNX/engine formats if supported by the chosen runtime.
- Raise a clear error for an invalid/missing local model path.
- Use CUDA device 0 when available; otherwise use CPU.
- Use half precision only when supported and enabled.
- Return the exact detection schema in Section 6.1.
- Do not apply the final event-confirmation thresholds here; the tracker owns those thresholds.
- Do not draw boxes in the detector; drawing belongs to postprocessing.
- Provide a warmup method to reduce first-inference delay.

### 7.8 ML deliverables

- `models/fabric_defect.pt`
- `vision/detector.py`
- Class mapping and dataset-generation configuration
- Training/evaluation scripts and reproducible run records
- Validation/test metrics, confusion matrix, PR curves, and qualitative error analysis
- Threshold-sweep results and detector speed benchmark
- A concise model evaluation summary for the final report

---

## 8. Camera, Streaming, and Pipeline Workstream

### 8.1 Physical rig

- Fix the smartphone above the fabric, lens facing downward and parallel to the fabric.
- Starting camera height: about 25–40 cm; choose one height, mark it, and keep it constant between data collection and the demo.
- Use two white LED lights positioned roughly 45° from either side to reduce shadows and glare.
- Use a matte, plain-coloured board underneath the fabric and guide/tape marks to keep it straight.
- Mark the board edge at known intervals (e.g. every 10 cm) so manual movement speed can be estimated during experiments.
- Aim for 1280×720 at 30 FPS where the phone/app supports it.
- Lock focus and exposure if available; disable beauty filters/HDR enhancements where possible; keep the phone powered and prevent sleep.
- Motion blur is a key risk. Use adequate lighting and a faster shutter if the app allows it.

### 8.2 Camera input options

The camera interface must accept:
- A webcam index, e.g. `0` or `1`.
- A stream URL, e.g. an HTTP/MJPEG or RTSP URL supplied by a phone camera app.
- A local video file for repeatable testing; loop it when configured.

Prefer USB phone-as-webcam mode when supported. Wi-Fi streaming is an alternative. If the college network blocks device-to-device traffic, test using the phone’s hotspot or a personal router. Never hardcode a particular phone IP address.

### 8.3 Camera interface and latest-frame buffer

Implement:
- `streaming/frame_queue.py`: `FramePacket` and a thread-safe `LatestFrameBuffer`.
- `streaming/camera.py`: `BaseCamera`, source parsing, and `OpenCVCamera`.
- `streaming/test_camera.py`: probe webcam indices and preview a selected source with FPS and snapshot support.

`FramePacket` must contain:

```python
{
    "frame": bgr_numpy_array,
    "frame_id": increasing_integer,
    "timestamp": time.perf_counter_value
}
```

Use a dataclass or equivalent stable structure. `LatestFrameBuffer` should retain the newest frame; if a frame is overwritten before consumption, increment a dropped-frame counter. The consumer should be able to request a frame newer than its last frame ID with a timeout.

`OpenCVCamera` should expose `start()`, `stop()`, `get(last_id=-1, timeout=1.0)`, `fps`, `is_connected`, `dropped_frames`, and `last_error`. It should reconnect after transient stream failures where feasible and close resources cleanly.

### 8.4 Preprocessing and drawing

Implement `vision/preprocessing.py` for region-of-interest handling. ROI is optional and represented as normalized fractions `[x1, y1, x2, y2]`. If configured, crop for inference but correctly map detection coordinates back to full-frame coordinates before returning them or drawing them.

Implement `vision/postprocessing.py` for drawing bounding boxes, class labels, confidence, and any needed status overlays. Use clear labels, e.g. `HOLE 91%`. Keep image colour handling consistent: OpenCV frames are BGR internally; Streamlit display must explicitly use `channels="BGR"` or convert to RGB.

### 8.5 Temporal deduplication tracker

Implement `vision/tracker.py` as a model-independent `DefectEventTracker`.

Purpose: the same physical defect may remain in the camera view for many processed frames. It must create one event, not a new event per frame.

Required behaviour:
1. Ignore detections below `min_confidence`.
2. Match new detections to existing tracks using bounding-box IoU and/or normalized center distance.
3. Use one-to-one matching for each processed frame so a single detection cannot update multiple tracks.
4. Confirm a track only after it is seen for `confirm_frames` consecutive processed frames and its best confidence reaches `confirm_confidence`.
5. Assign one stable event ID to a confirmed track and emit the new-event callback exactly once.
6. Continue updating the event's last-seen time/frame, box, hit count, and class vote while visible.
7. Close a track after `max_missed` processed frames without a match. Notify the close callback so final values can be saved.
8. Match tracks class-agnostically where appropriate and use a confidence-weighted class vote to reduce class flicker.
9. Make the tracker resettable between inspection sessions.
10. Keep all counters and timestamps explicit and testable.

Initial defaults in `config/config.yaml`:

```yaml
tracker:
  min_confidence: 0.50
  confirm_confidence: 0.70
  confirm_frames: 3
  max_missed: 10
  iou_threshold: 0.20
  max_center_dist: 0.15
```

These are starting values, not guaranteed optimal thresholds. Tune them using validation and threshold-sweep results. `confirm_frames` and `max_missed` count processed inference frames, not every camera frame.

Known limitation: if fabric is pulled backwards and the same defect leaves the view long enough to close its track, it may be counted again when it re-enters. Document this limitation rather than claiming perfect physical defect identity.

Add unit tests covering: one defect over many frames creates one event; detections below threshold do not confirm; a track closes after enough missed frames; two nearby defects do not merge incorrectly; class flicker does not create a duplicate; tracker reset restarts session state correctly.

### 8.6 Inspection pipeline

Implement `vision/pipeline.py` as the owner of camera capture consumption, inference, tracking, annotated frames, and runtime stats. It should:

- Construct or receive camera, detector, and tracker dependencies.
- Start and stop cleanly and idempotently.
- Keep camera acquisition independent of inference.
- Process the newest available frame and allow stale frames to be dropped.
- Apply ROI, detect, track, draw, and publish a latest snapshot.
- Invoke `on_event` and `on_event_closed` callbacks at the correct transitions.
- Expose `latest()` with the schema in Section 6.3.
- Track camera FPS, inference FPS, inference time, end-to-end latency, frames processed/dropped, events total, camera connectivity, model status, and whether a defect is currently in view.
- Handle camera loss and model/configuration errors with readable status/errors rather than hanging silently.
- Avoid unbounded queues or unbounded growth of in-memory frame history.

### 8.7 Standalone runner and benchmarks

`run_pipeline.py` must run the real camera/model pipeline in an OpenCV window without Streamlit. It is the fastest way to debug integration. It should display the annotated feed and key performance stats, and exit cleanly on a key press.

`benchmark_pipeline.py` must run repeatable measurements and save CSV output under `logs/`. Record model/weights, device, image size, camera source, camera FPS, inference FPS, inference time, end-to-end latency, processed/dropped frames, and relevant test conditions. Do not report only model inference time if the project claims end-to-end latency.

---

## 9. Runtime Configuration

Use `config/config.yaml` as the common source of runtime settings. Start with:

```yaml
camera:
  source: 1                 # webcam index, stream URL, or local video path
  width: 1280
  height: 720
  fps: 30
  backend: auto             # auto | dshow | msmf on Windows
  loop_file: true

roi: null                   # or [0.10, 0.05, 0.90, 0.95] as frame fractions

model:
  weights: models/fabric_defect.pt
  imgsz: 640                 # must match the training/inference configuration
  conf: 0.25                 # low raw threshold; event thresholds are in tracker
  half: true

tracker:
  min_confidence: 0.50
  confirm_confidence: 0.70
  confirm_frames: 3
  max_missed: 10
  iou_threshold: 0.20
  max_center_dist: 0.15

logging:
  db_path: logs/inspection.db
  snapshot_dir: logs/snapshots

alerts:
  beep: true
  banner_seconds: 3
```

Validate configuration on startup. If a value is invalid, show a specific actionable error. The dashboard may allow overrides for camera source, model weights, and selected tracker thresholds for the current run; the defaults must still come from this file. Keep `run_pipeline.py` and the dashboard consistent.

---

## 10. Logging and Persistence Workstream

Implement `defect_logging/defect_logger.py` using SQLite. It must be thread-safe because inference callbacks write events while Streamlit reads tables.

### 10.1 Sessions table

At minimum, store:
- Session ID
- Start and end timestamps
- Fabric/roll name or ID
- Fabric length in metres, if supplied
- Notes, if supplied
- Total processed frames

### 10.2 Events table

At minimum, store:
- Database row ID
- Session ID
- Pipeline event ID
- Defect class and confidence
- First/last seen timestamps
- First/last frame IDs
- Number of processed frames in which the event was seen
- Bounding box coordinates `x1, y1, x2, y2`
- Snapshot file path, when a snapshot exists

Create an index on `session_id`. Use parameterized SQL, protect shared SQLite operations with a lock, and commit writes reliably. Avoid creating a new database connection on every Streamlit rerun if a shared runtime is already managing it.

### 10.3 Snapshots

When a defect is confirmed, save a JPEG snapshot under `logs/snapshots/`. Include a bounding box and readable class/confidence/event ID annotation. Store the path in the database. If a snapshot is unavailable, log the event without crashing.

### 10.4 Required logger API

Provide a simple interface equivalent to:

```python
logger = DefectLogger("logs/inspection.db", "logs/snapshots")
session_id = logger.start_session("roll-A", fabric_length_m=5)
logger.log_event(session_id, event)
logger.close_event(session_id, event)  # update final event timestamps/counts
logger.end_session(session_id, frames_processed=18492)
events_df = logger.events_df(session_id)
sessions_df = logger.sessions_df()
```

Also provide `close()` for clean shutdown. Offer CSV download from the dashboard; CSV is an export, not the primary database.

---

## 11. Dashboard and Runtime Workstream

### 11.1 Runtime separation

Implement `dashboard/runtime.py` as the glue layer. It owns the active `InspectionPipeline`, `DefectLogger`, current session ID, current run settings, latest event/alert state, and error status.

Required runtime actions:
- Start a session and pipeline.
- Stop the pipeline and close the session cleanly.
- Provide a safe `snapshot()` method to return latest frame, stats, recent events, running state, mock state, and last alert.
- Allow camera/model overrides and selected tracker threshold overrides at startup.
- Send confirmed/closed events to the logger.
- Trigger an optional beep once per newly confirmed event, not once per frame.
- Surface readable errors in the UI.
- Keep a cached runtime resource so Streamlit reruns do not repeatedly create camera threads or database instances.

### 11.2 Mock mode

Implement `dashboard/mocks.py` with:
- `SyntheticFabricCamera`: creates a scrolling synthetic woven texture with artificial dark “holes” and coloured “stains”, and exposes the same camera interface as the real camera.
- `BlobDetector`: a lightweight OpenCV-based stand-in that returns the same detection dictionaries as the real detector.

Mock mode is only for testing the UI, runtime, callbacks, logging, and event flow. It is **not** a model evaluation and must be visibly identified as mock mode. Its artificial detections must not be presented as real ML performance.

### 11.3 Streamlit dashboard

The app must run with:

```bat
streamlit run dashboard/app.py
```

Use three main tabs:

#### A. Live inspection
- Live annotated camera feed.
- Start and Stop controls.
- Mock mode toggle.
- Camera source input (device index, stream URL, or video file).
- Model weights input.
- Fabric/roll ID and inspected length in metres.
- Configurable detection thresholds applied when starting a session.
- System status: running/stopped, camera connected/disconnected, model loaded/not loaded, and mock/real mode.
- Camera FPS, AI/inference FPS, inference time, latency, processed frames, dropped frames.
- Total confirmed defects in the current session and per-class counts.
- A visible alert with class, confidence, and **“STOP MOVING FABRIC”** while a recent/active defect alert is relevant.
- Optional beep controlled by configuration.
- When scanning with no active alert, display a normal/scanning status.

#### B. Analytics
- Total confirmed defects.
- Mean confidence.
- Defects per metre when fabric length is known.
- Counts by defect class.
- Confidence distribution.
- Defects over time (e.g. per minute).
- Event history with time, event ID, class, confidence, and frames seen.
- Latest saved defect snapshots.
- CSV download.

#### C. Past sessions
- Table of previous sessions.
- Session details and event history for a selected session.
- Analytics for the selected session.

### 11.4 Streamlit real-time implementation rules

Streamlit reruns the script after interactions. Do not put an infinite camera loop in `app.py`. Use a cached `InspectionRuntime` and periodic fragments/polling (e.g. live view about every 0.15 seconds and analytics about every 2 seconds, supported by the selected Streamlit version). Display refresh rate and inference rate are different; report them separately.

The dashboard must remain responsive while inference is running. Do not store an unlimited sequence of frames in Streamlit session state. Handle absent snapshots, empty dataframes, disconnected cameras, missing weights, and stopped sessions gracefully.

---

## 12. Integration Requirements

Integrate incrementally; do not wait until all modules are complete.

1. **Mock-only milestone:** dashboard, runtime, SQLite, mock camera/detector, and UI work without phone or trained model.
2. **Camera-only milestone:** phone feed displays in OpenCV and camera test reports FPS/connection status.
3. **Generic-model milestone:** standalone pipeline works with `yolo26n.pt` (or another supported generic YOLO checkpoint) to validate model loading and inference plumbing. These detections are not fabric defects.
4. **Tracker milestone:** unit tests pass and a repeated detection across many frames creates exactly one event.
5. **Custom-model milestone:** integrate `models/fabric_defect.pt` and verify its class names.
6. **Dashboard integration:** run recorded video first, then phone camera with the generic model, then phone camera with the custom fabric model.
7. **Logging milestone:** each confirmed defect creates one database row and one snapshot; event closure updates final values.
8. **Resilience milestone:** Start → Stop → Start works at least five times without restarting Streamlit; camera disconnect/reconnect produces clear status; invalid weights path gives a readable error.
9. **Soak test:** dashboard remains responsive during at least 15 minutes of continuous operation.
10. **Demo readiness:** complete three consecutive clean demo dry-runs; keep a recorded video and mock mode as backup.

Never claim integration is complete just because imports succeed. Run the actual end-to-end flow and verify a known physical defect from camera capture through bounding box, event confirmation, alert, database row, and snapshot.

---

## 13. Evaluation and Experiments

Create `experiments.md` and record the setup and results for each run. The team should mark known physical defects with numbered tags outside the camera view so the true defect count can be compared with detections.

Record setup details:
- Phone model and camera app
- USB or Wi-Fi connection
- Resolution and frame rate
- Camera height and lighting arrangement
- Fabric types and measured manual movement speed
- Model/weight version and image size
- Confidence and tracker thresholds
- GPU/CPU used

For each experiment, record:
- Experiment ID and question being tested
- Fabric ID/type and length inspected
- Ground-truth number of physical defects
- True positives, false negatives, and false positives
- Duplicate event count
- Frames processed and dropped
- Camera FPS, inference FPS, inference time, end-to-end latency
- Notes about lighting, blur, fabric motion, and failure cases

Required evaluation:
- ML: precision, recall, F1, mAP@50, mAP@50–95 where applicable, confusion matrix, PR curves, and class-specific results.
- Runtime: camera FPS, inference FPS, inference time, end-to-end latency, dropped frames, and event deduplication behaviour.
- End-to-end: physical defects correctly detected, missed defects, false alarms, duplicate events, and defects per metre when length is measured.
- Include qualitative error analysis with representative false positives and false negatives.
- Keep the final test split untouched until final evaluation. Do not tune thresholds against final test results.
- Separate model-only metrics from end-to-end system metrics and distinguish mock-mode results from real-camera results.

Do not invent or promise a fixed accuracy/FPS target before measuring it. Report actual results with the exact model, hardware, dataset split, image size, thresholds, and conditions.

---

## 14. Four-Week Team Plan

### Week 1 — Baseline and parallel development

- All members set up Python 3.11, the repository, virtual environment, and CUDA-enabled PyTorch.
- Member 1: obtain initial public data, inspect annotations/classes/licenses, build the data preparation path, train baseline v0, and hand over initial weights even if performance is imperfect.
- Member 2: build the physical rig, test phone-to-OpenCV streaming, implement camera and latest-frame buffer, and run the generic model through the pipeline.
- Member 3: build the logger, mock camera/detector, runtime, and dashboard against the shared detection/event contracts.

### Week 2 — Own data and first end-to-end run

- Capture phone-camera videos on the final rig and annotate representative frames.
- Train model v1 on public and own data with leak-free splits.
- Implement and test temporal tracking.
- Connect the real camera feed to the dashboard and custom model v0/v1.

### Week 3 — Evaluation and integration hardening

- Run model metrics, threshold sweeps, error analysis, and speed benchmarks.
- Tune tracking thresholds against observed false alarms/duplicates.
- Complete SQLite events, snapshots, analytics, session history, and CSV export.
- Test reconnects, invalid configuration, repeated start/stop, and long-running responsiveness.

### Week 4 — Final model, experiments, report, demo

- Fix the largest failure modes with targeted data/annotation improvements.
- Freeze final weights at least three days before the demo.
- Run external tests and the experiment matrix.
- Complete the README, evaluation summary, limitations, and demo script.
- Rehearse the live demonstration three times and prepare recorded-video/mock backups.

### Ownership

| Member | Primary ownership | Required handoff |
|---|---|---|
| Member 1 — ML/CV | Dataset acquisition, annotation, data preparation, training, evaluation, threshold sweep, detector API | `models/fabric_defect.pt`, `vision/detector.py`, class list, evaluation report, thresholds, speed results |
| Member 2 — Camera/pipeline | Rig, camera streaming, frame buffer, preprocessing/ROI, tracker, pipeline, standalone runner, benchmarks | `streaming/`, `vision/preprocessing.py`, `vision/postprocessing.py`, `vision/tracker.py`, `vision/pipeline.py`, config, benchmarks |
| Member 3 — Dashboard/integration | SQLite logger, mock mode, runtime, Streamlit UI, analytics, experiment sheet, README/demo | `defect_logging/`, `dashboard/`, `experiments.md`, integration evidence, README and demo script |

All members should help annotate and test physical defects. Ownership does not mean modules can change the shared interface without coordinating with the other members.

---

## 15. Acceptance Criteria

The project is considered complete only when the following are demonstrated or documented.

### Functional
- [ ] A smartphone camera feed or repeatable video file can be opened by the camera module.
- [ ] The live feed displays bounding boxes, defect class names, and confidence values from the custom model.
- [ ] The four-class names and class IDs match across the model and UI.
- [ ] No `normal` class is required; no confirmed detection means normal/scanning.
- [ ] A confirmed defect triggers a visible alert and optional beep.
- [ ] One physical defect that stays in view creates one event rather than one event per frame.
- [ ] Events are stored in SQLite with timestamps, class, confidence, bounding box, frame counts, and snapshot path when available.
- [ ] The dashboard displays live performance metrics, class counts, analytics, session history, snapshots, and CSV export.
- [ ] Mock mode works independently and is clearly marked as mock.
- [ ] The pipeline can be run without the dashboard for debugging.

### Reliability
- [ ] Camera capture is decoupled from inference with a bounded/latest-frame buffer.
- [ ] Start/Stop/Start works repeatedly without leaking threads or database connections.
- [ ] Camera loss and invalid model paths produce readable errors.
- [ ] The dashboard remains responsive during a 15-minute run.
- [ ] Unit tests cover tracker confirmation, deduplication, closing, and reset.

### Evidence and documentation
- [ ] Dataset sources, licenses, class mapping, and split strategy are documented.
- [ ] Training settings and run history are saved.
- [ ] Test data is split by fabric/capture group to avoid leakage.
- [ ] ML metrics and runtime benchmarks are measured and reported honestly.
- [ ] End-to-end experiments include false positives, false negatives, duplicates, latency, and frame drops.
- [ ] Limitations and known failure modes are documented.
- [ ] README contains setup, camera configuration, training/inference instructions, dashboard startup, and demo instructions.
- [ ] Three successful demo dry-runs have been completed.

---

## 16. README and Final Demo

The final README should explain:
1. What the prototype does and what it does not do.
2. Architecture and data flow.
3. Setup instructions for Windows/Python/CUDA.
4. Dataset sources, annotation/class mapping, and licensing notes.
5. How to train or replace the model.
6. How to configure the camera source and run the standalone pipeline.
7. How to launch the Streamlit dashboard and use mock mode.
8. Measured model results and system benchmarks.
9. Limitations, troubleshooting, and future work.
10. Team member responsibilities.

### Suggested six-minute demo sequence

1. Introduce the manual fabric-inspection problem and explain that this is a software prototype.
2. Show the phone mount, two-sided lighting, and manually moved fabric.
3. Start the dashboard and confirm camera/model status.
4. Move fabric with known examples of a hole, stain, broken thread, and irregular yarn.
5. Show the bounding box/class/confidence, warning banner, and “STOP MOVING FABRIC” alert.
6. Explain how temporal tracking prevents counting the same visible defect every frame.
7. Open analytics, show class counts and saved snapshots, then export the CSV.
8. Show actual benchmark/evaluation results and explain one limitation.
9. If the camera or live model fails, switch to a recorded video or visibly labelled mock mode; do not represent mock detections as real model results.

---

## 17. Common Failure Modes to Avoid

- Training only on public datasets and expecting strong performance on a different phone camera/lighting setup.
- Mixing adjacent video frames or augmented copies across train/validation/test splits.
- Incorrect class mapping, missing boxes, or treating an unlabelled defect as background.
- Adding a `normal` class that conflicts with object detection's background semantics.
- Changing class names or data schemas without updating all modules.
- Putting camera loops or YOLO inference directly inside the Streamlit rerun path.
- Using an unbounded frame queue that increases latency.
- Counting every frame-level detection as a new physical defect.
- Triggering a beep/alert every frame instead of once per confirmed event.
- Confusing generic COCO detections or mock detections with fabric-defect model results.
- Assuming a public GitHub repository's code or weights can be reused without checking its license.
- Reporting mAP as the same thing as end-to-end physical defect recall.
- Reporting measured FPS/accuracy without recording the model, device, image size, thresholds, split, and test conditions.
- Treating this prototype as an automated industrial control or certified quality-inspection system.

---

## 18. Coding-Agent Instructions

When implementing this project:

1. Read this entire specification before writing code.
2. Build the repository in small, testable increments; do not generate the entire application as one opaque script.
3. Implement and test shared contracts first. Keep class names, detection dictionaries, event dictionaries, and pipeline stats consistent.
4. Start with mock mode and a generic model for plumbing; integrate the custom fabric model when it is available.
5. Use typed functions/dataclasses where they improve clarity, but do not over-engineer the prototype.
6. Add clear logging, actionable errors, path validation, and graceful shutdown for threads/camera/database resources.
7. Keep configuration out of hardcoded business logic. Use `config/config.yaml` for runtime defaults.
8. Add unit tests for the tracker and smoke tests for camera/model/dashboard integration wherever feasible.
9. Do not fabricate dataset availability, model accuracy, FPS, benchmark results, or successful test status. If a dataset, dependency, camera, or checkpoint is unavailable, report the blocker and provide the next actionable step.
10. Do not silently change the four-class schema or the module interfaces. If a change is necessary, update this specification and all affected modules together.
11. Do not add motors, Arduino, ESP32, PLCs, or automated fabric movement.
12. Before declaring completion, execute the acceptance checklist and report what passed, what failed, and what remains unverified.

**Definition of done:** A local Windows prototype where fabric is moved manually, a phone camera feed is processed by a custom four-class detector, confirmed physical defects are highlighted and deduplicated, the dashboard alerts the operator, each confirmed event is persisted with a snapshot, and evaluation/latency results are recorded honestly.
