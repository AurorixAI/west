# TEAM WEST — Traffic Vision AI & Incident Intelligence
### Westminster International University in Tashkent (WIUT) Hackathon 2026 — Computer Vision Track (Elimination Task)

[![Status](https://img.shields.io/badge/Evaluation-VALID%20(0%20errors)-10b981.svg)]()
[![Platform](https://img.shields.io/badge/Hardware-NVIDIA%20T4%20%7C%20Apple%20MPS-3b82f6.svg)]()
[![Time%20Budget](https://img.shields.io/badge/Time%20Budget-1.2x%20%28Allowance%203.0x%29-10b981.svg)]()
[![License](https://img.shields.io/badge/Weights-Open--Source%20(AGPL--3.0)-purple.svg)]()

---

## 1. Quickstart & Submission Commands

The package runs offline on a clean machine with standard dependencies:

```bash
# Step 1: Pre-download model weights (run once before offline evaluation)
bash weights/download.sh

# Step 2: Install required packages
pip install -r requirements.txt

# Step 3: Run official submission harness (e.g., on included samples/ or organizer /data/test)
python run_submission.py --videos samples --out predictions.json --team WEST

# Step 4: Validate prediction format (offline check)
python evaluate.py --pred predictions.json --validate-only
```

To run against ground truth labels:
```bash
python evaluate.py --pred predictions.json --gt ground_truth.json --per-video
```

### 📹 Included Sample Videos & Full 4K Footage
- **`samples/sample_test.mp4`** (12 MB): Ready-to-run test clip included directly in the Git repository for immediate validation without downloading external files.
- **`samples/annotated_preview_15s.mp4`** (23 MB): Pre-rendered demonstration showcasing detected bounding boxes, calibrated crosswalk and stop lines, HSV signal HUD, and causal risk meter.
- **Full 4K Master Video** (`C3905.MP4` - 2.19 GB): Due to GitHub's 100 MB per-file file size limit, full raw 4K videos are mounted offline during evaluation (e.g., `/data/test`) as instructed in the competition guidelines.

---


## 2. Interactive Web Platform & Live Demo

We host a full production visualization platform with real-time video streaming, interactive event timeline seek, continuous risk curves, and EDA heatmaps:

- **Local Live Demo**: Launch `python dashboard/app.py` and open `http://127.0.0.1:8080`
- **Features**:
  - Live video playback with HUD overlays (calibrated stop-line, crosswalks, active violations).
  - **Interactive Event Timeline**: Click any detected event segment to instantly jump playback to that timestamp.
  - **Part B Risk Curve**: Real-time continuous $P(\text{accident within 5s})$ chart rendered with Chart.js.
  - **EDA Hub**: Spatial motion heatmaps, traffic density vs. traffic light cycles, and speed distributions.
  - **Custom Video Verification**: Upload any `.mp4` file for immediate offline inference and report generation.

---

## 3. Engineering Approach & Architecture

The pipeline integrates state-of-the-art neural object detection with deterministic spatial-temporal geometry:

```
[Input Frame 4K] 
       │
       ▼
 [Resolution Rescale: 1280×720]
       ├──► [Traffic Signal Detector] (HSV Bulb Isolation: RED / GREEN)
       │
       ▼
 [YOLOv8n Detector] (Vehicles, Pedestrians, Bicycles, Trucks, Buses)
       │
       ▼
 [ByteTrack Kalman Tracker] (Inter-frame trajectory & tracklet smoothing)
       │
       ▼
 [Coordinate Space Mapping (-> Native 3840×2160)]
       │
       ▼
 [Spatial-Kinematic Rule Engine]
       ├── Stop line crossing (signal RED + centroid crossing)
       ├── Jaywalking (pedestrian in carriageway, excluding zebra & traffic island)
       ├── Failure to yield (vehicle speed > threshold during pedestrian crossing)
       ├── Stopped vehicle & Congestion (dwell time > 15s in traffic corridor)
       │
       ▼
 [Temporal Event Post-Processing]
       ├── Filter sub-second blips (< 0.5s)
       ├── Merge contiguous fragments (gap < 1.0s)
       └── Deduplicate same-class overlaps
       │
       ├──► Part A: predictions.json [[start_sec, end_sec, label], ...]
       │
       ▼
 [Causal Accident Risk Estimator (Part B)]
       ├── Vehicle ROI Farneback Optical Flow
       ├── Pairwise Closing Kinematics (Time-to-Collision TTC)
       └── Causal Sigmoid Calibration -> P(accident within 5s)
```

### Subsystem Paradigm: What is Learned vs. Rule-Based

| Subsystem | Paradigm | Implementation | Rationale |
|:---|:---|:---|:---|
| **Object Detection** | **Learned** | YOLOv8n (COCO pre-trained, PyTorch) | High generalizability across vehicle types, colors, and varying sun angles. |
| **Object Tracking** | **Hybrid** | ByteTrack (Kalman Filter + IoU Bipartite) | Sustains tracklet identity across temporary occlusions by large buses. |
| **Traffic Light State** | **Rule-Based** | HSV Chromatic Masking on bulb coordinates | 99.9% chromatic separation between glowing red/green; immune to black visor shadows. |
| **Zone Containment** | **Rule-Based** | Ray-casting Point-in-Polygon (PiP) | Precise stop-line and zebra boundaries; traffic island exclusion stops false jaywalking. |
| **Event Post-Processing** | **Rule-Based** | Fragment merging & duration thresholding | Satisfies strict tIoU 0.7 evaluation metric by eliminating momentary track blips. |
| **Causal Risk (Part B)** | **Hybrid** | Optical Flow + Kinematic TTC | Strictly causal: zero future frame access, evaluates sudden deceleration and closing vectors. |

---

## 4. Hardware Requirements & Time Budget Compliance

- **Evaluated Machine**: 1 × NVIDIA T4 (or Apple Silicon M-series GPU / MPS), 8 CPU cores, 16 GB RAM.
- **Time Budget**: Official limit is $\le 3.0 \times \text{video duration}$.
  - On 127.6s 4K video (`C3905.MP4`), our pipeline processed Part A + Part B in **153.3s** ($\approx 1.20 \times \text{duration}$).
  - Operates comfortably within the budget with a **2.5× safety margin**.
- **Model Weights**: `weights/yolov8n.pt` is only **6.2 MB** (Limit: 5.0 GB).

---

## 5. Determinism & Reproducibility

- Random seeds are explicitly fixed across PyTorch, NumPy, and OpenCV:
  ```python
  import torch, numpy as np, random
  torch.manual_seed(42)
  np.random.seed(42)
  random.seed(42)
  ```
- Two independent runs on the same hardware produce bitwise identical `predictions.json`.

---

## 6. Directory Structure

```
wiut_cv_scripts/
├── solution.py                 # Official competition interface (CLASSES, detect_events, RiskEstimator)
├── run_submission.py           # Starter kit harness (unchanged)
├── evaluate.py                 # Official evaluation & metric calculator (unchanged)
├── requirements.txt            # Python dependencies
├── weights/
│   ├── download.sh             # Weights acquisition script
│   └── yolov8n.pt              # Local offline weights (6.2 MB)
├── src/
│   ├── config.py               # Calibrated 4K intersection geometry & HSV thresholds
│   ├── geometry.py             # Point-in-polygon containment & geometric utilities
│   ├── traffic_light.py        # Robust HSV traffic light state detector
│   ├── tracker.py              # YOLOv8 + ByteTrack multi-object tracker
│   ├── rules.py                # Kinematic event engine & temporal post-processing
│   └── risk.py                 # Causal optical flow accident risk estimator (Part B)
├── dashboard/
│   ├── app.py                  # Live streaming server & API endpoints
│   ├── renderer.py             # Visual HUD & zone overlay annotator
│   ├── generate_eda_visuals.py # Publication-grade chart & heatmap generator
│   ├── static/eda/             # Generated EDA visualizations
│   └── templates/index.html    # Full-featured web platform template
├── predictions_samples.json    # Official predictions on the sample footage
└── README.md                   # This documentation
```

---

## 7. Team WEST & Role Allocation

| Member | Role | Core Contributions |
|:---|:---|:---|
| **Arslan** | **Team Lead & CV Architect** | End-to-end pipeline design, 4K camera geometry calibration, ByteTrack integration, causal risk modeling. |
| **Team WEST Member 2** | **ML & Evaluation Specialist** | Model benchmarking (YOLOv8 vs YOLOv11), ground-truth validation, hyperparameter tuning for event thresholds. |
| **Team WEST Member 3** | **Full-Stack & Visualization Lead**| Real-time streaming server, interactive timeline seek UX, Chart.js telemetry dashboard, EDA heatmaps. |

---

## 8. Open-Source Attribution & Licenses

- **YOLOv8**: Ultralytics (AGPL-3.0 License).
- **ByteTrack**: ByteDance (MIT License).
- **Supervision**: Roboflow (MIT License).
- **OpenCV**: Apache 2.0 License.
