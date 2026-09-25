# Team WEST: traffic event detection and accident anticipation

WIUT Hackathon 2026, Computer Vision track. For a fixed road camera, `solution.py` returns every
traffic event as `[start_sec, end_sec, label]` (Part A) and a causal per-frame accident risk
(Part B).

## Run

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json --team WEST
python evaluate.py --pred predictions.json --validate-only
```

**Weights** are committed in `weights/` (`yolov8s.pt` 22 MB, `yolov8n.pt` 6 MB; 28 MB in total).
`bash weights/download.sh` restores them and verifies their SHA-256. Nothing is downloaded at run
time: `src/__init__.py` sets `YOLO_OFFLINE`, so Ultralytics neither probes the network nor sends
analytics.

**Hardware.** On a CUDA GPU the pipeline runs YOLOv8s at 1280 px, 10 fps for Part A and 6 fps for
Part B. Without a GPU it switches to YOLOv8n (960 px, 5 fps for Part A; 640 px, 4 fps for Part B),
so a CPU-only machine still keeps inside the 3× budget. A 10 s 4K clip took 9.5 s for Parts A+B
together on a 4-core CPU (0.95× real time). Both parts also watch their own clock: if a video runs
late they thin detection instead of overrunning.

`run_submission.py` and `evaluate.py` are the organisers' files, unchanged.

## Approach

```
Part A (offline, whole video)
  video ─► strided prefetching reader ─► YOLOv8 (COCO) ─► ByteTrack per category
        ─► smoothing + ID-switch stitching ─► scene model (learned from traffic)
        ─► 10 rules ─► segment merge ─► [[start, end, label], ...]
             ▲ signal state from the lit bulb's colour in a fixed ROI

Part B (causal, frame by frame; never sees Part A)
  frame ─► own YOLOv8 + ByteTrack ─► per-pair time & distance of closest approach,
        hard-braking cue ─► held hazard ─► P(accident starts within 5 s)
```

| Stage | Implementation | Learned or rule |
|---|---|---|
| Detection | YOLOv8s/n, COCO-pretrained (`src/detection.py`) | learned, not fine-tuned |
| Tracking | ByteTrack from `supervision`, one tracker per category (vehicle, person, animal); offline centred smoothing, speeds in object sizes per second, stitching of fragments of one object (`src/tracking.py`) | algorithmic |
| Scene | carriageway mask, per-cell lane heading and main traffic directions, all from each video's moving vehicles; crossings, stop line and signal ROI placed by hand in 3840×2160 coordinates (`src/scene.py`, `src/config.py`) | estimated from data (no labels) + hand geometry |
| Signal | lit red/green pixels in the ROI, 1 s majority vote; signal rules switch off if red and green are never both seen (`src/signal_state.py`) | rule |
| Events | one function per class over complete trajectories; boundaries follow the annotation conventions (`src/rules.py`) | rules |
| Risk | constant-velocity closest approach for every nearby pair, same-direction pairs down-weighted, hard-braking cue (`src/risk.py`) | hand-calibrated model |

Classes emitted: accident, red_light, stop_line, wrong_way, illegal_u_turn, stopped_vehicle,
jaywalking, failure_to_yield, congestion, road_obstacle. Not emitted: near_miss, illegal_turn,
solid_line_crossing and fire_smoke. Under macro-F1, a predicted class that the test set lacks adds a
zero to the average, so we only emit classes whose rule is specific. `ENABLED_CLASSES` in
`src/config.py` switches classes on and off. The website's Approach section has the full rule for
every class.

### Code map

| Path | Purpose |
|---|---|
| `solution.py` | the competition interface |
| `src/pipeline.py` | `observe` (decode, detect, track: expensive, cacheable) and `interpret` (scene + rules: seconds) |
| `src/config.py` | every threshold, in one place |
| `src/viz.py` | drawing for rendered videos, thumbnails and EDA images |
| `scripts/dev_eval.py` | score Part A on our own labels with `evaluate.py`'s metric; tracks are cached in `.cache/` |
| `scripts/build_scene_prior.py` | sum the samples' flow fields into `weights/scene_prior.npz` (optional prior) |
| `scripts/build_site.py` | annotated sample videos, event thumbnails, EDA data and images for the website |
| `web/` | website and live demo (FastAPI + static page) |
| `tests/` | rule tests on synthetic trajectories, component tests, and an end-to-end harness run |

## Sample videos

- `samples/sample_test.mp4`: a 20 s, 1080p cut of the camera, used to calibrate the scene geometry
  (crossings, stop line, carriageway and islands in `src/config.py`) and to generate
  `predictions_samples.json`.
- `samples/annotated_preview_15s.mp4`: a rendering from our first pipeline (boxes burned in). It is
  not an input: run the harness on `samples/sample_test.mp4` (or the organisers' originals), not on
  the whole folder.
- The organisers' four full-length 4K sample videos are linked from their Drive folder and are too
  large for git.

## Development loop

```bash
pip install -r requirements-web.txt
python -m pytest                                   # 35 tests, about 15 s on CPU
python scripts/dev_eval.py --videos samples/ --gt labels/dev_labels.json      # per-class F1
python scripts/dev_eval.py --videos samples/ --gt labels/dev_labels.json --disable congestion
python scripts/build_scene_prior.py --videos samples/                        # optional prior
python run_submission.py --videos samples/ --out predictions_samples.json --team WEST
python scripts/build_site.py --videos samples/ --pred predictions_samples.json
uvicorn web.app:app --port 7860                    # http://127.0.0.1:7860
```

Dev labels can be made in the website's Label section, which exports the `ground_truth.json` format.

## Website

`web/` is the team website: demo, results, EDA, approach, report, team and label tool. To publish
it on a Hugging Face Space (Docker SDK, free CPU), run
`bash deploy/huggingface/deploy.sh https://huggingface.co/spaces/<user>/<space>`. Team members and
links live in `web/static/team.json`; failure notes live in `web/static/failures.json`.

## Determinism

Seeds are fixed (`random`, NumPy, PyTorch; `cudnn.deterministic`, no benchmark mode) in
`src/detection.py`. Tracking and rules are deterministic, and Part B keeps no random state.
`tests/test_end_to_end.py` runs the harness twice and asserts identical output (checked on CPU).
Results differ between the GPU and CPU profiles, which use different models, and fp16 GPU
inference can differ slightly from CPU. `predictions_samples.json` must be generated on the
evaluation-class GPU.

## Datasets, models and licences

- No dataset was used for training. The only data are the organisers' sample videos, used for
  development and the optional scene prior.
- YOLOv8 weights and the `ultralytics` package, Ultralytics, AGPL-3.0.
- ByteTrack as implemented in `supervision`, Roboflow, MIT.
- OpenCV (Apache-2.0), PyTorch (BSD-3), FastAPI (MIT), imageio-ffmpeg (BSD-2; bundles FFmpeg, LGPL/GPL).

## Team

| Member | Role | Contributions |
|---|---|---|
| Arslan | Team lead, computer vision | pipeline, calibration, rules, risk model |
| _Member 2_ | ML and evaluation | dev labels, threshold tuning |
| _Member 3_ | website and visualisation | website, demo, EDA |

_Names, contributions and links are to be completed by the team (also in `web/static/team.json`)._
