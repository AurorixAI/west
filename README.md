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

`run_submission.py` and `evaluate.py` are the organisers' files, unchanged (byte-identical to the
starter kit).

`predictions_samples.json` was produced with the GPU configuration forced on a CPU:
`WEST_PROFILE=gpu WEST_NO_THIN=1 python run_submission.py --videos samples/sample_test.mp4 --out predictions_samples.json --team WEST --time-factor 20`.
`WEST_PROFILE` forces a profile, `WEST_NO_THIN` disables the time guard, and `--time-factor` is the
harness's own development option. On a T4 the plain command gives the same configuration within the
normal budget; fp16 may shift boundaries by a frame.

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
| Scene | crossings, stop line, carriageway, signal heads and signs placed once on a reference frame (`src/config.py`), then aligned to every video by ORB feature matching and a RANSAC homography (`src/registration.py`, reference image `weights/scene_reference.jpg`); another camera switches the zone rules off. Lane headings and main traffic directions learned from each video's moving vehicles (`src/scene.py`) | hand geometry, auto-aligned; estimated from data (no labels) |
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
| `scripts/eval_public.py` | score Parts A/B on public clips from other cameras (robustness check) |
| `scripts/build_scene_prior.py` | sum the samples' flow fields into `weights/scene_prior.npz` (optional prior) |
| `scripts/build_site.py` | annotated sample videos, event thumbnails, EDA data and images for the website |
| `web/` | website and live demo (FastAPI + static page) |
| `tests/` | rule tests on synthetic trajectories, component tests, and an end-to-end harness run |

## Sample videos

- `samples/sample_test.mp4`: a 20 s, 1080p cut of the camera, used to calibrate the scene geometry
  (crossings, stop line, traced carriageway and islands in `src/config.py`) and to generate
  `predictions_samples.json`. On it the pipeline reports two jaywalkers (checked frame by frame),
  two cars standing at the bus-stop kerb (`stopped_vehicle`, debatable), a car crossing the corner
  zebra at speed past a pedestrian (`failure_to_yield`), and no accident risk above 0.40.
- `samples/annotated_preview_15s.mp4`: a rendering from our first pipeline (boxes burned in). It is
  not an input: run the harness on `samples/sample_test.mp4` (or the organisers' originals), not on
  the whole folder.
- The organisers' four full-length 4K sample videos are linked from their Drive folder and are too
  large for git.

## Robustness on other cameras

The geometry is calibrated on one camera; everything else has to work on any. We scored the
pipeline on 63 public CCTV clips from other cameras with `scripts/eval_public.py` (evaluate.py's
metric, the evaluation-GPU configuration run on a CPU): 51 crashes from TAD, with accident times
from NVIDIA's AI City Challenge 2026 Track 3 temporal annotations, and 12 accident-free TAD clips.

| | before | after the fixes below |
|---|---|---|
| Part B score | 0.000 | 0.096 |
| crashes with an alarm before impact | 0 / 51 | 6 / 51 |
| alarms that were right | – | 6 / 7 |
| alarms on accident-free clips | 0 / 12 | 0 / 12 |

What the clips exposed:

- **Start-up charged to the first video.** `run_submission.py` starts each video's clock after
  importing `solution.py`. With lazy loading the first clip carried the weights load and predictor
  set-up (84 s against a 22 s budget) and was scored as empty. `solution.py` now warms the detectors
  up at import.
- **Fast cars were invisible to the tracker.** At a few detections per second a car moves more
  than its own length, its boxes stop overlapping, and ByteTrack never confirms it. Cars are now
  associated with buffered IoU (boxes widened by half their size for matching only). Two-wheelers
  have their own tracker with a small buffer, because widened boxes swapped the ids of a motorcycle
  and a bicycle riding side by side on our clip.
- **Late accidents could not fire.** The accident rule wanted 5 s of standing wrecks from 3 s after
  impact; near the end of a video it now uses what is left (at least 1.5 s).

What they did not fix: alarms come late (0.08 s before impact on average; in 20 of the clips the
crash happens in the first 1.5 s), and the accident rule stays strict (0 of 14 crashes on a
subset, no false events), because a knocked-down pedestrian leaves the detector's view while the
rule wants both parties seen standing. Loosening either would add false alarms on the competition
camera, where our own clip peaks at 0.40.

```bash
python scripts/eval_public.py --videos <clips> --gt <ground_truth.json>            # Part B
WEST_PROFILE=gpu WEST_NO_THIN=1 python scripts/eval_public.py --videos <clips> --gt <gt.json> --part-a
```

## Development loop

```bash
pip install -r requirements-web.txt
python -m pytest                                   # 53 tests, about 35 s on CPU
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

- No dataset was used for training. The organisers' sample videos were used for development and the
  optional scene prior.
- Evaluation only (not in the repository, not used for training): TAD (Traffic Anomaly Dataset,
  Kaggle `nikanvasei/traffic-anomaly-dataset-tad`) clips, with accident times from NVIDIA's
  `PhysicalAI-Traffic-Anomaly-Reasoning` annotations (CC-BY-4.0).
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
