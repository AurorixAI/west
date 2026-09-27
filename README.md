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

**Weights** are committed in `weights/`: `west_yolo11s.pt` (19 MB, our detector, YOLO11s fine-tuned
on this camera; see [Detector training](#detector-training)), `yolov8n.pt` (6 MB, CPU profile) and
`yolov8s.pt` (22 MB, the COCO baseline we compare against); 47 MB in total.
`bash weights/download.sh` restores the COCO ones and verifies all SHA-256 sums. Nothing is downloaded at run
time: `src/__init__.py` sets `YOLO_OFFLINE`, so Ultralytics neither probes the network nor sends
analytics.

**Hardware.** On a CUDA GPU the pipeline runs our fine-tuned YOLO11s at 1280 px, 10 fps for Part A and 6 fps for
Part B. Without a GPU it switches to YOLOv8n (960 px, 5 fps for Part A; 640 px, 4 fps for Part B),
so a CPU-only machine still keeps inside the 3× budget. A 10 s 4K clip took 9.5 s for Parts A+B
together on a 4-core CPU (0.95× real time). Both parts also watch their own clock: if a video runs
late they thin detection instead of overrunning. A full 128 s 4K sample (C3905) took 177 s for
Parts A+B in the GPU configuration on an Apple M5 (1.4× real time; the budget is 3×).

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
  video ─► strided prefetching reader ─► YOLO11s (ours) ─► ByteTrack per category
        ─► smoothing + ID-switch stitching ─► scene model (learned from traffic)
        ─► 10 rules ─► segment merge ─► [[start, end, label], ...]
             ▲ signal state from the lit bulb's colour in a fixed ROI

Part B (causal, frame by frame; never sees Part A)
  frame ─► own YOLO11s + ByteTrack ─► per-pair time & distance of closest approach,
        hard-braking cue ─► held hazard ─► P(accident starts within 5 s)
```

| Stage | Implementation | Learned or rule |
|---|---|---|
| Detection | YOLO11s fine-tuned on this camera by self-training from a YOLO11x teacher (`src/detection.py`, `scripts/train_detector.py`); YOLOv8n COCO on CPU | learned |
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
| `scripts/train_detector.py` | detector self-training: key frames, teacher pseudo-labels, student fine-tuning, comparison |
| `scripts/dev_eval.py` | score Part A on our own labels with `evaluate.py`'s metric; tracks are cached in `.cache/` |
| `scripts/eval_public.py` | score Parts A/B on public clips from other cameras (robustness check) |
| `scripts/build_scene_prior.py` | sum the samples' flow fields into `weights/scene_prior.npz` (optional prior) |
| `scripts/build_site.py` | annotated sample videos, event thumbnails, EDA data and images for the website |
| `web/` | website and live demo (FastAPI + static page) |
| `tests/` | rule tests on synthetic trajectories, component tests, and an end-to-end harness run |

## Sample videos

- `samples/sample_test.mp4`: a 20 s, 1080p cut of the camera, used to calibrate the scene geometry
  (crossings, stop line, traced carriageway and islands in `src/config.py`) and to generate
  `predictions_samples.json`. On it the pipeline reports three jaywalking segments, a car standing
  at the bus-stop kerb (`stopped_vehicle`, debatable), one `failure_to_yield` at the end, and a peak
  accident risk of 0.30 (the alarm is 0.5).
- `samples/annotated_preview_15s.mp4`: a rendering from our first pipeline (boxes burned in). It is
  not an input: run the harness on `samples/sample_test.mp4` (or the organisers' originals), not on
  the whole folder.
- The organisers' four full-length 4K sample videos (C3896, C3897, C3902, C3905; 18.4 min in total,
  3840x2160 at 29.97 fps, H.264 4:2:2 10-bit) are linked from their Drive folder and are too large
  for git; the detector is trained on their key frames. C3896/C3897 and C3902 are framed differently
  from C3905 and `sample_test.mp4` (zoomed in or shifted), and C3905 is at dusk.

## Robustness on other cameras

The geometry is calibrated on one camera; everything else has to work on any. We scored the
pipeline on 63 public CCTV clips from other cameras with `scripts/eval_public.py` (evaluate.py's
metric, the evaluation-GPU configuration run on a CPU): 51 crashes from TAD, with accident times
from NVIDIA's AI City Challenge 2026 Track 3 temporal annotations, and 12 accident-free TAD clips.

With the COCO YOLOv8s detector the clips first scored 0.000 for Part B. What they exposed:

- **Start-up charged to the first video.** `run_submission.py` starts each video's clock after
  importing `solution.py`. With lazy loading the first clip carried the weights load and predictor
  set-up (84 s against a 22 s budget) and was scored as empty. `solution.py` now warms the detectors
  up at import.
- **Late accidents could not fire.** The accident rule wanted 5 s of standing wrecks from 3 s after
  impact; near the end of a video it now uses what is left (at least 1.5 s).
- **A car can inherit a bicycle's id.** Two-wheelers now have a ByteTrack of their own.
- **Fast cars are lost at low sample rates.** At 4–6 detections per second a car can move more
  than its own length, its boxes stop overlapping, and ByteTrack never confirms it. Buffered IoU
  (boxes widened for matching) and re-linking lost tracks raised TAD's Part B score to 0.096 (alarms
  before 6 of 51 crashes, 6 of 7 alarms right, none on the calm clips). **We did not ship it.**
  With the fine-tuned detector, on the competition camera, both swapped ids between neighbouring
  cars: a small car's track jumped onto a bus, its speed in sizes per second collapsed, and the
  accident rule saw a crash; the risk reached 0.91 on a calm clip (0.30 without them). The
  competition camera decides. The shipped pipeline (fine-tuned YOLO11s, plain tracker) scores 0.039
  on TAD: alarms before 2 of 51 crashes, 2 of 5 alarms right, none on the calm clips.

What stays open: alarms on other cameras come late or not at all, and the accident rule is strict
(0 of 14 TAD crashes on a subset, no false events): a knocked-down pedestrian leaves the detector's
view, while the rule wants both parties seen standing. Loosening it would fire on every car that
stops in front of a pedestrian on the competition camera.

```bash
python scripts/eval_public.py --videos <clips> --gt <ground_truth.json>            # Part B
WEST_PROFILE=gpu WEST_NO_THIN=1 python scripts/eval_public.py --videos <clips> --gt <gt.json> --part-a
```

## Development loop

```bash
pip install -r requirements-web.txt
python -m pytest                                   # 55 tests, about 30 s on CPU
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

## Detector training

The COCO detector does not know this camera: far pedestrians on the pavement, vans and trucks seen
from above, and dusk. With no labels, we self-train (`scripts/train_detector.py`):

1. **Frames.** 1086 key frames of the four sample videos (one a second, decoded without the frames
   between them). The last 20 % of every video, after a 5 s gap, is held out, so all viewpoints and
   the dusk light are in both halves.
2. **Teacher.** YOLO11x labels every 4K frame twice: whole at 1920 px and as a 3x2 grid of
   overlapping tiles at native resolution. Weighted box fusion merges the views; car/bus/truck are
   fused together (a van is one or the other depending on scale).
3. **Clean-up.** Only the classes the rules use are labelled. People seen through a windscreen are
   dropped (28 % of the teacher's "persons"; the rules would take drivers for pedestrians on the
   road). Boxes with confidence 0.2-0.4 and boxes under 6 px are painted grey, so they teach neither
   "object" nor "background".
4. **Student.** YOLO11s from COCO, 80-class head kept (class ids unchanged), 10 epochs on 640 px
   tiles cut from frames shrunk to 1280 px, the scale the pipeline runs at. Full 1280 px frames need
   more than 16 GB on an Apple GPU; tiles keep object sizes identical. About 16 min per epoch on an
   Apple M5.

Held-out frames, whole, at 1280 px, against the teacher's labels:

| Model | mAP50 | mAP50-95 | Recall | person AP50 | truck AP50 | bicycle AP50 |
|---|---|---|---|---|---|---|
| YOLOv8s, COCO (previous detector) | 0.714 | 0.559 | 0.658 | 0.818 | 0.569 | 0.423 |
| YOLO11s, COCO | 0.738 | 0.594 | 0.628 | 0.819 | 0.577 | 0.448 |
| **YOLO11s, fine-tuned (ours)** | **0.850** | **0.731** | **0.803** | **0.966** | **0.794** | **0.581** |

The reference is the teacher, not people: the table says how close the student gets to the teacher
on unseen frames; the teacher itself was checked by eye on samples.

The sharper detector exposed three rules that had relied on the old one missing things; each now
has a regression test: a "car" reflected in a glass facade made a U-turn (U-turns must now be on
the carriageway), a courier walking his scooter over the zebra "failed to yield" to himself, and a
cyclist and his bicycle, tracked separately, raised the accident risk to 0.88.

```bash
python scripts/train_detector.py extract --videos videos/ samples/sample_test.mp4
python scripts/train_detector.py label                      # ~3 s per frame on an Apple M5
python scripts/train_detector.py train                      # YOLO11s, 10 epochs
python scripts/train_detector.py compare --models yolov8s.pt yolo11s.pt runs/west_11s/weights/best.pt
cp runs/west_11s/weights/best.pt weights/west_yolo11s.pt
```

## Datasets, models and licences

- No public dataset was used for training. The only training data are the organisers' sample
  videos: their key frames, labelled by the YOLO11x teacher, train our detector; they are also used
  for development and the optional scene prior.
- Evaluation only (not in the repository, not used for training): TAD (Traffic Anomaly Dataset,
  Kaggle `nikanvasei/traffic-anomaly-dataset-tad`) clips, with accident times from NVIDIA's
  `PhysicalAI-Traffic-Anomaly-Reasoning` annotations (CC-BY-4.0).
- YOLO11x (teacher, not shipped), YOLO11s (student base), YOLOv8s/n weights and the `ultralytics`
  package, Ultralytics, AGPL-3.0.
- ByteTrack as implemented in `supervision`, Roboflow, MIT.
- OpenCV (Apache-2.0), PyTorch (BSD-3), FastAPI (MIT), imageio-ffmpeg (BSD-2; bundles FFmpeg, LGPL/GPL).

## Team

| Member | Role | Contributions |
|---|---|---|
| Arslan | Team lead, computer vision | pipeline, calibration, rules, risk model |
| _Member 2_ | ML and evaluation | dev labels, threshold tuning |
| _Member 3_ | website and visualisation | website, demo, EDA |

_Names, contributions and links are to be completed by the team (also in `web/static/team.json`)._
