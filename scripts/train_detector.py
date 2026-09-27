#!/usr/bin/env python3
"""Adapt the detector to the competition camera by self-training on the sample videos.

    python scripts/train_detector.py extract --videos videos/ samples/sample_test.mp4
    python scripts/train_detector.py label             # teacher pseudo-labels (slow: ~3 s/frame on MPS)
    python scripts/train_detector.py train --student yolo11s.pt --name west_11s   # ~16 min/epoch on an M5
    python scripts/train_detector.py compare --models yolov8s.pt yolo11s.pt runs/west_11s/weights/best.pt

There are no labels for this camera, so a large open-weights teacher (YOLO11x,
COCO) labels frames of the sample videos with test-time tiling: the whole 4K
frame at 1920 px plus a 3x2 grid of tiles at native resolution, fused by
weighted box fusion. The small student that runs in the submission is then
fine-tuned on those labels at its deployment scale (in 640-px tiles), so it learns this
viewpoint, its scale range and its lighting from the teacher.

Only the classes the pipeline uses are labelled (config.CATEGORY_OF_CLASS);
the student keeps the 80-class COCO head so class ids stay unchanged. Teacher
boxes of middling confidence are neither positives nor negatives: they are
painted grey in the training image so the student is not taught either way.

Split: the last 20 % of every video (after a 5 s gap) is validation, so every
lighting condition and viewpoint is in both halves and no frame of the
validation time span is ever trained on.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402

WORK = ROOT / ".cache" / "det"
FRAMES = WORK / "frames"          # 4K key frames + index.json
LABELS = WORK / "teacher.json"    # fused teacher boxes per frame
DATASET = WORK / "dataset"        # YOLO-format dataset for the student
RUNS = ROOT / "runs"

VAL_FRACTION, VAL_GAP_SEC = 0.2, 5.0
KEEP_CLASSES = sorted(C.CATEGORY_OF_CLASS)

TEACHER = "yolo11x.pt"
TEACHER_FULL_SZ, TEACHER_TILE_SZ = 1920, 1280
TILE_GRID = (3, 2)                # columns, rows over the frame
TILE_OVERLAP = 0.15
TEACHER_MIN_CONF = 0.15
FUSE_IOU = 0.55
POS_CONF, IGNORE_CONF = 0.40, 0.20
MIN_BOX_PX = 6                    # at deployment scale: smaller boxes are ignored
OCCUPANT_IOA, OCCUPANT_HEIGHT = 0.85, 0.6

STUDENT_IMGSZ = C.DETECT_WIDTH    # frames are shrunk to the pipeline's detection width...
TILE = 640                        # ...and cut into tiles of this size for training
TILE_MIN_VISIBLE = 0.5
GREY = (114, 114, 114)            # Ultralytics' letterbox colour


def resolve_weights(name: str) -> str:
    """weights/ first, then the work folder; otherwise Ultralytics downloads it."""
    for folder in (C.WEIGHTS_DIR, WORK):
        if (folder / name).exists():
            return str(folder / name)
    return name


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


# --------------------------------------------------------------------------
# 1. key frames
# --------------------------------------------------------------------------
def extract(videos: list[Path]) -> None:
    """Decode only the key frames (about one a second): fast, and the best-quality pictures."""
    FRAMES.mkdir(parents=True, exist_ok=True)
    index = {}
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
        cap.release()
        out = FRAMES / video.stem
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir()
        cmd = [ffmpeg_exe(), "-hide_banner", "-v", "info", "-skip_frame", "nokey", "-i", str(video),
               "-vf", "showinfo", "-fps_mode", "passthrough",
               "-q:v", "2", str(out / "%05d.jpg")]
        log = subprocess.run(cmd, capture_output=True, text=True, check=True).stderr
        times = [float(t) for t in re.findall(r"pts_time:([0-9.]+)", log)]
        files = sorted(out.glob("*.jpg"))
        assert len(files) == len(times), (video, len(files), len(times))
        split_t = duration * (1 - VAL_FRACTION)
        for f, t in zip(files, times):
            split = "val" if t >= split_t else "train" if t < split_t - VAL_GAP_SEC else None
            if split:
                index[f"{video.stem}/{f.name}"] = {"video": video.name, "t": round(t, 3), "split": split}
            else:
                f.unlink()
        n_val = sum(v["split"] == "val" and v["video"] == video.name for v in index.values())
        print(f"{video.name}: {duration:.0f} s, {len(files)} key frames kept, {n_val} for validation")
    (FRAMES / "index.json").write_text(json.dumps(index, indent=1))


# --------------------------------------------------------------------------
# 2. teacher
# --------------------------------------------------------------------------
def tiles(w: int, h: int) -> list[tuple[int, int, int, int]]:
    cols, rows = TILE_GRID
    tw, th = int(w / cols * (1 + TILE_OVERLAP)), int(h / rows * (1 + TILE_OVERLAP))
    xs = np.linspace(0, w - tw, cols).astype(int)
    ys = np.linspace(0, h - th, rows).astype(int)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def ioa_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Share of each box of ``a`` that lies inside each box of ``b``."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    return inter / (np.prod(a[:, 2:] - a[:, :2], axis=1)[:, None] + 1e-9)


def occupants(boxes: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """People seen through a car or bus window: inside the vehicle's box and much
    shorter than it. The teacher finds drivers; the rules would take them for
    pedestrians on the carriageway, so they are not labelled at all. A pedestrian
    beside a car is about as tall as the car's box at this angle and is kept."""
    person = cls == C.COCO_PERSON
    vehicle = np.isin(cls, C.COCO_VEHICLE)
    out = np.zeros(len(boxes), bool)
    if not person.any() or not vehicle.any():
        return out
    inside = ioa_matrix(boxes[person], boxes[vehicle]) >= OCCUPANT_IOA
    heights = boxes[:, 3] - boxes[:, 1]
    shorter = heights[person][:, None] < OCCUPANT_HEIGHT * heights[vehicle][None, :]
    out[np.where(person)[0]] = (inside & shorter).any(1)
    return out


def fuse(boxes: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted box fusion. Vehicle classes (car/bus/truck) are fused together:
    the same van is often a car at one scale and a truck at another."""
    group = np.array([2 if c in C.COCO_VEHICLE else c for c in cls])
    order = np.argsort(-conf)
    boxes, conf, cls, group = boxes[order], conf[order], cls[order], group[order]
    used = np.zeros(len(boxes), bool)
    out_b, out_c, out_k = [], [], []
    iou = iou_matrix(boxes, boxes) if len(boxes) else np.zeros((0, 0))
    for i in range(len(boxes)):
        if used[i]:
            continue
        members = np.where(~used & (group == group[i]) & (iou[i] >= FUSE_IOU))[0]
        used[members] = True
        w = conf[members]
        out_b.append((boxes[members] * w[:, None]).sum(0) / w.sum())
        # a box seen by several views is more certain than any single view says
        out_c.append(min(1.0, conf[members].max() * (1 + 0.1 * (len(members) - 1))))
        # class: the confidence-weighted vote among the members
        votes: dict[int, float] = {}
        for k, c in zip(cls[members], w):
            votes[int(k)] = votes.get(int(k), 0.0) + float(c)
        out_k.append(max(votes, key=votes.get))
    return (np.array(out_b, np.float32).reshape(-1, 4), np.array(out_c, np.float32),
            np.array(out_k, np.int32))


def teacher_predict(model, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    views = [(0, 0, w, h)] + tiles(w, h)
    crops = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in views]
    results = model.predict(crops[:1], imgsz=TEACHER_FULL_SZ, conf=TEACHER_MIN_CONF, classes=KEEP_CLASSES,
                            verbose=False)
    results += model.predict(crops[1:], imgsz=TEACHER_TILE_SZ, conf=TEACHER_MIN_CONF, classes=KEEP_CLASSES,
                             verbose=False)
    B, S, K = [], [], []
    for (x1, y1, x2, y2), r in zip(views, results):
        b = r.boxes.xyxy.cpu().numpy()
        keep = np.ones(len(b), bool)
        if (x1, y1, x2, y2) != (0, 0, w, h):
            # a box cut by an inner tile border is a fragment: the neighbouring tile or the full view has it whole
            m = 4
            keep &= ~((b[:, 0] < m) & (x1 > 0)) & ~((b[:, 2] > x2 - x1 - m) & (x2 < w))
            keep &= ~((b[:, 1] < m) & (y1 > 0)) & ~((b[:, 3] > y2 - y1 - m) & (y2 < h))
        B.append(b[keep] + [x1, y1, x1, y1])
        S.append(r.boxes.conf.cpu().numpy()[keep])
        K.append(r.boxes.cls.cpu().numpy().astype(int)[keep])
    return fuse(np.concatenate(B), np.concatenate(S), np.concatenate(K))


def label() -> None:
    from ultralytics import YOLO
    from src.detection import pick_device, seed_everything
    seed_everything()
    index = json.loads((FRAMES / "index.json").read_text())
    done = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    model = YOLO(resolve_weights(TEACHER))
    model.to(pick_device())
    todo = [k for k in sorted(index) if k not in done]
    for n, key in enumerate(todo, 1):
        img = cv2.imread(str(FRAMES / key))
        b, s, k = teacher_predict(model, img)
        done[key] = {"size": [img.shape[1], img.shape[0]], "boxes": np.round(b, 1).tolist(),
                     "conf": np.round(s, 3).tolist(), "cls": k.tolist()}
        if n % 20 == 0 or n == len(todo):
            LABELS.write_text(json.dumps(done))
            print(f"  {n}/{len(todo)} frames labelled", flush=True)


# --------------------------------------------------------------------------
# 3. dataset + student
# --------------------------------------------------------------------------
def yolo_rows(boxes: np.ndarray, cls: np.ndarray, w: int, h: int) -> str:
    return "\n".join(f"{k} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                     for (x1, y1, x2, y2), k in zip(boxes, cls))


def train_tiles(w: int, h: int) -> list[tuple[int, int]]:
    """Top-left corners of overlapping TILE x TILE crops that cover a w x h frame."""
    nx, ny = int(np.ceil(w / TILE * 1.5)), int(np.ceil(h / TILE))
    return [(int(x), int(y)) for y in np.linspace(0, h - TILE, ny) for x in np.linspace(0, w - TILE, nx)]


def build_dataset() -> Path:
    """Write the YOLO dataset at deployment scale (4K frames shrunk to DETECT_WIDTH).

    Training and validation images are TILE x TILE crops: a full frame carries
    ~370 objects, and the label assigner's memory grows with objects x anchors,
    which does not fit a 16 GB Mac at 1280 px. Objects keep exactly the pixel
    size they have in the pipeline. ``val_full`` holds the whole validation
    frames for ``compare``, which scores the models as the pipeline runs them.
    Confident teacher boxes are labels; uncertain ones are greyed out."""
    from ultralytics.utils import yaml_load
    from ultralytics.utils.checks import check_yaml
    index = json.loads((FRAMES / "index.json").read_text())
    teacher = json.loads(LABELS.read_text())
    shutil.rmtree(DATASET, ignore_errors=True)
    for sub in ("train", "val", "val_full"):
        (DATASET / "images" / sub).mkdir(parents=True)
        (DATASET / "labels" / sub).mkdir(parents=True)
    counts = {"train": 0, "val": 0, "val_full": 0, "boxes": 0, "ignored": 0, "occupants": 0}

    def write(sub: str, name: str, img: np.ndarray, boxes: np.ndarray, cls: np.ndarray) -> None:
        cv2.imwrite(str(DATASET / "images" / sub / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (DATASET / "labels" / sub / name.replace(".jpg", ".txt")).write_text(
            yolo_rows(boxes, cls, img.shape[1], img.shape[0]))
        counts[sub] += 1

    for key, meta in sorted(index.items()):
        if key not in teacher:
            continue
        lab = teacher[key]
        img = cv2.imread(str(FRAMES / key))
        s = STUDENT_IMGSZ / img.shape[1]
        img = cv2.resize(img, (STUDENT_IMGSZ, round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        boxes = np.array(lab["boxes"], np.float32).reshape(-1, 4) * s
        conf, cls = np.array(lab["conf"]), np.array(lab["cls"])
        keep = ~occupants(boxes, cls)
        boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
        counts["occupants"] += int((~keep).sum())
        small = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) < MIN_BOX_PX
        pos = (conf >= POS_CONF) & ~small
        ign = ~pos & ((conf >= IGNORE_CONF) | small)
        for x1, y1, x2, y2 in boxes[ign]:
            # do not paint over a confident object: neither one it overlaps nor one it lies on
            box = np.array([[x1, y1, x2, y2]])
            if pos.any() and (iou_matrix(box, boxes[pos]).max() > 0.3 or ioa_matrix(box, boxes[pos]).max() > 0.3):
                continue
            cv2.rectangle(img, (int(x1), int(y1)), (int(np.ceil(x2)), int(np.ceil(y2))), GREY, -1)
            counts["ignored"] += 1
        boxes, cls = boxes[pos], cls[pos]
        counts["boxes"] += len(boxes)
        stem = key.replace("/", "_")[:-4]
        split = meta["split"]
        if split == "val":
            write("val_full", f"{stem}.jpg", img, boxes, cls)
        for x, y in train_tiles(w, h):
            tile = img[y:y + TILE, x:x + TILE].copy()
            clipped = np.clip(boxes - [x, y, x, y], 0, TILE)
            area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
            seen = np.prod(np.clip(clipped[:, 2:] - clipped[:, :2], 0, None), axis=1) / np.maximum(area, 1e-6)
            for x1, y1, x2, y2 in clipped[(seen > 0) & (seen < TILE_MIN_VISIBLE)]:
                # a sliver of an object cut by the tile edge is neither a label nor background
                cv2.rectangle(tile, (int(x1), int(y1)), (int(np.ceil(x2)), int(np.ceil(y2))), GREY, -1)
            kept = seen >= TILE_MIN_VISIBLE
            write(split, f"{stem}_{x}_{y}.jpg", tile, clipped[kept], cls[kept])

    names = yaml_load(check_yaml("coco.yaml"))["names"]     # keep COCO's 80 ids
    for yaml_name, val in (("data.yaml", "images/val"), ("data_full.yaml", "images/val_full")):
        (DATASET / yaml_name).write_text(json.dumps({"path": str(DATASET), "train": "images/train",
                                                     "val": val, "names": names}))
    print(f"dataset: {counts['train']} train / {counts['val']} val tiles, {counts['val_full']} full val frames, "
          f"{counts['boxes']} boxes, {counts['ignored']} uncertain regions greyed, "
          f"{counts['occupants']} vehicle occupants dropped", flush=True)
    return DATASET / "data.yaml"


def train(student: str, name: str, epochs: int, batch: int, freeze: int) -> None:
    from ultralytics import YOLO
    from src.detection import pick_device
    data = build_dataset()
    model = YOLO(resolve_weights(student))
    model.train(data=str(data), imgsz=TILE, epochs=epochs, batch=batch, freeze=freeze,
                device=pick_device(), project=str(RUNS), name=name, exist_ok=True,
                seed=C.SEED, deterministic=True, workers=2, cache=False,
                optimizer="AdamW", lr0=5e-4, cos_lr=True, warmup_epochs=1, patience=15,
                # the camera is fixed and upright: no vertical flips or rotations; mild scale jitter
                fliplr=0.5, flipud=0.0, degrees=0.0, scale=0.3, mosaic=0.5, close_mosaic=3,
                hsv_v=0.5, plots=True)


def compare(models: list[str]) -> None:
    """mAP of each model against the teacher's labels on the held-out time spans."""
    from ultralytics import YOLO
    from src.detection import pick_device
    data = DATASET / "data_full.yaml"
    rows = []
    for m in models:
        res = YOLO(resolve_weights(m)).val(data=str(data), imgsz=STUDENT_IMGSZ, batch=4, device=pick_device(),
                                  classes=KEEP_CLASSES, conf=0.001, plots=False, verbose=False,
                                  project=str(RUNS), name="compare", exist_ok=True)
        per = {res.names[int(c)]: round(float(ap), 3) for c, ap in zip(res.box.ap_class_index, res.box.ap50)}
        rows.append((m, res.box.map50, res.box.map, res.box.mp, res.box.mr, per))
    print(f"\n{'model':45s} mAP50  mAP50-95  P      R")
    for m, a50, a, p, r, per in rows:
        print(f"{m:45s} {a50:.3f}  {a:.3f}     {p:.3f}  {r:.3f}  {per}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--videos", nargs="+", type=Path, required=True, help="video files or folders")
    sub.add_parser("label")
    t = sub.add_parser("train")
    t.add_argument("--student", default="yolo11s.pt")
    t.add_argument("--name", default="west_11s")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--freeze", type=int, default=0, help="freeze the first N layers")
    c = sub.add_parser("compare")
    c.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    if args.cmd == "extract":
        videos = []
        for p in args.videos:
            videos += sorted(q for q in p.iterdir() if q.suffix.lower() == ".mp4") if p.is_dir() else [p]
        extract(videos)
    elif args.cmd == "label":
        label()
    elif args.cmd == "train":
        train(args.student, args.name, args.epochs, args.batch, args.freeze)
    else:
        compare(args.models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
