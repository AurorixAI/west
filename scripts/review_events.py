#!/usr/bin/env python3
"""Export every reported event for review: a contact sheet per event plus the cached tracks.

    WEST_PROFILE=gpu WEST_NO_THIN=1 python scripts/review_events.py --videos videos/ --out review/

For each event the pipeline reports, one JPG of six frames from 1 s before the
event to 1 s after it, cropped around the actors, with their boxes and track
ids, the crossings and stop line, the signal state and a red border on frames
inside the event. Alongside: events.json (every event with its actors),
durations.json, and the Observation of every video (tracks, signal, camera
transform) as a gzipped pickle, so rule thresholds can be tuned on another
machine without the videos or a GPU:

    obs = pickle.loads(gzip.open("review/obs/C3905.pkl.gz").read())
    events = interpret(obs).events

Reviewed verdicts become dev labels in evaluate.py's ground-truth format.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.dev_eval import cached_observation  # noqa: E402
from src.pipeline import interpret  # noqa: E402
from src.scene import SceneGeometry  # noqa: E402
from src.signal_state import NAMES  # noqa: E402
from src.viz import TrackIndex, draw_geometry, draw_objects  # noqa: E402

PAD_SEC = 1.0
N_FRAMES = 6
TILE_W, TILE_H, COLS = 800, 450, 2          # 2 x 3 tiles: a 1600 px wide sheet
MIN_CROP_W = 1280                           # full-resolution pixels: never zoom in further than 1.6x
HEADER = 64


def crop_box(tracks: dict, actors: tuple, t0: float, t1: float, w: int, h: int) -> tuple[int, int, int, int]:
    """16:9 window around everything the actors cover during [t0, t1]."""
    boxes = []
    for tid in actors:
        tr = tracks.get(tid)
        if tr is not None:
            m = (tr.t >= t0) & (tr.t <= t1)
            boxes.append(tr.box[m] if m.any() else tr.box)
    if not boxes:
        return 0, 0, w, h
    b = np.concatenate(boxes)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    cw = max(MIN_CROP_W, (x2 - x1) * 1.6, (y2 - y1) * 1.6 * 16 / 9)
    cw = min(cw, w)
    ch = min(cw * 9 / 16, h)
    cw = ch * 16 / 9
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X1 = int(np.clip(cx - cw / 2, 0, w - cw))
    Y1 = int(np.clip(cy - ch / 2, 0, h - ch))
    return X1, Y1, int(X1 + cw), int(Y1 + ch)


def sheet(cap: cv2.VideoCapture, analysis, ev: tuple, title: str) -> np.ndarray:
    s, e, label, actors = ev
    info = analysis.info
    t0, t1 = max(0.0, s - PAD_SEC), min(info.duration - 1 / info.fps, e + PAD_SEC)
    tracks = {tr.tid: tr for tr in analysis.tracks}
    index = TrackIndex(analysis.tracks)
    geom = SceneGeometry(info.width, info.height, analysis.obs.transform, analysis.obs.camera_known)
    X1, Y1, X2, Y2 = crop_box(tracks, actors, t0, t1, info.width, info.height)
    scale = (X2 - X1) / TILE_W
    highlight = {tid: label for tid in actors}
    rows = int(np.ceil(N_FRAMES / COLS))
    out = np.full((HEADER + rows * TILE_H, COLS * TILE_W, 3), 24, np.uint8)
    for k, t in enumerate(np.linspace(t0, t1, N_FRAMES)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * info.fps)))
        ok, full = cap.read()
        if not ok:
            continue
        # draw on the full frame so geometry and boxes keep their coordinates, then crop
        img = cv2.resize(full[Y1:Y2, X1:X2], (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
        shift = np.array([X1, Y1, X1, Y1], float)
        moved = SceneGeometry.__new__(SceneGeometry)
        moved.__dict__.update(geom.__dict__)
        moved.crosswalks = [c - [X1, Y1] for c in geom.crosswalks]
        moved.stop_line = None if geom.stop_line is None else geom.stop_line - [X1, Y1]
        draw_geometry(img, moved, scale)
        samples = []
        for tr, i in index.at(t):
            if tr.tid in highlight or tr.category == "person":
                tr_view = _Shifted(tr, shift)
                samples.append((tr_view, i))
        draw_objects(img, samples, scale, highlight)
        inside = s <= t <= e
        cv2.rectangle(img, (0, 0), (TILE_W - 1, TILE_H - 1), (40, 40, 230) if inside else (90, 90, 90), 5)
        sig = NAMES[analysis.signal.at(t)] if analysis.signal.observable else "-"
        cv2.putText(img, f"t={t:.2f}s  signal {sig}", (10, TILE_H - 12), cv2.FONT_HERSHEY_DUPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        r, c = divmod(k, COLS)
        out[HEADER + r * TILE_H:HEADER + (r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = img
    cv2.putText(out, title, (12, 42), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


class _Shifted:
    """A track seen through a crop: same samples, boxes moved by the crop's origin."""

    def __init__(self, tr, shift: np.ndarray):
        self.tid, self.category, self.cls = tr.tid, tr.category, tr.cls
        self.box = tr.box - shift


def export(videos: list[Path], out: Path) -> None:
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    (out / "obs").mkdir(exist_ok=True)
    events, durations = [], {}
    for path in videos:
        obs = cached_observation(path)
        analysis = interpret(obs)
        durations[path.name] = {"duration": round(obs.info.duration, 3), "fps": obs.info.fps,
                                "width": obs.info.width, "height": obs.info.height}
        with gzip.open(out / "obs" / f"{path.stem}.pkl.gz", "wb", compresslevel=6) as f:
            pickle.dump(obs, f)
        cap = cv2.VideoCapture(str(path))
        for n, (s, e, label, actors) in enumerate(sorted(analysis.detailed), 1):
            name = f"{path.stem}_{n:03d}_{label}_{s:.1f}-{e:.1f}.jpg"
            title = (f"{path.name}  #{n}  {label}  {s:.2f}-{e:.2f} s ({e - s:.1f} s)  "
                     f"actors {', '.join(map(str, actors))}")
            cv2.imwrite(str(out / "sheets" / name), sheet(cap, analysis, (s, e, label, actors), title),
                        [cv2.IMWRITE_JPEG_QUALITY, 72])
            events.append({"video": path.name, "n": n, "label": label, "start": round(s, 3), "end": round(e, 3),
                           "actors": [int(a) for a in actors], "sheet": f"sheets/{name}"})
        cap.release()
        print(f"{path.name}: {sum(ev['video'] == path.name for ev in events)} events", flush=True)
    (out / "events.json").write_text(json.dumps(events, indent=1))
    (out / "durations.json").write_text(json.dumps(durations, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", type=Path, required=True, help="video files or folders")
    ap.add_argument("--out", type=Path, default=ROOT / "review")
    args = ap.parse_args()
    videos = []
    for p in args.videos:
        videos += sorted(q for q in p.iterdir() if q.suffix.lower() == ".mp4") if p.is_dir() else [p]
    export(videos, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
