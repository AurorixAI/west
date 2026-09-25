#!/usr/bin/env python3
"""Render annotated sample videos and the EDA/result data behind the website.

    python scripts/build_site.py --videos samples/ --pred predictions_samples.json

For every sample video, into web/static/data/<video stem>/:
  annotated.mp4     1280x720 H.264: zones, tracks, active events, signal, risk
  data.json         metadata, events (+ actors), risk curve, signal phases,
                    counts / brightness per second, speed histogram, directions
  thumbs/<k>.jpg    one frame per event, the actors highlighted
  background.jpg, heatmap.jpg, trajectories.jpg, flow.jpg   EDA images
and web/static/data/manifest.json listing them. Events and risk come from the
harness output (--pred) when given, so the site shows exactly what we submit.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.dev_eval import cached_observation  # noqa: E402
from src import viz  # noqa: E402
from src.pipeline import interpret  # noqa: E402
from src.signal_state import NAMES  # noqa: E402
from src.tracking import CATEGORIES  # noqa: E402

OUT = ROOT / "web" / "static" / "data"
WIDTH = 1280
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
              15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow"}


def downsample_risk(curve: list, hz: float = 5.0) -> list:
    """Max per bin keeps short alarms visible on a chart."""
    out = {}
    for t, s in curve:
        k = int(t * hz)
        out[k] = max(out.get(k, 0.0), s)
    return [[round(k / hz, 2), round(v, 3)] for k, v in sorted(out.items())]


def per_second_counts(tracks, duration: float) -> dict:
    n = int(np.ceil(duration)) + 1
    groups = {"car": {2}, "bus_truck": {5, 7}, "two_wheeler": {1, 3}, "person": {0}}
    out = {g: np.zeros(n) for g in groups}
    samples = np.zeros(n)
    times = sorted({round(float(t), 3) for tr in tracks for t in tr.t})
    for t in times:
        samples[int(t)] += 1
    for tr in tracks:
        for g, cls in groups.items():
            if tr.cls in cls:
                np.add.at(out[g], tr.t.astype(int), 1)
    return {g: np.round(v / np.maximum(samples, 1), 2).tolist() for g, v in out.items()}


def direction_summary(flow, tracks) -> list[dict]:
    dirs, groups = flow.direction_groups()
    out = []
    for gi, d in enumerate(dirs):
        cells = groups == gi
        out.append({"angle_deg": round(float(np.degrees(np.arctan2(d[1], d[0]))), 1),
                    "cells": int(cells.sum()), "samples": int(flow.n[cells].sum())})
    return out


def build(video: Path, pred: dict | None, render: bool) -> dict:
    obs = cached_observation(video)
    an = interpret(obs)
    info = obs.info
    out = OUT / video.stem
    if out.exists():
        shutil.rmtree(out)
    (out / "thumbs").mkdir(parents=True)

    entry = (pred or {}).get("videos", {}).get(video.name)
    events = [list(e) + [[]] for e in entry["events"]] if entry else [[s, e, l, list(a)] for s, e, l, a in an.detailed]
    if entry:   # attach actors from our own detailed list where the segments coincide
        for ev in events:
            for s, e, lbl, actors in an.detailed:
                if lbl == ev[2] and abs(s - ev[0]) < 0.05 and abs(e - ev[1]) < 0.05:
                    ev[3] = list(actors)
    risk = entry["risk"] if entry else []
    risk_t = np.array([r[0] for r in risk]) if risk else np.zeros(0)
    risk_v = np.array([r[1] for r in risk]) if risk else np.zeros(0)

    index = viz.TrackIndex(obs.tracks)
    scale = info.width / WIDTH
    size = (WIDTH, int(round(info.height / scale)) // 16 * 16)
    writer = None
    if render:
        import imageio_ffmpeg
        writer = imageio_ffmpeg.write_frames(str(out / "annotated.mp4"), size, fps=info.fps, codec="libx264",
                                             pix_fmt_out="yuv420p", macro_block_size=16,
                                             output_params=["-crf", "27", "-preset", "veryfast",
                                                            "-movflags", "+faststart"])
        writer.send(None)
    thumbs_due = sorted(((ev[0] + ev[1]) / 2, k) for k, ev in enumerate(events))
    bg_every = max(1, info.n_frames // 40)
    bg, brightness = [], np.zeros(int(np.ceil(info.duration)) + 1)
    bright_n = np.zeros_like(brightness)

    cap = cv2.VideoCapture(str(video))
    idx = 0
    while True:
        need = render or idx % bg_every == 0 or (thumbs_due and idx / info.fps >= thumbs_due[0][0])
        if not need:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, full = cap.read()
        if not ok:
            break
        t = idx / info.fps
        small = cv2.resize(full, size, interpolation=cv2.INTER_AREA)
        brightness[int(t)] += float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
        bright_n[int(t)] += 1
        if idx % bg_every == 0:
            bg.append(small)
        active = [ev for ev in events if ev[0] <= t <= ev[1]]
        highlight = {a: ev[2] for ev in active for a in ev[3]}
        if writer is not None or (thumbs_due and t >= thumbs_due[0][0]):
            frame = small.copy()
            viz.draw_geometry(frame, an.geom, scale)
            samples = index.at(t)
            viz.draw_objects(frame, samples, scale, highlight)
            r = float(risk_v[min(np.searchsorted(risk_t, t), len(risk_v) - 1)]) if len(risk_v) else None
            viz.draw_hud(frame, t, obs.signal.at(t), sum(tr.category == "vehicle" for tr, _ in samples),
                         sum(tr.category == "person" for tr, _ in samples),
                         sorted({ev[2] for ev in active}), r)
            if writer is not None:
                writer.send(np.ascontiguousarray(frame[:, :, ::-1]).tobytes())
            while thumbs_due and t >= thumbs_due[0][0]:
                _, k = thumbs_due.pop(0)
                cv2.imwrite(str(out / "thumbs" / f"{k}.jpg"), _thumb(frame, samples, events[k][3], scale),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
        idx += 1
    cap.release()
    if writer is not None:
        writer.close()

    background = np.median(np.stack(bg), axis=0).astype(np.uint8) if bg else np.zeros(size[::-1] + (3,), np.uint8)
    cv2.imwrite(str(out / "background.jpg"), background, [cv2.IMWRITE_JPEG_QUALITY, 88])
    vehicles = [tr for tr in obs.tracks if tr.category == "vehicle"]
    feet = np.concatenate([tr.foot for tr in vehicles]) if vehicles else np.zeros((0, 2))
    people = [tr for tr in obs.tracks if tr.category == "person"]
    pfeet = np.concatenate([tr.foot for tr in people]) if people else np.zeros((0, 2))
    cv2.imwrite(str(out / "heatmap.jpg"), viz.heatmap(background, feet, scale), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(out / "heatmap_people.jpg"), viz.heatmap(background, pfeet, scale, 4.0),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(out / "trajectories.jpg"), viz.trajectories(background, obs.tracks, scale),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(out / "flow.jpg"), viz.flow_field(background, an.flow, scale), [cv2.IMWRITE_JPEG_QUALITY, 85])

    moving = np.concatenate([tr.speed[tr.speed > 0.12] for tr in vehicles]) if vehicles else np.zeros(0)
    hist, edges = np.histogram(np.clip(moving, 0, 8), bins=32, range=(0, 8))
    cls_counts = {}
    for tr in obs.tracks:
        name = COCO_NAMES.get(tr.cls, str(tr.cls))
        cls_counts[name] = cls_counts.get(name, 0) + 1
    data = {
        "video": video.name,
        "meta": {"width": info.width, "height": info.height, "fps": round(info.fps, 3), "frames": info.n_frames,
                 "duration": round(info.duration, 2), "size_mb": round(video.stat().st_size / 2**20, 1),
                 "tracks": {c: sum(t.category == c for t in obs.tracks) for c in CATEGORIES},
                 "classes": cls_counts, "observe_sec": round(obs.seconds, 1)},
        "events": [{"start": ev[0], "end": ev[1], "label": ev[2], "actors": ev[3], "thumb": f"thumbs/{k}.jpg"}
                   for k, ev in enumerate(events)],
        "risk": downsample_risk(risk),
        "signal": [[round(a, 2), round(b, 2), s] for a, b, s in obs.signal.phases()],
        "signal_observable": obs.signal.observable,
        "counts": per_second_counts(obs.tracks, info.duration),
        "brightness": np.round(brightness / np.maximum(bright_n, 1), 1).tolist(),
        "speed_hist": {"edges": np.round(edges, 2).tolist(), "counts": hist.tolist()},
        "directions": direction_summary(an.flow, obs.tracks),
        "annotated": "annotated.mp4" if render else None,
    }
    (out / "data.json").write_text(json.dumps(data))
    return {"video": video.name, "dir": video.stem, "duration": data["meta"]["duration"],
            "events": len(events), "labels": sorted({ev[2] for ev in events})}


def _thumb(frame: np.ndarray, samples, actors: list[int], scale: float) -> np.ndarray:
    boxes = [tr.box[i] / scale for tr, i in samples if tr.tid in actors]
    if not boxes:
        return cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
    b = np.array(boxes)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max(x2 - x1, (y2 - y1) * 16 / 9, 240) * 0.9
    half_h = half_w * 9 / 16
    h, w = frame.shape[:2]
    xa, xb = int(max(0, cx - half_w)), int(min(w, cx + half_w))
    ya, yb = int(max(0, cy - half_h)), int(min(h, cy + half_h))
    return cv2.resize(frame[ya:yb, xa:xb], (640, 360))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--pred", type=Path, help="predictions_samples.json from run_submission.py")
    ap.add_argument("--no-render", action="store_true", help="skip the annotated videos")
    args = ap.parse_args()
    pred = json.loads(args.pred.read_text()) if args.pred else None
    videos = sorted(p for p in args.videos.iterdir() if p.suffix.lower() == ".mp4")
    manifest = []
    for v in videos:
        print(f"[{v.name}]", flush=True)
        manifest.append(build(v, pred, not args.no_render))
    (OUT / "manifest.json").write_text(json.dumps({"videos": manifest, "signal_names": list(NAMES.values())},
                                                  indent=1))
    print(f"wrote {OUT / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
