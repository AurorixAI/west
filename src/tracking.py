"""Multi-category ByteTrack tracking and offline track post-processing."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from src import config as C
from src.detection import Detections

CATEGORIES = ("vehicle", "person", "animal")


def window_mean(t: np.ndarray, x: np.ndarray, half: float) -> np.ndarray:
    """Mean of x over samples within +-half seconds of each sample (irregular t)."""
    if len(t) == 0:
        return x.copy()
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    cs = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    cnt = (hi - lo).reshape((-1,) + (1,) * (x.ndim - 1))
    return (cs[hi] - cs[lo]) / cnt


def window_velocity(t: np.ndarray, xy: np.ndarray, half: float) -> np.ndarray:
    """Velocity from the first to the last sample within +-half seconds of each sample.

    The window always reaches at least the neighbouring sample on each side,
    so sparse sampling (a thinned or low-rate run) never reads as standing still.
    """
    n = len(t)
    if n < 2:
        return np.zeros((n, 2))
    idx = np.arange(n)
    lo = np.minimum(np.searchsorted(t, t - half, side="left"), np.maximum(idx - 1, 0))
    hi = np.maximum(np.searchsorted(t, t + half, side="right") - 1, np.minimum(idx + 1, n - 1))
    dt = t[hi] - t[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(dt[:, None] > 0, (xy[hi] - xy[lo]) / dt[:, None], 0.0)


def object_size(box: np.ndarray, category: str) -> np.ndarray:
    w = np.maximum(box[:, 2] - box[:, 0], 1.0)
    h = np.maximum(box[:, 3] - box[:, 1], 1.0)
    return h if category == "person" else np.sqrt(w * h)


@dataclass
class Track:
    tid: int
    category: str
    cls: int
    t: np.ndarray                   # (n,) seconds
    box: np.ndarray                 # (n, 4) full-resolution xyxy
    cls_votes: Counter = field(default_factory=Counter)
    conf: np.ndarray = None         # (n,) detector confidence, if known
    # derived by finalize()
    foot: np.ndarray = None         # (n, 2) smoothed bottom-centre (ground contact)
    size: np.ndarray = None         # (n,) smoothed object size in pixels
    vel: np.ndarray = None          # (n, 2) pixels / s
    speed: np.ndarray = None        # (n,) sizes / s

    @property
    def start(self) -> float:
        return float(self.t[0])

    @property
    def end(self) -> float:
        return float(self.t[-1])

    def finalize(self, smooth_sec: float = C.SMOOTH_SEC) -> "Track":
        half = smooth_sec / 2
        raw_foot = np.stack([(self.box[:, 0] + self.box[:, 2]) / 2, self.box[:, 3]], axis=1)
        self.foot = window_mean(self.t, raw_foot, half)
        self.size = window_mean(self.t, object_size(self.box, self.category), half)
        self.vel = window_velocity(self.t, self.foot, half)
        self.speed = np.linalg.norm(self.vel, axis=1) / np.maximum(self.size, 1.0)
        return self

    def index_at(self, t: float) -> int | None:
        """Index of the sample at time t (within half a sampling step), else None."""
        i = int(np.searchsorted(self.t, t))
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.t) and abs(self.t[j] - t) < 0.06 and (best is None or abs(self.t[j] - t) < abs(self.t[best] - t)):
                best = j
        return best


class MultiTracker:
    """One ByteTrack per category, so a person can never inherit a car's ID."""

    def __init__(self, fps: float):
        buf = max(1, int(round(C.TRACK_BUFFER_SEC * fps)))
        self.trackers = {c: sv.ByteTrack(track_activation_threshold=C.TRACK_ACTIVATION, lost_track_buffer=buf,
                                         minimum_matching_threshold=0.8, frame_rate=max(1, int(round(fps))))
                         for c in CATEGORIES}
        self.records: dict[tuple[str, int], list] = defaultdict(list)

    def update(self, t: float, det: Detections) -> list[tuple[str, int, np.ndarray, int]]:
        """Feed one frame; returns [(category, track_id, box, coco_cls)] for this frame."""
        out = []
        cats = np.array([C.CATEGORY_OF_CLASS.get(int(c), "") for c in det.cls])
        for cat, tracker in self.trackers.items():
            m = cats == cat
            sd = sv.Detections(xyxy=det.xyxy[m].astype(np.float32), confidence=det.conf[m], class_id=det.cls[m])
            tracked = tracker.update_with_detections(sd)
            for box, tid, cls, conf in zip(tracked.xyxy, tracked.tracker_id, tracked.class_id, tracked.confidence):
                self.records[(cat, int(tid))].append((t, box.astype(np.float32), int(cls), float(conf)))
                out.append((cat, int(tid), box, int(cls)))
        return out

    def tracks(self, min_samples: int = 3) -> list[Track]:
        out, next_id = [], 1
        for (cat, _), rec in sorted(self.records.items()):
            if len(rec) < min_samples:
                continue
            votes = Counter(r[2] for r in rec)
            out.append(Track(next_id, cat, votes.most_common(1)[0][0],
                             np.array([r[0] for r in rec], np.float64),
                             np.stack([r[1] for r in rec]).astype(np.float64), votes,
                             np.array([r[3] for r in rec], np.float32)))
            next_id += 1
        return out


def stitch(tracks: list[Track], max_gap: float = C.STITCH_MAX_GAP_SEC,
           max_dist: float = C.STITCH_MAX_DIST) -> list[Track]:
    """Join fragments of one object split by an ID switch (occlusion, missed frames).

    A fragment B continues A when it starts within ``max_gap`` seconds after A
    ends, in the same category, and B's first position is within ``max_dist``
    sizes of where A would be by then (A's end velocity, extrapolated for at
    most one second, so a moving object is not propelled into a random match).
    Greedy on distance; each track is joined at most once at each end.
    """
    for tr in tracks:
        if tr.foot is None:
            tr.finalize()
    by_start = sorted(tracks, key=lambda x: x.start)
    starts = np.array([x.start for x in by_start])
    cands = []
    for a in tracks:
        lo = np.searchsorted(starts, a.end, side="right")
        hi = np.searchsorted(starts, a.end + max_gap, side="right")
        for b in by_start[lo:hi]:
            if b.category != a.category:
                continue
            gap = b.start - a.end
            pred = a.foot[-1] + a.vel[-1] * min(gap, 1.0)
            d = np.linalg.norm(b.foot[0] - pred) / max(a.size[-1], 1.0)
            if d <= max_dist:
                cands.append((d, gap, a.tid, b.tid))
    cands.sort()
    by_id = {t.tid: t for t in tracks}
    nxt, prv = {}, {}
    for _, _, a, b in cands:          # links always go forward in time: no cycles
        if a not in nxt and b not in prv:
            nxt[a], prv[b] = b, a
    out = []
    for tid, tr in by_id.items():
        if tid in prv:
            continue
        chain = [tr]
        while chain[-1].tid in nxt:
            chain.append(by_id[nxt[chain[-1].tid]])
        if len(chain) == 1:
            out.append(tr)
            continue
        votes = sum((c.cls_votes for c in chain), Counter())
        confs = None if any(c.conf is None for c in chain) else np.concatenate([c.conf for c in chain])
        merged = Track(tr.tid, tr.category, votes.most_common(1)[0][0],
                       np.concatenate([c.t for c in chain]), np.concatenate([c.box for c in chain]), votes, confs)
        out.append(merged.finalize())
    out.sort(key=lambda x: x.tid)
    return out
