"""Part B: causal accident anticipation from tracked road users.

Strictly causal: the estimator owns its own detector and tracker, sees frames
only through ``step`` and keeps nothing but the last few seconds of each track.

Risk signal, per sampled frame:
  * for every pair of nearby road users (at least one a vehicle, at least one
    faster than walking pace), the time and distance of closest approach under
    constant velocity (TTC-style), weighted by the deceleration needed to avoid
    it (DRAC): approaching a queue needs gentle braking and scores nothing;
    pairs driving in the same direction (a platoon) are down-weighted;
  * a hard-braking cue (speed falling by >60% within ~0.6 s) near another user.
The raw hazard is held with a short exponential decay and mapped to [0, 1] so
that 0.5 (the alarm threshold) needs a near-simultaneous arrival at a common
point about 2 s ahead; a hard stop on its own stays below the threshold.
"""
from __future__ import annotations

import os
import time
from collections import deque

import cv2
import numpy as np

from src import config as C
from src.detection import Detector
from src.pipeline import GPU, default_profile
from src.tracking import MultiTracker, object_size

# (weights, detector rate inside step(), frame width) with and without a GPU
GPU_SETTINGS = (C.DETECTOR_WEIGHTS, 6.0, 960)
CPU_SETTINGS = ("yolov8n.pt", 4.0, 640)
HISTORY_SEC = 2.5
HORIZON_SEC = 4.0         # closest approaches further ahead are ignored
TTC_SCALE = 2.9           # seconds: a dead-on conflict 2 s ahead scores 0.5
MISS_SCALE = 0.25         # sizes: passing in the next lane misses by ~0.4 at this camera angle
# Deceleration rate to avoid the collision (DRAC, sizes/s^2): a normal approach
# to a queue needs gentle braking and scores 0; only harder-than-normal counts.
DRAC_LOW, DRAC_HIGH = 0.4, 1.2
WALKING_PACE = 1.0        # sizes/s: two users both slower than this cannot crash hard
FOLLOW_WEIGHT = 0.35      # same-direction pairs
DECAY_SEC = 1.5
BRAKING_HAZARD = 0.4      # on its own a hard stop is suspicious, not an alarm
MAX_REAL_TIME_SHARE = 0.5 # step() may spend this share of video time before thinning


def fit_velocity(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Least-squares velocity (px/s) of the samples; zero if they span < 0.3 s."""
    if len(t) < 3 or t[-1] - t[0] < 0.3:
        return np.zeros(2)
    tc = t - t.mean()
    return (tc[:, None] * (xy - xy.mean(0))).sum(0) / (tc ** 2).sum()


def pair_hazard(p: np.ndarray, v: np.ndarray, size: float) -> tuple[float, float, float]:
    """Hazard in [0, 1], time of closest approach (s) and miss distance (sizes).

    ``p`` is the relative position, ``v`` the relative velocity (pixels, px/s)."""
    vv = float(v @ v)
    if vv < 1e-9:
        return 0.0, np.inf, np.inf
    t_star = -float(p @ v) / vv
    if t_star <= 0 or t_star > HORIZON_SEC:
        return 0.0, t_star, np.inf
    size = max(size, 1.0)
    miss = float(np.linalg.norm(p + v * t_star)) / size
    dist = float(np.linalg.norm(p))
    closing = -float(p @ v) / max(dist, 1e-9) / size             # sizes/s
    gap = max(dist / size - 0.5, 0.25)                              # sizes to contact
    drac = closing ** 2 / (2 * gap)
    urgency = float(np.clip((drac - DRAC_LOW) / (DRAC_HIGH - DRAC_LOW), 0.0, 1.0))
    return float(urgency * np.exp(-t_star / TTC_SCALE) * np.exp(-(miss / MISS_SCALE) ** 2)), t_star, miss


def to_probability(raw: float) -> float:
    """Monotone map of the held hazard to P(accident within 5 s), floor 0.02."""
    return float(np.clip(0.02 + 0.96 * raw, 0.0, 1.0))


class _Hist:
    __slots__ = ("t", "foot", "size", "cls", "category", "last")

    def __init__(self, category: str, cls: int):
        self.t, self.foot, self.size = deque(), deque(), deque()
        self.category, self.cls, self.last = category, cls, -1.0


class CausalRiskEstimator:
    def __init__(self, settings: tuple[str, float, int] | None = None):
        weights, fps, width = settings or (GPU_SETTINGS if default_profile() is GPU else CPU_SETTINGS)
        self.detector = Detector(weights=weights, imgsz=width)
        self.target_fps, self.width = fps, width

    def reset(self, meta: dict) -> None:
        self.fps = float(meta.get("fps") or 25.0)
        self.stride = max(1, int(round(self.fps / self.target_fps)))
        self.tracker = MultiTracker(self.fps / self.stride)
        self.hist: dict[tuple[str, int], _Hist] = {}
        self.n = 0
        self.held, self.score, self.last_t = 0.0, to_probability(0.0), None
        self.busy = 0.0

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        n, self.n = self.n, self.n + 1
        if n % self.stride:
            return self.score
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        scale = w / float(self.width)
        small = cv2.resize(frame, (self.width, int(round(h / scale))), interpolation=cv2.INTER_AREA)
        det = self.detector([small], [scale])[0]
        raw = self._update(t_sec, det)
        dt = 0.0 if self.last_t is None else t_sec - self.last_t
        self.held = max(raw, self.held * float(np.exp(-dt / DECAY_SEC)))
        self.last_t = t_sec
        self.score = to_probability(self.held)
        self.busy += time.perf_counter() - t0
        # stay far inside the time budget on slow hardware
        if not os.environ.get("WEST_NO_THIN") and t_sec > 5 and self.busy > MAX_REAL_TIME_SHARE * t_sec:
            self.stride *= 2
            self.busy = 0.0
            self.tracker = MultiTracker(self.fps / self.stride)
            self.hist.clear()
        return self.score

    def _update(self, t: float, det) -> float:
        live = []
        for cat, tid, box, cls in self.tracker.update(t, det):
            key = (cat, tid)
            hs = self.hist.get(key) or self.hist.setdefault(key, _Hist(cat, cls))
            b = np.asarray(box, float)[None]
            hs.t.append(t)
            hs.foot.append(np.array([(b[0, 0] + b[0, 2]) / 2, b[0, 3]]))
            hs.size.append(float(object_size(b, cat)[0]))
            hs.last = t
            while hs.t and hs.t[0] < t - HISTORY_SEC:
                hs.t.popleft(), hs.foot.popleft(), hs.size.popleft()
            live.append(hs)
        for k in [k for k, hs in self.hist.items() if hs.last < t - HISTORY_SEC]:
            del self.hist[k]

        users = []
        for hs in live:
            if hs.category == "animal" or len(hs.t) < 4:
                continue
            tt, ff = np.array(hs.t), np.array(hs.foot)
            recent = tt >= t - 1.0
            v = fit_velocity(tt[recent], ff[recent])
            size = float(np.median(hs.size))
            before = (tt >= t - 1.2) & (tt < t - 0.5)
            v_before = fit_velocity(tt[before], ff[before]) if before.sum() >= 3 else v
            users.append((hs, ff[-1], v, size, v_before))

        raw = 0.0
        for i in range(len(users)):
            hi, pi, vi, si, vbi = users[i]
            for j in range(i + 1, len(users)):
                hj, pj, vj, sj, vbj = users[j]
                if hi.category == "person" and hj.category == "person":
                    continue
                size = 0.5 * (si + sj)
                rel, vrel = pj - pi, vj - vi
                if np.linalg.norm(rel) > np.linalg.norm(vrel) * HORIZON_SEC + 2 * size:
                    continue                  # cannot meet within the horizon
                ni, nj = np.linalg.norm(vi) / si, np.linalg.norm(vj) / sj
                if max(ni, nj) < WALKING_PACE:
                    continue                  # people and a walked scooter, a creeping queue
                hz, _, _ = pair_hazard(rel, vrel, size)
                if hz > 0 and "person" not in (hi.category, hj.category):
                    if ni > C.MOVE_SPEED and nj > C.MOVE_SPEED and vi @ vj / (np.linalg.norm(vi) * np.linalg.norm(vj)) > 0.9:
                        hz *= FOLLOW_WEIGHT
                raw = max(raw, hz)
                # hard braking right next to somebody else
                if np.linalg.norm(rel) < 2.5 * size:
                    for s_, v_, vb_ in ((si, vi, vbi), (sj, vj, vbj)):
                        before, now = np.linalg.norm(vb_) / s_, np.linalg.norm(v_) / s_
                        if before > 1.5 and now < 0.4 * before:
                            raw = max(raw, BRAKING_HAZARD)
        return raw
