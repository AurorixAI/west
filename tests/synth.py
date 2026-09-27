"""Synthetic tracks on the 3840x2160 reference frame, for testing the rules without video.

Background traffic teaches the flow field three lanes, placed on the
competition camera's hand-calibrated geometry:
  * SOUTH: x ~ 900, driving +y over the stop line (y ~ 1061) and the main crossing (y 1221-1326);
  * NORTH: x ~ 1250, driving -y;
  * EAST:  y ~ 1800, x from 2000 to 3800, driving +x (away from the crossings).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from src.rules import Context
from src.scene import FlowField, SceneGeometry
from src.signal_state import GREEN, RED, SignalTimeline
from src.tracking import Track

W, H = 3840, 2160
DT = 0.1
CAR, BUS, PERSON, BICYCLE, MOTORCYCLE, DOG = 2, 5, 0, 1, 3, 16


def times(t0: float, t1: float) -> np.ndarray:
    return np.round(np.arange(round(t0 / DT), round(t1 / DT) + 1) * DT, 3)


class Scene:
    def __init__(self, duration: float = 120.0):
        self.duration = duration
        self.tracks: list[Track] = []
        self.rng = np.random.default_rng(0)
        self.signal = SignalTimeline(times(0, duration), np.full(len(times(0, duration)), GREEN))

    # -- actors -------------------------------------------------------------
    def add(self, category: str, cls: int, t: np.ndarray, foot: np.ndarray, size: float) -> Track:
        foot = foot + self.rng.normal(0, 0.01 * size, foot.shape)          # detector jitter
        w = size / 3 if category == "person" else size
        h = size
        box = np.stack([foot[:, 0] - w / 2, foot[:, 1] - h, foot[:, 0] + w / 2, foot[:, 1]], 1)
        tr = Track(len(self.tracks) + 1, category, cls, t, box, Counter({cls: len(t)}))
        self.tracks.append(tr.finalize())
        return tr

    def path(self, category, cls, t0, waypoints, speed, size=120.0, dwell=None):
        """Drive through (x, y) waypoints at ``speed`` px/s; ``dwell`` = {waypoint index: seconds}."""
        pts, ts, t = [np.array(waypoints[0], float)], [t0], t0
        dwell = dwell or {}
        for k, (a, b) in enumerate(zip(waypoints[:-1], waypoints[1:])):
            if k in dwell:
                t += dwell[k]
                pts.append(np.array(a, float))
                ts.append(t)
            d = np.linalg.norm(np.subtract(b, a))
            t += d / speed
            pts.append(np.array(b, float))
            ts.append(t)
        tt = times(t0, ts[-1])
        foot = np.stack([np.interp(tt, ts, [p[0] for p in pts]), np.interp(tt, ts, [p[1] for p in pts])], 1)
        return self.add(category, cls, tt, foot, size)

    def background(self, t0: float = 0.0, t1: float | None = None, every: float = 4.0) -> None:
        t1 = self.duration - 12 if t1 is None else t1
        for k, t in enumerate(np.arange(t0, t1, every)):
            dx = [-120, 0, 120][k % 3]
            self.path("vehicle", CAR, t, [(900 + dx, 500), (900 + dx, 2100)], 400)
            self.path("vehicle", CAR, t + 1, [(1250 + dx, 2100), (1250 + dx, 500)], 400)
            self.path("vehicle", CAR, t + 2, [(2000, 1800 + dx), (3800, 1800 + dx)], 400)

    def set_signal(self, *phases: tuple[float, float, int]) -> None:
        st = self.signal.state.copy()
        for a, b, s in phases:
            st[(self.signal.t >= a) & (self.signal.t <= b)] = s
        self.signal = SignalTimeline(self.signal.t, st)

    # -- context ------------------------------------------------------------
    def context(self) -> Context:
        flow = FlowField(W, H).add_tracks(self.tracks)
        return Context(self.tracks, SceneGeometry(W, H), flow, self.signal, self.duration)


__all__ = ["Scene", "times", "CAR", "BUS", "PERSON", "BICYCLE", "MOTORCYCLE", "DOG", "RED", "GREEN"]
