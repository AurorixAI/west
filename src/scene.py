"""Scene model: hand-placed painted features + a flow field learned from traffic.

Only features that are painted on the road (crossings, stop line) and the
signal head are placed by hand. Where the carriageway is, and which way each
lane flows, is learned from the moving vehicles of the video itself: every
cell of a coarse grid accumulates the unit heading of the vehicles whose
ground point passes through it. That needs no calibration and does not break
if a lane is closed or the camera is nudged.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src import config as C
from src.tracking import Track


def _scaled(poly: np.ndarray, w: int, h: int) -> np.ndarray:
    return (poly * np.array([w / C.REF_W, h / C.REF_H], np.float32)).astype(np.float32)


def in_poly(poly: np.ndarray, pts: np.ndarray, margin: float | np.ndarray = 0.0) -> np.ndarray:
    """Vectorised point-in-polygon; ``margin`` > 0 grows the polygon (pixels)."""
    pts = np.atleast_2d(pts)
    margin = np.broadcast_to(np.asarray(margin, np.float64), (len(pts),))
    c = poly.reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(c, (float(x), float(y)), True) >= -m
                     for (x, y), m in zip(pts, margin)], bool)


@dataclass
class SceneGeometry:
    width: int
    height: int

    def __post_init__(self):
        w, h = self.width, self.height
        self.stop_line = _scaled(C.STOP_LINE, w, h)
        self.crosswalks = [_scaled(C.CROSSWALK_MAIN, w, h), _scaled(C.CROSSWALK_CORNER, w, h)]
        self.stop_zone = _scaled(C.STOP_ZONE, w, h)
        y0, y1, x0, x1 = C.TL_ROI
        sx, sy = w / C.REF_W, h / C.REF_H
        self.tl_roi = (int(y0 * sy), int(y1 * sy), int(x0 * sx), int(x1 * sx))

    def crosswalk_index(self, pts: np.ndarray, margin: float | np.ndarray = 0.0) -> np.ndarray:
        """Index of the crossing each point is on, -1 if none."""
        out = np.full(len(np.atleast_2d(pts)), -1)
        for i, cw in enumerate(self.crosswalks):
            out[(out < 0) & in_poly(cw, pts, margin)] = i
        return out

    def stop_line_side(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance (pixels) of points to the stop line, positive on the camera side."""
        a, b = self.stop_line
        d = b - a
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        if n[1] < 0:
            n = -n
        return (np.atleast_2d(pts) - a) @ n

    def on_stop_line_span(self, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Whether the projection of the points falls within the stop line's length."""
        a, b = self.stop_line
        d = b - a
        s = (np.atleast_2d(pts) - a) @ d / (d @ d)
        m = margin / np.linalg.norm(d)
        return (s >= -m) & (s <= 1 + m)


class FlowField:
    """Grid of vehicle headings learned from moving tracks."""

    def __init__(self, width: int, height: int, cols: int = C.GRID_COLS):
        self.width, self.height = width, height
        self.cell = width / cols
        self.cols, self.rows = cols, int(np.ceil(height / self.cell))
        shape = (self.rows, self.cols)
        self.n = np.zeros(shape)              # moving samples
        self.vec = np.zeros(shape + (2,))     # sum of unit headings
        self.tracks = np.zeros(shape)         # distinct moving vehicles
        self.contrib: dict[int, tuple] = {}   # per-track contributions (for leave-one-out)

    def cells(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.atleast_2d(pts)
        c = np.clip((pts[:, 0] / self.cell).astype(int), 0, self.cols - 1)
        r = np.clip((pts[:, 1] / self.cell).astype(int), 0, self.rows - 1)
        return r, c

    def add_tracks(self, tracks: list[Track]) -> "FlowField":
        for tr in tracks:
            if tr.category != "vehicle":
                continue
            m = tr.speed >= C.MOVE_SPEED
            if m.sum() < 3:
                continue
            u = tr.vel[m] / np.linalg.norm(tr.vel[m], axis=1, keepdims=True)
            r, c = self.cells(tr.foot[m])
            np.add.at(self.n, (r, c), 1)
            np.add.at(self.vec, (r, c), u)
            uniq = np.unique(r * self.cols + c)
            np.add.at(self.tracks, (uniq // self.cols, uniq % self.cols), 1)
            self.contrib[tr.tid] = (r, c, u)
        self._derive()
        return self

    def add_prior(self, path) -> None:
        """Add counts learned offline from the sample videos (scripts/build_scene_prior.py)."""
        d = np.load(path)
        if d["n"].shape != self.n.shape:
            n = cv2.resize(d["n"], (self.cols, self.rows), interpolation=cv2.INTER_AREA)
            vec = cv2.resize(d["vec"], (self.cols, self.rows), interpolation=cv2.INTER_AREA)
            tracks = cv2.resize(d["tracks"], (self.cols, self.rows), interpolation=cv2.INTER_AREA)
        else:
            n, vec, tracks = d["n"], d["vec"], d["tracks"]
        self.n += n
        self.vec += vec
        self.tracks += tracks
        self._derive()

    @staticmethod
    def _box3(a: np.ndarray) -> np.ndarray:
        # neighbourhood sums give each cell enough support without blurring lanes away
        return cv2.filter2D(a.astype(np.float32), -1, np.ones((3, 3), np.float32),
                            borderType=cv2.BORDER_CONSTANT)

    def _smoothed(self, r: np.ndarray, c: np.ndarray, u: np.ndarray):
        n = np.zeros(self.n.shape)
        vec = np.zeros(self.vec.shape)
        np.add.at(n, (r, c), 1)
        np.add.at(vec, (r, c), u)
        return self._box3(n), np.stack([self._box3(vec[..., i]) for i in range(2)], -1)

    def _derive(self) -> None:
        self.n_s = self._box3(self.n)
        self.vec_s = np.stack([self._box3(self.vec[..., i]) for i in range(2)], -1)
        road = (self.tracks >= C.ROAD_MIN_TRACKS).astype(np.uint8)
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        self.road = road.astype(bool)
        self.road_core = cv2.erode(road, np.ones((3, 3), np.uint8)).astype(bool)

    def is_road(self, pts: np.ndarray, core: bool = False) -> np.ndarray:
        r, c = self.cells(pts)
        return (self.road_core if core else self.road)[r, c]

    def direction(self, pts: np.ndarray, exclude_tid: int | None = None):
        """Dominant unit heading and whether it is reliable, per point.

        ``exclude_tid`` removes that track's own votes (leave-one-out), so a
        wrong-way vehicle cannot vote for its own direction.
        """
        r, c = self.cells(pts)
        n_s, vec_s = self.n_s, self.vec_s
        if exclude_tid is not None and exclude_tid in self.contrib:
            own_n, own_vec = self._smoothed(*self.contrib[exclude_tid])
            n_s, vec_s = n_s - own_n, vec_s - own_vec
        n = n_s[r, c].astype(np.float64)
        v = vec_s[r, c].astype(np.float64)
        norm = np.linalg.norm(v, axis=1)
        coherence = np.where(n > 0, norm / np.maximum(n, 1e-9), 0.0)
        unit = v / np.maximum(norm, 1e-9)[:, None]
        ok = (n >= C.FLOW_MIN_SAMPLES) & (coherence >= C.FLOW_MIN_COHERENCE)
        return unit, ok

    def direction_groups(self, max_groups: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """Main traffic directions (unit vectors) and each road cell's group (-1 = none).

        Peaks of the support-weighted histogram of cell headings, at least 60
        degrees apart. Used to evaluate congestion per direction of travel.
        """
        norm = np.linalg.norm(self.vec_s, axis=-1)
        coh = np.where(self.n_s > 0, norm / np.maximum(self.n_s, 1e-9), 0)
        good = self.road & (self.n_s >= C.FLOW_MIN_SAMPLES) & (coh >= C.FLOW_MIN_COHERENCE)
        groups = np.full(self.n.shape, -1)
        if not good.any():
            return np.zeros((0, 2)), groups
        ang = np.arctan2(self.vec_s[..., 1], self.vec_s[..., 0])[good]
        hist, edges = np.histogram(ang, bins=36, range=(-np.pi, np.pi), weights=self.n_s[good])
        hist = np.convolve(np.r_[hist[-1], hist, hist[0]], [1, 2, 1], "valid")
        centers = (edges[:-1] + edges[1:]) / 2
        peaks = []
        for i in np.argsort(-hist):
            if hist[i] < 0.1 * hist.max() or len(peaks) >= max_groups:
                break
            if all(abs(np.angle(np.exp(1j * (centers[i] - p)))) > np.radians(60) for p in peaks):
                peaks.append(centers[i])
        dirs = np.array([[np.cos(p), np.sin(p)] for p in peaks])
        cell_ang = np.arctan2(self.vec_s[..., 1], self.vec_s[..., 0])
        diff = np.abs(np.angle(np.exp(1j * (cell_ang[..., None] - np.array(peaks)))))
        best = diff.argmin(-1)
        groups[good & (diff.min(-1) < np.radians(40))] = best[good & (diff.min(-1) < np.radians(40))]
        return dirs, groups

    def to_npz(self, path) -> None:
        np.savez_compressed(path, n=self.n, vec=self.vec, tracks=self.tracks)
