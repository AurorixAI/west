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


def in_poly(poly: np.ndarray, pts: np.ndarray, margin: float | np.ndarray = 0.0) -> np.ndarray:
    """Vectorised point-in-polygon; ``margin`` > 0 grows the polygon (pixels)."""
    pts = np.atleast_2d(pts)
    margin = np.broadcast_to(np.asarray(margin, np.float64), (len(pts),))
    c = poly.reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(c, (float(x), float(y)), True) >= -m
                     for (x, y), m in zip(pts, margin)], bool)


@dataclass
class SceneGeometry:
    """The hand-placed geometry, carried into this video's pixels.

    ``transform`` maps the reference view (3840x2160) to the video; by default
    it is plain scaling. ``known=False`` (another camera) leaves every zone
    empty, so zone-based rules stay silent and nothing is drawn in the wrong place.
    """
    width: int
    height: int
    transform: np.ndarray | None = None
    known: bool = True

    def __post_init__(self):
        w, h = self.width, self.height
        T = np.diag([w / C.REF_W, h / C.REF_H, 1.0]) if self.transform is None else np.asarray(self.transform, float)
        if not self.known:
            self.stop_line = self.stop_zone = self.carriageway = self.tl_roi = None
            self.crosswalks, self.not_carriageway = [], []
            self.signal_heads, self.road_signs = {}, {}
            return

        def pts(p):
            return cv2.perspectiveTransform(np.asarray(p, np.float32).reshape(-1, 1, 2), T).reshape(-1, 2)

        def box(b):
            x1, y1, x2, y2 = b
            q = pts([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            return int(q[:, 0].min()), int(q[:, 1].min()), int(np.ceil(q[:, 0].max())), int(np.ceil(q[:, 1].max()))

        self.stop_line = pts(C.STOP_LINE)
        self.crosswalks = [pts(c) for c in (C.CROSSWALK_MAIN, C.CROSSWALK_RIGHT, C.CROSSWALK_CORNER)]
        self.stop_zone = pts(C.STOP_ZONE)
        self.carriageway = pts(C.CARRIAGEWAY)
        self.not_carriageway = [pts(p) for p in C.NOT_CARRIAGEWAY]
        self.signal_heads = {k: box(b) for k, b in C.SIGNAL_HEADS.items()}
        self.road_signs = {k: box(b) for k, b in C.ROAD_SIGNS.items()}
        y0, y1, x0, x1 = C.TL_ROI
        bx0, by0, bx1, by1 = box((x0, y0, x1, y1))
        self.tl_roi = (by0, by1, bx0, bx1)

    def on_carriageway(self, pts: np.ndarray, inset: float | np.ndarray = 0.0,
                       island_margin: float | np.ndarray = 0.0) -> np.ndarray:
        """Inside the traced carriageway by ``inset`` pixels, and ``island_margin`` pixels off every island.

        The outer kerbs are traced less precisely far from the camera than the
        islands are, so the two tolerances are separate.
        """
        pts = np.atleast_2d(pts)
        if self.carriageway is None:
            return np.zeros(len(pts), bool)
        inside = in_poly(self.carriageway, pts, -np.asarray(inset, float))
        for p in self.not_carriageway:
            inside &= ~in_poly(p, pts, island_margin)
        return inside

    def vehicle_crosswalk_index(self, box: np.ndarray) -> np.ndarray:
        """Crossing each vehicle box's ground footprint touches, -1 if none.

        The bottom-centre point alone is the rear bumper of a car driving away
        from the camera, still short of the zebra while its front is on it, so
        three points up the lower part of the box are tested.
        """
        box = np.atleast_2d(box)
        cx = (box[:, 0] + box[:, 2]) / 2
        h = box[:, 3] - box[:, 1]
        out = np.full(len(box), -1)
        for f in (0.0, 0.2, 0.4):
            idx = self.crosswalk_index(np.stack([cx, box[:, 3] - f * h], 1))
            out = np.where(out < 0, idx, out)
        return out

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
