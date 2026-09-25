"""Drawing: annotated frames, event thumbnails and EDA images."""
from __future__ import annotations

import cv2
import numpy as np

from src.scene import FlowField, SceneGeometry
from src.signal_state import GREEN, RED
from src.tracking import Track

# BGR
CATEGORY_COLOR = {"vehicle": (235, 190, 60), "person": (90, 220, 110), "animal": (0, 200, 255)}
EVENT_COLOR = {
    "accident": (40, 40, 240), "near_miss": (60, 120, 255), "red_light": (70, 70, 255),
    "wrong_way": (200, 60, 230), "illegal_u_turn": (230, 90, 180), "stopped_vehicle": (0, 170, 255),
    "jaywalking": (0, 215, 255), "failure_to_yield": (60, 60, 200), "illegal_turn": (180, 120, 255),
    "solid_line_crossing": (255, 160, 60), "stop_line": (40, 110, 240), "congestion": (200, 200, 60),
    "road_obstacle": (30, 140, 200), "fire_smoke": (50, 80, 160),
}


class TrackIndex:
    """Which track samples exist at (or nearest to) a given time."""

    def __init__(self, tracks: list[Track]):
        self.by_time: dict[int, list[tuple[Track, int]]] = {}
        for tr in tracks:
            for i, t in enumerate(tr.t):
                self.by_time.setdefault(int(round(t * 1000)), []).append((tr, i))
        self.keys = np.array(sorted(self.by_time))

    def at(self, t: float, tol: float = 0.2) -> list[tuple[Track, int]]:
        if not len(self.keys):
            return []
        k = int(np.clip(np.searchsorted(self.keys, t * 1000), 0, len(self.keys) - 1))
        if k > 0 and abs(self.keys[k - 1] - t * 1000) < abs(self.keys[k] - t * 1000):
            k -= 1
        return self.by_time[int(self.keys[k])] if abs(self.keys[k] - t * 1000) <= tol * 1000 else []


def _poly(img, pts, scale, color, alpha=0.22, thickness=2):
    p = (pts / scale).astype(np.int32)
    over = img.copy()
    cv2.fillPoly(over, [p], color)
    cv2.addWeighted(over, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [p], True, color, thickness, cv2.LINE_AA)


def draw_geometry(img: np.ndarray, geom: SceneGeometry, scale: float) -> None:
    """Crossings, stop line and signal ROI; ``scale`` = full-res pixels per image pixel."""
    for cw in geom.crosswalks:
        _poly(img, cw, scale, (120, 230, 120), 0.16, 1)
    a, b = (geom.stop_line / scale).astype(int)
    cv2.line(img, tuple(a), tuple(b), (40, 40, 255), 3, cv2.LINE_AA)
    y0, y1, x0, x1 = (np.array(geom.tl_roi) / scale).astype(int)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 1)


def draw_objects(img: np.ndarray, samples: list[tuple[Track, int]], scale: float,
                 highlight: dict[int, str]) -> None:
    for tr, i in samples:
        x1, y1, x2, y2 = (tr.box[i] / scale).astype(int)
        label = highlight.get(tr.tid)
        color = EVENT_COLOR.get(label, (0, 0, 255)) if label else CATEGORY_COLOR.get(tr.category, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if label else 1, cv2.LINE_AA)
        if label or y2 - y1 > 28:
            text = f"{tr.tid} {label.replace('_', ' ')}" if label else str(tr.tid)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)


def draw_hud(img: np.ndarray, t: float, signal: int, n_vehicles: int, n_people: int,
             active: list[str], risk: float | None) -> None:
    h, w = img.shape[:2]
    over = img.copy()
    cv2.rectangle(over, (0, 0), (w, 34), (18, 18, 22), -1)
    cv2.addWeighted(over, 0.8, img, 0.2, 0, img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"{int(t // 60):02d}:{t % 60:05.2f}", (12, 23), font, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    sig_col = {RED: (60, 60, 255), GREEN: (90, 220, 90)}.get(signal, (150, 150, 150))
    cv2.circle(img, (150, 17), 8, sig_col, -1, cv2.LINE_AA)
    cv2.putText(img, {RED: "RED", GREEN: "GREEN"}.get(signal, "signal ?"), (164, 23), font, 0.55,
                sig_col, 1, cv2.LINE_AA)
    cv2.putText(img, f"vehicles {n_vehicles}   people {n_people}", (280, 23), font, 0.55,
                (220, 220, 220), 1, cv2.LINE_AA)
    if risk is not None:
        x0 = w - 250
        cv2.putText(img, "risk", (x0, 23), font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x0 + 45, 10), (x0 + 235, 24), (80, 80, 80), 1)
        col = (60, 60, 255) if risk >= 0.5 else (0, 200, 255) if risk >= 0.2 else (90, 220, 90)
        cv2.rectangle(img, (x0 + 46, 11), (x0 + 46 + int(188 * risk), 23), col, -1)
    for k, lbl in enumerate(active):
        text = lbl.replace("_", " ").upper()
        (tw, th), _ = cv2.getTextSize(text, font, 0.6, 2)
        y = h - 16 - k * 34
        cv2.rectangle(img, (12, y - th - 12), (12 + tw + 20, y + 6), EVENT_COLOR.get(lbl, (0, 0, 255)), -1)
        cv2.putText(img, text, (22, y - 2), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def heatmap(background: np.ndarray, pts: np.ndarray, scale: float, sigma: float = 6.0) -> np.ndarray:
    """Density of ground points over a (dimmed) background image."""
    h, w = background.shape[:2]
    acc = np.zeros((h, w), np.float32)
    if len(pts):
        p = (pts / scale).astype(int)
        ok = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
        np.add.at(acc, (p[ok, 1], p[ok, 0]), 1.0)
    acc = cv2.GaussianBlur(acc, (0, 0), sigma)
    acc = np.clip(acc / max(np.percentile(acc[acc > 0], 99) if (acc > 0).any() else 1.0, 1e-6), 0, 1)
    col = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    out = (background * 0.45).astype(np.uint8)
    a = acc[..., None] ** 0.6
    return (out * (1 - a) + col * a).astype(np.uint8)


def trajectories(background: np.ndarray, tracks: list[Track], scale: float) -> np.ndarray:
    """Every vehicle and pedestrian path, hue = direction of travel."""
    out = (background * 0.5).astype(np.uint8)
    for tr in tracks:
        if len(tr.t) < 5 or tr.category == "animal":
            continue
        pts = (tr.foot / scale).astype(np.int32)
        d = tr.foot[-1] - tr.foot[0]
        if np.linalg.norm(d) < 2 * np.median(tr.size):
            continue
        hue = int((np.degrees(np.arctan2(d[1], d[0])) % 360) / 2)
        color = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.polylines(out, [pts], False, color, 1 if tr.category == "vehicle" else 2, cv2.LINE_AA)
    return out


def flow_field(background: np.ndarray, flow: FlowField, scale: float) -> np.ndarray:
    """Learned carriageway (tint) and dominant lane direction per cell (arrows)."""
    out = (background * 0.55).astype(np.uint8)
    cell = flow.cell / scale
    over = out.copy()
    for r, c in zip(*np.nonzero(flow.road)):
        cv2.rectangle(over, (int(c * cell), int(r * cell)), (int((c + 1) * cell), int((r + 1) * cell)),
                      (160, 110, 40), -1)
    cv2.addWeighted(over, 0.35, out, 0.65, 0, out)
    centers = np.array([[(c + 0.5) * flow.cell, (r + 0.5) * flow.cell]
                        for r in range(flow.rows) for c in range(flow.cols)])
    unit, ok = flow.direction(centers)
    for (x, y), u, good in zip(centers, unit, ok):
        if not good:
            continue
        hue = int((np.degrees(np.arctan2(u[1], u[0])) % 360) / 2)
        color = cv2.cvtColor(np.uint8([[[hue, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        p0 = np.array([x, y]) / scale
        p1 = p0 + u * cell * 0.8
        cv2.arrowedLine(out, tuple(p0.astype(int)), tuple(p1.astype(int)), color, 1, cv2.LINE_AA, tipLength=0.4)
    return out
