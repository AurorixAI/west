"""Drawing: annotated frames, event thumbnails and EDA images."""
from __future__ import annotations

import cv2
import numpy as np

from src.scene import FlowField, SceneGeometry
from src.signal_state import GREEN, RED, bulb_pixels, instant_state
from src.tracking import Track

# BGR. Event colours follow the website's class families; objects stay quiet.
CATEGORY_COLOR = {"vehicle": (214, 196, 150), "person": (150, 214, 160), "animal": (120, 200, 240)}
_FAMILY = {"collision": (52, 104, 235), "vehicle": (229, 135, 57), "pedestrian": (122, 175, 27), "flow": (0, 161, 237)}
EVENT_COLOR = {
    "accident": _FAMILY["collision"], "near_miss": _FAMILY["collision"],
    "red_light": _FAMILY["vehicle"], "wrong_way": _FAMILY["vehicle"], "illegal_u_turn": _FAMILY["vehicle"],
    "illegal_turn": _FAMILY["vehicle"], "solid_line_crossing": _FAMILY["vehicle"], "stop_line": _FAMILY["vehicle"],
    "failure_to_yield": _FAMILY["vehicle"], "jaywalking": _FAMILY["pedestrian"],
    "stopped_vehicle": _FAMILY["flow"], "congestion": _FAMILY["flow"], "road_obstacle": _FAMILY["flow"],
    "fire_smoke": _FAMILY["flow"],
}
INK = (28, 24, 22)
FONT = cv2.FONT_HERSHEY_DUPLEX


def _label(img, text: str, x: int, y: int, bg, scale: float = 0.45, fg=INK) -> None:
    """A filled plaque with its baseline at y, clamped inside the image."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    x = int(np.clip(x, 0, img.shape[1] - tw - 10))
    y = int(np.clip(y, th + 8, img.shape[0] - 2))
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y + 2), bg, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x + 5, y - 3), FONT, scale, fg, 1, cv2.LINE_AA)


def _brackets(img, x1: int, y1: int, x2: int, y2: int, color, t: int = 1) -> None:
    """Corner brackets: marks a tracked object without boxing it in."""
    L = max(4, int(0.28 * min(x2 - x1, y2 - y1)))
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, t, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, t, cv2.LINE_AA)


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
    """Crossings and stop line, kept faint so the road stays readable."""
    for cw in geom.crosswalks:
        _poly(img, cw, scale, (235, 235, 235), 0.10, 1)
    a, b = (geom.stop_line / scale).astype(int)
    cv2.line(img, tuple(a), tuple(b), (70, 70, 230), 2, cv2.LINE_AA)


SIGN_BLUE = (191, 95, 31)


def draw_signals(img: np.ndarray, full: np.ndarray, geom: SceneGeometry, scale: float) -> None:
    """Signal heads facing the camera, in the colour their own lamps show now, and the road signs.

    ``full`` is the full-resolution frame the lamps are read from; ``img`` is drawn on.
    """
    for name, (x1, y1, x2, y2) in geom.signal_heads.items():
        red, green = bulb_pixels(full, (y1, y2, x1, x2))
        state = instant_state(red, green, max(3, (x2 - x1) * (y2 - y1) // 150))
        color = {RED: (72, 72, 235), GREEN: (90, 200, 70)}.get(state, (170, 170, 170))
        a = (int(x1 / scale) - 3, int(y1 / scale) - 3)
        b = (int(x2 / scale) + 3, int(y2 / scale) + 3)
        cv2.rectangle(img, a, b, color, 2, cv2.LINE_AA)
        text = {RED: "RED", GREEN: "GREEN"}.get(state, "--")
        _label(img, text, a[0], a[1] - 4, color, 0.4, (255, 255, 255))
    for name, (x1, y1, x2, y2) in geom.road_signs.items():
        a = (int(x1 / scale) - 2, int(y1 / scale) - 2)
        b = (int(x2 / scale) + 2, int(y2 / scale) + 2)
        cv2.rectangle(img, a, b, (240, 240, 240), 1, cv2.LINE_AA)
        _label(img, name.upper(), b[0] + 4, b[1], SIGN_BLUE, 0.36, (255, 255, 255))


def draw_objects(img: np.ndarray, samples: list[tuple[Track, int]], scale: float,
                 highlight: dict[int, str]) -> None:
    """Brackets for every tracked object; a solid box and a plaque for event actors."""
    for tr, i in samples:
        if tr.tid in highlight:
            continue
        x1, y1, x2, y2 = (tr.box[i] / scale).astype(int)
        _brackets(img, x1, y1, x2, y2, CATEGORY_COLOR.get(tr.category, (200, 200, 200)))
    for tr, i in samples:                       # actors on top
        label = highlight.get(tr.tid)
        if not label:
            continue
        x1, y1, x2, y2 = (tr.box[i] / scale).astype(int)
        color = EVENT_COLOR.get(label, (0, 0, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        # in failure_to_yield the person is the one not given way, not the offender
        text = "PEDESTRIAN" if label == "failure_to_yield" and tr.category == "person" \
            else label.replace("_", " ").upper()
        _label(img, f"{text}  {tr.tid}", x1, y1 - 3, color)


def draw_hud(img: np.ndarray, t: float, signal: int, n_vehicles: int, n_people: int,
             active: list[str], risk: float | None) -> None:
    h, w = img.shape[:2]
    over = img.copy()
    cv2.rectangle(over, (0, 0), (w, 36), (20, 18, 16), -1)
    cv2.addWeighted(over, 0.78, img, 0.22, 0, img)
    cv2.putText(img, f"{int(t // 60):02d}:{t % 60:05.2f}", (14, 24), FONT, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    sig_col = {RED: (72, 72, 230), GREEN: (90, 190, 70)}.get(signal, (140, 140, 140))
    cv2.circle(img, (132, 18), 6, sig_col, -1, cv2.LINE_AA)
    cv2.putText(img, {RED: "SIGNAL RED", GREEN: "SIGNAL GREEN"}.get(signal, "SIGNAL --"), (146, 24), FONT, 0.5,
                (225, 225, 225), 1, cv2.LINE_AA)
    cv2.putText(img, f"{n_vehicles} VEHICLES   {n_people} PEOPLE", (310, 24), FONT, 0.5, (185, 185, 185), 1,
                cv2.LINE_AA)
    if risk is not None:
        x0 = w - 236
        cv2.putText(img, "RISK", (x0, 24), FONT, 0.5, (185, 185, 185), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x0 + 48, 13), (x0 + 218, 23), (70, 66, 62), -1, cv2.LINE_AA)
        col = (72, 72, 230) if risk >= 0.5 else (30, 176, 242) if risk >= 0.2 else (90, 190, 70)
        cv2.rectangle(img, (x0 + 48, 13), (x0 + 48 + max(2, int(170 * risk)), 23), col, -1, cv2.LINE_AA)
    for k, lbl in enumerate(active):
        _label(img, lbl.replace("_", " ").upper(), 14, h - 16 - k * 34, EVENT_COLOR.get(lbl, (0, 0, 255)), 0.62)


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
