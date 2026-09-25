"""Traffic-signal state from the glowing bulb colour inside a fixed ROI."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

UNKNOWN, RED, GREEN = 0, 1, 2
NAMES = {UNKNOWN: "UNKNOWN", RED: "RED", GREEN: "GREEN"}


def bulb_pixels(full_frame: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[int, int]:
    """Count lit red and lit green pixels in the signal ROI of a full-resolution frame."""
    y0, y1, x0, x1 = roi
    crop = full_frame[y0:y1, x0:x1]
    if crop.size == 0:
        return 0, 0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 80, 90), (12, 255, 255)) | cv2.inRange(hsv, (168, 80, 90), (180, 255, 255))
    green = cv2.inRange(hsv, (40, 70, 90), (95, 255, 255))
    return int(cv2.countNonZero(red)), int(cv2.countNonZero(green))


def instant_state(red: int, green: int, min_pixels: int) -> int:
    if red >= min_pixels and red > 2 * green:
        return RED
    if green >= min_pixels and green > 2 * red:
        return GREEN
    return UNKNOWN


@dataclass
class SignalTimeline:
    t: np.ndarray        # sample times
    state: np.ndarray    # smoothed state per sample

    @staticmethod
    def from_counts(t: np.ndarray, counts: np.ndarray, frame_area: int,
                    vote_sec: float = 1.0, fill_sec: float = 2.0) -> "SignalTimeline":
        """Smooth per-frame bulb counts into a state series.

        Majority vote over a centred ``vote_sec`` window removes single-frame
        flicker (LED refresh, occlusion by a passing bus); UNKNOWN gaps shorter
        than ``fill_sec`` take the state on both sides when it agrees.
        """
        t = np.asarray(t, float)
        min_px = max(6, int(25 * frame_area / (3840 * 2160)))
        raw = np.array([instant_state(r, g, min_px) for r, g in counts], int) if len(counts) else np.zeros(0, int)
        half = vote_sec / 2
        lo = np.searchsorted(t, t - half, "left")
        hi = np.searchsorted(t, t + half, "right")
        cr = np.concatenate([[0], np.cumsum(raw == RED)])
        cg = np.concatenate([[0], np.cumsum(raw == GREEN)])
        nr, ng = cr[hi] - cr[lo], cg[hi] - cg[lo]
        st = np.where((nr > ng) & (nr > 0), RED, np.where((ng > nr) & (ng > 0), GREEN, UNKNOWN))
        # bridge short unknown gaps whose two sides agree
        i, n = 0, len(st)
        while i < n:
            if st[i] != UNKNOWN:
                i += 1
                continue
            j = i
            while j < n and st[j] == UNKNOWN:
                j += 1
            if 0 < i and j < n and st[i - 1] == st[j] and t[j] - t[i - 1] <= fill_sec:
                st[i:j] = st[j]
            i = j
        return SignalTimeline(t, st)

    @property
    def observable(self) -> bool:
        """True if both red and green were seen: otherwise signal rules are skipped."""
        return bool(len(self.state) and (self.state == RED).any() and (self.state == GREEN).any())

    def at(self, t: float) -> int:
        if not len(self.t):
            return UNKNOWN
        i = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        if i > 0 and abs(self.t[i - 1] - t) < abs(self.t[i] - t):
            i -= 1
        return int(self.state[i])

    def red_since(self, t: float) -> float | None:
        """Start time of the red phase containing t, or None if not red at t."""
        i = int(np.clip(np.searchsorted(self.t, t, "right") - 1, 0, len(self.t) - 1)) if len(self.t) else -1
        if i < 0 or self.state[i] != RED:
            return None
        while i > 0 and self.state[i - 1] == RED:
            i -= 1
        return float(self.t[i])

    def next_green(self, t: float) -> float | None:
        idx = np.nonzero((self.t > t) & (self.state == GREEN))[0]
        return float(self.t[idx[0]]) if len(idx) else None

    def phases(self) -> list[tuple[float, float, str]]:
        out = []
        if not len(self.t):
            return out
        s = 0
        for i in range(1, len(self.t) + 1):
            if i == len(self.t) or self.state[i] != self.state[s]:
                out.append((float(self.t[s]), float(self.t[i - 1]), NAMES[int(self.state[s])]))
                s = i
        return out
