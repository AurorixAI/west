"""Tracking post-processing, signal smoothing and the Part B risk estimator."""
from __future__ import annotations

from collections import Counter

import numpy as np

from src import risk
from src.detection import Detections
from src.signal_state import GREEN, RED, UNKNOWN, SignalTimeline
from src.tracking import Track, stitch
from tests.synth import CAR, times


def _track(tid, t, x, y=1000.0, size=100.0):
    foot = np.stack([x, np.full_like(t, y)], 1)
    box = np.stack([foot[:, 0] - size / 2, foot[:, 1] - size, foot[:, 0] + size / 2, foot[:, 1]], 1)
    return Track(tid, "vehicle", CAR, t, box, Counter({CAR: len(t)})).finalize()


# -- tracking ------------------------------------------------------------------
def test_stitch_joins_an_id_switch_of_a_parked_car():
    a = _track(1, times(0, 10), np.full(101, 500.0))
    b = _track(2, times(13, 30), np.full(171, 505.0))
    far = _track(3, times(12, 20), np.full(81, 2000.0))
    out = stitch([a, b, far])
    assert len(out) == 2
    joined = next(t for t in out if t.tid == 1)
    assert joined.start == 0 and joined.end == 30


def test_stitch_does_not_join_distant_fragments():
    a = _track(1, times(0, 5), np.linspace(0, 500, 51))
    b = _track(2, times(6, 10), np.linspace(1500, 1900, 41))
    assert len(stitch([a, b])) == 2


def test_speed_is_in_sizes_per_second():
    tr = _track(1, times(0, 5), np.linspace(0, 1000, 51), size=100)
    assert abs(np.median(tr.speed) - 2.0) < 0.05


# -- signal --------------------------------------------------------------------
def test_signal_votes_out_flicker_and_bridges_short_gaps():
    t = times(0, 20)
    counts = np.zeros((len(t), 2), int)
    counts[t < 10, 0] = 100                  # red
    counts[t >= 10, 1] = 100                 # green
    counts[np.isin(np.round(t, 1), [3.0, 3.1]), :] = [0, 100]   # 0.2 s green flicker
    counts[(t > 14) & (t < 15)] = 0          # bulb occluded for 1 s
    sig = SignalTimeline.from_counts(t, counts, 3840 * 2160)
    assert sig.observable
    assert sig.at(3.05) == RED and sig.at(14.5) == GREEN
    assert [(round(a), p) for a, _, p in sig.phases()] == [(0, "RED"), (10, "GREEN")]
    assert abs(sig.red_since(8.0) - 0.0) < 1e-9 and sig.red_since(12.0) is None
    assert abs(sig.next_green(5.0) - 10.0) < 0.11


def test_signal_without_a_lit_bulb_is_unobservable():
    t = times(0, 10)
    sig = SignalTimeline.from_counts(t, np.zeros((len(t), 2), int), 3840 * 2160)
    assert not sig.observable and sig.at(5) == UNKNOWN


# -- Part B ----------------------------------------------------------------------
def test_pair_hazard_ranks_conflicts():
    head_on, t_star, _ = risk.pair_hazard(np.array([400.0, 0]), np.array([-800.0, 0]), 100)
    passing, _, _ = risk.pair_hazard(np.array([400.0, 300]), np.array([-800.0, 0]), 100)
    diverging, _, _ = risk.pair_hazard(np.array([400.0, 0]), np.array([800.0, 0]), 100)
    assert abs(t_star - 0.5) < 1e-9 and head_on > 0.6
    assert passing < 0.05 and diverging == 0.0


class ScriptedDetector:
    """Stands in for YOLO: boxes come from a function of time."""

    def __init__(self, script):
        self.script, self.t = script, 0.0

    def __call__(self, images, scales):
        boxes = self.script(self.t)
        xyxy = np.array(boxes, np.float32).reshape(-1, 4)
        return [Detections(xyxy, np.full(len(xyxy), 0.9, np.float32), np.full(len(xyxy), CAR, np.int32))]


def _run(script, duration=10.0, fps=25.0):
    est = risk.CausalRiskEstimator.__new__(risk.CausalRiskEstimator)
    est.detector = ScriptedDetector(script)
    est.target_fps, est.width = risk.GPU_SETTINGS[1], 64
    est.reset({"fps": fps, "video_id": "x", "width": 64, "height": 36, "n_frames": int(duration * fps)})
    frame = np.zeros((36, 64, 3), np.uint8)
    out = []
    for i in range(int(duration * fps)):
        t = i / fps
        est.detector.t = t
        out.append((t, est.step(frame, t)))
    return np.array(out)


def _box(x, y, s=100.0):
    return [x - s / 2, y - s, x + s / 2, y]


def test_crossing_conflict_raises_the_alarm_before_impact():
    # A drives east, B drives south; both reach (2000, 1000) at t = 8 s.
    def script(t):
        return [_box(2000 - 300 * (8 - t), 1000), _box(2000, 1000 - 300 * (8 - t))] if t < 8 else []
    curve = _run(script)
    before = curve[(curve[:, 0] >= 3) & (curve[:, 0] < 8)]
    alarm = before[before[:, 1] >= 0.5]
    assert len(alarm), "no alarm before the impact"
    assert 8 - alarm[0, 0] >= 1.5             # warned well before the impact
    assert curve[curve[:, 0] < 3, 1].max() < 0.5


def test_fast_cars_whose_boxes_never_overlap_are_still_followed():
    """At 6 detections/s a car at 6 lengths/s jumps a full length: its boxes never overlap."""
    def script(t):
        return [_box(2000 - 600 * (8 - t), 1000), _box(2000, 1000 - 600 * (8 - t))] if t < 8 else []
    curve = _run(script)
    alarm = curve[(curve[:, 0] < 8) & (curve[:, 1] >= 0.5)]
    assert len(alarm) and 8 - alarm[0, 0] >= 0.5


def test_platoon_does_not_raise_the_alarm():
    # Two cars one and a half lengths apart at the same speed, for 10 s.
    def script(t):
        return [_box(500 + 300 * t, 1000), _box(350 + 300 * t, 1000)]
    assert _run(script)[:, 1].max() < 0.5


def test_score_is_causal():
    """The score at time t does not depend on anything after t."""
    def a(t):
        return [_box(2000 - 300 * (8 - t), 1000), _box(2000, 1000 - 300 * (8 - t))] if t < 8 else []

    def b(t):                                 # identical until 5 s, then B turns away
        return a(t) if t < 5 else [_box(2000 - 300 * (8 - t), 1000), _box(2000 + 300 * (t - 5), 100)]
    ca, cb = _run(a), _run(b)
    early = ca[:, 0] < 5
    assert np.array_equal(ca[early], cb[early])
