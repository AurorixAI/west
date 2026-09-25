"""Part A pipeline: video -> detections -> tracks -> scene model -> events."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src import config as C
from src import rules
from src.detection import Detector, pick_device
from src.registration import median_frame, register
from src.scene import FlowField, SceneGeometry
from src.signal_state import SignalTimeline, bulb_pixels
from src.tracking import MultiTracker, Track, stitch
from src.video import Frame, VideoInfo, probe, read_frames, stride_for

PRIOR_PATH = C.WEIGHTS_DIR / "scene_prior.npz"

# Part A may use this share of the 3x-duration budget; the rest is Part B's.
PART_A_BUDGET = 1.8


@dataclass
class Profile:
    """Speed/accuracy settings; ``default_profile()`` picks one for the hardware."""
    weights: str = C.DETECTOR_WEIGHTS
    analysis_fps: float = C.ANALYSIS_FPS
    detect_width: int = C.DETECT_WIDTH
    batch: int = C.BATCH_SIZE


GPU = Profile()
# Without a GPU the full profile cannot keep to the time budget: a lighter
# detector at half the sampling rate still does (~0.4 s per video second).
CPU = Profile(weights="yolov8n.pt", analysis_fps=5.0, detect_width=960, batch=4)


def default_profile() -> Profile:
    """GPU profile on CUDA, CPU profile otherwise; WEST_PROFILE=gpu|cpu forces one.

    Forcing ``gpu`` on a CPU reproduces the evaluation machine's configuration
    (slowly: run the harness with a larger --time-factor).
    """
    forced = os.environ.get("WEST_PROFILE", "").lower()
    if forced in ("gpu", "cpu"):
        return GPU if forced == "gpu" else CPU
    return GPU if pick_device().startswith("cuda") else CPU


@dataclass
class Observation:
    """Everything measured from the pixels: the expensive, cacheable stage."""
    info: VideoInfo
    tracks: list[Track]
    signal: SignalTimeline
    seconds: float
    transform: np.ndarray | None = None   # reference view -> this video (None: plain scaling)
    camera_known: bool = True             # False: another camera, zones switched off
    registration: str = ""


@dataclass
class Analysis:
    obs: Observation
    geom: SceneGeometry
    flow: FlowField
    detailed: list[rules.Segment]      # merged events with the track ids involved

    @property
    def events(self) -> list[list]:
        """The official output: [[start_sec, end_sec, label], ...]."""
        return [[s, e, lbl] for s, e, lbl, _ in self.detailed]

    @property
    def info(self) -> VideoInfo:
        return self.obs.info

    @property
    def tracks(self) -> list[Track]:
        return self.obs.tracks

    @property
    def signal(self) -> SignalTimeline:
        return self.obs.signal


def observe(path: str, profile: Profile | None = None,
            progress: Callable[[float], None] | None = None,
            on_frame: Callable[[Frame, list], None] | None = None) -> Observation:
    """Decode, detect and track. ``on_frame(frame, tracked)`` sees every analysed frame."""
    t_start = time.perf_counter()
    profile = profile or default_profile()
    info = probe(path)
    reg = register(median_frame(path), info.width, info.height)
    geom = SceneGeometry(info.width, info.height, reg.transform, reg.known)
    stride = stride_for(info.fps, profile.analysis_fps)
    detector = Detector(weights=profile.weights, imgsz=profile.detect_width)
    tracker = MultiTracker(info.fps / stride)
    sig_t, sig_counts = [], []
    batch, keep_every = [], 1

    def flush():
        dets = detector([f.image for f in batch], [f.scale for f in batch])
        for f, d in zip(batch, dets):
            tracked = tracker.update(f.t, d)
            if on_frame:
                on_frame(f, tracked)
        batch.clear()

    for n, frame in enumerate(read_frames(info, stride, profile.detect_width,
                                          full_res_hook=lambda im: bulb_pixels(im, geom.tl_roi))):
        sig_t.append(frame.t)
        sig_counts.append(frame.extra)
        if n % keep_every:
            continue
        batch.append(frame)
        if len(batch) >= profile.batch:
            flush()
            done = frame.t / max(info.duration, 1e-6)
            if progress:
                progress(min(done, 1.0))
            # running late (slow GPU, or none): halve the detection rate, repeatedly if needed
            elapsed = time.perf_counter() - t_start
            if (not os.environ.get("WEST_NO_THIN") and done > 0.05 and keep_every < 8
                    and elapsed / done > 0.9 * PART_A_BUDGET * info.duration):
                keep_every *= 2
    if batch:
        flush()

    tracks = stitch([t.finalize() for t in tracker.tracks()])
    signal = SignalTimeline.from_counts(np.array(sig_t), np.array(sig_counts).reshape(-1, 2),
                                        info.width * info.height)
    return Observation(info, tracks, signal, time.perf_counter() - t_start, reg.transform, reg.known, reg.note)


def interpret(obs: Observation, enabled=C.ENABLED_CLASSES) -> Analysis:
    """Scene model + rules: cheap, so thresholds can be tuned on cached observations."""
    info = obs.info
    geom = SceneGeometry(info.width, info.height, obs.transform, obs.camera_known)
    flow = FlowField(info.width, info.height)
    if PRIOR_PATH.exists():
        flow.add_prior(PRIOR_PATH)
    flow.add_tracks(obs.tracks)
    ctx = rules.Context(obs.tracks, geom, flow, obs.signal, info.duration)
    return Analysis(obs, geom, flow, rules.merge(rules.detect(ctx, enabled), info.duration))


def analyze(path: str, profile: Profile | None = None,
            progress: Callable[[float], None] | None = None) -> Analysis:
    analysis = interpret(observe(path, profile, progress))
    if progress:
        progress(1.0)
    return analysis
