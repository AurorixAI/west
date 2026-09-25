"""
solution.py — WIUT Hackathon 2026: Computer Vision track submission.

Implements:
    detect_events(video_path) -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# Add directory to sys.path so src imports work seamlessly
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import cv2
import numpy as np

from src.config import CLASSES, FRAME_STRIDE
from src.traffic_light import TrafficLightDetector
from src.tracker import VideoTracker
from src.rules import EventEngine
from src.risk import CausalRiskEstimator

__all__ = ["CLASSES", "detect_events", "RiskEstimator"]


def detect_events(video_path: str) -> list[list]:
    """Part A — traffic event detection.

    Args:
        video_path: path to one .mp4 file.

    Returns:
        List of events: [[start_sec, end_sec, label], ...]
    """
    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    tl_detector = TrafficLightDetector(history_len=5)
    tracker = VideoTracker(model_name="yolov8n.pt", imgsz=1280, conf=0.25)
    engine = EventEngine()

    stride = max(1, int(FRAME_STRIDE))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            t_sec = frame_idx / fps
            # 1. Detect traffic light state (RED / GREEN)
            tl_state = tl_detector.detect_frame(frame)
            # 2. Multi-object tracking (YOLOv8 + ByteTrack)
            detections = tracker.step(frame, t_sec)
            # 3. Normalize coordinates to calibrated 3840x2160 geometry
            H, W = frame.shape[:2]
            if W != 3840 or H != 2160:
                sx = 3840.0 / W
                sy = 2160.0 / H
                for d in detections:
                    d["centroid"] = (d["centroid"][0] * sx, d["centroid"][1] * sy)
            # 4. Rule evaluation across zones and trajectories
            engine.process_frame(detections, tl_state, t_sec)

        frame_idx += 1

    cap.release()

    # Post-process, merge contiguous events, clamp times
    events = engine.finalize(duration)
    return events


class RiskEstimator:
    """Part B — causal accident anticipation."""

    def __init__(self):
        self.estimator = CausalRiskEstimator(stride=3)

    def reset(self, meta: dict) -> None:
        """Called once before the first frame of each video."""
        self.estimator.reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        """Return P(accident starts within next 5s) in [0, 1]."""
        return self.estimator.step(frame, t_sec)
