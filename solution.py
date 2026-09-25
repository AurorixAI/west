"""Team WEST submission: traffic event detection (Part A) and accident anticipation (Part B).

    detect_events(video_path) -> [[start_sec, end_sec, label], ...]
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float

The implementation lives in src/: see README.md for the architecture.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CLASSES  # noqa: E402
from src.detection import warm_up  # noqa: E402
from src.pipeline import GPU, analyze, default_profile  # noqa: E402
from src.risk import CPU_SETTINGS, GPU_SETTINGS, CausalRiskEstimator  # noqa: E402

__all__ = ["CLASSES", "detect_events", "RiskEstimator"]


def _warm_up() -> None:
    """Pay the one-off start-up cost now: the harness times each video from its first call."""
    part_a = default_profile()
    weights_b, _, width_b = GPU_SETTINGS if part_a is GPU else CPU_SETTINGS
    try:
        warm_up([(part_a.weights, part_a.detect_width), (weights_b, width_b)])
    except Exception as exc:  # a failed warm-up only costs time later; never fail the import
        print(f"warning: detector warm-up failed: {exc}", file=sys.stderr)


_warm_up()


def detect_events(video_path: str) -> list[list]:
    """Part A. Return [[start_sec, end_sec, label], ...] for one .mp4."""
    return analyze(video_path).events


class RiskEstimator:
    """Part B. Causal: step() sees frames in order and nothing else."""

    def __init__(self) -> None:
        self._est = CausalRiskEstimator()

    def reset(self, meta: dict) -> None:
        self._est.reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return self._est.step(frame, t_sec)
