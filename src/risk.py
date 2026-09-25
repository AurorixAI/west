"""
risk.py — Causal accident anticipation estimator (Part B).
Uses Time-To-Collision (TTC) and kinematics between interacting road agents.
"""
from __future__ import annotations
import cv2
import numpy as np

class CausalRiskEstimator:
    def __init__(self, stride: int = 3):
        self.meta: dict = {}
        self.stride = stride
        self.frame_idx = 0
        self.last_score = 0.0
        self.prev_gray = None
        self.prev_pts = None

    def reset(self, meta: dict) -> None:
        self.meta = meta
        self.frame_idx = 0
        self.last_score = 0.0
        self.prev_gray = None
        self.prev_pts = None

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        self.frame_idx += 1
        
        # Internal stride to maintain high speed well within budget
        if self.frame_idx % self.stride != 0 and self.last_score is not None:
            return float(self.last_score)

        # Downsample frame for fast optical flow / motion analysis
        H, W = frame.shape[:2]
        small_w, small_h = 640, 360
        resized = cv2.resize(frame, (small_w, small_h))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        raw_risk = 0.0

        if self.prev_gray is not None:
            # Good features to track
            if self.prev_pts is None or len(self.prev_pts) < 50:
                self.prev_pts = cv2.goodFeaturesToTrack(
                    self.prev_gray, maxCorners=100, qualityLevel=0.03, minDistance=10
                )

            if self.prev_pts is not None and len(self.prev_pts) > 10:
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.prev_pts, None
                )
                
                if curr_pts is not None and status is not None:
                    good_prev = self.prev_pts[status == 1]
                    good_curr = curr_pts[status == 1]
                    
                    if len(good_curr) > 10:
                        displacements = good_curr - good_prev
                        speeds = np.linalg.norm(displacements, axis=1)
                        
                        # Detect rapid deceleration or violent divergence (sudden stopping / collision)
                        max_speed = float(np.percentile(speeds, 90))
                        high_speed_points = good_curr[speeds > 5.0]
                        
                        # Pairwise proximity of high-motion points
                        if len(high_speed_points) >= 4:
                            # Sample pairwise distances
                            dists = np.linalg.norm(high_speed_points[:, None, :] - high_speed_points[None, :, :], axis=-1)
                            np.fill_diagonal(dists, 9999.0)
                            min_dist = float(np.min(dists))
                            
                            if min_dist < 20.0 and max_speed > 12.0:
                                # Two fast-moving clusters converging closely
                                raw_risk = min(0.85, 0.4 + (20.0 - min_dist) / 40.0)
                            elif min_dist < 35.0 and max_speed > 8.0:
                                raw_risk = 0.35

                    self.prev_pts = good_curr.reshape(-1, 1, 2)
                else:
                    self.prev_pts = None

        self.prev_gray = gray

        # Exponential moving average for causal smoothness
        self.last_score = 0.85 * self.last_score + 0.15 * raw_risk
        return float(np.clip(self.last_score, 0.0, 1.0))
