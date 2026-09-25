"""
geometry.py — Geometric calculations for zone membership, line crossings, and kinematics.
"""
import cv2
import numpy as np

def point_in_poly(point: tuple[float, float], polygon: np.ndarray) -> bool:
    """Check if point (x, y) is inside polygon."""
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def line_segment_intersects(A, B, C, D):
    """Return True if line segment AB intersects CD."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def line_crossing_direction(p_prev: tuple[float, float], p_curr: tuple[float, float],
                            line_start: tuple[float, float], line_end: tuple[float, float]) -> bool:
    """
    Check if object moved across the line.
    """
    return line_segment_intersects(p_prev, p_curr, line_start, line_end)

def calculate_speed(p1: tuple[float, float], p2: tuple[float, float], dt: float) -> float:
    """Speed in pixels per second."""
    if dt <= 0:
        return 0.0
    dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
    return dist / dt
