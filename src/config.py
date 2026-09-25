"""
config.py — Calibrated scene geometry, classes, and parameters for WIUT 2026.
"""
import numpy as np

CLASSES: list[str] = [
    "accident",
    "near_miss",
    "red_light",
    "wrong_way",
    "illegal_u_turn",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "illegal_turn",
    "solid_line_crossing",
    "stop_line",
    "congestion",
    "road_obstacle",
    "fire_smoke",
]

# Precise pixel-aligned geometry on 3840x2160:
STOP_LINE = np.array([
    [450, 1060],
    [1400, 1050]
], dtype=np.int32)

# Full avenue crosswalk covering all painted zebra stripes from left curb to median island
CROSSWALK_MAIN = np.array([
    [550, 1080],
    [2450, 1020],
    [2550, 1220],
    [600, 1340]
], dtype=np.int32)

# Corner zebra crossing covering all bottom-left painted zebra stripes
CROSSWALK_CORNER = np.array([
    [0, 1280],
    [520, 1330],
    [1750, 2160],
    [300, 2160],
    [0, 1800]
], dtype=np.int32)

ROAD_SURFACE = np.array([
    [0, 950],
    [2800, 880],
    [3840, 1100],
    [3840, 2160],
    [0, 2160]
], dtype=np.int32)

# Traffic Light ROI on 3840x2160 (ymin, ymax, xmin, xmax)
TL_ROI = (680, 880, 2260, 2380)

FRAME_STRIDE = 3
YOLO_MODEL = "yolov8n.pt"
CONF_THRESHOLD = 0.3
