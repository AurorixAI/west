"""Scene geometry, class list and every tunable threshold of the pipeline.

Geometry is given in the 3840x2160 reference frame of the competition camera and
scaled to the actual frame size at run time (see ``scene.SceneGeometry``).

Kinematic thresholds are expressed in *object sizes per second* rather than
pixels: an object's size is sqrt(w*h) of its box for vehicles and the box
height for people. This makes the rules independent of resolution and of the
strong perspective of the camera (a far car is ~5x smaller than a near one).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
SEED = 42

CLASSES: list[str] = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]

# Classes the rule engine is allowed to emit. Macro-F1 adds every predicted
# class to the average, so a class whose detector is not specific enough costs
# more than it earns. near_miss, illegal_turn, solid_line_crossing and
# fire_smoke have no rule precise enough on this camera yet (see the report).
ENABLED_CLASSES: frozenset[str] = frozenset({
    "accident", "red_light", "wrong_way", "illegal_u_turn", "stopped_vehicle",
    "jaywalking", "failure_to_yield", "stop_line", "congestion", "road_obstacle",
})

# --------------------------------------------------------------------------
# Hand-calibrated geometry (3840x2160 reference frame)
# --------------------------------------------------------------------------
REF_W, REF_H = 3840, 2160

STOP_LINE = np.array([[450, 1060], [1400, 1050]], dtype=np.float32)

# Zebra crossing from the left kerb to the central island.
CROSSWALK_MAIN = np.array(
    [[550, 1080], [2450, 1020], [2550, 1220], [600, 1340]], dtype=np.float32)

# Zebra crossing in the bottom-left corner.
CROSSWALK_CORNER = np.array(
    [[0, 1280], [520, 1330], [1750, 2160], [300, 2160], [0, 1800]], dtype=np.float32)

# Between the stop line and the far edge of the main crossing: a vehicle that
# stops here on red has passed the line without entering the intersection.
STOP_ZONE = np.array(
    [[450, 1060], [1400, 1050], [1450, 1320], [600, 1340]], dtype=np.float32)

# Traffic-signal head facing the camera: (ymin, ymax, xmin, xmax).
TL_ROI = (680, 880, 2260, 2380)

# --------------------------------------------------------------------------
# Detection / tracking
# --------------------------------------------------------------------------
DETECTOR_WEIGHTS = "yolov8s.pt"
ANALYSIS_FPS = 10.0          # Part A samples the video at this rate
DETECT_WIDTH = 1280          # frames are resized to this width before YOLO
DETECT_CONF = 0.10           # low: ByteTrack uses low-score boxes for its second pass
TRACK_ACTIVATION = 0.30
TRACK_BUFFER_SEC = 3.0
BATCH_SIZE = 8

COCO_PERSON = 0
COCO_TWO_WHEELER = (1, 3)                 # bicycle, motorcycle
COCO_VEHICLE = (2, 5, 7)                  # car, bus, truck
COCO_ANIMAL = (15, 16, 17, 18, 19)        # cat, dog, horse, sheep, cow
CATEGORY_OF_CLASS = {COCO_PERSON: "person"}
CATEGORY_OF_CLASS.update({c: "vehicle" for c in COCO_VEHICLE + COCO_TWO_WHEELER})
CATEGORY_OF_CLASS.update({c: "animal" for c in COCO_ANIMAL})

# --------------------------------------------------------------------------
# Kinematics (sizes / second)
# --------------------------------------------------------------------------
SMOOTH_SEC = 1.0             # centred smoothing window for offline tracks
STOP_SPEED = 0.12            # below: stationary
MOVE_SPEED = 0.45            # above: clearly moving
RIDER_SPEED = 2.5            # a "person" faster than this is riding something

# Track stitching (closes ByteTrack ID switches of slow/stationary objects)
STITCH_MAX_GAP_SEC = 6.0
STITCH_MAX_DIST = 0.6        # in sizes

# --------------------------------------------------------------------------
# Scene model learned from each video's own vehicle tracks
# --------------------------------------------------------------------------
GRID_COLS = 64               # flow-field grid (cells are square-ish)
ROAD_MIN_TRACKS = 3          # distinct moving vehicles needed to call a cell road
FLOW_MIN_SAMPLES = 8
FLOW_MIN_COHERENCE = 0.75    # mean resultant length of headings in a cell

# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
STOPPED_MIN_SEC = 10.0
STOPPED_MIN_PASSERS = 2      # moving vehicles passing a stopped one -> not a queue
STOPPED_QUEUE_EXEMPT_SEC = 90.0

JAYWALK_MIN_SEC = 1.5
CROSSWALK_MARGIN = 0.5       # crossing polygons are grown by this many person heights

YIELD_MAX_DIST = 3.0         # vehicle-pedestrian distance on the crossing, in vehicle sizes

RED_MIN_SEC = 1.0            # signal must have been red this long (no amber cases)
RED_EVENT_MAX_SEC = 6.0      # red_light ends when the vehicle leaves the frame or after this
STOP_LINE_MIN_STOP_SEC = 2.0

WRONG_WAY_MIN_SEC = 2.0
WRONG_WAY_COS = -0.5

UTURN_MIN_DEG = 150.0
UTURN_MAX_SEC = 20.0

CONGESTION_MIN_VEHICLES = 6
CONGESTION_SLOW_FRAC = 0.8
CONGESTION_MIN_SEC = 45.0

ACCIDENT_DROP_RATIO = 0.35   # speed after contact <= ratio * speed before
ACCIDENT_STAY_SEC = 5.0      # involved vehicles stay stationary this long afterwards

OBSTACLE_MIN_SEC = 2.0
OBSTACLE_MIN_CONF = 0.5

# Temporal post-processing: same-class segments closer than this are merged.
MERGE_GAP_SEC = 0.5
MIN_EVENT_SEC = 0.5
