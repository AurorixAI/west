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

# Measured on the organisers' sample frame (1080p coordinates x 2).
# Stop line across the near carriageway, before the main crossing.
STOP_LINE = np.array([[570, 1104], [1870, 936]], dtype=np.float32)

# Zebra crossing from the left kerb to the central island (a narrow diagonal band).
CROSSWALK_MAIN = np.array(
    [[660, 1252], [2360, 1030], [2370, 1116], [744, 1348]], dtype=np.float32)

# Its continuation from the island to the right-hand kerb.
CROSSWALK_RIGHT = np.array(
    [[2580, 996], [3720, 924], [3740, 984], [2660, 1096]], dtype=np.float32)

# Zebra crossing in the bottom-left corner. Lens distortion bends it at this corner of
# the frame, so both edges are traced point by point.
CROSSWALK_CORNER = np.array(
    [[356, 1562], [674, 1474], [874, 1586], [1250, 1748], [1500, 1868], [1612, 1980], [1724, 2080], [1780, 2160], [1200, 2160], [1162, 2080], [1100, 1980], [906, 1848], [612, 1692]],
    dtype=np.float32)

# Between the stop line and the far edge of the main crossing: a vehicle that
# stops here on red has passed the line without entering the intersection.
STOP_ZONE = np.array(
    [[570, 1104], [1870, 936], [2000, 1164], [744, 1348]], dtype=np.float32)

# Carriageway traced on the sample frame: both carriageways of the avenue and
# the intersection, bounded by the kerbs (parking strip on the left excluded).
CARRIAGEWAY = np.array(
    [[200, 180], [1200, 296], [2000, 464], [2600, 596], [2700, 660], [2800, 790], [3600, 820], [3680, 900], [3840, 1000], [3840, 2160],
     [0, 2160], [0, 1700], [320, 1600], [500, 1520], [660, 1400], [670, 1280], [580, 1140],
     [660, 1110], [120, 370]], dtype=np.float32)

# Raised areas inside that outline where people legitimately stand.
NOT_CARRIAGEWAY = [
    np.array([[180, 276], [2220, 880], [2320, 960], [2280, 1040], [1860, 940], [120, 364]], np.float32),  # median
    np.array([[2160, 880], [2300, 880], [2600, 1000], [2640, 1090], [2370, 1090], [2160, 1000]], np.float32),  # island
    np.array([[2370, 1080], [2640, 1080], [2630, 1160], [2380, 1160]], np.float32),              # round island
    np.array([[1010, 1590], [1250, 1366], [1520, 1504], [1470, 1530]], np.float32),              # triangle
    np.array([[240, 1930], [660, 1750], [850, 1870], [820, 1900], [260, 1940]], np.float32),     # triangle
    np.array([[1380, 1720], [1440, 1690], [1740, 1650], [1930, 1770], [1880, 1790],
              [1480, 1830], [1380, 1760]], np.float32),                                          # island
]

# For the rendered video only: signal heads that face the camera, and the road
# signs, as (x1, y1, x2, y2) boxes. The camera is fixed, so they are mapped
# once, not detected; each head's colour is read from its own lamps per frame.
SIGNAL_HEADS = {"island": (2300, 722, 2350, 834), "kerb": (510, 1010, 540, 1086)}
ROAD_SIGNS = {"pedestrian crossing": (2434, 708, 2498, 780), "keep right": (2550, 994, 2610, 1064)}

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
# Buffered IoU (Yang et al. 2021): boxes are widened by this share of their
# size on every side before association, so a car that moves more than its
# own length between two samples still overlaps its predicted box. The
# tracker hands back the original boxes. Two-wheelers ride side by side and
# among pedestrians: widened boxes would swap their ids, so they get a small
# buffer and a tracker of their own (a car never inherits a bike's id).
TRACK_IOU_BUFFER = {"vehicle": 0.5, "two_wheeler": 0.1, "person": 0.0, "animal": 0.0}
BATCH_SIZE = 8

COCO_PERSON = 0
COCO_TWO_WHEELER = (1, 3)                 # bicycle, motorcycle
COCO_VEHICLE = (2, 5, 7)                  # car, bus, truck
COCO_BUS = 5
COCO_BICYCLE = 1
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
JAYWALK_MIN_SPEED = 0.25     # heights/s; walking is ~0.6-0.8
CROSSWALK_MARGIN = 0.15      # crossing polygons are grown by this many person heights (box jitter)
KERB_INSET = 0.6             # a pedestrian must be this many heights inside the outer kerb
ISLAND_MARGIN = 0.2          # ... and this many heights off an island

YIELD_MAX_DIST = 3.0         # vehicle-pedestrian distance on the crossing, in vehicle sizes
YIELD_KERB_INSET = 0.3       # pedestrian heights inside the kerb: stepping onto the road, not waiting
WALKING_PACE = 1.0           # sizes/s; a two-wheeler slower than this is being wheeled

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
ACCIDENT_TAIL_SEC = 1.5      # ...or, near the end of a video, over at least this much of what is left

OBSTACLE_MIN_SEC = 2.0
OBSTACLE_MIN_CONF = 0.5

# Temporal post-processing: same-class segments closer than this are merged.
MERGE_GAP_SEC = 0.5
MIN_EVENT_SEC = 0.3          # a car at speed crosses a zebra in about this long
