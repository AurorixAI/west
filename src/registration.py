"""Align each video to the reference view of the competition camera.

The painted geometry (crossings, stop line, carriageway, signal heads, signs)
is placed once, on a reference frame of this camera. A test video may be
recorded at another resolution, or after the camera was nudged or zoomed;
matching static scenery (buildings, poles, kerbs, markings) between the
reference and the video gives a homography that carries the geometry over.
A video from a different camera does not match: its geometry is switched
off, and only the camera-independent parts of the pipeline run on it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src import config as C

REFERENCE_PATH = C.WEIGHTS_DIR / "scene_reference.jpg"
WORK_WIDTH = 1280                 # both images are matched at this width
MIN_INLIERS = 80
MIN_INLIER_RATIO = 0.25
SNAP_PX = 2.0                     # closer than this to pure scaling: use pure scaling


@dataclass
class Registration:
    known: bool                   # the video shows the reference camera's view
    transform: np.ndarray         # 3x3: reference (3840x2160) pixels -> video pixels
    inliers: int
    note: str


def _prepare(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Grey, contrast-equalised, WORK_WIDTH wide; returns the image and its scale factor."""
    grey = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = WORK_WIDTH / grey.shape[1]
    grey = cv2.resize(grey, (WORK_WIDTH, int(round(grey.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey), s


def scaling(width: int, height: int) -> np.ndarray:
    """The transform when the video shows exactly the reference view."""
    return np.diag([width / C.REF_W, height / C.REF_H, 1.0])


def median_frame(path: str, n: int = 7) -> np.ndarray | None:
    """Median of n frames spread over the video: parked scenery stays, traffic goes."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for k in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.resize(f, (WORK_WIDTH, int(round(f.shape[0] * WORK_WIDTH / f.shape[1])))))
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8) if frames else None


def register(frame: np.ndarray | None, width: int, height: int,
             reference: np.ndarray | None = None) -> Registration:
    """Homography from the reference view to this video, or known=False."""
    if reference is None:
        if not REFERENCE_PATH.exists():
            return Registration(True, scaling(width, height), 0, "no reference image: assuming the reference view")
        reference = cv2.imread(str(REFERENCE_PATH), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return Registration(False, scaling(width, height), 0, "could not read frames")
    cv2.setRNGSeed(C.SEED)                         # RANSAC is deterministic run to run
    ref, s_ref = _prepare(reference)
    vid, _ = _prepare(frame)
    s_vid = WORK_WIDTH / width        # from the video's own size: the frame may arrive downscaled
    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=10)
    kr, dr = orb.detectAndCompute(ref, None)
    kv, dv = orb.detectAndCompute(vid, None)
    if dr is None or dv is None or len(kr) < 50 or len(kv) < 50:
        return Registration(False, scaling(width, height), 0, "too little texture to match")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dr, dv, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < 0.8 * rest[0].distance]
    if len(good) < MIN_INLIERS:
        return Registration(False, scaling(width, height), len(good), f"only {len(good)} matches: another camera")
    src = np.float32([kr[m.queryIdx].pt for m in good])
    dst = np.float32([kv[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if H is None or inliers < MIN_INLIERS or inliers < MIN_INLIER_RATIO * len(good):
        return Registration(False, scaling(width, height), inliers, f"{inliers} consistent matches: another camera")
    # reference 3840x2160 pixels -> reference work image -> video work image -> video pixels
    ref_scale = s_ref * reference.shape[1] / C.REF_W
    full = np.diag([1 / s_vid, 1 / s_vid, 1.0]) @ H @ np.diag([ref_scale, ref_scale, 1.0])
    full /= full[2, 2]
    corners = np.float32([[0, 0], [C.REF_W, 0], [C.REF_W, C.REF_H], [0, C.REF_H]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, full).reshape(-1, 2)
    area = cv2.contourArea(mapped) / (width * height)
    if not 0.4 < area < 2.5 or not cv2.isContourConvex(mapped.astype(np.float32)):
        return Registration(False, scaling(width, height), inliers, "implausible alignment: another camera")
    plain = scaling(width, height)
    drift = np.abs(mapped - cv2.perspectiveTransform(corners, plain).reshape(-1, 2)).max()
    if drift < SNAP_PX * max(width, height) / 1920:
        return Registration(True, plain, inliers, "reference view")
    return Registration(True, full, inliers, f"camera moved: corners shifted by up to {drift:.0f} px")
