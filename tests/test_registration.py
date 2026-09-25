"""Each video is aligned to the reference view; a foreign camera switches the zones off."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src import registration as R
from src.scene import SceneGeometry
from tests.synth import CAR, PERSON, Scene

REF = cv2.imread(str(R.REFERENCE_PATH), cv2.IMREAD_GRAYSCALE)
PROBES = np.float32([[450, 1060], [1400, 1050], [2400, 1030], [700, 1500], [3000, 1800]]).reshape(-1, 1, 2)


def _error(reg: R.Registration, truth: np.ndarray) -> float:
    return float(np.abs(cv2.perspectiveTransform(PROBES, reg.transform) - cv2.perspectiveTransform(PROBES, truth)).max())


@pytest.mark.skipif(REF is None, reason="no reference image")
@pytest.mark.parametrize("angle,zoom,tx,ty", [(0, 1, 40, -25), (2, 1, 0, 0), (0, 1.08, -60, -30), (1.5, 0.94, 30, 20)])
def test_a_moved_camera_is_recovered(angle, zoom, tx, ty):
    h, w = REF.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, zoom)
    M[:, 2] += (tx, ty)
    moved = cv2.warpAffine(REF, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    reg = R.register(moved, w, h)
    assert reg.known and _error(reg, np.vstack([M, [0, 0, 1]]) @ R.scaling(w, h)) < 3.0


@pytest.mark.skipif(REF is None, reason="no reference image")
@pytest.mark.parametrize("gamma", [0.5, 1.8])
def test_daylight_and_darker_evenings_still_match(gamma):
    lit = np.clip(255 * (REF / 255.0) ** gamma, 0, 255).astype(np.uint8)
    reg = R.register(lit, REF.shape[1], REF.shape[0])
    assert reg.known and _error(reg, R.scaling(REF.shape[1], REF.shape[0])) < 3.0


@pytest.mark.skipif(REF is None, reason="no reference image")
def test_the_same_view_at_another_resolution_is_plain_scaling():
    small = cv2.resize(REF, (1280, 720), interpolation=cv2.INTER_AREA)
    reg = R.register(small, 1280, 720)
    assert reg.known and np.allclose(reg.transform, R.scaling(1280, 720))


@pytest.mark.skipif(REF is None, reason="no reference image")
def test_a_downscaled_frame_is_mapped_to_the_videos_own_size():
    """Regression: median_frame() hands over 1280-wide frames of a 1920-wide video."""
    small = cv2.resize(REF, (1280, 720), interpolation=cv2.INTER_AREA)
    reg = R.register(small, 1920, 1080)
    assert reg.known and np.allclose(reg.transform, R.scaling(1920, 1080))


@pytest.mark.skipif(REF is None, reason="no reference image")
def test_the_real_sample_clip_is_the_reference_view():
    reg = R.register(R.median_frame("samples/sample_test.mp4"), 1920, 1080)
    assert reg.known and np.allclose(reg.transform, R.scaling(1920, 1080)), reg.note


@pytest.mark.skipif(REF is None, reason="no reference image")
def test_another_camera_is_not_matched():
    rng = np.random.default_rng(0)
    other = cv2.GaussianBlur(rng.integers(0, 255, REF.shape, dtype=np.uint8), (0, 0), 3)
    cv2.rectangle(other, (300, 300), (900, 700), 255, -1)
    assert not R.register(other, REF.shape[1], REF.shape[0]).known
    assert not R.register(cv2.flip(REF, 1), REF.shape[1], REF.shape[0]).known


def test_unknown_camera_rules_stay_quiet_and_do_not_crash():
    s = Scene(120.0)
    s.background()
    s.path("person", PERSON, 50.0, [(700, 1300), (1150, 1240)], 60, size=150)
    s.path("vehicle", CAR, 60.0, [(900, 600), (900, 2100)], 400)
    ctx = s.context()
    ctx.geom = SceneGeometry(3840, 2160, known=False)
    from src import rules
    events = rules.detect(ctx)
    assert not any(lbl in ("failure_to_yield", "red_light", "stop_line") for _, _, lbl, _ in events)
