"""The organisers' harness runs our solution, the output validates, and two runs agree."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def video_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("videos")
    w = cv2.VideoWriter(str(d / "synthetic.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (1280, 720))
    rng = np.random.default_rng(0)
    base = rng.integers(60, 90, (720, 1280, 3), dtype=np.uint8)
    for i in range(100):                        # 4 s: a grey road with a moving block
        f = base.copy()
        cv2.rectangle(f, (100 + 8 * i, 400), (220 + 8 * i, 470), (200, 200, 210), -1)
        w.write(f)
    w.release()
    return d


def _run(video_dir: Path, out: Path) -> dict:
    # A 4 s clip gets a 12 s budget, less than a cold start's model load; this test checks
    # validity and determinism, so it relaxes the budget with the harness's own dev option.
    subprocess.run([sys.executable, "run_submission.py", "--videos", str(video_dir), "--out", str(out),
                    "--team", "WEST", "--time-factor", "30"], cwd=ROOT, check=True, capture_output=True,
                   timeout=600)
    return json.loads(out.read_text())


def test_harness_output_is_valid_and_deterministic(video_dir, tmp_path):
    a = _run(video_dir, tmp_path / "a.json")
    log = a["log"]["synthetic.mp4"]
    assert log["errors"] == [], log["errors"]
    entry = a["videos"]["synthetic.mp4"]
    assert len(entry["risk"]) == 100 and all(0 <= s <= 1 for _, s in entry["risk"])
    check = subprocess.run([sys.executable, "evaluate.py", "--pred", str(tmp_path / "a.json"), "--validate-only"],
                           cwd=ROOT, capture_output=True, text=True)
    assert check.returncode == 0 and "VALID" in check.stdout, check.stdout + check.stderr

    b = _run(video_dir, tmp_path / "b.json")
    assert a["videos"] == b["videos"]
