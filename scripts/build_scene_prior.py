#!/usr/bin/env python3
"""Learn the lane-direction prior from the sample videos.

    python scripts/build_scene_prior.py --videos samples/

Sums the flow fields of all sample videos into weights/scene_prior.npz. At
test time the pipeline adds each video's own flow on top, so even a short
test clip with little traffic knows every lane's direction. The prior is
built from our own tracks of the organisers' sample videos only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.dev_eval import cached_observation  # noqa: E402
from src.pipeline import PRIOR_PATH  # noqa: E402
from src.scene import FlowField  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, type=Path)
    args = ap.parse_args()
    videos = sorted(p for p in args.videos.iterdir() if p.suffix.lower() == ".mp4")
    total = None
    for path in videos:
        obs = cached_observation(path)
        flow = FlowField(obs.info.width, obs.info.height).add_tracks(obs.tracks)
        if total is None:
            total = flow
        else:
            total.n += flow.n
            total.vec += flow.vec
            total.tracks += flow.tracks
        print(f"{path.name}: {len(flow.contrib)} moving vehicles")
    if total is None:
        print("no .mp4 files found", file=sys.stderr)
        return 2
    total.to_npz(PRIOR_PATH)
    print(f"wrote {PRIOR_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
