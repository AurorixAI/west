#!/usr/bin/env python3
"""Score Part A on our own labels of the sample videos, with cached tracks.

    python scripts/dev_eval.py --videos samples/ --gt labels/dev_labels.json
    python scripts/dev_eval.py --videos samples/ --gt labels/dev_labels.json --disable congestion,illegal_u_turn

Detection and tracking run once per video and are cached in .cache/ (keyed on
the file and the detector profile), so re-scoring after changing a rule
threshold in src/config.py takes seconds. The metric is evaluate.py's own.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluate  # noqa: E402
from src import config as C  # noqa: E402
from src.pipeline import Observation, default_profile, interpret, observe  # noqa: E402

CACHE = ROOT / ".cache"


def cached_observation(path: Path) -> Observation:
    profile = default_profile()
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{profile}".encode()).hexdigest()[:16]
    f = CACHE / f"{path.stem}-{key}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    print(f"  observing {path.name} ...", flush=True)
    obs = observe(str(path), profile)
    CACHE.mkdir(exist_ok=True)
    f.write_bytes(pickle.dumps(obs))
    return obs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--disable", default="", help="comma-separated classes to switch off")
    ap.add_argument("--out", type=Path, help="also write the predictions here")
    args = ap.parse_args()

    gt = json.loads(args.gt.read_text())
    enabled = C.ENABLED_CLASSES - set(filter(None, args.disable.split(",")))
    pred = {"team": "WEST", "videos": {}}
    for name in sorted(gt):
        obs = cached_observation(args.videos / name)
        pred["videos"][name] = {"events": interpret(obs, enabled).events, "risk": []}
    if args.out:
        args.out.write_text(json.dumps(pred, indent=1))
    evaluate.print_report(evaluate.evaluate(gt, pred, per_video=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
