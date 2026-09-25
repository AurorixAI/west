#!/usr/bin/env python3
"""Score the pipeline on public CCTV clips from other cameras (robustness check).

    python scripts/eval_public.py --videos data/public/TAD --gt data/public/tad_gt.json
    python scripts/eval_public.py --videos data/public/TAD --gt data/public/tad_gt.json --part-a

The ground truth uses evaluate.py's format ({<file name>: {"duration", "events"}}).
Part B runs every clip through the harness's own frame loop (run_submission.run_risk),
Part A (optional, slow on a CPU) through solution.detect_events. Results are
appended to --out after each clip, so an interrupted run resumes where it
stopped. The scores are evaluate.py's; alongside them the script prints the
highest risk in the 5 s before each accident and on accident-free clips, which
says more than the pooled numbers on a small set.

Set WEST_PROFILE=gpu WEST_NO_THIN=1 to score the configuration of the
evaluation GPU on a CPU (slowly).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluate  # noqa: E402
import run_submission  # noqa: E402
import solution  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default=".cache/public_pred.json")
    ap.add_argument("--part-a", action="store_true", help="also run detect_events (Part A)")
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text())
    out = Path(args.out)
    pred = json.loads(out.read_text()) if out.exists() else {"team": "WEST", "videos": {}}
    for name in sorted(gt):
        entry = pred["videos"].get(name)
        if entry is not None and (not args.part_a or "part_a" in entry):
            continue
        path = Path(args.videos) / name
        meta = run_submission.video_meta(path)
        t0 = time.perf_counter()
        curve, _ = run_submission.run_risk(solution.RiskEstimator(), path, meta, 1, float("inf"))
        entry = {"events": [], "risk": curve}
        if args.part_a:
            entry["events"], _ = run_submission.clean_events(solution.detect_events(str(path)),
                                                             solution.CLASSES, meta["duration"])
            entry["part_a"] = True
        pred["videos"][name] = entry
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(pred))
        print(f"{name}: {len(entry['events'])} events, max risk {max(s for _, s in curve):.2f}, "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    rep = evaluate.evaluate(gt, pred)
    b = rep["part_b"]
    pre, calm = [], []
    for name, g in gt.items():
        r = np.array(pred["videos"][name]["risk"])
        accidents = evaluate.segs(g["events"], "accident")
        if not accidents:
            calm.append(r[:, 1].max())
        for s, _ in accidents:
            m = (r[:, 0] >= s - evaluate.H) & (r[:, 0] < s)
            if m.any():
                pre.append(r[m, 1].max())
    print(f"\n{len(gt)} clips, {b['n_accidents'] if b else 0} accidents")
    if args.part_a:
        acc = rep["part_a"]["per_class"].get("accident", {})
        print("Part A accident F1 @0.3/0.5/0.7:", [round(acc.get(str(t), {}).get("f1", 0.0), 3)
                                                  for t in evaluate.TIOU_THRESHOLDS])
        fp = sum(len(v["events"]) for k, v in pred["videos"].items() if not gt[k]["events"])
        print("events on accident-free clips:", fp)
    if b:
        print(f"Part B: score {b['score_b']:.3f}  AP {b['ap']:.3f} (raw {b['ap_raw']:.3f}, chance "
              f"{b['positive_rate']:.3f})  F1_alarm {b['f1_alarm']:.3f}  mTTA {b['mtta_sec']:.2f}s  "
              f"alarms {b['n_alarms']} matched {b['n_matched']}")
    if pre:
        pre = np.array(pre)
        print(f"max risk in the 5 s before an accident: median {np.median(pre):.2f}, "
              f">=0.3 on {int((pre >= 0.3).sum())}/{len(pre)}, >=0.5 on {int((pre >= 0.5).sum())}/{len(pre)}")
    if calm:
        calm = np.array(calm)
        print(f"max risk on accident-free clips: median {np.median(calm):.2f}, "
              f">=0.5 on {int((calm >= 0.5).sum())}/{len(calm)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
