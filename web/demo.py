"""Live-demo jobs: one uploaded video -> events, risk curve, playback overlay, thumbnails."""
from __future__ import annotations

import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from src import viz
from src.pipeline import CPU, default_profile, interpret, observe
from src.risk import CPU_SETTINGS, CausalRiskEstimator
from src.video import probe, stride_for

JOBS_DIR = Path("/tmp/west_jobs")
MAX_MB = 500
MAX_SEC = 150.0
KEEP_SEC = 2 * 3600
CATEGORY_CODE = {"vehicle": 0, "person": 1, "animal": 2}


class Jobs:
    """In-memory job table; one worker so concurrent uploads queue instead of thrashing the CPU."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        JOBS_DIR.mkdir(parents=True, exist_ok=True)

    def submit(self, upload_path: Path, name: str) -> str:
        self._cleanup()
        info = probe(str(upload_path))           # raises on a file OpenCV cannot read
        if info.duration > MAX_SEC:
            raise ValueError(f"video is {info.duration:.0f} s long; the demo accepts up to {MAX_SEC:.0f} s")
        jid = uuid.uuid4().hex[:12]
        d = JOBS_DIR / jid
        d.mkdir()
        path = d / "input.mp4"
        shutil.move(str(upload_path), path)
        with self.lock:
            self.jobs[jid] = {"id": jid, "name": name, "state": "queued", "progress": 0.0,
                              "stage": "waiting for a free worker", "created": time.time()}
        self.pool.submit(self._run, jid, path)
        return jid

    def get(self, jid: str) -> dict | None:
        with self.lock:
            j = self.jobs.get(jid)
            return dict(j) if j else None

    def _set(self, jid: str, **kw) -> None:
        with self.lock:
            self.jobs[jid].update(kw)

    def _run(self, jid: str, path: Path) -> None:
        try:
            self._set(jid, state="running", stage="detecting and tracking")
            result = analyse_upload(path, lambda p, stage: self._set(jid, progress=round(p, 3), stage=stage))
            self._set(jid, state="done", progress=1.0, stage="done", result=result)
        except Exception as exc:                       # reported to the page, never crashes the server
            traceback.print_exc()
            self._set(jid, state="error", error=f"{type(exc).__name__}: {exc}")

    def _cleanup(self) -> None:
        now = time.time()
        with self.lock:
            old = [j for j, v in self.jobs.items() if now - v["created"] > KEEP_SEC]
            for j in old:
                del self.jobs[j]
        for j in old:
            shutil.rmtree(JOBS_DIR / j, ignore_errors=True)


def analyse_upload(path: Path, report) -> dict:
    """Part A and a causal Part B curve in a single decode pass.

    Part B's estimator still only sees frames in order, as in the harness; it
    runs on the analysed frames instead of every frame to halve the decoding.
    """
    info = probe(str(path))
    profile = default_profile()
    eff_fps = info.fps / stride_for(info.fps, profile.analysis_fps)
    est = CausalRiskEstimator(CPU_SETTINGS if profile is CPU else None)
    est.reset({"video_id": path.name, "fps": eff_fps, "width": info.width, "height": info.height,
               "n_frames": info.n_frames})
    risk: list[list[float]] = []

    def on_frame(frame, _tracked):
        risk.append([round(frame.t, 2), round(est.step(frame.image, frame.t), 3)])

    obs = observe(str(path), profile, progress=lambda p: report(0.92 * p, "detecting and tracking"),
                  on_frame=on_frame)
    report(0.93, "applying the rules")
    an = interpret(obs)
    report(0.95, "rendering event thumbnails")
    thumbs = _thumbnails(path, an)
    return {
        "video": {"width": info.width, "height": info.height, "fps": round(info.fps, 3),
                  "duration": round(info.duration, 2), "frames": info.n_frames},
        "events": [{"start": s, "end": e, "label": lbl, "actors": list(a), "thumb": thumbs.get(k)}
                   for k, (s, e, lbl, a) in enumerate(an.detailed)],
        "official": an.events,
        "risk": risk,
        "signal": [[round(a, 2), round(b, 2), st] for a, b, st in obs.signal.phases()],
        "overlay": _overlay(obs.tracks, an.geom, info.width, info.height),
        "camera": {"known": obs.camera_known, "note": obs.registration},
        "seconds": round(obs.seconds, 1),
        "profile": {"weights": profile.weights, "analysis_fps": profile.analysis_fps},
    }


WARNING_CODE = {"PEDESTRIAN AHEAD": 1, "CLOSE PASS": 2}


def _overlay(tracks, geom, w: int, h: int) -> dict:
    """Per analysed frame: [x1, y1, x2, y2 (0..10000), track id, category code, warning code]."""
    index = viz.TrackIndex(tracks)
    frames = []
    for key in index.keys:
        samples = index.by_time[int(key)]
        warn = viz.crossing_conflicts(samples, geom)
        frames.append([[int(tr.box[i, 0] / w * 1e4), int(tr.box[i, 1] / h * 1e4), int(tr.box[i, 2] / w * 1e4),
                        int(tr.box[i, 3] / h * 1e4), tr.tid, CATEGORY_CODE[tr.category],
                        WARNING_CODE.get(warn.get(tr.tid), 0)]
                       for tr, i in samples])
    return {"t": [round(k / 1000, 3) for k in index.keys.tolist()], "boxes": frames}


def _thumbnails(path: Path, an) -> dict[int, str]:
    """JPEG of each event's middle frame, actors highlighted, as a data URL."""
    import base64
    out, index = {}, viz.TrackIndex(an.tracks)
    cap = cv2.VideoCapture(str(path))
    for k, (s, e, lbl, actors) in enumerate(an.detailed[:60]):
        t = (s + e) / 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * an.info.fps))
        ok, full = cap.read()
        if not ok:
            continue
        scale = full.shape[1] / 960
        img = cv2.resize(full, (960, int(full.shape[0] / scale)), interpolation=cv2.INTER_AREA)
        viz.draw_geometry(img, an.geom, scale)
        viz.draw_signals(img, full, an.geom, scale)
        viz.draw_objects(img, index.at(t), scale, {a: lbl for a in actors})
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            out[k] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    cap.release()
    return out
