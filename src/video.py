"""Video metadata and a prefetching, strided frame reader."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    # Same fallback as the organisers' harness, so our timestamps match theirs.
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    info = VideoInfo(path, float(fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


@dataclass
class Frame:
    index: int
    t: float
    image: np.ndarray        # resized to the analysis width
    scale: float             # full-resolution pixels per analysis pixel
    extra: Any = None        # whatever the full-resolution hook returned


def stride_for(fps: float, target_fps: float) -> int:
    return max(1, int(round(fps / target_fps)))


def read_frames(info: VideoInfo, stride: int, width: int,
                full_res_hook: Callable[[np.ndarray], Any] | None = None,
                prefetch: int = 16) -> Iterator[Frame]:
    """Yield every ``stride``-th frame, resized to ``width``.

    Decoding runs in a background thread (OpenCV releases the GIL) so it
    overlaps with inference. Skipped frames are ``grab()``-ed, which avoids the
    colour conversion.
    """
    q: queue.Queue = queue.Queue(maxsize=prefetch)
    stop = threading.Event()

    def worker() -> None:
        cap = cv2.VideoCapture(info.path)
        idx = 0
        try:
            while not stop.is_set():
                if idx % stride:
                    if not cap.grab():
                        break
                    idx += 1
                    continue
                ok, full = cap.read()
                if not ok:
                    break
                extra = full_res_hook(full) if full_res_hook else None
                h, w = full.shape[:2]
                scale = w / float(width)
                small = full if w == width else cv2.resize(
                    full, (width, int(round(h / scale))), interpolation=cv2.INTER_AREA)
                q.put(Frame(idx, idx / info.fps, small, scale, extra))
                idx += 1
        finally:
            cap.release()
            q.put(None)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    try:
        while True:
            item = q.get()
            if item is None:
                break
            yield item
    finally:
        stop.set()
        while th.is_alive():          # drain so the worker can exit
            try:
                q.get(timeout=0.1)
            except queue.Empty:
                pass
        th.join()
