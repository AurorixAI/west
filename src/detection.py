"""YOLO object detector (COCO-pretrained, open weights) with batched inference."""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src import config as C


def seed_everything(seed: int = C.SEED) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=4)
def load_model(weights: str, device: str):
    from ultralytics import YOLO
    path = C.WEIGHTS_DIR / weights
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `bash weights/download.sh` once")
    model = YOLO(str(path))
    model.to(device)
    return model


@dataclass
class Detections:
    xyxy: np.ndarray    # (N, 4) float32, full-resolution pixels
    conf: np.ndarray    # (N,)
    cls: np.ndarray     # (N,) COCO class ids

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32))


class Detector:
    KEEP = tuple(sorted(C.CATEGORY_OF_CLASS))

    def __init__(self, weights: str = C.DETECTOR_WEIGHTS, imgsz: int = C.DETECT_WIDTH,
                 conf: float = C.DETECT_CONF, device: str | None = None):
        seed_everything()
        self.device = device or pick_device()
        self.model = load_model(weights, self.device)
        self.imgsz = imgsz
        self.conf = conf
        self.half = self.device.startswith("cuda")

    def __call__(self, images: list[np.ndarray], scales: list[float]) -> list[Detections]:
        if not images:
            return []
        results = self.model.predict(images, imgsz=self.imgsz, conf=self.conf, classes=list(self.KEEP),
                                     half=self.half, device=self.device, verbose=False)
        out = []
        for res, s in zip(results, scales):
            b = res.boxes
            if b is None or len(b) == 0:
                out.append(Detections.empty())
                continue
            out.append(Detections(b.xyxy.cpu().numpy().astype(np.float32) * s,
                                  b.conf.cpu().numpy().astype(np.float32),
                                  b.cls.cpu().numpy().astype(np.int32)))
        return out


def warm_up(settings: list[tuple[str, int]]) -> None:
    """Load each (weights, imgsz) detector and run it once on a blank frame.

    The first prediction builds the predictor (layer fusion, CUDA context,
    kernels) and takes seconds. Done here, at import, it is not charged to
    the first video's time budget.
    """
    blank = np.full((720, 1280, 3), 114, np.uint8)
    for weights, imgsz in settings:
        Detector(weights=weights, imgsz=imgsz)([blank], [1.0])

