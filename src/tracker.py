"""
tracker.py — Multi-object detection and trajectory tracking using YOLOv8 + ByteTrack.
"""
from __future__ import annotations
import numpy as np
import torch
from ultralytics import YOLO

from pathlib import Path

class VideoTracker:
    def __init__(self, model_name: str = "yolov8n.pt", imgsz: int = 1280, conf: float = 0.25):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Offline weights resolution
        local_weights = Path(__file__).resolve().parent.parent / "weights" / "yolov8n.pt"
        target_model = str(local_weights) if local_weights.exists() else model_name
        self.model = YOLO(target_model)
        self.imgsz = imgsz
        self.conf = conf
        self.tracks: dict[int, dict] = {} # id -> track info
        
        # Vehicle class IDs in COCO: 2: car, 3: motorcycle, 5: bus, 7: truck
        # Person class ID: 0: person, 1: bicycle
        self.VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        self.PED_CLASSES = {0: "person", 1: "bicycle"}

    def step(self, frame: np.ndarray, t_sec: float) -> list[dict]:
        """
        Process a single frame. Updates all active tracks and returns current frame detections.
        """
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,
            imgsz=self.imgsz,
            conf=self.conf
        )
        
        current_detections = []
        res = results[0]
        
        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            track_ids = res.boxes.id.int().cpu().numpy()
            class_ids = res.boxes.cls.int().cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            
            for bbox, tid, cid, conf in zip(boxes, track_ids, class_ids, confs):
                cls_name = None
                category = None
                if cid in self.VEHICLE_CLASSES:
                    cls_name = self.VEHICLE_CLASSES[cid]
                    category = "vehicle"
                elif cid in self.PED_CLASSES:
                    cls_name = self.PED_CLASSES[cid]
                    category = "pedestrian"
                else:
                    continue
                    
                x1, y1, x2, y2 = bbox
                # Bottom-center point represents ground position
                centroid = ((x1 + x2) / 2.0, y2)
                
                # Update track history
                if tid not in self.tracks:
                    self.tracks[tid] = {
                        "id": tid,
                        "class_name": cls_name,
                        "category": category,
                        "first_seen": t_sec,
                        "last_seen": t_sec,
                        "history": [], # list of (t_sec, centroid, bbox, speed)
                        "stopped_since": None,
                        "stopped_duration": 0.0,
                    }
                
                track = self.tracks[tid]
                speed = 0.0
                if track["history"]:
                    prev_t, prev_pt, _, _ = track["history"][-1]
                    dt = t_sec - prev_t
                    if dt > 0:
                        dist = np.hypot(centroid[0] - prev_pt[0], centroid[1] - prev_pt[1])
                        speed = dist / dt
                        
                # Update stopped state (low speed threshold)
                if speed < 15.0: # pixels per second
                    if track["stopped_since"] is None:
                        track["stopped_since"] = t_sec
                    track["stopped_duration"] = t_sec - track["stopped_since"]
                else:
                    track["stopped_since"] = None
                    track["stopped_duration"] = 0.0
                    
                track["last_seen"] = t_sec
                track["history"].append((t_sec, centroid, bbox, speed))
                
                # Keep history reasonable size
                if len(track["history"]) > 100:
                    track["history"].pop(0)
                    
                current_detections.append({
                    "id": tid,
                    "class_name": cls_name,
                    "category": category,
                    "bbox": bbox,
                    "centroid": centroid,
                    "speed": speed,
                    "stopped_duration": track["stopped_duration"],
                    "t_sec": t_sec
                })
                
        return current_detections
