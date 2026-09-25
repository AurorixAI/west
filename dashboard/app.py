"""
app.py — Production FastAPI live streaming server and full demo platform for WIUT CV 2026.
"""
import os
import sys
import time
import json
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.traffic_light import TrafficLightDetector
from src.tracker import VideoTracker
from src.rules import EventEngine
from src.risk import CausalRiskEstimator
from dashboard.renderer import HUDAnnotator

app = FastAPI(title="WIUT CV Real-Time Vision AI — Team WEST")

VIDEO_DIR = Path("/Users/arslan/Desktop/WEST/test video")
TEMPLATES_DIR = CURRENT_DIR / "templates"
STATIC_DIR = CURRENT_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Shared telemetry state
latest_telemetry = {
    "signal": "GREEN",
    "risk": 0.05,
    "vehicles": 0,
    "pedestrians": 0,
    "active_events": [],
    "fps": 30.0,
    "current_time": 0.0
}

playback_speed = 1.0


@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = TEMPLATES_DIR / "index.html"
    return html_file.read_text()


@app.get("/api/videos")
async def get_videos():
    exts = {".mp4", ".MP4"}
    videos = []
    if VIDEO_DIR.exists():
        for p in sorted(VIDEO_DIR.iterdir()):
            if p.suffix in exts:
                size_gb = round(p.stat().st_size / (1024**3), 2)
                videos.append({"filename": p.name, "size_gb": size_gb})
    return JSONResponse(videos)


@app.get("/api/telemetry")
async def get_telemetry():
    return JSONResponse(latest_telemetry)


@app.get("/api/events")
async def get_events():
    pred_file = PROJECT_DIR / "predictions_samples.json"
    if not pred_file.exists():
        pred_file = PROJECT_DIR / "predictions_c3905_v3.json"
    if pred_file.exists():
        with open(pred_file, "r") as f:
            data = json.load(f)
        vkey = list(data["videos"].keys())[0]
        return JSONResponse(data["videos"][vkey].get("events", []))
    return JSONResponse([])


@app.get("/api/risk_profile")
async def get_risk_profile():
    pred_file = PROJECT_DIR / "predictions_samples.json"
    if not pred_file.exists():
        pred_file = PROJECT_DIR / "predictions_c3905_v3.json"
    if pred_file.exists():
        with open(pred_file, "r") as f:
            data = json.load(f)
        vkey = list(data["videos"].keys())[0]
        full_risk = data["videos"][vkey].get("risk", [])
        # Downsample for web chart rendering (every 3rd sample ~ 10 fps)
        downsampled = full_risk[::3]
        return JSONResponse(downsampled)
    return JSONResponse([])


@app.get("/api/speed")
async def set_speed(val: float = 1.0):
    global playback_speed
    playback_speed = max(0.2, min(5.0, val))
    return {"speed": playback_speed}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    save_path = VIDEO_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename, "size": save_path.stat().st_size}


def generate_frames(video_name: str, start_sec: float = 0.0):
    global latest_telemetry, playback_speed

    video_path = str(VIDEO_DIR / video_name)
    if not os.path.exists(video_path):
        video_path = str(VIDEO_DIR / "C3905.MP4")

    cap = cv2.VideoCapture(video_path)
    
    # Check if video opens properly
    if not cap.isOpened():
        target_w, target_h = 1280, 720
        msg_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        cv2.putText(msg_frame, f"VIDEO: {video_name}", (100, 260),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(msg_frame, "FILE DOWNLOAD INCOMPLETE (Missing moov atom)", (100, 340),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 165, 255), 2)
        cv2.putText(msg_frame, "Please complete downloading this file from Google Drive.", (100, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        cv2.putText(msg_frame, "Switch to C3905.MP4 or C3905_clip15s.mp4 above to view live AI.", (100, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 100), 2)
        
        ret, buf = cv2.imencode(".jpg", msg_frame)
        frame_bytes = buf.tobytes()
        while True:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(1.0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    frame_delay = 1.0 / fps

    start_frame = max(0, int(start_sec * fps))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    tl_detector = TrafficLightDetector(history_len=4)
    tracker = VideoTracker(model_name="yolov8n.pt", imgsz=640, conf=0.3)
    engine = EventEngine()
    risk_est = CausalRiskEstimator(stride=2)
    risk_est.reset({"fps": fps})
    annotator = HUDAnnotator()

    frame_idx = start_frame
    target_w, target_h = 1280, 720

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            continue

        t_sec = frame_idx / fps
        scaled_frame = cv2.resize(frame, (target_w, target_h))

        # 1. Traffic signal detection
        tl_state = tl_detector.detect_frame(frame)

        # 2. Tracking on scaled frame
        detections = tracker.step(scaled_frame, t_sec)

        # 3. Dynamic risk calculation
        risk_score = risk_est.step(scaled_frame, t_sec)

        # 4. Map scaled detections to native 3840x2160 for rules engine
        native_dets = []
        for d in detections:
            d_nat = dict(d)
            d_nat["centroid"] = (
                d["centroid"][0] * (3840.0 / target_w),
                d["centroid"][1] * (2160.0 / target_h)
            )
            native_dets.append(d_nat)

        engine.process_frame(native_dets, tl_state, t_sec)
        active_evs = [ev["label"] for ev in engine.active_events.values()]

        # Object counts
        veh_count = sum(1 for d in detections if d["category"] == "vehicle")
        ped_count = sum(1 for d in detections if d["category"] == "pedestrian")

        # Telemetry updates
        latest_telemetry["signal"] = tl_state
        latest_telemetry["risk"] = round(float(risk_score), 3)
        latest_telemetry["vehicles"] = veh_count
        latest_telemetry["pedestrians"] = ped_count
        latest_telemetry["active_events"] = active_evs
        latest_telemetry["current_time"] = round(t_sec, 1)

        # 5. Overlays
        scaled_frame = annotator.draw_zones(scaled_frame)
        scaled_frame = annotator.draw_detections(scaled_frame, detections, active_evs)
        scaled_frame = annotator.draw_telemetry(
            scaled_frame, t_sec, tl_state, risk_score, active_evs, veh_count, ped_count
        )

        ret, buffer = cv2.imencode(".jpg", scaled_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        frame_idx += 1

        dt = time.time() - t0
        target_delay = (frame_delay / playback_speed)
        if target_delay > dt:
            time.sleep(target_delay - dt)

    cap.release()


@app.get("/stream")
def video_stream(video: str = "C3905.MP4", start_sec: float = 0.0):
    return StreamingResponse(
        generate_frames(video, start_sec=start_sec),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
