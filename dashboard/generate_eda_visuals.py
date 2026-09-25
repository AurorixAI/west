"""
Script to generate publication-grade EDA charts and heatmaps from C3905.MP4
for the WIUT Hackathon 2026 Team WEST Platform.
"""
import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Output directory
OUT_DIR = "/Users/arslan/Desktop/WEST/wiut_cv_scripts/dashboard/static/eda"
os.makedirs(OUT_DIR, exist_ok=True)

# Set style
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

VIDEO_PATH = "/Users/arslan/Desktop/WEST/test video/C3905.MP4"
PRED_PATH = "/Users/arslan/Desktop/WEST/wiut_cv_scripts/predictions_c3905_v3.json"

print("Generating EDA Visualizations...")

# 1. Load predictions
with open(PRED_PATH, 'r') as f:
    pred_data = json.load(f)

video_key = list(pred_data["videos"].keys())[0]
events = pred_data["videos"][video_key]["events"]
risk_series = pred_data["videos"][video_key]["risk"]

# -------------------------------------------------------------
# Chart 1: Part B Accident Risk Anticipation Curve
# -------------------------------------------------------------
print("1. Plotting Risk Anticipation Curve...")
risk_t = [pt[0] for pt in risk_series]
risk_v = [pt[1] for pt in risk_series]

fig, ax = plt.subplots(figsize=(12, 4.5), dpi=200)
fig.patch.set_facecolor('#0b0f19')
ax.set_facecolor('#111827')

# Plot risk curve
ax.plot(risk_t, risk_v, color='#38bdf8', lw=1.8, label='P(accident within 5s)')
ax.fill_between(risk_t, 0, risk_v, color='#38bdf8', alpha=0.15)

# Highlight high-risk threshold
ax.axhline(0.5, color='#ef4444', linestyle='--', alpha=0.7, label='Critical Alarm Threshold (0.50)')
ax.axhline(0.2, color='#f59e0b', linestyle=':', alpha=0.6, label='Elevated Risk (0.20)')

# Mark event intervals as vertical shaded spans
event_colors = {
    'failure_to_yield': '#f43f5e',
    'jaywalking': '#fb923c',
    'stopped_vehicle': '#a855f7',
    'congestion': '#38bdf8'
}
seen_labels = set()
for ev in events:
    start_t, end_t, label = ev
    c = event_colors.get(label, '#64748b')
    lbl = f"Event: {label}" if label not in seen_labels else None
    seen_labels.add(label)
    ax.axvspan(start_t, end_t, color=c, alpha=0.18, label=lbl)

ax.set_title("Part B: Continuous Accident Anticipation Risk Profile (C3905.MP4)", fontsize=13, fontweight='bold', pad=12, color='#f8fafc')
ax.set_xlabel("Video Timeline (Seconds)", fontsize=10, color='#94a3b8')
ax.set_ylabel("Causal Risk Score ∈ [0, 1]", fontsize=10, color='#94a3b8')
ax.set_xlim(0, max(risk_t))
ax.set_ylim(-0.02, 1.0)
ax.grid(True, linestyle='--', alpha=0.15, color='#ffffff')
ax.legend(loc='upper right', framealpha=0.8, facecolor='#1e293b', edgecolor='#334155', fontsize=8)

plt.tight_layout()
risk_out = os.path.join(OUT_DIR, "eda_risk_profile.png")
plt.savefig(risk_out, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print("Saved:", risk_out)

# -------------------------------------------------------------
# Chart 2: Object Counts & Traffic Density Over Time
# -------------------------------------------------------------
print("2. Sampling frame detections for object counts over time...")
# Let's sample every 1 sec across C3905.MP4
from ultralytics import YOLO
model = YOLO("/Users/arslan/Desktop/WEST/wiut_cv_scripts/weights/yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

sample_times = []
vehicle_counts = []
ped_counts = []
heatmap_accum = np.zeros((1080, 1920), dtype=np.float32)

step_frames = int(fps * 2.0) # every 2 seconds for high resolution
frame_idx = 0

bg_frame = None

while cap.isOpened() and frame_idx < total_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break
    
    t_sec = frame_idx / fps
    sample_times.append(t_sec)
    
    # Store background reference frame at 10s
    if bg_frame is None and t_sec >= 10.0:
        bg_frame = cv2.resize(frame, (1920, 1080))

    # Inference (downscaled for fast EDA)
    small_frame = cv2.resize(frame, (1280, 720))
    res = model.predict(small_frame, imgsz=640, verbose=False, conf=0.25, device='mps' if cv2.ocl.haveOpenCL() else 'cpu')[0]
    
    v_cnt = 0
    p_cnt = 0
    
    if res.boxes is not None and len(res.boxes) > 0:
        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        
        # Scale back to 1920x1080
        sx = 1920.0 / 1280.0
        sy = 1080.0 / 720.0
        
        for box, cls_id in zip(boxes, classes):
            x1, y1, x2, y2 = box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy
            cx, cy = int((x1 + x2) / 2), int(y2)
            
            if cls_id in [2, 3, 5, 7]: # car, motorcycle, bus, truck
                v_cnt += 1
                if 0 <= cx < 1920 and 0 <= cy < 1080:
                    cv2.circle(heatmap_accum, (cx, cy), 18, 1.0, -1)
            elif cls_id == 0: # person
                p_cnt += 1
                if 0 <= cx < 1920 and 0 <= cy < 1080:
                    cv2.circle(heatmap_accum, (cx, cy), 10, 0.6, -1)
                    
    vehicle_counts.append(v_cnt)
    ped_counts.append(p_cnt)
    frame_idx += step_frames

cap.release()

# Plot Object Counts
fig, ax = plt.subplots(figsize=(12, 4.5), dpi=200)
fig.patch.set_facecolor('#0b0f19')
ax.set_facecolor('#111827')

ax.plot(sample_times, vehicle_counts, color='#3b82f6', lw=2.2, label='Active Vehicles (Car/Bus/Truck)', marker='o', markersize=3)
ax.plot(sample_times, ped_counts, color='#10b981', lw=2.0, label='Pedestrians on Intersection', marker='s', markersize=3)

# Mark signal transitions
ax.axvspan(0, 48, color='#ef4444', alpha=0.08, label='Signal Phase: RED (Queue Accumulation)')
ax.axvspan(48, 105, color='#10b981', alpha=0.08, label='Signal Phase: GREEN (Discharge Flow)')
ax.axvspan(105, 127, color='#ef4444', alpha=0.08)

ax.set_title("EDA: Dynamic Traffic Density & Pedestrian Volume vs. Signal Phases", fontsize=13, fontweight='bold', pad=12, color='#f8fafc')
ax.set_xlabel("Video Timeline (Seconds)", fontsize=10, color='#94a3b8')
ax.set_ylabel("Count of Tracked Objects", fontsize=10, color='#94a3b8')
ax.set_xlim(0, max(sample_times))
ax.grid(True, linestyle='--', alpha=0.15, color='#ffffff')
ax.legend(loc='upper right', framealpha=0.85, facecolor='#1e293b', edgecolor='#334155', fontsize=8.5)

plt.tight_layout()
density_out = os.path.join(OUT_DIR, "eda_object_counts.png")
plt.savefig(density_out, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print("Saved:", density_out)

# -------------------------------------------------------------
# Chart 3: Spatial Heatmap Overlaid on Intersection
# -------------------------------------------------------------
print("3. Generating Spatial Motion Heatmap Overlay...")
if bg_frame is not None:
    # Normalize heatmap
    norm_heat = cv2.GaussianBlur(heatmap_accum, (41, 41), 0)
    if norm_heat.max() > 0:
        norm_heat = norm_heat / norm_heat.max()
    
    heat_color = cv2.applyColorMap((norm_heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_overlay = cv2.addWeighted(bg_frame, 0.6, heat_color, 0.4, 0)
    
    # Save image
    heat_out = os.path.join(OUT_DIR, "eda_motion_heatmap.png")
    cv2.imwrite(heat_out, heat_overlay)
    print("Saved:", heat_out)

# -------------------------------------------------------------
# Chart 4: Event Class Breakdown & Duration Analysis
# -------------------------------------------------------------
print("4. Plotting Event Distribution & Durations...")
from collections import defaultdict
class_durations = defaultdict(list)
for ev in events:
    s, e, lbl = ev
    class_durations[lbl].append(e - s)

labels = list(class_durations.keys())
counts = [len(class_durations[k]) for k in labels]
total_dur = [sum(class_durations[k]) for k in labels]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=200)
fig.patch.set_facecolor('#0b0f19')
ax1.set_facecolor('#111827')
ax2.set_facecolor('#111827')

colors = ['#f43f5e', '#fb923c', '#38bdf8', '#a855f7']

# Bar 1: Event Frequency
bars1 = ax1.bar(labels, counts, color=colors[:len(labels)], width=0.55, edgecolor='#ffffff', linewidth=0.5)
ax1.set_title("Incident Frequency by Class", fontsize=11, fontweight='bold', color='#f8fafc', pad=10)
ax1.set_ylabel("Count of Detected Segments", fontsize=9, color='#94a3b8')
ax1.grid(axis='y', linestyle='--', alpha=0.15)
for bar in bars1:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, y + 0.1, f"{int(y)}", ha='center', va='bottom', fontsize=9, color='#f8fafc')

# Bar 2: Cumulative Duration
bars2 = ax2.bar(labels, total_dur, color=colors[:len(labels)], width=0.55, edgecolor='#ffffff', linewidth=0.5)
ax2.set_title("Total Incident Duration (Seconds)", fontsize=11, fontweight='bold', color='#f8fafc', pad=10)
ax2.set_ylabel("Cumulative Seconds", fontsize=9, color='#94a3b8')
ax2.grid(axis='y', linestyle='--', alpha=0.15)
for bar in bars2:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, y + 1.0, f"{y:.1f}s", ha='center', va='bottom', fontsize=9, color='#f8fafc')

plt.tight_layout()
dist_out = os.path.join(OUT_DIR, "eda_event_breakdown.png")
plt.savefig(dist_out, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print("Saved:", dist_out)

print("EDA Generation Complete!")
