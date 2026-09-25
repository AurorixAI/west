"""
renderer.py — High-precision visualizer with clean aesthetics, zero clutter, and accurate zones.
"""
import cv2
import numpy as np
from src.config import STOP_LINE, CROSSWALK_MAIN, CROSSWALK_CORNER

class HUDAnnotator:
    def __init__(self):
        # Color definitions (BGR)
        self.C_VEHICLE = (255, 200, 0)      # Ice Cyan / Blue
        self.C_PED = (80, 235, 100)          # Vibrant Emerald Green
        self.C_VIOLATION = (40, 40, 255)     # Neon Red
        self.C_ZONE_GREEN = (40, 220, 90)    # Crosswalk green
        self.C_STOPLINE = (0, 0, 255)        # Stopline red

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw transparent zones for stop line and crosswalks with clean borders."""
        overlay = frame.copy()
        H, W = frame.shape[:2]
        sx, sy = W / 3840.0, H / 2160.0

        cw_main = (CROSSWALK_MAIN * [sx, sy]).astype(np.int32)
        cw_corner = (CROSSWALK_CORNER * [sx, sy]).astype(np.int32)
        st_line = (STOP_LINE * [sx, sy]).astype(np.int32)

        # Crosswalk polygons (clean emerald tint)
        cv2.fillPoly(overlay, [cw_main], self.C_ZONE_GREEN)
        cv2.fillPoly(overlay, [cw_corner], self.C_ZONE_GREEN)

        # Smooth blend
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

        # Crisp polygon borders
        cv2.polylines(frame, [cw_main], True, (60, 240, 110), 1)
        cv2.polylines(frame, [cw_corner], True, (60, 240, 110), 1)

        # Stop line (bright red bar)
        cv2.polylines(frame, [st_line], False, self.C_STOPLINE, max(2, int(4 * sy)))
        return frame

    def draw_detections(self, frame: np.ndarray, detections: list[dict], active_events: list[str]) -> np.ndarray:
        """
        Draw sleek, non-cluttering bounding boxes.
        Filters out tiny background objects that cause overlapping mess.
        """
        H, W = frame.shape[:2]

        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            box_h = y2 - y1
            box_w = x2 - x1

            # Filter distant clutter in far background (Y < 220 or tiny boxes)
            if y1 < int(180 * (H / 720.0)) or box_h < 18 or box_w < 18:
                continue

            tid = d["id"]
            cname = d["class_name"]
            cat = d["category"]

            is_ped = (cat == "pedestrian")
            color = self.C_PED if is_ped else self.C_VEHICLE

            # Highlight violators in neon red
            if is_ped and any("jaywalking" in ev.lower() for ev in active_events):
                color = self.C_VIOLATION
            elif not is_ped and any("yield" in ev.lower() or "red" in ev.lower() for ev in active_events):
                if d.get("speed", 0) > 10:
                    color = self.C_VIOLATION

            # Thin crisp bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            # Minimal pill label (only for significant objects to avoid clutter)
            if box_h > 35 or is_ped:
                label = f"#{tid} {cname}"
                font_scale = 0.4
                thickness = 1
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Small rounded pill
                pill_y1 = max(0, y1 - th - 6)
                pill_y2 = y1
                cv2.rectangle(frame, (x1, pill_y1), (x1 + tw + 6, pill_y2), (15, 20, 30), -1)
                cv2.rectangle(frame, (x1, pill_y1), (x1 + tw + 6, pill_y2), color, 1)
                cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        return frame

    def draw_telemetry(self, frame: np.ndarray, t_sec: float, tl_state: str,
                       risk_score: float, active_events: list[str],
                       vehicle_count: int, ped_count: int) -> np.ndarray:
        """Draw streamlined HUD telemetry bar."""
        H, W = frame.shape[:2]

        # Top dark bar
        bar_h = int(45 * (H / 720.0))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, bar_h), (8, 12, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Time
        time_str = f"TIME: {int(t_sec)//60:02d}:{int(t_sec)%60:02d}"
        cv2.putText(frame, time_str, (16, int(bar_h * 0.68)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65 * (H / 720.0), (240, 240, 240), 1)

        # Traffic light badge
        tl_x = int(W * 0.40)
        tl_color = (0, 0, 255) if tl_state == "RED" else ((0, 230, 80) if tl_state == "GREEN" else (160, 160, 160))
        cv2.circle(frame, (tl_x, int(bar_h / 2)), 8, tl_color, -1)
        cv2.putText(frame, f"SIGNAL: {tl_state}", (tl_x + 16, int(bar_h * 0.68)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6 * (H / 720.0), tl_color, 1)

        # Counters
        stats_str = f"VEHICLES: {vehicle_count}  |  PEDESTRIANS: {ped_count}"
        cv2.putText(frame, stats_str, (int(W * 0.60), int(bar_h * 0.68)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55 * (H / 720.0), (180, 210, 240), 1)

        # Active incident banner
        if active_events:
            alert_text = "  |  ".join(f"ALERT: {ev.upper().replace('_', ' ')}" for ev in active_events)
            banner_h = int(38 * (H / 720.0))
            banner_y = H - banner_h - 16

            overlay_b = frame.copy()
            cv2.rectangle(overlay_b, (20, banner_y), (W - 20, banner_y + banner_h), (10, 15, 160), -1)
            cv2.addWeighted(overlay_b, 0.88, frame, 0.12, 0, frame)
            cv2.rectangle(frame, (20, banner_y), (W - 20, banner_y + banner_h), (0, 0, 255), 1)

            cv2.putText(frame, alert_text, (36, banner_y + int(banner_h * 0.68)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6 * (H / 720.0), (255, 255, 255), 1)

        return frame
