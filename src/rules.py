"""
rules.py — Traffic incident rules engine and temporal segment post-processing.
"""
from __future__ import annotations
import numpy as np
from src.config import STOP_LINE, CROSSWALK_MAIN, CROSSWALK_CORNER, ROAD_SURFACE
from src.geometry import point_in_poly, line_crossing_direction

class EventEngine:
    def __init__(self):
        # Ongoing candidate events: {event_key: [start_t, last_seen_t, label, meta]}
        self.active_events: dict[str, dict] = {}
        self.completed_events: list[list] = [] # [start_sec, end_sec, label]
        
        # Track history of stopped states
        self.stopped_past_stopline: dict[int, float] = {} # tid -> start_t
        self.jaywalking_tracks: dict[int, float] = {}      # tid -> start_t

    def process_frame(self, detections: list[dict], tl_state: str, t_sec: float):
        """
        Evaluate rules on the current frame.
        """
        vehicles = [d for d in detections if d["category"] == "vehicle"]
        pedestrians = [d for d in detections if d["category"] == "pedestrian"]
        
        # Check pedestrians on crosswalk
        peds_on_crosswalk = [
            p for p in pedestrians
            if point_in_poly(p["centroid"], CROSSWALK_MAIN) or point_in_poly(p["centroid"], CROSSWALK_CORNER)
        ]
        
        # 1. Red Light & Stop Line Violations
        for v in vehicles:
            tid = v["id"]
            pt = v["centroid"]
            
            # Check crossing stop line
            if len(v.get("history", [])) >= 2:
                prev_pt = v["history"][-2][1]
                # Did vehicle move past stop line?
                if line_crossing_direction(prev_pt, pt, tuple(STOP_LINE[0]), tuple(STOP_LINE[1])):
                    if tl_state == "RED":
                        self.completed_events.append([round(max(0.0, t_sec - 0.5), 2), round(t_sec + 2.5, 2), "red_light"])
            
            # Check stop_line violation: stopped past the stop line on red
            # In region: Y between 1080 and 1350, X between 450 and 1500
            if tl_state == "RED" and 450 <= pt[0] <= 1500 and 1080 <= pt[1] <= 1350:
                if v["speed"] < 15.0: # stopped
                    if tid not in self.stopped_past_stopline:
                        self.stopped_past_stopline[tid] = t_sec
                else:
                    if tid in self.stopped_past_stopline:
                        start_t = self.stopped_past_stopline.pop(tid)
                        if t_sec - start_t >= 3.0:
                            self.completed_events.append([round(start_t, 2), round(t_sec, 2), "stop_line"])
            else:
                if tid in self.stopped_past_stopline:
                    start_t = self.stopped_past_stopline.pop(tid)
                    if t_sec - start_t >= 3.0:
                        self.completed_events.append([round(start_t, 2), round(t_sec, 2), "stop_line"])
                        
            # 2. Failure to Yield
            # Vehicle traverses crosswalk while pedestrian is on it
            if peds_on_crosswalk and point_in_poly(pt, CROSSWALK_MAIN):
                ev_key = f"yield_{tid}"
                if ev_key not in self.active_events:
                    self.active_events[ev_key] = {"start": t_sec, "last": t_sec, "label": "failure_to_yield"}
                else:
                    self.active_events[ev_key]["last"] = t_sec

            # 3. Stopped vehicle on carriageway (>= 10s when light is GREEN)
            if tl_state == "GREEN" and point_in_poly(pt, ROAD_SURFACE):
                if v["stopped_duration"] >= 10.0:
                    ev_key = f"stopped_{tid}"
                    if ev_key not in self.active_events:
                        self.active_events[ev_key] = {"start": t_sec - v["stopped_duration"], "last": t_sec, "label": "stopped_vehicle"}
                    else:
                        self.active_events[ev_key]["last"] = t_sec

        # 4. Jaywalking (only on actual asphalt carriageway outside crossings and islands)
        # Island zone (X: 2240..2850, Y: 950..1150)
        for p in pedestrians:
            tid = p["id"]
            pt = p["centroid"]
            
            # Check if on sidewalk or island
            on_island = (2200 <= pt[0] <= 2900 and 900 <= pt[1] <= 1180)
            on_sidewalk_left = (pt[0] < 450 and pt[1] < 1350)
            on_crosswalk = point_in_poly(pt, CROSSWALK_MAIN) or point_in_poly(pt, CROSSWALK_CORNER)
            on_road = point_in_poly(pt, ROAD_SURFACE)
            
            # Legitimate jaywalking: on actual driving lanes, not on crossing, not on island/sidewalk
            if on_road and not on_crosswalk and not on_island and not on_sidewalk_left:
                if tid not in self.jaywalking_tracks:
                    self.jaywalking_tracks[tid] = t_sec
                else:
                    dur = t_sec - self.jaywalking_tracks[tid]
                    if dur >= 3.0: # sustained presence on vehicle carriageway
                        ev_key = f"jay_{tid}"
                        if ev_key not in self.active_events:
                            self.active_events[ev_key] = {"start": self.jaywalking_tracks[tid], "last": t_sec, "label": "jaywalking"}
                        else:
                            self.active_events[ev_key]["last"] = t_sec
            else:
                self.jaywalking_tracks.pop(tid, None)

        # 5. Congestion (all lanes queuing)
        stopped_vehicles_count = sum(1 for v in vehicles if v["speed"] < 10.0 and point_in_poly(v["centroid"], ROAD_SURFACE))
        if stopped_vehicles_count >= 5:
            ev_key = "congestion"
            if ev_key not in self.active_events:
                self.active_events[ev_key] = {"start": t_sec, "last": t_sec, "label": "congestion"}
            else:
                self.active_events[ev_key]["last"] = t_sec

        # Flush inactive events
        to_del = []
        for k, ev in self.active_events.items():
            if t_sec - ev["last"] > 1.5: # 1.5 seconds without update
                if ev["last"] - ev["start"] >= 0.5: # minimum duration
                    self.completed_events.append([round(ev["start"], 2), round(ev["last"], 2), ev["label"]])
                to_del.append(k)
        for k in to_del:
            del self.active_events[k]

    def finalize(self, duration: float) -> list[list]:
        """Flush remaining events and return clean, merged list."""
        for k, ev in self.active_events.items():
            if ev["last"] - ev["start"] >= 0.5:
                self.completed_events.append([round(ev["start"], 2), round(min(duration, ev["last"]), 2), ev["label"]])
        self.active_events.clear()
        
        return post_process_events(self.completed_events, duration)


def post_process_events(events: list[list], duration: float) -> list[list]:
    """
    1. Filter out blips < 0.5s
    2. Clamp to [0, duration]
    3. Merge contiguous events of the same class (gap < 1.0s)
    4. Remove same-class overlaps
    """
    if not events:
        return []

    # Filter invalid and clamp
    valid = []
    for s, e, lbl in events:
        s = max(0.0, float(s))
        e = min(duration, float(e))
        if e - s >= 0.5:
            valid.append([s, e, str(lbl)])

    # Group by label
    by_class: dict[str, list[list]] = {}
    for s, e, lbl in valid:
        by_class.setdefault(lbl, []).append([s, e])

    merged_all = []
    for lbl, segs in by_class.items():
        segs.sort(key=lambda x: x[0])
        merged = []
        for s, e in segs:
            if not merged:
                merged.append([s, e])
            else:
                prev_s, prev_e = merged[-1]
                # Merge if overlapping or gap < 1.0s
                if s <= prev_e + 1.0:
                    merged[-1][1] = max(prev_e, e)
                else:
                    merged.append([s, e])
        for s, e in merged:
            merged_all.append([round(s, 2), round(e, 2), lbl])

    # Sort chronologically
    merged_all.sort(key=lambda x: (x[0], x[1]))
    return merged_all
