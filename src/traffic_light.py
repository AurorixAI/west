"""
traffic_light.py — Robust traffic signal detector using HSV glowing-bulb segmentation.
"""
import cv2
import numpy as np

class TrafficLightDetector:
    def __init__(self, history_len: int = 5):
        self.history = []
        self.history_len = history_len
        self.last_state = "GREEN"

    def detect_frame(self, frame: np.ndarray) -> str:
        """
        Detects current traffic light state from frame (supports any resolution).
        """
        H, W = frame.shape[:2]
        sx = W / 3840.0
        sy = H / 2160.0

        ymin, ymax = int(680 * sy), int(880 * sy)
        xmin, xmax = int(2260 * sx), int(2380 * sx)

        roi = frame[ymin:ymax, xmin:xmax]
        if roi.size == 0:
            return self.last_state

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Red hue wraps around 0 and 180
        mask_r1 = cv2.inRange(hsv, np.array([0, 80, 90]), np.array([12, 255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([168, 80, 90]), np.array([180, 255, 255]))
        mask_red = mask_r1 | mask_r2

        # Green hue
        mask_green = cv2.inRange(hsv, np.array([40, 70, 90]), np.array([95, 255, 255]))

        red_pixels = int(cv2.countNonZero(mask_red))
        green_pixels = int(cv2.countNonZero(mask_green))

        # Minimum pixel threshold adjusted for scale
        thresh = max(10, int(25 * (W / 3840.0) * (H / 2160.0)))

        instant_state = "UNKNOWN"
        if red_pixels > thresh and red_pixels > green_pixels:
            instant_state = "RED"
        elif green_pixels > thresh and green_pixels > red_pixels:
            instant_state = "GREEN"

        if instant_state != "UNKNOWN":
            self.history.append(instant_state)
            if len(self.history) > self.history_len:
                self.history.pop(0)

            counts = {"RED": self.history.count("RED"), "GREEN": self.history.count("GREEN")}
            if counts["RED"] > counts["GREEN"]:
                self.last_state = "RED"
            elif counts["GREEN"] > counts["RED"]:
                self.last_state = "GREEN"

        return self.last_state
