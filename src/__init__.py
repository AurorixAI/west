"""Team WEST traffic event detection package."""
import os

# Ultralytics probes the internet at import and sends usage analytics when it
# thinks it is online. Evaluation is offline: never let it try.
os.environ.setdefault("YOLO_OFFLINE", "True")
