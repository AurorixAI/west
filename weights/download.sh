#!/usr/bin/env bash
# Fetch the detector weights (COCO-pretrained YOLOv8, Ultralytics, AGPL-3.0) and verify ours.
# They are also committed in weights/; this restores them and verifies checksums.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
BASE="https://github.com/ultralytics/assets/releases/download/v8.3.0"
fetch() {  # name sha256
  if [ ! -f "$1" ] || ! echo "$2  $1" | sha256sum -c --quiet - 2>/dev/null; then
    echo "downloading $1"
    curl -fL --retry 3 -o "$1" "$BASE/$1"
  fi
  echo "$2  $1" | sha256sum -c -
}
fetch yolov8s.pt 1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a
fetch yolov8n.pt f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36

# Our detector, YOLO11s fine-tuned on the sample videos: not downloadable, it is committed here.
# scripts/train_detector.py rebuilds it (extract -> label -> train).
echo "8ff17d6d2902fa192501c5cd1b682f869ed5f7424a63c082e7a89b49e2df1050  west_yolo11s.pt" | sha256sum -c -
