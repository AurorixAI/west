#!/usr/bin/env bash
# Download model weights before offline evaluation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}"

echo "Checking model weights in ${SCRIPT_DIR}..."

if [ ! -f "${SCRIPT_DIR}/yolov8n.pt" ]; then
    echo "Downloading yolov8n.pt..."
    curl -L -o "${SCRIPT_DIR}/yolov8n.pt" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
fi

echo "All weights ready in ${SCRIPT_DIR}."
