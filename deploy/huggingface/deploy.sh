#!/usr/bin/env bash
# Push the website + demo to a Hugging Face Space.
#   1. create a Space at https://huggingface.co/new-space (SDK: Docker, hardware: CPU basic)
#   2. bash deploy/huggingface/deploy.sh https://huggingface.co/spaces/<user>/<space>
# Needs git-lfs (the weights and annotated sample videos are binary) and an HF token
# with write access (git will ask for it; use the token as the password).
set -euo pipefail
SPACE_URL="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(mktemp -d)"
git clone "$SPACE_URL" "$WORK/space"
cd "$WORK/space"
git lfs install --local
git lfs track "*.pt" "*.mp4" "*.jpg"
rsync -a --delete --exclude .git --exclude .cache --exclude samples --exclude '__pycache__' \
      --exclude '.gitattributes' "$ROOT/" ./
cp deploy/huggingface/Dockerfile ./Dockerfile
cat > README.md <<'MD'
---
title: WEST Traffic Events
emoji: 🚦
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---
Team WEST: traffic event detection and accident anticipation. Source: https://github.com/AurorixAI/west
MD
git add -A
git commit -m "Deploy $(cd "$ROOT" && git rev-parse --short HEAD)"
git push
echo "deployed: $SPACE_URL"
