#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

RUN_DIR="runs/detect/models/bottle_v1"

yolo detect train \
  model=yolov8n.pt \
  data=data/bottle/data.yaml \
  epochs=30 \
  imgsz=640 \
  batch=16 \
  device=mps \
  patience=8 \
  project=runs/detect/models \
  name=bottle_v1

yolo detect val \
  model="${RUN_DIR}/weights/best.pt" \
  data=data/bottle/data.yaml \
  conf=0.25

mkdir -p models/bottle_v1/weights
cp "${RUN_DIR}/weights/best.pt" models/bottle_v1/weights/best.pt
echo "Copied trained model to models/bottle_v1/weights/best.pt"
