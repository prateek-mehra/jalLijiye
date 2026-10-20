#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/data/bottle_checkered/data.yaml"
PROJECT="${ROOT}/runs/detect/checkered_bottle_comparison"

for model_name in yolov8n yolo26n; do
  model_weights="${model_name}.pt"
  run_name="${model_name}_batch4"
  run_dir="${PROJECT}/${run_name}"

  echo "Training ${model_weights} on MPS..."
  yolo detect train \
    model="${model_weights}" \
    data="${DATA}" \
    epochs=30 \
    imgsz=640 \
    batch=4 \
    device=mps \
    workers=0 \
    patience=8 \
    seed=42 \
    optimizer=AdamW \
    lr0=0.001 \
    lrf=0.01 \
    weight_decay=0.0005 \
    project="${PROJECT}" \
    name="${run_name}"

  echo "Evaluating ${model_name} on the held-out test split..."
  yolo detect val \
    model="${run_dir}/weights/best.pt" \
    data="${DATA}" \
    split=test \
    imgsz=640 \
    batch=4 \
    device=mps \
    conf=0.25 \
    plots=True \
    project="${PROJECT}" \
    name="${run_name}_test"

  mkdir -p "models/checkered_${run_name}/weights"
  cp "${run_dir}/weights/best.pt" "models/checkered_${run_name}/weights/best.pt"
done
