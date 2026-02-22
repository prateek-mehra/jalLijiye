# Bottle Dataset Quickstart

## 1) Collect videos

Put 3 short webcam videos into `data/raw_videos/`.

## 2) Extract frames

```bash
source .venv/bin/activate
python scripts/extract_frames.py --input-dir data/raw_videos --output-dir data/bottle/images/raw --sample-every-seconds 1.0
```

## 3) Annotate

Annotate images in `data/bottle/images/raw` as YOLO labels and place `.txt` files in `data/bottle/labels/raw`.

Class mapping:
- `0` -> `my_bottle`

## 4) Split train/val

```bash
source .venv/bin/activate
python scripts/split_dataset.py --images-raw data/bottle/images/raw --labels-raw data/bottle/labels/raw --out-root data/bottle --train-ratio 0.8
```

## 5) Train + validate

```bash
bash scripts/train_bottle_model.sh
```

After training, expected model path:
- `models/bottle_v1/weights/best.pt`
