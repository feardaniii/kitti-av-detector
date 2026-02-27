# AV Detector Pipeline (KITTI + TensorFlow)

Production-style, modular computer vision pipeline for:

1. Training an object detector on KITTI (`Car`, `Pedestrian`, `Cyclist`)
2. Running MP4 video inference
3. Writing annotated MP4 output with bounding boxes and labels

## Project structure

```text
project_root/
  config/
    config.py
  data/
    kitti_loader.py
    generator.py
  models/
    detector.py
  training/
    train.py
  inference/
    video_inference.py
    utils.py
  scripts/
    run_training.py
    run_inference.py
  outputs/
  requirements.txt
  README.md
```

## Dataset layout

Expected KITTI paths:

```text
datasets/kitti/
  training/image_2/*.png
  training/label_2/*.txt
```

## Setup

Python requirement:

- Use Python `3.10` to `3.12` (TensorFlow is not available for Python `3.14` yet).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
python scripts/run_training.py
```

If KITTI is stored outside this repository, pass its root path:

```bash
python scripts/run_training.py --dataset-root "D:\path\to\kitti"
```

The path must contain:

- `training/image_2/*.png`
- `training/label_2/*.txt`

Alternative: set `KITTI_DATASET_ROOT` and run training without the flag.

Speed-up options for long CPU training runs:

```bash
# Quick debug run (recommended first pass on CPU)
python scripts/run_training.py --dataset-root "D:\path\to\kitti" --quick

# Custom fast run
python scripts/run_training.py --dataset-root "D:\path\to\kitti" --stage1-epochs 4 --stage2-epochs 0 --skip-fine-tune --input-size 160 --max-samples 10000 --workers 6
```

Useful training flags:

- `--stage1-epochs`, `--stage2-epochs`
- `--skip-fine-tune`
- `--max-samples`
- `--input-size`
- `--workers`
- `--batch-size`

Training includes:

- Deterministic 80/20 train/val split
- Stage 1: frozen MobileNetV2 backbone
- Stage 2: fine-tune last 30 backbone layers
- Callbacks: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`

Output model:

- `outputs/models/av_perception_final.keras`

## Inference

```bash
python scripts/run_inference.py --input "path/to/video.mp4"
```

Optional args:

- `--output outputs/videos/annotated_output.mp4`
- `--model outputs/models/av_perception_final.keras`
- `--confidence 0.5`

Default output:

- `outputs/videos/annotated_output.mp4`

## Notes

- Bounding boxes are normalized during training and denormalized during inference.
- Low-confidence predictions are skipped using configurable confidence threshold.
- Inference preserves original FPS and resolution.
- All paths and hyperparameters are centralized in `config/config.py`.

## YOLO Multi-Object Inference (Recommended for all objects per frame)

This path uses pretrained YOLO weights and detects multiple objects per frame.
It maps COCO labels to project labels:

- `car` -> `Car`
- `person` -> `Pedestrian`
- `bicycle` -> `Cyclist`

Run:

```bash
python scripts/run_yolo_inference.py --input "path/to/video.mp4"
```

Optional args:

- `--weights yolov8n.pt`
- `--output outputs/videos/annotated_output_yolo.mp4`
- `--confidence 0.3`
- `--iou 0.45`
- `--imgsz 640`
- `--device cpu`

Note: first run may download YOLO weights automatically.


## Final Results

Final long-video inference command used:

```bash
python scripts/run_yolo_inference.py --input "testingvideos/1/Crashes caught on Seattle traffic cameras #14! - SeattleTraffic Cams (720p, h264).mp4" --output "outputs/videos/annotated_seattle_long.mp4" --weights yolov8s.pt --imgsz 640 --confidence 0.25 --iou 0.5 --device cpu
```

Final short-video inference command used:

```bash
python scripts/run_yolo_inference.py --input "testingvideos/2/FASTEST AMBULANCE EVER! - HyperXZ (720p, h264).mp4" --output "outputs/videos/annotated_ambulance_test.mp4" --weights yolov8s.pt --imgsz 640 --confidence 0.25 --iou 0.5 --device cpu
```

Final output videos used for review:

- `outputs/videos/annotated_seattle_long_fixed.mp4`
- `outputs/videos/annotated_ambulance_test_fixed.mp4`

