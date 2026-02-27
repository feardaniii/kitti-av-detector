# Specs Index

This directory is the source of truth for implementation scope in this repository.

## Active spec

- End-to-end AV detection pipeline on KITTI using TensorFlow/Keras MobileNetV2:
  - Train on KITTI object detection labels/images.
  - Classes: `Car`, `Pedestrian`, `Cyclist`.
  - Produce an annotated MP4 from an input MP4.

## Current constraints

- Modular structure with dedicated `config`, `data`, `models`, `training`, `inference`, and `scripts` packages.
- CPU-compatible inference path with OpenCV frame processing.
- Reproducible training split/shuffle and robust error handling.
- Runtime requirement: use Python `3.10` to `3.12` for TensorFlow compatibility on Windows (Python `3.13+` unsupported for this project setup).
- Dataset root is configurable via CLI (`--dataset-root`) or env var (`KITTI_DATASET_ROOT`) rather than requiring in-repo placement.
- Inference auto-detects model input size from the saved Keras model to avoid training/inference shape mismatches.
- Added pretrained YOLO inference path for multi-object per-frame detection (Car/Pedestrian/Cyclist mapping) to satisfy full-scene annotation requirement.
- YOLO inference includes a person+bicycle overlap relabel rule to reduce cyclist-to-pedestrian misclassification.

