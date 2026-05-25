# GP-YOLO11

Experimental code, model modifications, and result artifacts for the paper:

> **A Catalogue of Green Pea Galaxies in the Kilo-Degree Survey DR5 with the GP-YOLO11 Deep-Learning Framework**

This repository is not a generic Ultralytics mirror. It is the working research codebase used to develop the GP-YOLO11 framework, run the KiDS DR5 experiments, and store representative outputs used in the paper.

## Overview

Green Pea galaxies (GPs) are compact extreme emission-line galaxies that are difficult to identify from wide-field imaging because they are rare, compact, and easily confused with stars, QSOs, compact galaxies, and image artifacts.

This project builds a **source-centered GP detection framework** on top of **YOLO11** for KiDS DR5 `u/g/r/i` image cutouts. The paper applies the model to **1,725,117** photometrically pre-selected KiDS sources and constructs a catalog of **5,699** high-confidence GP candidates.

The main idea is to combine:

- multiband image information from KiDS DR5,
- compact-source localization with a YOLO-style detector,
- receptive-field attention feature extraction for compact morphology,
- NWD-based box regression for tiny astronomical targets.

## Paper Summary

The GP-YOLO11 framework is designed for **compact GP-like source identification** rather than blind full-tile detection. Each input is a source-centered cutout, and the detector outputs both:

- a localized GP-like region,
- a confidence score for candidate ranking and follow-up.

According to the manuscript, the method:

- uses `512 x 512 x 4` KiDS `u/g/r/i` cutouts as model input,
- benchmarks against color-cut selection, XGBoost, CNN classification, Faster R-CNN, SSD, EfficientDet, and YOLOv8n,
- recovers **59 known GPs**,
- identifies **61 newly spectroscopically confirmed GPs** from archival DESI and SDSS spectra.

## Main Results

### Source-Level Test Performance

| Method | Precision (%) | Recall (%) | F1-score (%) |
| --- | ---: | ---: | ---: |
| Color-cut Selection | 42.25 | 100.00 | 59.41 |
| XGBoost | 82.14 | 92.00 | 86.79 |
| Simple CNN classifier | 90.63 | 96.67 | 93.55 |
| Faster R-CNN | 92.13 | 91.73 | 91.93 |
| SSD | 92.56 | 92.31 | 92.43 |
| EfficientDet | 94.82 | 93.97 | 94.39 |
| YOLOv8n | 94.64 | 94.80 | 94.72 |
| **GP-YOLO11** | **97.97** | **97.61** | **97.79** |

### Ablation Results

| Model | RFAConv | NWD Loss | Precision (%) | Recall (%) | mAP@0.5 (%) |
| --- | :---: | :---: | ---: | ---: | ---: |
| Baseline (YOLOv11n) |  |  | 96.19 | 95.47 | 95.78 |
| Variant 1 | yes |  | 96.89 | 96.45 | 96.45 |
| Variant 2 |  | yes | 96.62 | 96.71 | 97.48 |
| **GP-YOLO11** | yes | yes | **97.97** | **97.61** | **98.36** |

## Method Components in This Repository

The paper-related implementation is mainly distributed across the following files:

- [`train_cvs_yolo11_multi.py`](./train_cvs_yolo11_multi.py): training entry script used for the main experiment.
- [`data.yaml`](./data.yaml): dataset configuration for the GP detection task.
- [`ultralytics/cfg/models/11/yolov11_RFAConv.yaml`](./ultralytics/cfg/models/11/yolov11_RFAConv.yaml): GP-oriented YOLO11 variant with RFAConv and a small-object head.
- [`ultralytics/nn/conv/RFAConv.py`](./ultralytics/nn/conv/RFAConv.py): custom receptive-field attention convolution implementations.
- [`ultralytics/utils/loss.py`](./ultralytics/utils/loss.py): NWD-related box loss support.
- [`ultralytics/nn/tasks.py`](./ultralytics/nn/tasks.py): custom module registration for the modified architecture.

## Repository Contents

This repository currently contains the **experimental codebase and result artifacts** used during paper preparation.

### Core files

- `train_cvs_yolo11_multi.py`: multi-GPU training script.
- `data.yaml`: single-class GP detection dataset config (`nc: 1`, class name `cvs` in the current file).
- `results.csv`: root-level training log snapshot.

### Figures and paper assets

- `docs/figures/gp_yolo11_architecture.png`: paper architecture figure.
- `docs/figures/gp_yolo11_neck.png`: neck-level design illustration.
- `docs/figures/kids_ugri_stack.png`: KiDS multiband image example.

### Training outputs

- `runs/cvs_yolo11_multiGPU/`: representative training outputs, including:
  - `results.csv`
  - `results.png`
  - PR / P / R / F1 curves
  - confusion matrices
  - validation visualizations
  - `weights/best.pt`

### Spectral inspection assets

- `spectrum/spectrumfits/`: FITS spectra used in confirmation/inspection.
- `spectrum/spectrumpic/`: annotated spectral plots.
- `pic/`: supplementary image assets used during inspection and paper preparation.

## Experimental Setup

From the current repository state, the main training configuration is:

- detector family: YOLO11
- task: single-class detection
- image size: `512`
- epochs: `300`
- batch size: `32`
- optimizer: `SGD`
- device setting in script: `0,1,2,3`
- early stopping patience: `30`
- confidence threshold used in the paper for candidate filtering: `conf > 0.8`

## How To Run

### 1. Install dependencies

This codebase is built on top of Ultralytics and a local modified source tree.

```bash
pip install -e .
```

If you want a clean environment, use a fresh Python environment first.

### 2. Prepare the dataset

The KiDS DR5 training data are **not included** in this repository.

The current `data.yaml` expects:

```text
dataset/
  train/
  val/
```

You will need to prepare your own source-centered cutouts and labels in YOLO format.

### 3. Update local paths

The current training script still contains **machine-specific absolute paths**, for example:

- `/home/yangpengchao/caoxinyao/ultralytics-main/...`

Before running, update:

- `MODEL_CFG`
- `PRETRAINED_WEIGHTS`
- `DATA_YAML`
- `PROJECT_DIR`

in [`train_cvs_yolo11_multi.py`](./train_cvs_yolo11_multi.py), and adjust dataset paths in [`data.yaml`](./data.yaml).

### 4. Start training

```bash
python train_cvs_yolo11_multi.py
```

## Notes On Reproducibility

This repository should be understood as a **research working tree**, not yet a polished benchmark release.

Important points:

- the repository is based on a modified local Ultralytics source tree,
- some paths in scripts are still environment-specific,
- paper figures and result artifacts are included,
- the full KiDS parent sample and raw survey cutouts are not included,
- the repository currently does **not** include the final released 5,699-candidate catalog file itself.

If you want this repository to be used by other researchers directly, the next cleanup step should be:

1. remove hard-coded local paths,
2. add a minimal dataset format example,
3. provide the final inference/config command used for the paper,
4. add the released candidate catalog and column description.

## Representative Outputs

The repository already includes representative experiment outputs:

- architecture figure: [`docs/figures/gp_yolo11_architecture.png`](./docs/figures/gp_yolo11_architecture.png)
- training summary: [`runs/cvs_yolo11_multiGPU/results.png`](./runs/cvs_yolo11_multiGPU/results.png)
- PR curve: [`runs/cvs_yolo11_multiGPU/BoxPR_curve.png`](./runs/cvs_yolo11_multiGPU/BoxPR_curve.png)

## Citation

If you use this repository, please cite the corresponding paper:

```bibtex
@article{gp_yolo11_kids_dr5,
  title   = {A Catalogue of Green Pea Galaxies in the Kilo-Degree Survey DR5 with the GP-YOLO11 Deep-Learning Framework},
  author  = {Yang, Pengchao and collaborators},
  journal = {Draft manuscript},
  year    = {2026}
}
```

Update the BibTeX entry after the paper is formally submitted or accepted.

## Acknowledgement

This work is built on top of the Ultralytics YOLO codebase and extends it for Green Pea galaxy candidate identification in KiDS DR5.
