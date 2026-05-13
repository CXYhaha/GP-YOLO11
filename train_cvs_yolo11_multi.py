import os
import warnings
from pathlib import Path

from ultralytics import YOLO

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent

# Model and data
MODEL_CFG = ROOT / "ultralytics" / "cfg" / "models" / "11" / "gp_yolo11.yaml"
PRETRAINED_WEIGHTS = ROOT / "yolo11m.pt"
DATA_YAML = ROOT / "data.yaml"
PROJECT_DIR = ROOT / "runs"
EXPERIMENT_NAME = "cvs_gp_yolo11_4ch"

# Training
EPOCHS = 300
IMAGE_SIZE = 512
BATCH_SIZE = 32
WORKERS = 16
DEVICE = "0,1,2,3"
OPTIMIZER = "SGD"
AMP = True
CACHE = False

# Paper-aligned loss configuration
BOX_LOSS = "nwd"  # options in this repo: ciou | nwd | ciou+nwd
NWD_C = 10.0
NWD_WEIGHT = 1.0
NWD_WH_FACTOR = 12.0


def main():
    os.environ["RANK"] = "-1"
    os.environ["WORLD_SIZE"] = "1"

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")
    if not MODEL_CFG.exists():
        raise FileNotFoundError(f"Model YAML not found: {MODEL_CFG}")

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model config : {MODEL_CFG}")
    print(f"Dataset YAML : {DATA_YAML}")
    print(f"Using GPUs   : {DEVICE}")
    print(f"Box loss     : {BOX_LOSS}")

    model = YOLO(str(MODEL_CFG))

    if PRETRAINED_WEIGHTS.exists():
        print(f"Loading pretrained weights: {PRETRAINED_WEIGHTS}")
        model.load(str(PRETRAINED_WEIGHTS))
    else:
        print("Pretrained weights not found, training from scratch.")

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        optimizer=OPTIMIZER,
        amp=AMP,
        cache=CACHE,
        project=str(PROJECT_DIR),
        name=EXPERIMENT_NAME,
        exist_ok=True,
        rect=False,
        cos_lr=True,
        patience=30,
        verbose=True,
        save=True,
        save_period=10,
        box_loss=BOX_LOSS,
        nwd_c=NWD_C,
        nwd_weight=NWD_WEIGHT,
        nwd_wh_factor=NWD_WH_FACTOR,
    )


if __name__ == "__main__":
    main()
