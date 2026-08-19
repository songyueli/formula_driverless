"""YOLO cone detection — training script."""

from typing import Optional

import torch
from ultralytics import YOLO

# epochs/imgsz per hardware profile. "smoke" is a fast sanity-check config for
# CPU-only dev machines; "full" assumes a real GPU (e.g. the Jetson AGX Orin
# deployment target) and uses the real target imgsz.
PROFILES = {
    "smoke": {"epochs": 5, "imgsz": 640},
    "full": {"epochs": 100, "imgsz": 1440},
}


def train(
    data_yaml: str,
    model_variant: str = "yolo26n.pt",
    epochs: Optional[int] = None,
    imgsz: Optional[int] = None,
    profile: str = "auto",
    project: str = "runs/detect",
    name: str = "train",
):
    if profile == "auto":
        profile = "full" if torch.cuda.is_available() else "smoke"
    defaults = PROFILES[profile]
    epochs = defaults["epochs"] if epochs is None else epochs
    imgsz = defaults["imgsz"] if imgsz is None else imgsz
    print(f"[train] profile={profile} (cuda available: {torch.cuda.is_available()}) "
          f"-> epochs={epochs}, imgsz={imgsz}")

    model = YOLO(model_variant)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        name=name,
        exist_ok=True,  # overwrite the same run dir on rerun instead of train2, train3, ...
    )
    # No separate model.val() call: .train() already runs a final validation
    # pass internally and writes its plots/metrics into the same run dir.
    model.export(format="onnx", imgsz=imgsz, simplify=True)
    return model


if __name__ == "__main__":
    # Standalone usage: `python train.py` from within ml/, venv activated.
    train(data_yaml="data/fsoco_sample/dataset.yaml")
