"""Download a sample of FSOCO (bounding-box archive) and convert it to a
YOLO-format dataset with a real train/val split.

Logic lifted from perception/notebooks/yolo_cone_detection.ipynb's data-prep
cells, which validated the Supervisely->YOLO conversion against a 200-image
smoke-test sample. This adds an actual train/val split (the notebook's
smoke test deliberately used train==val, which is fine for confirming the
pipeline runs but produces meaningless validation metrics for a real run).
"""

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from remotezip import RemoteZip

FSOCO_BBOX_URL = "http://fsoco.cs.uni-freiburg.de/datasets/fsoco_bounding_boxes_train.zip"

# Pinned explicitly (rather than derived from meta.json's order) so class
# indices are stable across reruns. Drops unknown_cone (FSOCO's own docs
# call it non-rules-compliant) and the seg_* classes (segmentation-only,
# never appear on rectangle objects in this bounding-box archive).
CLASSES = ["blue_cone", "yellow_cone", "orange_cone", "large_orange_cone"]

# remotezip's RemoteZip re-fetches the archive's central directory on every
# open, and its read() isn't documented as thread-safe — so each worker
# thread gets its own long-lived client (opened once, reused for all of
# that thread's downloads) instead of sharing one or opening per-file.
_thread_local = threading.local()


def _client() -> RemoteZip:
    if not hasattr(_thread_local, "zf"):
        _thread_local.zf = RemoteZip(FSOCO_BBOX_URL)
    return _thread_local.zf


def supervisely_to_yolo(ann: dict, class_to_id: dict) -> List[str]:
    h, w = ann["size"]["height"], ann["size"]["width"]
    lines = []
    for obj in ann["objects"]:
        if obj["geometryType"] != "rectangle":
            continue
        if obj["classTitle"] not in class_to_id:
            continue
        (x1, y1), (x2, y2) = obj["points"]["exterior"]
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = abs(x2 - x1) / w, abs(y2 - y1) / h
        cls_id = class_to_id[obj["classTitle"]]
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def ann_path_for(img_path: str) -> str:
    return img_path.replace("/img/", "/ann/") + ".json"


def _fetch_pair(img_path: str, ann_path: str, img_out: Path, label_out: Path, class_to_id: dict) -> str:
    if img_out.exists() and label_out.exists():
        return "skipped"
    zf = _client()
    img_bytes = zf.read(img_path)
    ann = json.loads(zf.read(ann_path))
    img_out.write_bytes(img_bytes)
    label_out.write_text("\n".join(supervisely_to_yolo(ann, class_to_id)))
    return "downloaded"


def prepare(out_dir: Path, sample_size: int, val_fraction: float, seed: int, workers: int = 12):
    class_to_id = {name: i for i, name in enumerate(CLASSES)}

    for split in ("train", "val"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    print(f"[prepare_data] listing {FSOCO_BBOX_URL} ...")
    with RemoteZip(FSOCO_BBOX_URL) as zf:
        names = zf.namelist()

    img_entries = [n for n in names if "/img/" in n and not n.endswith("/")]
    print(f"[prepare_data] {len(img_entries)} total images in archive")

    random.seed(seed)
    sample_imgs = random.sample(img_entries, min(sample_size, len(img_entries)))
    sample_pairs = [(p, ann_path_for(p)) for p in sample_imgs]

    missing = [ann for _, ann in sample_pairs if ann not in names]
    assert not missing, f"{len(missing)} annotation paths not found — check ann_path_for()"

    n_val = max(1, int(len(sample_pairs) * val_fraction))
    val_pairs = sample_pairs[:n_val]
    train_pairs = sample_pairs[n_val:]
    print(f"[prepare_data] split: {len(train_pairs)} train / {len(val_pairs)} val")
    print(f"[prepare_data] downloading with {workers} parallel workers")

    totals = {"train": len(train_pairs), "val": len(val_pairs)}
    done = {"train": 0, "val": 0}
    skipped = {"train": 0, "val": 0}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for split, pairs in (("train", train_pairs), ("val", val_pairs)):
            for img_path, ann_path in pairs:
                out_name = Path(img_path).name
                img_out = out_dir / split / "images" / out_name
                label_out = out_dir / split / "labels" / Path(out_name).with_suffix(".txt").name
                fut = pool.submit(_fetch_pair, img_path, ann_path, img_out, label_out, class_to_id)
                futures[fut] = split

        for fut in as_completed(futures):
            split = futures[fut]
            status = fut.result()
            done[split] += 1
            if status == "skipped":
                skipped[split] += 1
            if done[split] % 20 == 0 or done[split] == totals[split]:
                print(f"[prepare_data]   {split}: {done[split]}/{totals[split]} ({skipped[split]} already on disk)")

    dataset_yaml = out_dir / "dataset.yaml"
    dataset_yaml.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"names:\n" + "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES)) + "\n"
    )
    print(f"[prepare_data] wrote {dataset_yaml}")
    print(dataset_yaml.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/fsoco_train")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    prepare(Path(args.out_dir), args.sample_size, args.val_fraction, args.seed, args.workers)
