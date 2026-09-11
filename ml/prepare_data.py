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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import cv2
import numpy as np
from remotezip import RemoteZip

# FSOCO's raw images are full dashcam-resolution originals (measured directly,
# 2026-09-02: a random 10-image sample averaged 2.17MB/image, mostly PNG) --
# storing them at native resolution is pure waste here, since training only
# ever resizes down to imgsz=640 anyway (see train.py). Re-encoding to JPEG
# at a resolution modestly above imgsz (960, not exactly 640) keeps some
# margin for a future higher-imgsz retrain without needing to re-download,
# while still cutting per-image size roughly 15-20x. Confirmed necessary,
# not just an optimization: at the raw average size, 8000 images would need
# ~17.4GB, essentially all of the Jetson's 17GB free space, leaving nothing
# for training checkpoints/temp files -- user caught this before any bytes
# were written to disk.
_MAX_IMAGE_SIDE = 960
_JPEG_QUALITY = 90


def _resize_and_encode(img_bytes: bytes) -> bytes:
    img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > _MAX_IMAGE_SIDE:
        scale = _MAX_IMAGE_SIDE / longest
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()

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


# FSOCO's own server has confirmed-real, non-rare transient connect/read
# timeouts (2026-09-02: hit repeatedly during both the annotation-scan pass
# -- which already tolerated them via its own try/except -- AND the
# download pass, which had NO handling at all and let an unhandled
# RemoteIOError propagate out of a worker thread, through
# ThreadPoolExecutor's fut.result(), and crash the whole script under
# overnight_train.sh's `set -e` -- confirmed live: the process was
# confirmed dead via pgrep with 6339/8000 images already on disk and
# training never reached). Retried with backoff here, matching the
# resilience the scan pass already had. A fresh RemoteZip client on retry,
# not the same cached one -- a timed-out connection isn't assumed healthy
# just because the exception was caught.
_FETCH_MAX_ATTEMPTS = 5
_FETCH_RETRY_BASE_DELAY = 2.0  # seconds, doubles each attempt


def _fetch_pair(img_path: str, ann_path: str, img_out: Path, label_out: Path, class_to_id: dict) -> str:
    # img_out is always given a .jpg suffix by the caller regardless of the
    # source format (some FSOCO images are PNG) -- every image gets
    # re-encoded to JPEG here (see _resize_and_encode), so the on-disk
    # extension must match what's actually written, not the original's.
    if img_out.exists() and label_out.exists():
        return "skipped"
    last_exc = None
    for attempt in range(_FETCH_MAX_ATTEMPTS):
        try:
            zf = _client()
            img_bytes = zf.read(img_path)
            ann = json.loads(zf.read(ann_path))
            img_out.write_bytes(_resize_and_encode(img_bytes))
            label_out.write_text("\n".join(supervisely_to_yolo(ann, class_to_id)))
            return "downloaded"
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see module comment above
            last_exc = exc
            if hasattr(_thread_local, "zf"):
                del _thread_local.zf  # don't retry on a connection that just failed
            if attempt < _FETCH_MAX_ATTEMPTS - 1:
                time.sleep(_FETCH_RETRY_BASE_DELAY * (2 ** attempt))
    print(f"[prepare_data]   WARNING: giving up on {img_path} after {_FETCH_MAX_ATTEMPTS} attempts: {last_exc}")
    return "failed"


# Classes rare enough per-image that a uniform random sample under-represents
# them badly -- orange/large_orange cones only appear at a track's start/
# finish and entry/exit zones, not along every straightaway the way blue/
# yellow boundary cones do, so a plain uniform sample over the whole archive
# systematically starves the model of orange examples relative to blue/
# yellow (2026-09-02, user report: live orange-cone recognition confidence
# is poor; a check of the prior run's own confusion matrix showed orange/
# large_orange recall in the same ballpark as blue/yellow ON THE FEW EXAMPLES
# THEY DID GET, so the fix that actually matters is making sure there ARE
# enough examples, not just training longer against the same skewed sample).
RARE_CLASSES = {"orange_cone", "large_orange_cone"}


def _annotation_classes(img_path: str) -> set:
    """Reads just the (small) annotation JSON for one image and returns the
    set of classTitle strings it contains -- used for a metadata-only scan
    pass, never downloads the (much larger) image itself."""
    zf = _client()
    ann = json.loads(zf.read(ann_path_for(img_path)))
    return {obj.get("classTitle") for obj in ann.get("objects", [])}


def _annotation_class_counts(img_path: str) -> dict:
    """Like _annotation_classes, but counts INSTANCES per class, not just
    presence -- see scan_and_balance's own comment for why presence alone
    (this module's first stratification attempt) wasn't enough: an image
    can contain orange_cone AND ten blue_cone boxes, so "does this image
    have orange" and "how many orange instances does the final sample have
    relative to blue/yellow" are genuinely different questions."""
    zf = _client()
    ann = json.loads(zf.read(ann_path_for(img_path)))
    counts = {}
    for obj in ann.get("objects", []):
        cls = obj.get("classTitle")
        if cls in CLASSES:
            counts[cls] = counts.get(cls, 0) + 1
    return counts


def scan_for_rare_classes(img_entries: List[str], workers: int = 24):
    """Scans every image's annotation (parallel, metadata-only -- no image
    bytes downloaded) and splits img_entries into (rare_imgs, other_imgs)
    based on whether at least one RARE_CLASSES annotation is present.
    Cheap relative to the actual download pass: annotation JSONs are tiny
    compared to full images, so scanning the WHOLE archive first to decide
    what to sample is affordable even though it wasn't done before."""
    rare_imgs, other_imgs = [], []
    done = 0
    print(f"[prepare_data] scanning {len(img_entries)} annotations for {sorted(RARE_CLASSES)} ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_annotation_classes, p): p for p in img_entries}
        for fut in as_completed(futures):
            img_path = futures[fut]
            try:
                classes = fut.result()
            except Exception as exc:
                print(f"[prepare_data]   WARNING: failed to read annotation for {img_path}: {exc}")
                continue
            (rare_imgs if classes & RARE_CLASSES else other_imgs).append(img_path)
            done += 1
            if done % 1000 == 0 or done == len(img_entries):
                print(f"[prepare_data]   scanned {done}/{len(img_entries)} -- {len(rare_imgs)} contain a rare class so far")
    return rare_imgs, other_imgs


def scan_and_balance(img_entries: List[str], workers: int = 24):
    """Builds a sample with roughly EQUAL total instance counts across
    blue_cone, yellow_cone, and the orange family (orange_cone +
    large_orange_cone combined -- the user's own framing was "orange, blue,
    yellow" as three buckets, and large_orange is already a further-rare
    subset of the same real-world scarcity: it only marks the start/finish
    gate) -- not just equal IMAGE counts.

    Confirmed necessary (2026-09-02), not redundant with the earlier
    per-image stratification: that fix guaranteed every orange-containing
    image was included, but did nothing to control how many blue/yellow
    instances rode along in the SAME images or in additional filler images
    -- the resulting train_v2 dataset's own validation output showed
    orange-family instances (5086+1587=6673 in just the val split) at
    roughly HALF blue's (10515) and yellow's (11970), because a typical
    photo has many boundary cones down a straightaway but only 1-2 orange
    gate cones. Per-image presence and per-instance balance are genuinely
    different questions.

    Algorithm: take EVERY image containing an orange-family instance (can't
    increase the real-world supply of orange examples, so use all of it --
    same principle as the earlier fix), tally the orange-family instance
    total that set already provides, then greedily add MORE images (from
    the non-orange pool, shuffled) ONLY until blue and yellow instance
    counts each reach that same total -- deliberately NOT adding further
    blue/yellow-heavy filler once parity is reached, which is what actually
    achieves balance: capping the abundant classes to match the scarce one,
    not trying to inflate the scarce one to match (there often isn't enough
    unique real orange data in the archive to do that at blue/yellow's
    natural scale).
    """
    print(f"[prepare_data] scanning {len(img_entries)} annotations for per-class instance counts ...")
    counts_by_img = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_annotation_class_counts, p): p for p in img_entries}
        for fut in as_completed(futures):
            img_path = futures[fut]
            try:
                counts_by_img[img_path] = fut.result()
            except Exception as exc:
                print(f"[prepare_data]   WARNING: failed to read annotation for {img_path}: {exc}")
                continue
            done += 1
            if done % 1000 == 0 or done == len(img_entries):
                print(f"[prepare_data]   scanned {done}/{len(img_entries)}")

    def orange_count(c):
        return c.get("orange_cone", 0) + c.get("large_orange_cone", 0)

    orange_imgs = [p for p, c in counts_by_img.items() if orange_count(c) > 0]
    other_imgs = [p for p, c in counts_by_img.items() if orange_count(c) == 0]

    blue_total = sum(counts_by_img[p].get("blue_cone", 0) for p in orange_imgs)
    yellow_total = sum(counts_by_img[p].get("yellow_cone", 0) for p in orange_imgs)
    orange_total = sum(orange_count(counts_by_img[p]) for p in orange_imgs)
    print(f"[prepare_data] {len(orange_imgs)} images contain an orange-family instance "
          f"({100 * len(orange_imgs) / len(img_entries):.1f}%) -- "
          f"orange={orange_total} blue={blue_total} yellow={yellow_total} instances from this set alone")

    target = orange_total  # the parity ceiling every other class is capped at, not raised to
    random.shuffle(other_imgs)
    filler = []
    for p in other_imgs:
        if blue_total >= target and yellow_total >= target:
            break
        c = counts_by_img[p]
        filler.append(p)
        blue_total += c.get("blue_cone", 0)
        yellow_total += c.get("yellow_cone", 0)
    print(f"[prepare_data] added {len(filler)} filler images to reach blue={blue_total} "
          f"yellow={yellow_total} (target={target}, from {len(orange_imgs)} orange-family images)")

    sample_imgs = orange_imgs + filler
    random.shuffle(sample_imgs)
    return sample_imgs, {"orange": orange_total, "blue": blue_total, "yellow": yellow_total,
                          "orange_images": len(orange_imgs), "filler_images": len(filler)}


def prepare(out_dir: Path, sample_size: int, val_fraction: float, seed: int, workers: int = 12,
            mode: str = "balance"):
    """mode: "balance" (default, 2026-09-02) -- roughly equal blue/yellow/
    orange-family INSTANCE counts, see scan_and_balance's own comment;
    sample_size is ignored (the balanced sample's size is a RESULT of the
    algorithm, not an input to it). "stratify" -- the earlier, weaker fix:
    guarantees every orange-containing IMAGE is included but doesn't control
    instance ratios, sample_size caps the total. "uniform" -- plain random
    sample over the whole archive, no class awareness at all, sample_size
    is the exact count."""
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
    if mode == "balance":
        sample_imgs, stats = scan_and_balance(img_entries, workers=max(workers, 24))
        print(f"[prepare_data] final balanced sample: {len(sample_imgs)} images "
              f"({stats['orange_images']} orange-family + {stats['filler_images']} filler) -- "
              f"instance totals: orange={stats['orange']} blue={stats['blue']} yellow={stats['yellow']}")
    elif mode == "stratify":
        rare_imgs, other_imgs = scan_for_rare_classes(img_entries, workers=max(workers, 24))
        print(f"[prepare_data] {len(rare_imgs)}/{len(img_entries)} images contain a rare class "
              f"({100 * len(rare_imgs) / len(img_entries):.1f}%)")
        # Take EVERY rare-class image (there's no reason to leave orange/
        # large_orange examples on the table when they're already this
        # scarce), then fill the remaining budget with a random sample of
        # the rest so blue/yellow-only images still make up the majority --
        # matching real track composition, just no longer starving the rare
        # classes down to whatever a uniform draw happened to include.
        random.shuffle(rare_imgs)
        remaining_budget = max(0, sample_size - len(rare_imgs))
        filler = random.sample(other_imgs, min(remaining_budget, len(other_imgs)))
        sample_imgs = rare_imgs + filler
        random.shuffle(sample_imgs)  # mix rare/common before the train/val slice below
        print(f"[prepare_data] final sample: {len(rare_imgs)} rare-class + {len(filler)} other "
              f"= {len(sample_imgs)} total")
    else:
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
    failed = {"train": 0, "val": 0}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for split, pairs in (("train", train_pairs), ("val", val_pairs)):
            for img_path, ann_path in pairs:
                # Always .jpg regardless of source extension -- _fetch_pair
                # re-encodes every image to JPEG (see _resize_and_encode).
                out_name = Path(img_path).with_suffix(".jpg").name
                img_out = out_dir / split / "images" / out_name
                label_out = out_dir / split / "labels" / Path(out_name).with_suffix(".txt").name
                fut = pool.submit(_fetch_pair, img_path, ann_path, img_out, label_out, class_to_id)
                futures[fut] = split

        # _fetch_pair itself never raises (it retries internally and returns
        # "failed" as a status string -- see its own comment) -- fut.result()
        # here is expected to always succeed; a network error that survives
        # ALL retries becomes a counted "failed" entry, not a crashed script.
        for fut in as_completed(futures):
            split = futures[fut]
            status = fut.result()
            done[split] += 1
            if status == "skipped":
                skipped[split] += 1
            elif status == "failed":
                failed[split] += 1
            if done[split] % 20 == 0 or done[split] == totals[split]:
                print(f"[prepare_data]   {split}: {done[split]}/{totals[split]} "
                      f"({skipped[split]} already on disk, {failed[split]} failed)")

    total_failed = failed["train"] + failed["val"]
    if total_failed:
        print(f"[prepare_data] WARNING: {total_failed} image(s) permanently failed to download "
              f"after {_FETCH_MAX_ATTEMPTS} attempts each -- neither the image nor its label file "
              f"gets written on failure (both writes happen only after both reads succeed), so the "
              f"final dataset is just {total_failed} image(s) smaller than requested, not corrupted "
              f"with orphaned files. Not fatal; re-run prepare_data.py again later (already-downloaded "
              f"images are skipped) to fill the gap if the failure count is large.")

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
    parser.add_argument("--sample-size", type=int, default=8000,
                         help="ignored in --mode balance, where sample size is a result, not an input")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--mode", choices=["balance", "stratify", "uniform"], default="balance",
                         help="balance: equal blue/yellow/orange-family instance counts (default). "
                              "stratify: every orange-containing image included, no instance-count "
                              "control. uniform: plain random sample, no class awareness.")
    args = parser.parse_args()

    prepare(Path(args.out_dir), args.sample_size, args.val_fraction, args.seed, args.workers,
            mode=args.mode)
