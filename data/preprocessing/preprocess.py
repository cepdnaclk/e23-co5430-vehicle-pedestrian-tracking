"""
=============================================================================
MOT16 Dataset Preprocessing Pipeline
Project: Vehicle & Pedestrian Tracking (CO5430 – Group G07)
=============================================================================

WHAT THIS SCRIPT DOES (step-by-step):
  Step 1  – Explore & validate the raw MOT16 dataset structure
  Step 2  – Parse seqinfo.ini to extract sequence metadata
  Step 3  – Parse gt.txt annotations (filter valid classes & visible objects)
  Step 4  – Convert MOT16 bounding boxes → YOLO format (normalized xywh)
  Step 5  – Copy/organize images into train/val splits
  Step 6  – Write YOLO .txt label files alongside images
  Step 7  – Generate a dataset.yaml for YOLOv8
  Step 8  – Produce a summary report (frame count, class distribution, etc.)

OUTPUT FOLDER LAYOUT:
  data/
  └── yolo_dataset/
      ├── images/
      │   ├── train/   ← 80 % of frames
      │   └── val/     ← 20 % of frames
      ├── labels/
      │   ├── train/   ← matching .txt label files
      │   └── val/
      └── dataset.yaml

HOW TO RUN:
  python preprocess.py --mot_root data/raw/MOT16 --output data/yolo_dataset

=============================================================================
"""

import os
import sys
import shutil
import argparse
import configparser
import csv
from pathlib import Path
from collections import defaultdict

# ── optional, only needed for Step 8 visualisation ────────────────────────────
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# =============================================================================
# MOT16 CLASS IDs  (from the official devkit)
# =============================================================================
MOT16_CLASSES = {
    1: "pedestrian",
    2: "person_on_vehicle",
    3: "car",
    4: "bicycle",
    5: "motorbike",
    6: "non_mot_vehicle",
    7: "static_person",
    8: "distractor",
    9: "occluder",
    10: "occluder_on_ground",
    11: "occluder_full",
    12: "reflection",
    13: "crowd",
}

# Classes we actually want to keep for tracking
KEEP_CLASSES = {1: 0, 3: 1}   # pedestrian→0, car→1   (YOLO class ids)
CLASS_NAMES  = ["pedestrian", "car"]

MIN_VISIBILITY = 0.3          # skip objects less than 30 % visible
CONF_THRESHOLD = 1            # 0 = ignore in evaluation; we keep only 1


# =============================================================================
# STEP 1 – Validate raw dataset structure
# =============================================================================
def validate_mot16(mot_root: Path):
    """Check that the expected MOT16 folder structure is present."""
    print("\n" + "="*60)
    print("STEP 1 – Validating MOT16 dataset structure")
    print("="*60)

    if not mot_root.exists():
        sys.exit(f"[ERROR] Dataset root not found: {mot_root}\n"
                 f"  → Download from Kaggle first (see README).")

    sequences = []
    for split in ["train", "test"]:
        split_dir = mot_root / split
        if not split_dir.exists():
            print(f"  [SKIP] '{split}' split not found – skipping.")
            continue
        for seq in sorted(split_dir.iterdir()):
            if not seq.is_dir():
                continue
            img_dir = seq / "img1"
            gt_file = seq / "gt" / "gt.txt"
            seq_info = seq / "seqinfo.ini"

            ok = img_dir.exists() and seq_info.exists()
            has_gt = gt_file.exists()
            sequences.append({
                "split": split,
                "name": seq.name,
                "path": seq,
                "img_dir": img_dir,
                "gt_file": gt_file if has_gt else None,
                "seq_info": seq_info,
                "has_gt": has_gt,
            })
            status = "✓" if ok else "✗"
            gt_tag = " [has GT]" if has_gt else " [no GT – test split]"
            print(f"  {status}  {split}/{seq.name}{gt_tag}")

    if not sequences:
        sys.exit("[ERROR] No sequences found. Check the dataset path.")

    train_seqs = [s for s in sequences if s["has_gt"]]
    print(f"\n  → Found {len(sequences)} total sequences, "
          f"{len(train_seqs)} with ground-truth labels.")
    return sequences


# =============================================================================
# STEP 2 – Parse seqinfo.ini
# =============================================================================
def parse_seqinfo(seq_info_path: Path) -> dict:
    """Read seqinfo.ini and return a metadata dict."""
    cfg = configparser.ConfigParser()
    cfg.read(seq_info_path)
    sec = cfg["Sequence"]
    return {
        "name":        sec.get("name"),
        "fps":         int(sec.get("frameRate", 30)),
        "seq_length":  int(sec.get("seqLength", 0)),
        "img_width":   int(sec.get("imWidth", 1920)),
        "img_height":  int(sec.get("imHeight", 1080)),
        "img_ext":     sec.get("imExt", ".jpg"),
    }


# =============================================================================
# STEP 3 – Parse gt.txt
# =============================================================================
def parse_gt(gt_file: Path, img_width: int, img_height: int) -> dict:
    """
    Parse gt.txt and return a dict keyed by frame number.
    Each value is a list of YOLO-format annotations:
      [class_id, cx, cy, w, h]   (all normalised 0–1)
    """
    frame_annots = defaultdict(list)

    with open(gt_file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 9:
                continue

            frame   = int(row[0])
            # track_id = int(row[1])  # not needed for detection training
            bb_left = float(row[2])
            bb_top  = float(row[3])
            bb_w    = float(row[4])
            bb_h    = float(row[5])
            conf    = int(float(row[6]))
            cls_id  = int(float(row[7]))
            vis     = float(row[8])

            # ── Filters ──────────────────────────────────────────────
            if conf < CONF_THRESHOLD:        # ignore-region marker
                continue
            if cls_id not in KEEP_CLASSES:   # only pedestrian + car
                continue
            if vis < MIN_VISIBILITY:          # too occluded
                continue

            # ── Convert bbox  ─────────────────────────────────────────
            # MOT16: bb_left, bb_top = top-left corner (pixel, 1-based)
            cx = (bb_left + bb_w / 2) / img_width
            cy = (bb_top  + bb_h / 2) / img_height
            nw = bb_w / img_width
            nh = bb_h / img_height

            # clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            yolo_cls = KEEP_CLASSES[cls_id]
            frame_annots[frame].append([yolo_cls, cx, cy, nw, nh])

    return frame_annots


# =============================================================================
# STEP 4 + 5 + 6 – Organise images & write label files
# =============================================================================
def process_sequence(seq: dict, output_root: Path, val_ratio: float = 0.2):
    """
    For one MOT16 sequence:
      - Parse metadata & GT
      - Split frames into train / val
      - Copy images and write YOLO label .txt files
    Returns a stats dict.
    """
    if not seq["has_gt"]:
        print(f"  [SKIP] {seq['name']} – no ground-truth (test split).")
        return None

    meta = parse_seqinfo(seq["seq_info"])
    frame_annots = parse_gt(
        seq["gt_file"], meta["img_width"], meta["img_height"]
    )

    # Collect all frame image paths that exist
    img_paths = sorted(seq["img_dir"].glob(f"*{meta['img_ext']}"))
    if not img_paths:
        print(f"  [WARN] No images found in {seq['img_dir']}")
        return None

    n_val   = max(1, int(len(img_paths) * val_ratio))
    val_set = set(img_paths[-n_val:])     # last N frames → val

    stats = defaultdict(int)

    for img_path in img_paths:
        frame_num = int(img_path.stem)   # e.g. "000042" → 42
        split_tag = "val" if img_path in val_set else "train"

        # Destination paths
        dest_img = output_root / "images" / split_tag / f"{seq['name']}_{img_path.name}"
        dest_lbl = output_root / "labels" / split_tag / f"{seq['name']}_{img_path.stem}.txt"

        dest_img.parent.mkdir(parents=True, exist_ok=True)
        dest_lbl.parent.mkdir(parents=True, exist_ok=True)

        # Copy image
        shutil.copy2(img_path, dest_img)

        # Write label file (empty if no annotations for this frame)
        annotations = frame_annots.get(frame_num, [])
        with open(dest_lbl, "w") as f:
            for ann in annotations:
                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

        stats[split_tag] += 1
        for ann in annotations:
            stats[f"class_{ann[0]}"] += 1

    print(f"  ✓  {seq['name']:15s}  "
          f"train={stats['train']:4d}  val={stats['val']:4d}  "
          f"ped={stats['class_0']:5d}  car={stats['class_1']:5d}")
    return stats


# =============================================================================
# STEP 7 – Write dataset.yaml
# =============================================================================
def write_dataset_yaml(output_root: Path):
    """Generate dataset.yaml for YOLOv8 training."""
    yaml_content = f"""# YOLOv8 Dataset Config – MOT16 (Vehicle & Pedestrian Tracking)
# Auto-generated by preprocess.py  (Group G07, CO5430)

path: {output_root.resolve()}
train: images/train
val:   images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\n  ✓  dataset.yaml written → {yaml_path}")


# =============================================================================
# STEP 8 – Summary report
# =============================================================================
def print_summary(all_stats: list, output_root: Path):
    total_train = sum(s["train"]   for s in all_stats if s)
    total_val   = sum(s["val"]     for s in all_stats if s)
    total_ped   = sum(s["class_0"] for s in all_stats if s)
    total_car   = sum(s["class_1"] for s in all_stats if s)

    print("\n" + "="*60)
    print("STEP 8 – Preprocessing Summary")
    print("="*60)
    print(f"  Train frames : {total_train}")
    print(f"  Val frames   : {total_val}")
    print(f"  Total frames : {total_train + total_val}")
    print(f"  Pedestrian labels : {total_ped}")
    print(f"  Car labels        : {total_car}")
    print(f"  Output directory  : {output_root.resolve()}")
    print("="*60)
    print("  ✅  Preprocessing complete! Run YOLOv8 training next.")
    print("="*60 + "\n")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MOT16 dataset → YOLO format for YOLOv8"
    )
    parser.add_argument(
        "--mot_root",
        type=Path,
        default=Path("data/raw/MOT16"),
        help="Path to the downloaded MOT16 root folder",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/yolo_dataset"),
        help="Where to save the preprocessed YOLO dataset",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of frames per sequence used for validation (default: 0.2)",
    )
    args = parser.parse_args()

    # Step 1
    sequences = validate_mot16(args.mot_root)

    # Steps 2–6
    print("\n" + "="*60)
    print("STEPS 2-6 – Parsing annotations & organising YOLO dataset")
    print("="*60)
    all_stats = []
    for seq in sequences:
        stats = process_sequence(seq, args.output, args.val_ratio)
        all_stats.append(stats)

    # Step 7
    print("\n" + "="*60)
    print("STEP 7 – Writing dataset.yaml")
    print("="*60)
    write_dataset_yaml(args.output)

    # Step 8
    print_summary(all_stats, args.output)


if __name__ == "__main__":
    main()
