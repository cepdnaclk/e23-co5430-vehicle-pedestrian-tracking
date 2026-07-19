"""
verify_labels.py  –  Quick sanity-check on the preprocessed YOLO dataset.

What it checks:
  1. Every image in images/ has a matching .txt in labels/
  2. No label file has values outside [0, 1]
  3. Prints class distribution per split
  4. (Optional) draws bounding boxes on a few sample images

Usage:
    python verify_labels.py --dataset data/yolo_dataset --samples 5
"""

import argparse
import random
from pathlib import Path
from collections import Counter

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

CLASS_NAMES  = ["pedestrian", "car"]
COLORS       = [(0, 255, 0), (0, 100, 255)]   # green=ped, orange=car


def verify(dataset_root: Path, num_samples: int = 5):
    errors = 0
    class_counts = {"train": Counter(), "val": Counter()}

    for split in ["train", "val"]:
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split

        img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

        for img_path in img_files:
            lbl_path = lbl_dir / (img_path.stem + ".txt")

            if not lbl_path.exists():
                print(f"[MISSING LABEL] {lbl_path}")
                errors += 1
                continue

            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"[BAD LINE] {lbl_path}: {line.strip()}")
                        errors += 1
                        continue
                    cls_id, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                    class_counts[split][cls_id] += 1

                    for v in (cx, cy, w, h):
                        if not (0.0 <= v <= 1.0):
                            print(f"[OUT OF RANGE] {lbl_path}: {line.strip()}")
                            errors += 1

    print("\n── Class distribution ──────────────────────────")
    for split in ["train", "val"]:
        print(f"  {split}:")
        for cls_id, name in enumerate(CLASS_NAMES):
            print(f"    {name:12s}: {class_counts[split][cls_id]}")
    print(f"\n  Total errors found: {errors}")

    # ── Optional: visualise random samples ─────────────────────────────────
    if CV2_AVAILABLE and num_samples > 0:
        all_imgs = (
            list((dataset_root / "images" / "train").glob("*.jpg")) +
            list((dataset_root / "images" / "val").glob("*.jpg"))
        )
        samples = random.sample(all_imgs, min(num_samples, len(all_imgs)))
        vis_dir = dataset_root / "previews"
        vis_dir.mkdir(exist_ok=True)

        for img_path in samples:
            split_tag = "train" if "train" in str(img_path) else "val"
            lbl_path = dataset_root / "labels" / split_tag / (img_path.stem + ".txt")

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    color = COLORS[cls_id % len(COLORS)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, CLASS_NAMES[cls_id],
                                (x1, max(y1-4, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            out_path = vis_dir / img_path.name
            cv2.imwrite(str(out_path), img)
            print(f"  [Preview saved] {out_path}")

        if samples:
            print(f"\n  Preview images saved to: {vis_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/yolo_dataset"))
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of random sample images to visualise (needs opencv)")
    args = parser.parse_args()
    verify(args.dataset, args.samples)


if __name__ == "__main__":
    main()
