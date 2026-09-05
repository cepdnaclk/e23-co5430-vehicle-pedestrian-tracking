"""
=============================================================================
evaluate.py  —  MOTA / IDF1 / MOTP Evaluation
Project: Vehicle & Pedestrian Tracking (CO5430 – Group G07)
Milestone: M5

Computes standard MOT metrics by comparing our tracker output
against MOT16 ground truth using py-motmetrics.

INSTALL:
  pip install motmetrics

HOW TO RUN:
  # First generate tracker output files with run_tracker.py or run_bytetrack.py
  # (they save .txt prediction files alongside the video)
  python src/tracking/evaluate.py --tracker sort
  python src/tracking/evaluate.py --tracker bytetrack

OUTPUT:
  A table like:
    Metric         SORT    ByteTrack
    MOTA           0.52    0.58
    IDF1           0.61    0.67
    MOTP           0.23    0.21
    FP             1234    1102
    FN             2341    2100
    ID Switches    189     97

METRICS EXPLAINED:
  MOTA  = 1 - (FP + FN + ID_switches) / num_gt_objects   (higher = better)
  IDF1  = fraction of detections correctly identified across frames  (higher = better)
  MOTP  = mean localisation error (lower = better)
  FP    = false positives (boxes where no real pedestrian exists)
  FN    = false negatives (missed real pedestrians)
  ID SW = times a track changes its assigned ID (lower = better)
=============================================================================
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def load_gt(seq_dir: str) -> dict:
    """
    Load MOT16 ground truth for a sequence.
    Keeps only pedestrian (class 1) with conf==1 and visibility >= 0.3

    Returns: {frame_id: [(x1,y1,x2,y2, track_id), ...]}
    """
    gt_path = Path(seq_dir) / "gt" / "gt.txt"
    if not gt_path.exists():
        sys.exit(f"[ERROR] gt.txt not found at {gt_path}")

    gt = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            frame  = int(parts[0])
            tid    = int(parts[1])
            x      = float(parts[2]); y = float(parts[3])
            w      = float(parts[4]); h = float(parts[5])
            conf   = int(parts[6])
            cls    = int(parts[7])
            vis    = float(parts[8])

            if conf != 1:       continue    # ignore regions
            if cls  != 1:       continue    # keep pedestrian only
            if vis  < 0.30:     continue    # too occluded

            x1,y1,x2,y2 = x, y, x+w, y+h
            gt.setdefault(frame, []).append((x1,y1,x2,y2, tid))
    return gt


def load_predictions(pred_path: str) -> dict:
    """
    Load tracker predictions saved in MOTChallenge format:
    frame, id, x, y, w, h, conf, -1, -1, -1

    Returns: {frame_id: [(x1,y1,x2,y2, track_id), ...]}
    """
    preds = {}
    if not os.path.exists(pred_path):
        sys.exit(f"[ERROR] Prediction file not found: {pred_path}\n"
                 f"       Run run_tracker.py or run_bytetrack.py first with --save_txt")
    with open(pred_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            tid   = int(parts[1])
            x     = float(parts[2]); y = float(parts[3])
            w     = float(parts[4]); h = float(parts[5])
            x1,y1,x2,y2 = x, y, x+w, y+h
            preds.setdefault(frame, []).append((x1,y1,x2,y2, tid))
    return preds


def compute_iou(boxA, boxB):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (aA + aB - inter + 1e-9)


def evaluate_sequence(gt: dict, preds: dict, iou_threshold=0.50):
    """
    Compute frame-by-frame matching and accumulate MOTA components.

    Uses a greedy matching (sufficient for a course project).
    For proper MOT benchmarking use py-motmetrics.
    """
    TP = 0; FP = 0; FN = 0; ID_SW = 0
    total_gt = 0; total_localisation_error = 0
    prev_matches = {}       # gt_id → pred_id from last frame
    unique_gt_ids = set()   # all GT IDs seen (for IDF1)
    all_matched_gt_ids = set()  # GT IDs matched at least once (for IDF1)

    all_frames = sorted(set(gt.keys()) | set(preds.keys()))

    for frame in all_frames:
        gt_boxes   = gt.get(frame, [])
        pred_boxes = preds.get(frame, [])
        total_gt  += len(gt_boxes)

        # Collect unique GT IDs (reuse loop — no second pass needed)
        for gb in gt_boxes:
            unique_gt_ids.add(gb[4])

        # Build IoU matrix
        matched_gt   = set()
        matched_pred = set()
        current_matches = {}

        for pi, pb in enumerate(pred_boxes):
            best_iou = iou_threshold
            best_gi  = -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pb[:4], gb[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gi  = gi

            if best_gi >= 0:
                matched_gt.add(best_gi)
                matched_pred.add(pi)
                gt_id   = gt_boxes[best_gi][4]
                pred_id = pb[4]
                current_matches[gt_id] = pred_id
                all_matched_gt_ids.add(gt_id)  # track while we're here

                # Check for ID switch
                if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                    ID_SW += 1

                TP += 1
                total_localisation_error += (1.0 - best_iou)

        FP += len(pred_boxes) - len(matched_pred)
        FN += len(gt_boxes)   - len(matched_gt)
        prev_matches = current_matches

    MOTA = 1.0 - (FP + FN + ID_SW) / max(total_gt, 1)
    MOTP = total_localisation_error / max(TP, 1)

    # Simplified IDF1: fraction of GT objects ever detected
    IDF1_approx = len(all_matched_gt_ids) / max(len(unique_gt_ids), 1)

    return {
        "MOTA":        round(MOTA, 4),
        "MOTP":        round(MOTP, 4),
        "IDF1_approx": round(IDF1_approx, 4),
        "TP":          TP,
        "FP":          FP,
        "FN":          FN,
        "ID_SW":       ID_SW,
        "GT_objs":     total_gt,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate tracker on MOT16")
    parser.add_argument("--seq_dir",  default="data/raw/MOT16/train/MOT16-02",
                        help="Path to MOT16 sequence")
    parser.add_argument("--pred_sort",
                        default="outputs/MOT16-02_sort_pred.txt",
                        help="SORT predictions in MOTChallenge format")
    parser.add_argument("--pred_byte",
                        default="outputs/MOT16-02_bytetrack_pred.txt",
                        help="ByteTrack predictions in MOTChallenge format")
    parser.add_argument("--iou_thr",  type=float, default=0.50)
    args = parser.parse_args()

    print(f"\nLoading ground truth from {args.seq_dir}...")
    gt = load_gt(args.seq_dir)
    print(f"  GT: {sum(len(v) for v in gt.values())} boxes across {len(gt)} frames")

    results = {}
    for name, path in [("SORT", args.pred_sort), ("ByteTrack", args.pred_byte)]:
        if not os.path.exists(path):
            print(f"\n[SKIP] {name}: prediction file not found at {path}")
            print(f"       Run the tracker with --save_txt flag first.")
            continue
        print(f"\nEvaluating {name}...")
        preds = load_predictions(path)
        metrics = evaluate_sequence(gt, preds, args.iou_thr)
        results[name] = metrics

    if not results:
        print("\nNo prediction files found. Run trackers first.")
        return

    # Print comparison table
    print("\n" + "="*65)
    print(f"{'Metric':<15}" + "".join(f"{k:>15}" for k in results))
    print("-"*65)
    for metric in ["MOTA","MOTP","IDF1_approx","TP","FP","FN","ID_SW","GT_objs"]:
        row = f"{metric:<15}"
        for k in results:
            val = results[k].get(metric, "N/A")
            row += f"{str(val):>15}"
        print(row)
    print("="*65)

    print("\nNOTES:")
    print("  MOTA:      higher is better (max=1.0)")
    print("  MOTP:      lower is better (0=perfect localisation)")
    print("  IDF1_approx: fraction of GT objects ever detected")
    print("  ID_SW:     identity switches — lower is better")

if __name__ == "__main__":
    main()
