"""
=============================================================================
run_bytetrack.py  —  YOLOv8 + ByteTrack Tracking Pipeline
Project: Vehicle & Pedestrian Tracking (CO5430 – Group G07)
Milestone: M4

ByteTrack is built into ultralytics — no extra library needed.

DIFFERENCE FROM SORT (M3):
  SORT:       IoU-only matching. Loses ID when person is occluded.
  ByteTrack:  Uses HIGH-confidence AND LOW-confidence detections in two
              separate association passes. Recovers tracks that were
              temporarily lost behind occlusions → fewer ID switches.

HOW TO RUN:
  python src/tracking/run_bytetrack.py \
      --model   models/best.pt \
      --seq_dir data/raw/MOT16/train/MOT16-02 \
      --output  outputs/MOT16-02_bytetrack.mp4

=============================================================================
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ── Colour palette: one stable colour per track ID ───────────────────────────
COLOURS = [
    (255,56,56),(255,157,151),(255,112,31),(255,178,29),(207,210,49),
    (72,249,10),(146,204,23),(61,219,134),(26,147,52),(0,212,187),
    (44,153,168),(0,194,255),(52,69,147),(100,115,255),(0,24,236),
    (132,56,255),(82,0,133),(203,56,255),(255,149,200),(255,55,199),
]

def get_colour(tid):
    return COLOURS[int(tid) % len(COLOURS)]


def draw_tracks(frame, result):
    """Draw ByteTrack results (stored in result.boxes with track IDs)."""
    if result.boxes is None or len(result.boxes) == 0:
        return frame

    for box in result.boxes:
        if box.id is None:          # no track ID assigned yet
            continue
        tid  = int(box.id.item())
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf.item())
        colour = get_colour(tid)
        label  = f"ID {tid} | ped  {conf:.2f}"

        cv2.rectangle(frame, (x1,y1), (x2,y2), colour, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), colour, -1)
        cv2.putText(frame, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)
    return frame


def run(model_path, seq_dir, output_path, conf=0.30, device="cpu"):
    from ultralytics import YOLO

    print(f"\n[1/5] Loading model: {model_path}")
    model = YOLO(model_path)

    img_dir = Path(seq_dir) / "img1"
    if not img_dir.exists():
        sys.exit(f"[ERROR] img1/ not found in {seq_dir}")

    frame_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    print(f"[2/5] Found {len(frame_paths)} frames")

    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]

    fps = 30
    seqinfo = Path(seq_dir) / "seqinfo.ini"
    if seqinfo.exists():
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(seqinfo)
        fps = int(cfg["Sequence"].get("frameRate", 30))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output_path,
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"[3/5] Output: {output_path}  ({w}x{h} @ {fps}fps)")
    print(f"[4/5] ByteTrack: conf_threshold={conf}, device={device}")
    print(f"[5/5] Tracking {len(frame_paths)} frames...\n")

    # Save predictions in MOTChallenge format for evaluate.py
    txt_path = str(output_path).replace(".mp4", "_pred.txt")

    t_start = time.time()
    total_dets = 0

    with open(txt_path, "w") as txt_file:
        for idx, fpath in enumerate(frame_paths):
        frame = cv2.imread(str(fpath))
        if frame is None:
            continue

        # ultralytics ByteTrack: tracker="bytetrack.yaml" activates it
        results = model.track(
            frame,
            conf=conf,
            tracker="bytetrack.yaml",   # ← the only difference from detection
            persist=True,               # keep track state between calls
            verbose=False,
            device=device,
        )
        result = results[0]

        n_dets = len(result.boxes) if result.boxes else 0
        total_dets += n_dets

        frame = draw_tracks(frame, result)

        # Overlay stats
        seq_name = Path(seq_dir).name
        n_tracks = sum(1 for b in result.boxes if b.id is not None) if result.boxes else 0
        cv2.putText(frame, f"{seq_name}  Frame {idx+1}/{len(frame_paths)} [ByteTrack]",
                    (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Active tracks: {n_tracks}",
                    (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,144), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Dets: {n_dets}",
                    (10,84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,0), 2, cv2.LINE_AA)

        writer.write(frame)

        # Save predictions in MOTChallenge format: frame,id,x,y,w,h,conf,-1,-1,-1
        if result.boxes:
            for box in result.boxes:
                if box.id is None:
                    continue
                tid = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w_box, h_box = x2 - x1, y2 - y1
                txt_file.write(f"{idx+1},{tid},{x1},{y1},{w_box},{h_box},1,-1,-1,-1\n")

            if (idx+1) % 50 == 0 or idx == 0:
                elapsed = time.time()-t_start
                speed = (idx+1)/elapsed
                eta = (len(frame_paths)-idx-1)/speed
                print(f"  Frame {idx+1:4d}/{len(frame_paths)} | "
                      f"tracks={n_tracks:3d} | speed={speed:.1f} fps | ETA={eta:.0f}s")

    writer.release()
    elapsed = time.time()-t_start
    print(f"\n=== DONE ===")
    print(f"  Total frames : {len(frame_paths)}")
    print(f"  Total dets   : {total_dets}")
    print(f"  Time elapsed : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Avg speed    : {len(frame_paths)/elapsed:.1f} fps")
    print(f"  Output saved : {Path(output_path).resolve()}")
    print(f"  Predictions  : {Path(txt_path).resolve()}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 + ByteTrack")
    parser.add_argument("--model",   default="models/best.pt")
    parser.add_argument("--seq_dir", default="data/raw/MOT16/train/MOT16-02")
    parser.add_argument("--output",  default="outputs/MOT16-02_bytetrack.mp4")
    parser.add_argument("--conf",    type=float, default=0.30)
    parser.add_argument("--device",  default="cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("YOLOv8 + ByteTrack  |  CO5430 Group G07")
    print("=" * 60)
    print(f"  Model   : {args.model}")
    print(f"  Sequence: {args.seq_dir}")
    print(f"  Output  : {args.output}")
    print("=" * 60)

    run(args.model, args.seq_dir, args.output, args.conf, args.device)

if __name__ == "__main__":
    main()
