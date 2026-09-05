"""
=============================================================================
run_tracker.py  —  YOLOv8 + SORT End-to-End Tracking Pipeline
Project: Vehicle & Pedestrian Tracking (CO5430 – Group G07)
Milestone: M3
=============================================================================

WHAT THIS SCRIPT DOES:
  1. Loads a trained YOLOv8 model (best.pt from Colab)
  2. Runs it frame-by-frame on a MOT16 sequence
  3. Passes detections into the SORT tracker
  4. Draws coloured bounding boxes + track IDs on each frame
  5. Saves output as a video file (.mp4)

HOW TO RUN:
  python src/tracking/run_tracker.py \\
      --model   models/best.pt \\
      --seq_dir data/raw/MOT16/train/MOT16-02 \\
      --output  outputs/MOT16-02_tracked.mp4

DEPENDENCIES (install first):
  pip install ultralytics filterpy scipy

=============================================================================
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path so we can import sort_tracker
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "tracking"))

from sort_tracker import Sort  # our SORT implementation


# =============================================================================
# COLOUR PALETTE  —  one distinct colour per track ID
# =============================================================================
COLOURS = [
    (255, 56,  56),   # red
    (255, 157, 151),  # pink
    (255, 112, 31),   # orange
    (255, 178, 29),   # yellow
    (207, 210, 49),   # lime
    (72,  249, 10),   # green
    (146, 204, 23),   # yellow-green
    (61,  219, 134),  # mint
    (26,  147, 52),   # dark green
    (0,   212, 187),  # teal
    (44,  153, 168),  # cyan-dark
    (0,   194, 255),  # light blue
    (52,  69,  147),  # navy
    (100, 115, 255),  # blue
    (0,   24,  236),  # deep blue
    (132, 56,  255),  # purple
    (82,  0,   133),  # dark purple
    (203, 56,  255),  # magenta
    (255, 149, 200),  # light pink
    (255, 55,  199),  # hot pink
]

def get_colour(track_id: int) -> tuple:
    """Return a stable RGB colour for a given track ID."""
    return COLOURS[int(track_id) % len(COLOURS)]


# =============================================================================
# DRAW  —  render one frame's tracks onto the image
# =============================================================================
# Model detects pedestrians only (fine-tuned on MOT16 pedestrian class)
CLASS_NAMES = ["pedestrian"]

def draw_tracks(frame: np.ndarray, tracks: np.ndarray, class_id: int = 0) -> np.ndarray:
    """
    Draw bounding boxes + IDs on the frame.

    tracks : (N, 5) — [x1, y1, x2, y2, track_id]
    """
    for track in tracks:
        x1, y1, x2, y2, tid = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
        colour = get_colour(tid)
        label  = f"ID {tid} | {CLASS_NAMES[class_id]}"

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Label background pill
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)  # filled
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# =============================================================================
# MAIN TRACKING LOOP
# =============================================================================
def run(model_path: str,
        seq_dir: str,
        output_path: str,
        conf_threshold: float = 0.30,
        iou_threshold: float  = 0.45,
        max_age: int          = 3,
        min_hits: int         = 3,
        device: str           = "cpu"):
    """
    Full tracking pipeline for one MOT16 sequence.

    Parameters
    ----------
    model_path     : path to best.pt
    seq_dir        : path to MOT16 sequence folder (contains img1/)
    output_path    : where to save the output .mp4
    conf_threshold : YOLOv8 confidence cutoff
    iou_threshold  : SORT IoU threshold for matching
    max_age        : SORT — frames before a lost track is deleted
    min_hits       : SORT — min detections before a track is reported
    device         : 'cpu' or '0' (GPU id)
    """
    from ultralytics import YOLO

    # ── 1. Load model ───────────────────────────────────────────────────────
    print(f"\n[1/5] Loading model: {model_path}")
    model = YOLO(model_path)
    model.to(device)

    # ── 2. Collect frame paths ──────────────────────────────────────────────
    img_dir = Path(seq_dir) / "img1"
    if not img_dir.exists():
        sys.exit(f"[ERROR] img1/ not found in {seq_dir}")

    frame_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not frame_paths:
        sys.exit(f"[ERROR] No images found in {img_dir}")

    print(f"[2/5] Found {len(frame_paths)} frames in {img_dir}")

    # ── 3. Set up video writer ──────────────────────────────────────────────
    first_frame = cv2.imread(str(frame_paths[0]))
    h, w = first_frame.shape[:2]

    # Read FPS from seqinfo.ini if available
    fps = 30
    seqinfo = Path(seq_dir) / "seqinfo.ini"
    if seqinfo.exists():
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(seqinfo)
        fps = int(cfg["Sequence"].get("frameRate", 30))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    print(f"[3/5] Output video: {output_path}  ({w}x{h} @ {fps}fps)")

    # ── 4. Initialise SORT tracker ──────────────────────────────────────────
    tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    print(f"[4/5] SORT tracker: max_age={max_age}, min_hits={min_hits}, iou_thr={iou_threshold}")

    # Optional: save predictions as MOTChallenge .txt for evaluation
    txt_path = str(output_path).replace(".mp4", "_pred.txt")
    print(f"      Predictions will be saved to: {txt_path}")

    # ── 5. Frame-by-frame loop ──────────────────────────────────────────────
    print(f"[5/5] Tracking {len(frame_paths)} frames...\n")
    t_start = time.time()
    total_dets = 0

    with open(txt_path, "w") as txt_file:
        for frame_idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            # ── 5a: Detect with YOLOv8 ─────────────────────────────────────
            results = model(frame, conf=conf_threshold, verbose=False)[0]

            # Convert YOLOv8 results → numpy [x1, y1, x2, y2, conf]
            dets_for_sort = np.empty((0, 5))
            if results.boxes is not None and len(results.boxes) > 0:
                boxes  = results.boxes.xyxy.cpu().numpy()   # (N, 4) pixel coords
                confs  = results.boxes.conf.cpu().numpy()   # (N,)
                dets_for_sort = np.hstack([boxes, confs.reshape(-1, 1)])
                total_dets += len(dets_for_sort)

            # ── 5b: Update SORT tracker ─────────────────────────────────────
            #  Output: (M, 5)  [x1, y1, x2, y2, track_id]
            tracks = tracker.update(dets_for_sort)

            # ── 5c: Draw tracks ─────────────────────────────────────────────
            frame = draw_tracks(frame, tracks, class_id=0)

            # ── 5d: Overlay stats ───────────────────────────────────────────
            seq_name = Path(seq_dir).name
            n_tracks = len(tracks)
            cv2.putText(frame, f"{seq_name}  Frame {frame_idx+1}/{len(frame_paths)}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Active tracks: {n_tracks}",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 144), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Dets: {len(dets_for_sort)}",
                        (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)

            # ── 5e: Write frame ─────────────────────────────────────────────
            writer.write(frame)

            # ── 5f: Save predictions (MOTChallenge format) ───────────────────
            # format: frame,id,x,y,w,h,conf,-1,-1,-1
            for track in tracks:
                x1,y1,x2,y2,tid = int(track[0]),int(track[1]),int(track[2]),int(track[3]),int(track[4])
                w_box = x2-x1
                h_box = y2-y1
                txt_file.write(f"{frame_idx+1},{tid},{x1},{y1},{w_box},{h_box},1,-1,-1,-1\n")

            # Progress print every 50 frames
            if (frame_idx + 1) % 50 == 0 or frame_idx == 0:
                elapsed = time.time() - t_start
                fps_proc = (frame_idx + 1) / elapsed
                eta = (len(frame_paths) - frame_idx - 1) / fps_proc
                print(f"  Frame {frame_idx+1:4d}/{len(frame_paths)} | "
                      f"tracks={n_tracks:3d} | "
                      f"speed={fps_proc:.1f} fps | "
                      f"ETA={eta:.0f}s")

    writer.release()
    elapsed = time.time() - t_start
    print(f"\n=== DONE ===")
    print(f"  Total frames   : {len(frame_paths)}")
    print(f"  Total dets     : {total_dets}")
    print(f"  Time elapsed   : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Avg speed      : {len(frame_paths)/elapsed:.1f} fps")
    print(f"  Output saved   : {Path(output_path).resolve()}")
    print(f"  Predictions txt: {Path(txt_path).resolve()}")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 + SORT tracker for MOT16 sequences"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best.pt",
        help="Path to trained YOLOv8 weights (best.pt)",
    )
    parser.add_argument(
        "--seq_dir",
        type=str,
        default="data/raw/MOT16/train/MOT16-02",
        help="Path to MOT16 sequence folder (must contain img1/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/MOT16-02_tracked.mp4",
        help="Output video path",
    )
    parser.add_argument("--conf",    type=float, default=0.30,  help="Detection confidence threshold")
    parser.add_argument("--iou",     type=float, default=0.45,  help="SORT IoU threshold")
    parser.add_argument("--max_age", type=int,   default=3,     help="SORT max frames before track deletion")
    parser.add_argument("--min_hits",type=int,   default=3,     help="SORT min detections before track is reported")
    parser.add_argument("--device",  type=str,   default="cpu", help="Device: 'cpu' or '0' (GPU)")

    args = parser.parse_args()

    print("=" * 60)
    print("YOLOv8 + SORT Tracker  |  CO5430 Group G07")
    print("=" * 60)
    print(f"  Model   : {args.model}")
    print(f"  Sequence: {args.seq_dir}")
    print(f"  Output  : {args.output}")
    print(f"  Device  : {args.device}")
    print("=" * 60)

    run(
        model_path    = args.model,
        seq_dir       = args.seq_dir,
        output_path   = args.output,
        conf_threshold= args.conf,
        iou_threshold = args.iou,
        max_age       = args.max_age,
        min_hits      = args.min_hits,
        device        = args.device,
    )


if __name__ == "__main__":
    main()
