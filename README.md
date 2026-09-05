# Vehicle & Pedestrian Tracking in Video Sequences
## CO5430 — Image Processing & Computer Vision | Group G07

---

## Team Members
| Name | Index |
|------|-------|
| M.R.A. Rahman | E/23/282 |
| A. Piraveen | E/23/273 |
| M.M.M.A. Nular | E/23/249 |
| V. Thanush | E/23/392 |

---

## Project Overview

This project implements a **multi-object tracking (MOT)** system for detecting and tracking **pedestrians** in video sequences. We use:

1. **YOLOv8n** — object detector fine-tuned on the MOT16 dataset
2. **SORT** — Simple Online and Realtime Tracking (Kalman Filter + Hungarian Algorithm)
3. **ByteTrack** — improved tracker with two-pass association for better occlusion handling

> **Note:** Although the project topic covers vehicles and pedestrians, the MOT16 dataset used for training is pedestrian-focused. Car annotations did not survive the visibility filter (`vis ≥ 0.3`), so the trained model detects **pedestrians only**.

**Dataset:** [MOT16 Benchmark](https://motchallenge.net/data/MOT16/) — 7 training sequences of urban street scenes

---

## Results (YOLOv8n on MOT16 validation split)

| Metric | Value |
|--------|-------|
| mAP@50 | **0.9058** |
| mAP@50-95 | **0.5915** |
| Precision | **0.9237** |
| Recall | **0.8308** |
| Training epochs | 50 |
| Training hardware | Google Colab T4 GPU |

---

## Repository Structure

```
vehicle-pedestrian-tracking/
├── data/
│   ├── preprocessing/
│   │   ├── preprocess.py        ← MOT16 → YOLO format converter (M2)
│   │   ├── verify_labels.py     ← Label sanity checker + visualiser
│   │   └── README.md
│   └── yolo_dataset/
│       ├── dataset.yaml         ← Local training config
│       └── dataset_colab.yaml   ← Google Colab training config
├── src/
│   └── tracking/
│       ├── sort_tracker.py      ← SORT implementation (Kalman + Hungarian)
│       ├── run_tracker.py       ← YOLOv8 + SORT tracking pipeline (M3)
│       ├── run_bytetrack.py     ← YOLOv8 + ByteTrack pipeline (M4)
│       ├── evaluate.py          ← MOTA/IDF1/MOTP evaluation (M5)
│       └── README.md            ← Detailed M3 instructions
├── models/
│   └── best.pt                  ← Trained YOLOv8n weights (not in git — too large)
├── outputs/
│   └── *.mp4                    ← Tracked videos (not in git — too large)
├── Docs/
│   ├── G07_M3_Presentation_v2.pptx
│   └── G07_Project_progress.pdf
├── ROADMAP.md                   ← Milestone timeline
└── README.md                    ← This file
```

---

## How to Run

### Prerequisites
```bash
pip install ultralytics filterpy scipy torch torchvision opencv-python numpy
```

### Step 1 — Preprocess MOT16 Data
```bash
python data/preprocessing/preprocess.py \
    --mot_root data/raw/MOT16 \
    --output   data/yolo_dataset
```

### Step 2 — Train YOLOv8 (Google Colab recommended)
Use the cells in `src/training/colab_cells_reference.py`.
Or locally (slow — CPU only):
```bash
yolo train model=yolov8n.pt data=data/yolo_dataset/dataset.yaml epochs=50 imgsz=640
```

### Step 3 — Run SORT Tracker
```bash
python src/tracking/run_tracker.py \
    --model   models/best.pt \
    --seq_dir data/raw/MOT16/train/MOT16-02 \
    --output  outputs/MOT16-02_sort.mp4
```

### Step 4 — Run ByteTrack Tracker
```bash
python src/tracking/run_bytetrack.py \
    --model   models/best.pt \
    --seq_dir data/raw/MOT16/train/MOT16-02 \
    --output  outputs/MOT16-02_bytetrack.mp4
```

### Step 5 — Evaluate (MOTA / IDF1)
```bash
python src/tracking/evaluate.py \
    --seq_dir   data/raw/MOT16/train/MOT16-02 \
    --pred_sort outputs/MOT16-02_sort_pred.txt \
    --pred_byte outputs/MOT16-02_bytetrack_pred.txt
```

---

## Methodology

### Detection — YOLOv8n
- Architecture: CSP-DarkNet backbone + PAN-FPN neck + decoupled detection head
- Pretrained on COCO (80 classes), fine-tuned on MOT16 (pedestrian class only)
- Input: 640×640 px | Output: [x1,y1,x2,y2, confidence, class_id]

### Tracking — SORT (M3 Baseline)
- **Kalman Filter**: models each track as constant-velocity motion, predicts next position
- **Hungarian Algorithm**: optimally matches predictions to new detections via IoU cost matrix
- Parameters: `max_age=3`, `min_hits=3`, `iou_threshold=0.45`

### Tracking — ByteTrack (M4 Upgrade)
- Two-pass association: first matches high-confidence detections, then uses remaining low-confidence detections to recover temporarily lost tracks
- Built into `ultralytics` — activated via `tracker="bytetrack.yaml"`
- Advantage: fewer ID switches during occlusion

### Evaluation Metrics
| Metric | Formula | Meaning |
|--------|---------|---------|
| **MOTA** | 1 - (FP+FN+IDS)/GT | Overall tracking accuracy |
| **IDF1** | TP_id / (2·GT + FP) | Identity-consistent detection rate |
| **MOTP** | avg(1-IoU) | Localisation precision |
| **ID Sw** | count | Identity switch count |

---

## Dataset

**MOT16 Training Sequences used:**
| Sequence | Frames | Resolution | FPS | Scene |
|----------|--------|------------|-----|-------|
| MOT16-02 | 600 | 1920×1080 | 30 | Venice plaza |
| MOT16-04 | 1050 | 1920×1080 | 30 | ETH Bahnhof |
| MOT16-05 | 837 | 640×480 | 14 | Venice people |
| MOT16-09 | 525 | 1920×1080 | 30 | ETH Sunnyday |
| MOT16-10 | 654 | 1920×1080 | 30 | ETH Crossing |
| MOT16-11 | 900 | 1920×1080 | 30 | PETS09-S2L1 |
| MOT16-13 | 750 | 1920×1080 | 25 | TownCentre |

**Preprocessing:**
- Filter: pedestrian class (id=1), conf≥1, visibility≥0.3
- Convert: MOT16 pixel coords → YOLO normalized [cx,cy,w,h]
- Split: last 20% frames → val, first 80% → train
- **Result:** 4,254 train images | 1,062 val images | 79,799 pedestrian labels

---

## Challenges

1. **No car annotations**: MOT16 is pedestrian-focused; car labels did not survive the visibility filter
2. **Occlusion**: crowded scenes cause ID switches in SORT — addressed in M4 with ByteTrack
3. **Small objects**: distant pedestrians (~10px) are hard to detect with YOLOv8n
4. **CPU-only inference**: no CUDA GPU on local machine; training done on Google Colab T4

---

## References

- Bewley et al. (2016). *Simple Online and Realtime Tracking (SORT)*. ICASSP 2016.
- Zhang et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. ECCV 2022.
- Jocher et al. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
- Dendorfer et al. (2020). *MOT16: A Benchmark for Multi-Object Tracking*. MOTChallenge.
