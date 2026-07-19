# Data Preprocessing – Step-by-Step Guide
## Project: Vehicle & Pedestrian Tracking (CO5430 – Group G07)
## Milestone: M2 – Data Preparation

---

## 📦 Dataset: MOT16

MOT16 (Multiple Object Tracking 2016) is a standard benchmark dataset for multi-object tracking. It contains annotated video sequences of pedestrians and vehicles in various scenes.

| Property | Details |
|---|---|
| Sequences (train) | 7 sequences |
| Sequences (test) | 7 sequences |
| Resolution | Up to 1920 × 1080 |
| Annotation format | CSV (`gt.txt`) |
| Classes used by us | Pedestrian (id=1), Car (id=3) |

---

## 🗂️ Raw MOT16 Folder Structure

When you unzip the Kaggle download, you get:

```
MOT16/
├── train/
│   ├── MOT16-02/
│   │   ├── img1/           ← JPEG frames (000001.jpg, 000002.jpg, …)
│   │   ├── gt/
│   │   │   └── gt.txt      ← Ground truth annotations
│   │   └── seqinfo.ini     ← Sequence metadata (fps, resolution, etc.)
│   ├── MOT16-04/
│   └── … (7 sequences)
└── test/
    └── … (7 sequences, no gt.txt)
```

### What is `gt.txt`?

Each line = one object bounding box in one frame:

```
frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility, _
  1,    1,   912,    484,    97,        109,       1,    1,     0.86,       0
```

| Column | Meaning |
|---|---|
| frame | Frame number (1-based) |
| id | Track identity |
| bb_left, bb_top | Top-left pixel of bounding box |
| bb_width, bb_height | Box size in pixels |
| conf | 1 = valid, 0 = ignore |
| class | 1=pedestrian, 3=car, others |
| visibility | 0.0 (fully hidden) to 1.0 (fully visible) |

---

## 🔄 Preprocessing Steps Explained

### Step 1 – Validate Structure
Scan all sequence folders and confirm that `img1/`, `gt/gt.txt`, and `seqinfo.ini` exist. Skip test sequences (no ground truth).

### Step 2 – Parse `seqinfo.ini`
Read image resolution (width, height), FPS, and number of frames. This is needed for coordinate normalisation.

### Step 3 – Parse & Filter `gt.txt`
Read every row and apply filters:
- ✅ Keep only `conf == 1` (valid objects, not ignore regions)
- ✅ Keep only class `1` (pedestrian) and class `3` (car)
- ✅ Keep only objects with `visibility ≥ 0.3` (at least 30% visible)

This removes distractors, reflection artifacts, crowds, etc.

### Step 4 – Convert MOT16 → YOLO Bounding Boxes
MOT16 uses absolute pixel coordinates `(left, top, width, height)`.
YOLO needs normalised `(center_x, center_y, width, height)` in range `[0, 1]`.

```
center_x = (bb_left + bb_width/2)  / image_width
center_y = (bb_top  + bb_height/2) / image_height
norm_w   = bb_width  / image_width
norm_h   = bb_height / image_height
```

Class IDs are remapped: `pedestrian (1 → 0)`, `car (3 → 1)`.

### Step 5 – Train/Val Split
For each sequence, the **last 20% of frames** go to `val/`, the rest go to `train/`. This is temporal splitting — it avoids data leakage (val frames come after train frames chronologically).

### Step 6 – Write Label Files
For each frame image, one `.txt` file is created in `labels/train/` or `labels/val/`:

```
0 0.512000 0.461111 0.050521 0.100926   ← pedestrian
1 0.234375 0.388889 0.112500 0.175926   ← car
```

Empty `.txt` files are written for frames with no valid objects (required by YOLO).

### Step 7 – Generate `dataset.yaml`
YOLOv8 needs a config file pointing to train/val image folders and listing class names.

```yaml
path: data/yolo_dataset
train: images/train
val:   images/val
nc: 2
names: ["pedestrian", "car"]
```

### Step 8 – Verify
Run `verify_labels.py` to:
- Check every image has a label file
- Check all coordinates are in `[0, 1]`
- Print class distribution
- Draw boxes on sample images for visual confirmation

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install kaggle opencv-python
```

### 2. Download dataset from Kaggle
```bash
# First: put your kaggle.json in C:/Users/<you>/.kaggle/
python data/preprocessing/download_mot16.py
```
Or manually download from Kaggle and extract to `data/raw/MOT16/`.

### 3. Run preprocessing
```bash
python data/preprocessing/preprocess.py --mot_root data/raw/MOT16 --output data/yolo_dataset
```

### 4. Verify output
```bash
python data/preprocessing/verify_labels.py --dataset data/yolo_dataset --samples 5
```

---

## 📁 Output Structure

```
data/
└── yolo_dataset/
    ├── images/
    │   ├── train/   ← ~80% of frames from all train sequences
    │   └── val/     ← ~20% of frames
    ├── labels/
    │   ├── train/   ← .txt files, one per image
    │   └── val/
    ├── previews/    ← sample images with drawn bounding boxes
    └── dataset.yaml ← YOLOv8 config
```

---

## ⚠️ Notes

- `data/raw/` and `data/yolo_dataset/` are in `.gitignore` (do not commit raw data to GitHub)
- Only commit the scripts under `data/preprocessing/`
- Preprocessed stats should be logged in your lab notebook / report
