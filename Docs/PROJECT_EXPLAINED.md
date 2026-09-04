# Project Explained — A to Z
## Vehicle & Pedestrian Tracking | CO5430 Group G07

> This document explains the **complete project from scratch** — how the data comes in, what format it is, who drew the boxes, how the code works, what each script does, and what the final results mean.

---

## Table of Contents
1. [What Problem Are We Solving?](#1-what-problem-are-we-solving)
2. [Where the Data Comes From](#2-where-the-data-comes-from)
3. [Who Drew the Boxes?](#3-who-drew-the-boxes)
4. [What gt.txt Looks Like](#4-what-gttxt-looks-like)
5. [What preprocess.py Does](#5-what-preprocesspy-does-m2)
6. [What verify_labels.py Does](#6-what-verify_labelspy-does)
7. [What dataset.yaml Does](#7-what-datasetyaml-does)
8. [Training YOLOv8 on Google Colab](#8-training-yolov8-on-google-colab-m3-part-1)
9. [What run_tracker.py Does (SORT)](#9-what-run_trackerpy-does-sort--m3)
10. [What the sort_pred.txt File Is](#10-what-the-sort_predtxt-file-is)
11. [What run_bytetrack.py Does (ByteTrack)](#11-what-run_bytetrackpy-does-m4-upgrade)
12. [What evaluate.py Does](#12-what-evaluatepy-does-m5)
13. [Final Results](#13-final-results)
14. [Who Did What](#14-who-did-what)

---

## 1. What Problem Are We Solving?

Imagine a CCTV camera recording a busy street. Every second it captures 30 photographs (called **frames**). The question is:

> **"In this video, find every person, put a box around them, and track each one so that Person #5 in frame 1 is still called Person #5 in frame 200 — even if they walk behind a car or get hidden in a crowd."**

That is called **Multi-Object Tracking (MOT)**. It has two stages:

| Stage | Task | Tool |
|-------|------|------|
| **Detection** | Find where people are in each frame — draw a box | YOLOv8n |
| **Tracking** | Link those boxes across frames so each person keeps the same ID | SORT / ByteTrack |

The final output is a **video where every person has a coloured bounding box with a number** (e.g., "ID 4 | pedestrian") that stays consistent as they walk through the scene.

---

## 2. Where the Data Comes From

### The Dataset: MOT16

**MOT16** is a publicly available research benchmark published by motchallenge.net. A group of researchers physically placed cameras at busy public locations, recorded hours of footage, then hired human annotators to manually label every person in every frame.

**You downloaded it from the internet.** It is not your original footage — it is a standard benchmark that every tracking research paper in the world uses to compare results fairly.

**Locations recorded:**

| Sequence | Frames | Resolution | FPS | Scene |
|----------|--------|------------|-----|-------|
| MOT16-02 | 600 | 1920×1080 | 30 | Venice plaza (Italy) |
| MOT16-04 | 1,050 | 1920×1080 | 30 | ETH Bahnhof (Zurich) |
| MOT16-05 | 837 | 640×480 | 14 | Venice pedestrian street |
| MOT16-09 | 525 | 1920×1080 | 30 | ETH Sunnyday |
| MOT16-10 | 654 | 1920×1080 | 30 | ETH Crossing |
| MOT16-11 | 900 | 1920×1080 | 30 | PETS09 S2L1 (indoor mall) |
| MOT16-13 | 750 | 1920×1080 | 25 | Town Centre (UK street) |

**Total: 5,316 frames across 7 sequences.**

### What the Download Looks Like

```
data/raw/MOT16/train/
├── MOT16-02/
│   ├── img1/
│   │   ├── 000001.jpg   ← a real photograph of Venice plaza (1920×1080 pixels)
│   │   ├── 000002.jpg   ← same scene, 1/30th of a second later
│   │   ├── ...
│   │   └── 000600.jpg   ← last frame (600 photos total)
│   ├── gt/
│   │   └── gt.txt       ← the human-drawn annotations for all 600 frames
│   └── seqinfo.ini      ← metadata (resolution, fps, total frame count)
├── MOT16-04/
│   └── ...
```

It is **NOT a video file**. It is a folder of JPEG images, one per frame, plus one annotation file.

---

## 3. Who Drew the Boxes?

**Human annotators hired by the MOT16 research team drew every single box.**

You did NOT draw any boxes. The file `gt.txt` is called the **Ground Truth (GT)** — the authoritative, human-verified correct answer.

The annotators used professional annotation software to:
1. Watch each video frame by frame
2. Draw a rectangle around every pedestrian, car, and cyclist
3. Assign each object a unique ID number that stays the same across frames

This process took hundreds of hours for just these 7 sequences. The Ground Truth is used for two purposes:
- **Training:** teach YOLOv8 what pedestrians look like
- **Evaluation:** check how well our tracker performed compared to the human-verified truth

---

## 4. What gt.txt Looks Like

Every line in `gt.txt` describes one object in one frame:

```
frame,  id,  left,  top,  width, height, conf, class, visibility
    1,   1,   912,  484,     97,    109,    0,     7,       1.00
    1,   2,   445,  422,    112,    285,    1,     1,       0.86
    1,   3,   580,  440,     90,    270,    1,     1,       0.74
```

**Reading line 2 in plain English:**
- `frame=1` → this describes something in Frame 1
- `id=2` → it is Person #2 (same ID used in every frame where this person appears)
- `left=445, top=422` → the TOP-LEFT CORNER of the box is at pixel (445, 422)
- `width=112, height=285` → the box is 112px wide, 285px tall
- `conf=1` → this is a **real annotation** (conf=0 means "ignore region" — skip it)
- `class=1` → class 1 = pedestrian (class 3 = car, class 7 = static person)
- `visibility=0.86` → 86% of the person is visible, 14% is hidden behind something

**Filtering rules we applied:**

| Rule | Reason |
|------|--------|
| Keep only `conf == 1` | Remove "ignore regions" (areas where detections are not penalized) |
| Keep only `class == 1` (pedestrian) or `class == 3` (car) | Ignore cyclists, static people, distractors |
| Keep only `visibility >= 0.3` | Skip objects that are more than 70% hidden |

After filtering: **79,799 pedestrian annotations** survived. Car annotations = 0 (MOT16 is pedestrian-focused and the few cars present did not survive the visibility filter).

---

## 5. What preprocess.py Does (M2)

**The problem:** MOT16 format and YOLO format are completely different. YOLOv8 cannot read `gt.txt` directly. They must be converted.

### Format Comparison

| Property | MOT16 (raw) | YOLO (required) |
|----------|-------------|-----------------|
| Box anchor | **Top-left corner** in pixels | **Center** of box, normalized 0.0–1.0 |
| Example coordinates | `left=912, top=484, w=97, h=109` | `0.5003  0.4986  0.0505  0.1009` |
| Number of files | One `gt.txt` for all 600 frames | One `.txt` file per frame (600 files) |
| Class numbering | 1=ped, 3=car, 7=static… | Must start from 0: 0=ped, 1=car |

### The Coordinate Conversion Math

**Example: MOT16-02, Frame 1, Person 2 (image size: 1920×1080)**

```
MOT16 input: left=445, top=422, width=112, height=285

Step A — Find the center point (not the corner):
  center_x_pixel = left + width/2  = 445 + 112/2 = 445 + 56  = 501 pixels
  center_y_pixel = top + height/2  = 422 + 285/2 = 422 + 142 = 564 pixels

Step B — Normalize by dividing by image dimensions:
  cx = 501 / 1920 = 0.2609   (fraction of image width)
  cy = 564 / 1080 = 0.5222   (fraction of image height)
  nw = 112 / 1920 = 0.0583
  nh = 285 / 1080 = 0.2639

Step C — Remap class ID:
  MOT16 class 1 (pedestrian) → YOLO class 0

YOLO output line: 0  0.2609  0.5222  0.0583  0.2639
                  ^    ^       ^        ^       ^
                class  cx      cy       w       h
```

All five numbers are between 0.0 and 1.0. This is what YOLOv8 trains on.

### What preprocess.py Produces

```
data/yolo_dataset/
├── images/
│   ├── train/
│   │   ├── MOT16-02_000001.jpg   ← copy of the original photo
│   │   ├── MOT16-02_000002.jpg
│   │   └── ...  (4,254 files)
│   └── val/
│       └── ...  (1,062 files)
├── labels/
│   ├── train/
│   │   ├── MOT16-02_000001.txt   ← converted annotations for that frame
│   │   ├── MOT16-02_000002.txt
│   │   └── ...  (4,254 files)
│   └── val/
│       └── ...  (1,062 files)
└── dataset.yaml                   ← config file pointing to these folders
```

The sequence name `MOT16-02_` is prepended to each filename because all 7 sequences have a `000001.jpg` — without the prefix they would overwrite each other.

**Train/Val split logic:** The **last 20% of frames** of each sequence go to validation. For MOT16-02: frames 1–480 → train, frames 481–600 → val. This is important because consecutive video frames look almost identical — using random splitting would leak future frames into training (data leakage).

---

## 6. What verify_labels.py Does

After preprocessing, how do you know the math was correct? This script reverses the normalization and draws the YOLO boxes back onto the original images so you can visually confirm they land in the right place.

**Process:**
```
YOLO line: 0  0.2609  0.5222  0.0583  0.2639

Reverse math:
  cx_px = 0.2609 × 1920 = 501 pixels
  cy_px = 0.5222 × 1080 = 564 pixels
  w_px  = 0.0583 × 1920 = 112 pixels
  h_px  = 0.2639 × 1080 = 285 pixels
  x1    = 501 - 112/2   = 445
  y1    = 564 - 285/2   = 422
```

It then draws a green rectangle at those pixel coordinates on the image and saves 5 sample images to `data/yolo_dataset/previews/`. The green boxes you see on the Venice plaza and shopping mall images are from this script — they are **ground truth visualised**, not model predictions.

---

## 7. What dataset.yaml Does

A small config file that tells YOLOv8 where to find the training data:

```yaml
path: C:\Users\abdul\Desktop\vehicle-pedestrian-tracking\data\yolo_dataset
train: images/train
val:   images/val
nc: 2
names: ['pedestrian', 'car']
```

- `nc: 2` — number of classes (2: pedestrian and car)
- `names` — what class 0 and class 1 are called
- Without this file, YOLOv8 has no idea where your images or labels are

A second version `dataset_colab.yaml` has the Google Colab path (`/content/drive/MyDrive/G07/...`) for training on the cloud.

---

## 8. Training YOLOv8 on Google Colab (M3 Part 1)

### What YOLOv8 Is

YOLOv8 (You Only Look Once, version 8) is a **convolutional neural network** — a mathematical function with approximately 3.2 million learnable parameters (numbers). It takes a 640×640 pixel image as input and outputs bounding boxes with class labels and confidence scores.

It was NOT built specifically for pedestrian tracking. It came **pretrained on COCO** — a large dataset of 80 everyday objects (cats, cars, chairs, people, etc.) collected from the internet. We **fine-tuned** it: start from those pretrained weights and continue training on our MOT16 data so it specialises in pedestrians from street surveillance angles.

### What "Training" Actually Means — Step by Step

For each of 50 **epochs** (one complete pass through all 4,254 training images):

```
1. Take a batch of 16 training images

2. Run each image through the network → get predicted boxes
   e.g., "I think there's a pedestrian at [0.50, 0.50, 0.05, 0.10] with 0.71 confidence"

3. Compare predictions to the ground truth labels (.txt files)
   "The actual pedestrian was at [0.52, 0.51, 0.05, 0.10] — you were off by 0.02"

4. Compute loss (how wrong the prediction was):
   - box_loss   → how far the box coordinates are from ground truth
   - cls_loss   → how wrong the class probability is
   - dfl_loss   → distribution focal loss (box shape quality)

5. Backpropagation — calculates how to adjust each of the 3.2M parameters
   to make predictions slightly more accurate next time

6. Update all parameters and move to the next batch

Repeat: 266 batches × 50 epochs = 13,300 weight updates total
```

**Your training results (Epoch 1 vs Epoch 50):**

```
Metric        Epoch 1    Epoch 50    Change
──────────────────────────────────────────
box_loss       1.388  →   0.650     −53%  ✅
cls_loss       1.189  →   0.367     −69%  ✅
mAP@50         0.832  →   0.892     +7%   ✅
Precision      0.875  →   0.938     +7%   ✅
Recall         0.728  →   0.809     +11%  ✅
```

**Validation results (clean evaluation after training):**

| Metric | Value | Meaning |
|--------|-------|---------|
| mAP@50 | **0.9058** | Detection correct at IoU≥50% — excellent |
| mAP@50-95 | **0.5915** | Stricter averaged metric |
| Precision | **0.9237** | 92% of predicted boxes are correct |
| Recall | **0.8308** | Model finds 83% of all real pedestrians |

**The output of training: `models/best.pt`** — a 5.9 MB file containing all 3.2M trained parameter values. This is the trained model.

Training was done on **Google Colab T4 GPU** (~30 minutes). On a CPU it would take 8–12 hours.

---

## 9. What run_tracker.py Does (SORT — M3)

This is the M3 main script. It runs after training. It takes:
- The trained model (`models/best.pt`)
- A raw MOT16 sequence folder (`data/raw/MOT16/train/MOT16-02/`)

And produces:
- A tracked video (`outputs/MOT16-02_tracked.mp4`) with coloured boxes and ID numbers
- A prediction `.txt` file for evaluation

### What Happens for Each of the 600 Frames

**Using Frame 50 as an example:**

#### Step A: Load the JPEG
```python
frame = cv2.imread("...MOT16-02/img1/000050.jpg")
# frame = a 1920×1080 array of pixel RGB values
```

#### Step B: YOLOv8 Detection
```python
results = model(frame, conf=0.30)
```
YOLOv8 internally resizes the image to 640×640, runs it through the neural network in ~100ms, and outputs raw detection boxes. Example output for Frame 50:
```
[x1=360, y1=437, x2=457, y2=592, conf=0.89, class=0]   ← pedestrian
[x1=580, y1=427, x2=598, y2=473, conf=0.78, class=0]   ← pedestrian
[x1=972, y1=443, x2=1008, y2=546, conf=0.82, class=0]  ← pedestrian
... (17 total detections in this frame)
```
**These boxes have NO ID numbers yet.** The model has no memory — it only sees one image at a time.

#### Step C: SORT Tracker — Kalman Filter Prediction
SORT maintains a list of "active tracks" — one per person currently being tracked. Each track has its own **Kalman Filter** that stores:
- Estimated current position
- Estimated current velocity (pixels per frame)

Before looking at new detections, each Kalman Filter **predicts** where its person should be now:
```
Track ID 4:  last known center=(400,480), velocity=(+5,0)px/frame
             → predicted position this frame: (405, 480)

Track ID 7:  last known center=(579,428), velocity=(+2,−1)px/frame
             → predicted position: (581, 427)
```

#### Step D: SORT Tracker — Hungarian Algorithm Matching

SORT builds an **IoU matrix** — for every predicted position vs every new detection, it computes how much they overlap:

```
                  Det A    Det B    Det C    Det D
Track ID 4 pred:  0.85     0.03     0.00     0.01
Track ID 7 pred:  0.04     0.91     0.00     0.02
Track ID 11 pred: 0.00     0.01     0.00     0.72
(new person)      —        —        1.00     —
```

The **Hungarian Algorithm** finds the globally optimal assignment (minimum total cost = maximum total IoU):

```
Det A → Track ID 4   (IoU=0.85 → MATCH → ID stays 4, Kalman Filter updated)
Det B → Track ID 7   (IoU=0.91 → MATCH → ID stays 7)
Det D → Track ID 11  (IoU=0.72 → MATCH → ID stays 11)
Det C → no match     → NEW track created → ID 25 assigned
```

If a track has no match for 3 consecutive frames (`max_age=3`), it is deleted.  
If a new track hasn't been confirmed for 3 frames (`min_hits=3`), it isn't shown yet.

#### Step E: Draw and Write the Frame
For each confirmed track, draw a coloured rectangle with label `ID 4 | pedestrian`. Each ID always gets the same colour from a 20-colour palette. Save the frame to the output video.

#### Step F: Save Prediction to .txt
```
50, 4, 360, 437, 97, 155, 1, -1, -1, -1
↑   ↑   ↑    ↑   ↑    ↑
frame  ID  left top width height
```
This MOTChallenge format line is written to `outputs/MOT16-02_sort_pred.txt` for evaluation.

---

## 10. What the sort_pred.txt File Is

Looking at the first few lines of your actual file:
```
1, 19, 560, 440,  24,  45, 1, -1, -1, -1
1, 18, 579, 428,  18,  45, 1, -1, -1, -1
1, 17,1077, 479,  30, 115, 1, -1, -1, -1
1,  4, 440, 442, 114, 282, 1, -1, -1, -1
1,  1,1338, 418, 160, 360, 1, -1, -1, -1
```

**Format:** `frame, track_id, left, top, width, height, conf, -1, -1, -1`

**Reading line 1:** `1, 19, 560, 440, 24, 45, 1, -1, -1, -1`
- Frame **1**
- Track ID **19** — this person was assigned ID 19 by SORT
- Box: left=560, top=440, width=24, height=45 — a small distant pedestrian (24px wide!)
- conf=**1** — this track is confirmed
- `-1, -1, -1` — placeholder columns for 3D position (not used in 2D tracking)

**File statistics:**
- 8,825 lines = 8,825 person-frame detections across 600 frames
- Average: ~14.7 tracked people per frame
- This file is used by `evaluate.py` to compute MOTA

---

## 11. What run_bytetrack.py Does (M4 Upgrade)

ByteTrack fixes SORT's biggest weakness: **losing track ID when a person is briefly hidden (occluded)**.

### The Problem SORT Has

```
Frame 100: Person with ID 4 is fully visible ✅
Frame 101: Person walks behind a pillar
           → detector returns a low confidence box (conf=0.25)
           → SORT ignores it (our threshold is 0.30)
           → track ID 4 has no match → eventually deleted
Frame 103: Person reappears
           → detector sees a new detection
           → SORT creates a NEW track → assigns ID 47
           → THE SAME PERSON is now ID 4 in frame 100 and ID 47 in frame 103
                                 ↑ this is called an ID SWITCH — bad!
```

### How ByteTrack Fixes It

ByteTrack uses a **two-pass association**:

```
Pass 1 (same as SORT):
  Match high-confidence detections (conf ≥ 0.30) to active tracks
  → handles most normal cases

Pass 2 (ByteTrack's innovation):
  Take tracks that were NOT matched in Pass 1 (likely temporarily hidden)
  + low-confidence detections (conf 0.10–0.30)
  Try to match them together
  → Person behind pillar had conf=0.25 → used in Pass 2 → Track ID 4 KEPT
```

**The code difference is just one line:**
```python
# SORT (run_tracker.py):
results = model(frame, conf=0.30, verbose=False)
tracks  = sort_tracker.update(detections)

# ByteTrack (run_bytetrack.py):
results = model.track(frame, conf=0.30, tracker="bytetrack.yaml", persist=True)
# .track() activates ultralytics' built-in ByteTrack — no extra library needed
# persist=True  → keeps the track state between frames
```

### Result on MOT16-02

```
ID Switches:   SORT=98   ByteTrack=40   → 59% fewer with ByteTrack ✅
```

---

## 12. What evaluate.py Does (M5)

After running both trackers, you have two prediction `.txt` files. `evaluate.py` compares them against the ground truth `gt.txt` to produce standard tracking metrics.

### How It Works

For every frame, for every GT pedestrian box, the script checks all predicted boxes:

| Situation | Label |
|-----------|-------|
| Predicted box overlaps GT ≥ 50% AND same track ID as last time | **True Positive (TP)** |
| Predicted box where no GT pedestrian exists | **False Positive (FP)** |
| GT pedestrian with no matching predicted box | **False Negative (FN)** |
| Predicted box matches GT but track ID changed vs last frame | **ID Switch (IDS)** |

### The MOTA Formula

```
MOTA = 1 − (FP + FN + IDS) / total_GT_objects

SORT:      = 1 − (1338 + 1354 + 98) / 8840  = 1 − 0.316  = 0.684
ByteTrack: = 1 − (1827 + 1184 + 40) / 8840  = 1 − 0.345  = 0.655
```

Higher is better. 0.684 = approximately 68% accurate overall.

### Full Comparison Table (Your Real Results on MOT16-02)

| Metric | SORT | ByteTrack | Notes |
|--------|------|-----------|-------|
| **MOTA** | **0.684** | 0.655 | Overall tracking accuracy (higher=better) |
| **MOTP** | **0.151** | 0.161 | Box localisation error (lower=better) |
| **ID Switches** | 98 | **40** | Times a person's ID changed — ByteTrack wins ✅ |
| True Positives | 7,486 | **7,656** | ByteTrack finds more real pedestrians ✅ |
| False Positives | **1,338** | 1,827 | ByteTrack accepts more borderline boxes |
| False Negatives | 1,354 | **1,184** | ByteTrack misses fewer pedestrians ✅ |
| GT objects | 8,840 | 8,840 | Same ground truth for both |

**Interpretation:**  
SORT achieves a higher MOTA because it is more conservative (fewer FP). ByteTrack is more aggressive — it accepts more detections, finds more real pedestrians (lower FN), but also introduces more false alarms (higher FP). ByteTrack's **key advantage is ID switches: 59% fewer** — meaning the same person keeps their ID number much longer through occlusions and crowds. This is ByteTrack's primary design goal.

---

## 13. Final Results

### Detection Quality (YOLOv8n after 50 epochs)

| Metric | Value |
|--------|-------|
| mAP@50 | **0.9058** |
| mAP@50-95 | **0.5915** |
| Precision | **0.9237** |
| Recall | **0.8308** |
| Training dataset | MOT16 train split (4,254 frames) |
| Validation dataset | MOT16 val split (1,062 frames) |
| Pedestrian annotations | 79,799 |

### Tracking Quality (MOT16-02, 600 frames)

| Metric | SORT | ByteTrack |
|--------|------|-----------|
| MOTA | 0.684 | 0.655 |
| MOTP | 0.151 | 0.161 |
| ID Switches | 98 | **40** |

### Outputs Produced

| File | Description |
|------|-------------|
| `models/best.pt` | Trained YOLOv8n weights (5.9 MB) |
| `outputs/MOT16-02_tracked.mp4` | SORT tracked video (Venice plaza, 600 frames) |
| `outputs/MOT16-02_bytetrack.mp4` | ByteTrack tracked video |
| `outputs/MOT16-02_sort_pred.txt` | SORT predictions in MOTChallenge format (8,825 lines) |
| `outputs/MOT16-02_bytetrack_pred.txt` | ByteTrack predictions |

---

## 14. Who Did What

| Entity | Role |
|--------|------|
| **MOT16 researchers** | Placed cameras, recorded the videos at real public locations |
| **Human annotators** | Drew every box in `gt.txt` by hand, frame by frame |
| **`preprocess.py`** | Converted human boxes from MOT16 pixel format → YOLO normalised format |
| **YOLOv8n neural network** | Learned from the converted boxes to automatically detect pedestrians in new images |
| **`sort_tracker.py`** | Implements the SORT algorithm: Kalman Filter + Hungarian Algorithm |
| **`run_tracker.py`** | Runs the full SORT pipeline: load model → detect → track → draw → save video + txt |
| **`run_bytetrack.py`** | Runs the ByteTrack pipeline: same as above but with two-pass association |
| **`evaluate.py`** | Compares tracker predictions vs ground truth → produces MOTA/MOTP/ID Switch metrics |
| **`verify_labels.py`** | Draws ground truth boxes on images to visually confirm preprocessing was correct |
| **Group G07** | Wrote all code, trained the model, ran experiments, documented everything |

---

### Key Insight

> Nobody in the code "draws" boxes manually. The **human annotators** drew them once in `gt.txt`. The code **taught a machine to do the same thing automatically** using those human examples as a training signal — and then measured how well the machine learned by comparing its output to the human-drawn truth.
