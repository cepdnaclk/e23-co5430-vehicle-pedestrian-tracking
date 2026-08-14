# M3 — YOLOv8 + SORT Baseline
## Project: Vehicle & Pedestrian Tracking (CO5430 – Group G07)

---

## What M3 Delivers
- A trained YOLOv8n model (`.pt` file) that detects pedestrians in street scenes
- A SORT tracker that assigns consistent IDs across frames
- A tracked output video (`.mp4`) on at least one MOT16 sequence

---

## Time Estimates

| Platform | Training (50 epochs) | Inference (600 frames) |
|---|---|---|
| **Your CPU (i7-1185G7)** | ~8–12 hours | ~20–30 min |
| **Google Colab T4 GPU** | ~25–35 min ✅ | ~2–3 min |
| **Department GPU server** | ~10–20 min | ~1–2 min |

**Recommendation: Use Google Colab.** Instructions below.

---

## Step 1 — Prepare Google Drive

1. Open [drive.google.com](https://drive.google.com)
2. Create a folder called `G07`
3. Upload the entire `data/yolo_dataset/` folder inside it

Your Drive should look like:
```
MyDrive/
└── G07/
    └── data/
        └── yolo_dataset/
            ├── images/
            │   ├── train/   ← 4,254 .jpg files
            │   └── val/     ← 1,062 .jpg files
            ├── labels/
            │   ├── train/   ← 4,254 .txt files
            │   └── val/     ← 1,062 .txt files
            └── dataset_colab.yaml
```

> ⚠️ **Important:** Upload `dataset_colab.yaml` (not `dataset.yaml`).
> The Colab version has the correct `/content/drive/MyDrive/G07/...` path.

---

## Step 2 — Open Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **New Notebook**
3. Name it: `G07_M3_YOLOv8_Training`
4. Set GPU: **Runtime → Change runtime type → T4 GPU → Save**

---

## Step 3 — Run Training (Colab)

Copy each code block below into a new cell and run them **in order**.

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
> A popup will ask you to authorize. Click through it.

### Cell 2: Verify GPU
```python
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU!")
print("CUDA:", torch.cuda.is_available())
```
Expected output: `GPU: Tesla T4`

### Cell 3: Install YOLOv8
```python
!pip install ultralytics -q
import ultralytics
print("Ultralytics:", ultralytics.__version__)
```

### Cell 4: Verify Dataset Paths
```python
import os
root = "/content/drive/MyDrive/G07/data/yolo_dataset"
for p in [root, f"{root}/images/train", f"{root}/images/val",
          f"{root}/labels/train", f"{root}/labels/val",
          f"{root}/dataset_colab.yaml"]:
    tag = "OK" if os.path.exists(p) else "MISSING!"
    n = f" ({len(os.listdir(p))} files)" if os.path.isdir(p) else ""
    print(f"[{tag}] {p}{n}")
```
Expected output:
```
[OK] .../yolo_dataset (6 items)
[OK] .../images/train (4254 files)
[OK] .../images/val (1062 files)
[OK] .../labels/train (4254 files)
[OK] .../labels/val (1062 files)
[OK] .../dataset_colab.yaml
```

### Cell 5: TRAIN YOLOv8n (25–35 minutes)
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')   # auto-downloads pretrained weights

results = model.train(
    data    = '/content/drive/MyDrive/G07/data/yolo_dataset/dataset_colab.yaml',
    epochs  = 50,
    imgsz   = 640,
    batch   = 16,
    name    = 'mot16_pedestrian_v1',
    patience= 20,
    device  = 0,        # T4 GPU
    workers = 2,
    project = '/content/drive/MyDrive/G07/runs',
    save    = True,
    plots   = True,
)
print("Best model:", results.save_dir + "/weights/best.pt")
```

What you will see scrolling by:
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances
  1/50     3.21G      1.123      1.876      1.234        521
  2/50     3.21G      1.089      1.654      1.201        517
  ...
 50/50     3.21G      0.741      0.823      0.989        519

Training complete.  Best mAP50: 0.782 at epoch 48.
```

### Cell 6: Validate
```python
from ultralytics import YOLO

model = YOLO('/content/drive/MyDrive/G07/runs/mot16_pedestrian_v1/weights/best.pt')
val   = model.val(data='/content/drive/MyDrive/G07/data/yolo_dataset/dataset_colab.yaml')

print(f"mAP50:    {val.box.map50:.3f}")
print(f"mAP50-95: {val.box.map:.3f}")
print(f"Precision:{val.box.mp:.3f}")
print(f"Recall:   {val.box.mr:.3f}")
```

### Cell 7: Download best.pt
```python
from google.colab import files
files.download('/content/drive/MyDrive/G07/runs/mot16_pedestrian_v1/weights/best.pt')
```
> This downloads `best.pt` to your local machine. Save it in the project at `models/best.pt`.

---

## Step 4 — Install Local Dependencies

Back on your laptop, in the project folder:
```bash
pip install ultralytics filterpy scipy
```

---

## Step 5 — Run the SORT Tracker

```bash
python src/tracking/run_tracker.py \
    --model   models/best.pt \
    --seq_dir data/raw/MOT16/train/MOT16-02 \
    --output  outputs/MOT16-02_tracked.mp4
```

What you'll see:
```
YOLOv8 + SORT Tracker  |  CO5430 Group G07
  Model   : models/best.pt
  Sequence: data/raw/MOT16/train/MOT16-02
  Output  : outputs/MOT16-02_tracked.mp4
  Device  : cpu

[1/5] Loading model...
[2/5] Found 600 frames
[3/5] Output video: outputs/MOT16-02_tracked.mp4  (1920x1080 @ 30fps)
[4/5] SORT tracker: max_age=3, min_hits=3, iou_thr=0.45
[5/5] Tracking 600 frames...

  Frame   1/600 | tracks=  0 | speed= 1.2 fps | ETA=499s
  Frame  50/600 | tracks= 12 | speed= 2.1 fps | ETA=260s
  Frame 100/600 | tracks= 15 | speed= 2.3 fps | ETA=217s
  ...

=== DONE ===
  Total frames   : 600
  Output saved   : outputs/MOT16-02_tracked.mp4
```

The output video will have:
- **Coloured bounding boxes** around each person
- **ID labels** like `ID 3 | pedestrian` that stay consistent across frames
- **Frame counter** and **active track count** in the top-left

---

## Expected Results at M3 Completion

| Metric | Target | Notes |
|---|---|---|
| mAP50 (val) | > 0.65 | Depends on training, typical is 0.65–0.80 |
| mAP50-95 (val) | > 0.30 | Stricter metric |
| Tracking visible | ✅ | Video shows boxes with persistent IDs |
| ID switches | Some | Expected — SORT is basic, improved in M4 |
| Car tracking | ❌ | No car labels survived — pedestrian only |

---

## M3 File Structure After Completion

```
vehicle-pedestrian-tracking/
├── models/
│   └── best.pt              ← trained YOLOv8n weights (download from Colab)
├── outputs/
│   └── MOT16-02_tracked.mp4 ← the tracked output video
├── src/
│   └── tracking/
│       ├── sort_tracker.py  ← SORT implementation (Kalman + Hungarian)
│       └── run_tracker.py   ← main tracking pipeline
├── data/
│   └── yolo_dataset/
│       └── dataset_colab.yaml
└── runs/  (generated by ultralytics after local inference)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` |
| `ModuleNotFoundError: filterpy` | `pip install filterpy` |
| `FileNotFoundError: best.pt` | Make sure `models/` folder exists and put best.pt there |
| Video is blank / black | Check `img_dir` path — should point to the `img1/` folder |
| All tracks flash and disappear | Lower `--min_hits` to 1: `--min_hits 1` |
| Too many false positives | Raise `--conf` to 0.50 |

---

## After M3 — What Changes in M4

M4 replaces SORT with **DeepSORT** or **ByteTrack**:

| | SORT (M3) | DeepSORT (M4) |
|---|---|---|
| Matching | Box overlap (IoU) only | IoU + visual appearance features |
| Re-ID after occlusion | ❌ Fails (new ID assigned) | ✅ Can re-identify same person |
| ID switches | Many | Fewer |
