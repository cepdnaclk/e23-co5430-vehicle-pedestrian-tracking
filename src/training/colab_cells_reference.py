"""
=============================================================================
G07_M3_YOLOv8_Training.ipynb  —  Google Colab Training Notebook
Project: Vehicle & Pedestrian Tracking (CO5430 – Group G07)
=============================================================================

INSTRUCTIONS — READ BEFORE RUNNING:
1. Open Google Colab: https://colab.research.google.com
2. Create a new notebook, name it "G07_M3_YOLOv8_Training"
3. Set runtime: Runtime → Change runtime type → T4 GPU
4. Copy each CELL below into a Colab cell (the "# ── CELL N" marks the boundary)
5. Run cells one by one, top to bottom

WHAT NEEDS TO BE ON YOUR GOOGLE DRIVE FIRST:
  Upload this entire folder to your Google Drive:
    data/yolo_dataset/
      ├── images/train/   (4,254 .jpg files)
      ├── images/val/     (1,062 .jpg files)
      ├── labels/train/   (4,254 .txt files)
      ├── labels/val/     (1,062 .txt files)
      └── dataset_colab.yaml

  Put it at:  MyDrive/G07/data/yolo_dataset/
  (So the path in dataset_colab.yaml matches)
=============================================================================
"""

# ── CELL 1: Mount Google Drive ─────────────────────────────────────────────
"""
Paste into Colab Cell 1:
"""
# from google.colab import drive
# drive.mount('/content/drive')

CELL_1 = """
from google.colab import drive
drive.mount('/content/drive')
"""

# ── CELL 2: Verify GPU ─────────────────────────────────────────────────────
CELL_2 = """
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU Memory:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")
else:
    print("WARNING: No GPU! Go to Runtime → Change runtime type → T4 GPU")
"""

# ── CELL 3: Install YOLOv8 ────────────────────────────────────────────────
CELL_3 = """
!pip install ultralytics -q
import ultralytics
print("Ultralytics version:", ultralytics.__version__)
"""

# ── CELL 4: Check Dataset Paths ───────────────────────────────────────────
CELL_4 = """
import os

DATASET_ROOT = "/content/drive/MyDrive/G07/data/yolo_dataset"
YAML_PATH    = f"{DATASET_ROOT}/dataset_colab.yaml"

# Verify the paths exist
for path in [DATASET_ROOT, YAML_PATH,
             f"{DATASET_ROOT}/images/train",
             f"{DATASET_ROOT}/images/val",
             f"{DATASET_ROOT}/labels/train",
             f"{DATASET_ROOT}/labels/val"]:
    status = "OK" if os.path.exists(path) else "MISSING!"
    count = ""
    if os.path.isdir(path):
        count = f"  ({len(os.listdir(path))} files)"
    print(f"  [{status}]  {path}{count}")

# Print yaml content to confirm
print("\\n--- dataset_colab.yaml contents ---")
with open(YAML_PATH) as f:
    print(f.read())
"""

# ── CELL 5: TRAIN YOLOv8n ─────────────────────────────────────────────────
# This is the main training cell. Takes ~25-35 min on Colab T4 GPU.
CELL_5 = """
from ultralytics import YOLO

# Load the pre-trained YOLOv8 nano model (smallest, fastest to fine-tune)
# 'yolov8n.pt' will be auto-downloaded if not present
model = YOLO('yolov8n.pt')

# ── Training parameters ───────────────────────────────────────────────────
# data      : path to your dataset.yaml
# epochs    : 50 full passes through all training images
# imgsz     : resize all images to 640x640 (YOLOv8 standard)
# batch     : 16 images per gradient update (good for T4 16GB)
# name      : output folder name under runs/detect/
# patience  : stop early if val loss doesn't improve for 20 epochs
# device    : 0 = first GPU, 'cpu' = CPU only
# workers   : number of CPU threads for data loading

results = model.train(
    data    = '/content/drive/MyDrive/G07/data/yolo_dataset/dataset_colab.yaml',
    epochs  = 50,
    imgsz   = 640,
    batch   = 16,
    name    = 'mot16_pedestrian_v1',
    patience= 20,
    device  = 0,           # GPU (use 'cpu' if no GPU)
    workers = 2,
    project = '/content/drive/MyDrive/G07/runs',  # save to Drive
    save    = True,
    plots   = True,        # generate training curve plots
)

print("\\n=== Training complete! ===")
print(f"Best model saved to: {results.save_dir}/weights/best.pt")
"""

# ── CELL 6: Evaluate on Validation Set ────────────────────────────────────
CELL_6 = """
from ultralytics import YOLO
import os

# Load the best trained weights
best_model_path = '/content/drive/MyDrive/G07/runs/mot16_pedestrian_v1/weights/best.pt'
model = YOLO(best_model_path)

# Run validation
val_results = model.val(
    data='/content/drive/MyDrive/G07/data/yolo_dataset/dataset_colab.yaml',
    imgsz=640,
    batch=16,
)

print("\\n=== Validation Results ===")
print(f"mAP50       : {val_results.box.map50:.4f}")    # mAP at IoU=0.50
print(f"mAP50-95    : {val_results.box.map:.4f}")      # mAP at IoU=0.50:0.95
print(f"Precision   : {val_results.box.mp:.4f}")
print(f"Recall      : {val_results.box.mr:.4f}")
"""

# ── CELL 7: Run Inference on Sample Images ────────────────────────────────
CELL_7 = """
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import glob

best_model_path = '/content/drive/MyDrive/G07/runs/mot16_pedestrian_v1/weights/best.pt'
model = YOLO(best_model_path)

# Pick 5 random val images
import random
val_images = glob.glob('/content/drive/MyDrive/G07/data/yolo_dataset/images/val/*.jpg')
samples = random.sample(val_images, min(5, len(val_images)))

fig, axes = plt.subplots(1, len(samples), figsize=(20, 5))
fig.suptitle('YOLOv8n Detections on Validation Frames', fontsize=14)

for i, img_path in enumerate(samples):
    result = model(img_path, conf=0.3)[0]   # conf=0.3 threshold
    img_bgr = result.plot()                  # draw boxes on image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    axes[i].imshow(img_rgb)
    axes[i].axis('off')
    axes[i].set_title(os.path.basename(img_path)[:20], fontsize=8)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/G07/sample_detections.png', dpi=150)
plt.show()
print("Saved sample detections to Google Drive")
"""

# ── CELL 8: Download best.pt to Drive (already there from Cell 5) ─────────
CELL_8 = """
# The best model is already saved to your Drive at:
# /content/drive/MyDrive/G07/runs/mot16_pedestrian_v1/weights/best.pt
#
# Download it to your LOCAL machine from Drive for the SORT tracking step.
# Or just download from Colab directly:
from google.colab import files
files.download('/content/drive/MyDrive/G07/runs/mot16_pedestrian_v1/weights/best.pt')
print("Downloading best.pt ...")
"""

print("=" * 60)
print("COLAB NOTEBOOK CELLS — copy each CELL_N into a Colab cell")
print("=" * 60)
for name, code in [("CELL 1", CELL_1), ("CELL 2", CELL_2), ("CELL 3", CELL_3),
                   ("CELL 4", CELL_4), ("CELL 5", CELL_5), ("CELL 6", CELL_6),
                   ("CELL 7", CELL_7), ("CELL 8", CELL_8)]:
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(code.strip())
