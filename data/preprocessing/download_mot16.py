"""
download_mot16.py  –  Download MOT16 from Kaggle using the kaggle API
Usage:
    python download_mot16.py

Prerequisites:
    pip install kaggle
    Place your kaggle.json API token in C:/Users/<you>/.kaggle/kaggle.json
"""

import subprocess
import sys
from pathlib import Path

KAGGLE_DATASET = "takshmandar/mot16dataset"   # update if different
DEST = Path("data/raw")

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading {KAGGLE_DATASET} into {DEST} ...")
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", KAGGLE_DATASET, "-p", str(DEST), "--unzip"],
        check=True
    )
    print("[INFO] Download complete.")
    print(f"[INFO] Dataset extracted to: {DEST.resolve()}")

if __name__ == "__main__":
    main()
