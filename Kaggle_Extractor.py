"""
1b_Kaggle_Extractor.py — Kaggle Dataset Landmark Extractor
Reads hand sign images from a Kaggle dataset folder structure:
    kaggle_dataset/
        A/
            img1.jpg
            img2.jpg
        B/
            img1.jpg
        ...

Extracts MediaPipe hand landmarks from each image and merges
the results into sign_data/raw_data.csv (same format as 1_Data_Collector.py).
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

# ---------------------------------------------------------------
# SETTINGS — edit these to match your setup
# ---------------------------------------------------------------
KAGGLE_DIR = r"C:\Users\Deema\OneDrive\Desktop\Singlanguage\asl_alphabet_train"   # <- path to your Kaggle folder
DATA_DIR   = "sign_data"
DATA_FILE  = os.path.join(DATA_DIR, "raw_data.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# Signs to extract — set to None to auto-detect from folder names
SIGNS = None   # e.g. ["A","B","C"] or None for all subfolders

# Supported image extensions
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Max images per sign to extract (None = all)
MAX_PER_SIGN = 1000

# ---------------------------------------------------------------
# MediaPipe setup — static image mode for photos
# ---------------------------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,       # important: True for photos
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ---------------------------------------------------------------
def extract_landmarks(hand_landmarks):
    """Extract landmarks normalized relative to the wrist (same as collector)."""
    wrist = hand_landmarks.landmark[0]
    lm_list = []
    for lm in hand_landmarks.landmark:
        lm_list.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])
    return lm_list  # 63 values


# Validate Kaggle folder

print("=" * 60)
print("  Kaggle Landmark Extractor")
print("=" * 60)

if not os.path.exists(KAGGLE_DIR):
    print(f"\n[ERROR] Folder '{KAGGLE_DIR}' not found.")
    print(f"  Please set KAGGLE_DIR to your Kaggle dataset path.")
    print(f"  Expected structure:")
    print(f"    {KAGGLE_DIR}/")
    print(f"      A/  img1.jpg  img2.jpg ...")
    print(f"      B/  img1.jpg ...")
    exit()

# Auto-detect signs from subfolders
if SIGNS is None:
    SIGNS = sorted([
        d for d in os.listdir(KAGGLE_DIR)
        if os.path.isdir(os.path.join(KAGGLE_DIR, d))
    ])

if not SIGNS:
    print(f"[ERROR] No subfolders found in '{KAGGLE_DIR}'")
    exit()

print(f"\n  Kaggle folder : {KAGGLE_DIR}")
print(f"  Signs found   : {SIGNS}")
print(f"  Output file   : {DATA_FILE}")


# Load existing data (from webcam collection if any)

existing_rows = []
existing_counts = {}
if os.path.exists(DATA_FILE):
    df_existing = pd.read_csv(DATA_FILE)
    existing_rows = df_existing.to_dict('records')
    existing_counts = df_existing['label'].value_counts().to_dict()
    print(f"\n  Existing data : {len(existing_rows)} samples")
    for s, c in existing_counts.items():
        print(f"    {s}: {c}")
else:
    print("\n  No existing data found — starting fresh.")

# ---------------------------------------------------------------
# Extract landmarks from images
# ---------------------------------------------------------------
print(f"\n[Processing {len(SIGNS)} signs...]\n")

new_rows     = []
stats        = {}
total_images = 0
total_found  = 0
total_failed = 0

for sign in SIGNS:
    sign_folder = os.path.join(KAGGLE_DIR, sign)

    # Collect all image files in this folder
    img_files = [
        f for f in os.listdir(sign_folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    ]

    if MAX_PER_SIGN is not None:
        img_files = img_files[:MAX_PER_SIGN]

    found   = 0
    failed  = 0
    skipped = 0

    print(f"  [{sign}] — {len(img_files)} images", end="", flush=True)
    t0 = time.time()

    for img_name in img_files:
        img_path = os.path.join(sign_folder, img_name)

        img = cv2.imread(img_path)
        if img is None:
            failed += 1
            continue

        # Resize if too large (speeds up processing)
        h, w = img.shape[:2]
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            img   = cv2.resize(img, (int(w * scale), int(h * scale)))

        rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            lm_data = extract_landmarks(results.multi_hand_landmarks[0])
            if not any(np.isnan(v) or np.isinf(v) for v in lm_data):
                row = {f"x{i}": v for i, v in enumerate(lm_data)}
                row["label"] = sign
                new_rows.append(row)
                found += 1
            else:
                failed += 1
        else:
            failed += 1   # no hand detected in image

    elapsed = time.time() - t0
    rate    = len(img_files) / max(elapsed, 0.001)
    print(f"  →  detected: {found}  |  failed/no-hand: {failed}  |  {rate:.0f} img/s")

    stats[sign] = {"images": len(img_files), "found": found, "failed": failed}
    total_images += len(img_files)
    total_found  += found
    total_failed += failed

# ---------------------------------------------------------------
# Merge with existing data and save
# ---------------------------------------------------------------
print(f"\n[Merging and saving...]")

all_rows = existing_rows + new_rows
df_out   = pd.DataFrame(all_rows)

# Ensure column order
feat_cols = [f"x{i}" for i in range(63)]
df_out = df_out[feat_cols + ["label"]]

# Shuffle
df_out = df_out.sample(frac=1, random_state=42).reset_index(drop=True)
df_out.to_csv(DATA_FILE, index=False)

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("  Extraction Complete!")
print("=" * 60)
print(f"  Total images processed : {total_images:,}")
print(f"  Landmarks extracted    : {total_found:,}")
print(f"  Failed / no hand       : {total_failed:,}")
print(f"  Detection rate         : {total_found/max(total_images,1):.1%}")
print(f"\n  Existing (webcam) rows : {len(existing_rows):,}")
print(f"  New (Kaggle) rows      : {len(new_rows):,}")
print(f"  Total in CSV           : {len(df_out):,}")
print(f"\n  Saved to               : {DATA_FILE}")
print(f"\n  Final distribution:")
for label in sorted(df_out['label'].unique()):
    count = (df_out['label'] == label).sum()
    bar   = "█" * (count // 20)
    print(f"    {label:15s}: {count:5d}  {bar}")
print("=" * 60)
print("\n  Next step: run 2_Augment_Data.py")
print("=" * 60)

hands.close()
