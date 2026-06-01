"""
2_Augment_Data.py — Data Augmentation for Sign Language (A-Z)
Reads raw_data.csv and produces augmented_data.csv with more diverse samples.
"""

import numpy as np
import pandas as pd
import os

# --- Paths ---
DATA_DIR    = "sign_data"
INPUT_FILE  = os.path.join(DATA_DIR, "raw_data.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "augmented_data.csv")

# --- Augmentation settings ---
TARGET_SAMPLES  = 1000
NOISE_STD       = 0.005
SCALE_RANGE     = (0.85, 1.15)
ROTATION_DEG    = 15
TRANSLATION_MAX = 0.03
NUM_LANDMARKS   = 21
COORDS_PER_LM   = 3
TOTAL_FEATURES  = NUM_LANDMARKS * COORDS_PER_LM

print("=" * 60)
print("  Data Augmentation — Sign Language A-Z")
print("=" * 60)

if not os.path.exists(INPUT_FILE):
    print(f"[ERROR] File '{INPUT_FILE}' not found.")
    print("Please run 1_Data_Collector.py first.")
    exit()

# Load dataset
df_raw = pd.read_csv(INPUT_FILE)
feature_cols = [c for c in df_raw.columns if c != 'label']

for col in feature_cols:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

df_raw = df_raw.dropna()

labels = sorted(df_raw['label'].unique())
X_raw  = df_raw[feature_cols].values
y_raw  = df_raw['label'].values

print("\n[1/4] Original data:")
print(f"    Total samples : {len(df_raw):,}")
print(f"    Signs         : {len(labels)}")
print(f"    Features      : {len(feature_cols)}")
print(f"    Labels        : {labels}")

print("\n[2/4] Original sample distribution:")
for label in labels:
    count = (y_raw == label).sum()
    bar   = "█" * (count // 5)
    print(f"    {label:15s}: {count:5d}  {bar}")

# --- Augmentation functions ---
def add_noise(landmarks, std=NOISE_STD):
    noise = np.random.normal(0, std, landmarks.shape)
    return landmarks + noise

def scale_landmarks(landmarks, scale_range=SCALE_RANGE):
    scale = np.random.uniform(*scale_range)
    return landmarks * scale

def mirror_landmarks(landmarks):
    lm = landmarks.copy().reshape(NUM_LANDMARKS, COORDS_PER_LM)
    lm[:, 0] = -lm[:, 0]
    return lm.flatten()

def rotate_landmarks(landmarks, max_deg=ROTATION_DEG):
    angle = np.random.uniform(-max_deg, max_deg)
    rad   = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    lm = landmarks.copy().reshape(NUM_LANDMARKS, COORDS_PER_LM)
    x_new = lm[:, 0] * cos_a - lm[:, 1] * sin_a
    y_new = lm[:, 0] * sin_a + lm[:, 1] * cos_a
    lm[:, 0] = x_new
    lm[:, 1] = y_new
    return lm.flatten()

def translate_landmarks(landmarks, max_t=TRANSLATION_MAX):
    tx = np.random.uniform(-max_t, max_t)
    ty = np.random.uniform(-max_t, max_t)
    lm = landmarks.copy().reshape(NUM_LANDMARKS, COORDS_PER_LM)
    lm[:, 0] += tx
    lm[:, 1] += ty
    return lm.flatten()

def combined_aug(landmarks):
    lm = landmarks.copy()
    if np.random.rand() > 0.3:
        lm = add_noise(lm, std=np.random.uniform(0.002, 0.008))
    if np.random.rand() > 0.4:
        lm = scale_landmarks(lm)
    if np.random.rand() > 0.5:
        lm = rotate_landmarks(lm, max_deg=np.random.uniform(5, 15))
    if np.random.rand() > 0.5:
        lm = translate_landmarks(lm)
    return lm

AUGMENTATION_FUNCS = [
    (add_noise,           "Noise"),
    (scale_landmarks,     "Scale"),
    (mirror_landmarks,    "Mirror"),
    (rotate_landmarks,    "Rotation"),
    (translate_landmarks, "Translation"),
    (combined_aug,        "Combined"),
]

# --- Run augmentation ---
print(f"\n[3/4] Augmenting data (target: {TARGET_SAMPLES} samples per sign)...")
print(f"      Expected total: ~{TARGET_SAMPLES * len(labels):,} samples")

augmented_rows = []
aug_stats      = {}

for label in labels:
    mask       = y_raw == label
    X_label    = X_raw[mask]
    orig_count = len(X_label)

    for x in X_label:
        augmented_rows.append((*x, label))

    needed    = TARGET_SAMPLES - orig_count
    generated = 0

    if needed <= 0:
        print(f"    {label:15s}: {orig_count} originals (no augmentation needed)")
        aug_stats[label] = {"original": orig_count, "generated": 0, "total": orig_count}
        continue

    func_idx = 0
    while generated < needed:
        src  = X_label[np.random.randint(len(X_label))]
        func, _ = AUGMENTATION_FUNCS[func_idx % len(AUGMENTATION_FUNCS)]
        func_idx += 1

        aug = func(src)

        if not np.any(np.isnan(aug)) and not np.any(np.isinf(aug)):
            augmented_rows.append((*aug, label))
            generated += 1

    total = orig_count + generated
    aug_stats[label] = {"original": orig_count, "generated": generated, "total": total}

    print(f"    {label:15s}: {orig_count:5d} original + {generated:5d} generated = {total:5d}")

# --- Save results ---
print(f"\n[4/4] Saving augmented dataset...")

col_names = [f"x{i}" for i in range(TOTAL_FEATURES)] + ["label"]
df_aug    = pd.DataFrame(augmented_rows, columns=col_names)
df_aug    = df_aug.sample(frac=1, random_state=42).reset_index(drop=True)

df_aug.to_csv(OUTPUT_FILE, index=False)

total_orig = sum(s["original"] for s in aug_stats.values())
total_gen  = sum(s["generated"] for s in aug_stats.values())
total_all  = len(df_aug)

print(f"\n{'='*60}")
print("  Augmentation complete!")
print(f"{'='*60}")
print(f"  Original samples   : {total_orig:,}")
print(f"  Generated samples  : {total_gen:,}")
print(f"  Total samples      : {total_all:,}")
print(f"  Saved to           : {OUTPUT_FILE}")

print(f"\n  Final distribution:")
for label in labels:
    count = (df_aug['label'] == label).sum()
    bar   = "█" * (count // 20)
    print(f"    {label:15s}: {count:5d}  {bar}")

print(f"\n{'='*60}")
print("  Next step: run 3_Train_v2.py")
print(f"{'='*60}")