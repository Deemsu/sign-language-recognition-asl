"""
1_Data_Collector.py — Hand Sign Data Collector (A-Z)
Collects MediaPipe landmarks and saves them to CSV.
Collection method: Live webcam with countdown and progress bar.
"""
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
# --- Settings ---
DATA_DIR = "sign_data"
DATA_FILE = os.path.join(DATA_DIR, "raw_data.csv")
os.makedirs(DATA_DIR, exist_ok=True)
# All 26 letters A-Z
SIGNS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# Increased to 300 — 26 letters, many are similar
SAMPLES_PER_SIGN = 300
COUNTDOWN = 3 # countdown seconds before collection starts
# --- MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
static_image_mode=False,
max_num_hands=1,
min_detection_confidence=0.6,
min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
# --- Helper functions ---
def extract_landmarks(hand_landmarks):
    """Extract landmarks normalized relative to the wrist."""
    wrist = hand_landmarks.landmark[0]
    lm_list = []
    for lm in hand_landmarks.landmark:
        lm_list.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])
    return lm_list # 63 values

def draw_progress_bar(frame, x, y, w, h, value, max_val, color=(80, 220, 100)):
    pct = value / max(max_val, 1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x+int(w*pct), y+h), color, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (120, 120, 120), 1)
    cv2.putText(frame, f"{value}/{max_val}", (x + w//2 - 30, y + h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

def overlay_rect(frame, x, y, w, h, alpha=0.6):
    ov = frame.copy()
    cv2.rectangle(ov, (x, y), (x+w, y+h), (10, 10, 10), -1)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
# --- Load existing data ---
all_data = []
if os.path.exists(DATA_FILE):
    existing = pd.read_csv(DATA_FILE)
    all_data = existing.to_dict('records')
    existing_counts = existing['label'].value_counts().to_dict()
    print(f"[INFO] Existing data: {len(existing)} samples")
    for s, c in existing_counts.items():
        print(f" {s}: {c}")
else:
    print("[INFO] Starting fresh data collection")
# --- Camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("\n" + "="*55)
print(" Sign Language Data Collector — A to Z")
print("="*55)
print(f" Signs : {SIGNS}")
print(f" Samples : {SAMPLES_PER_SIGN} per sign")
print(f" Total : ~{len(SIGNS) * SAMPLES_PER_SIGN:,} samples")
print("="*55)
for sign_idx, sign in enumerate(SIGNS):

    existing_count = sum(1 for d in all_data if d.get('label') == sign)
    needed = SAMPLES_PER_SIGN - existing_count
    if needed <= 0:
        print(f"\n[SKIP] '{sign}' — already has {existing_count} samples ✓")
        continue
    print(f"\n[{sign_idx+1}/{len(SIGNS)}] Sign: '{sign}' — need {needed} samples")
    phase = "wait"
    countdown_start = None
    collected = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        hand_detected = False
        if results.multi_hand_landmarks:
            hand_detected = True
            hlm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                frame, hlm, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
        # --- Top panel ---
        overlay_rect(frame, 0, 0, W, 90)
        cv2.putText(frame, f"Sign: {sign}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (50, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(frame, f"[{sign_idx+1}/{len(SIGNS)}] Existing: {existing_count} Needed: {needed}",
                    (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
        # --- Wait phase ---
        if phase == "wait":
            overlay_rect(frame, 0, H-120, W, 120)
            cv2.putText(frame, "Press SPACE to start | Q to quit", (20, H-70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Show sign '{sign}' and place your hand in frame",
                        (20, H-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
            hand_color = (80, 220, 100) if hand_detected else (80, 80, 240)

            hand_msg = "Hand detected" if hand_detected else "No hand detected"
            cv2.putText(frame, hand_msg, (20, H-100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2, cv2.LINE_AA)
        # --- Countdown phase ---
        elif phase == "countdown":
            elapsed = time.time() - countdown_start
            remaining = max(0, COUNTDOWN - int(elapsed))
            cv2.putText(frame, str(remaining) if remaining > 0 else "GO!",
                        (W//2 - 60, H//2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 5.0,
                        (50, 220, 50) if remaining == 0 else (50, 220, 220),
                        8, cv2.LINE_AA)
            if elapsed >= COUNTDOWN:
                phase = "collect"
        # --- Collect phase ---
        elif phase == "collect":
            if hand_detected:
                lm_data = extract_landmarks(results.multi_hand_landmarks[0])
                row = {f"x{i}": v for i, v in enumerate(lm_data)}
                row["label"] = sign
                all_data.append(row)
                collected += 1
            overlay_rect(frame, 0, H-100, W, 100)
            draw_progress_bar(frame, 20, H-70, W-40, 30, collected, needed,
                              color=(80, 220, 100) if hand_detected else (80, 80, 220))
            status = "Collecting..." if hand_detected else "Show your hand!"
            color = (80, 220, 100) if hand_detected else (80, 80, 220)
            cv2.putText(frame, status, (20, H-80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
            if collected >= needed:
                phase = "done"
        # --- Done phase ---
        elif phase == "done":
            overlay_rect(frame, W//2-200, H//2-60, 400, 120)
            cv2.putText(frame, f"'{sign}' complete!", (W//2-150, H//2+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 220, 100), 2, cv2.LINE_AA)
            cv2.putText(frame, "Moving to next sign...", (W//2-130, H//2+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.imshow("Data Collector", frame)
            cv2.waitKey(1500)
            break

        cv2.imshow("Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q') or key == 27:
            print("\n[INFO] Stopped by user.")
            cap.release()
            cv2.destroyAllWindows()
            if all_data:
                df = pd.DataFrame(all_data)
                df.to_csv(DATA_FILE, index=False)
                print(f"[INFO] Saved {len(df)} samples to '{DATA_FILE}'")
            exit()

        elif key == ord(' ') and phase == "wait":
            phase = "countdown"
            countdown_start = time.time()
            collected = 0
# --- Save all data ---
cap.release()
cv2.destroyAllWindows()
if all_data:
    df = pd.DataFrame(all_data)
    df.to_csv(DATA_FILE, index=False)
    print("\n" + "="*55)
    print(f" Collection complete!")
    print(f" Total samples : {len(df):,}")
    print(f" Saved to : {DATA_FILE}")
    print(" Distribution:")
    for label, count in df['label'].value_counts().items():
        bar = "█" * (count // 10)
        print(f" {label:10s}: {count:5d} {bar}")
    print("="*55)
    print("\n Next step: run 2_Augment_Data.py")
else:
    print("[WARNING] No data was collected.")