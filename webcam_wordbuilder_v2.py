"""
webcam_wordbuilder_v2.py — Enhanced Word Builder
Features:
- Works with the new model (MLP/sklearn)
- Improved stability timer (1 second hold = add letter)
- Confidence threshold
- Delete and Space keys
- Per-letter confidence display
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time
from collections import deque, Counter

# --- Load model ---
MODEL_PATH = "sign_model.pkl"
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model '{MODEL_PATH}' not found. Run 3_Train_v2.py first.")
    exit()

model_data = joblib.load(MODEL_PATH)
model  = model_data["model"]
scaler = model_data["scaler"]
labels = model_data.get("labels", model.classes_.tolist())
print(f"Model loaded — Signs: {labels}")

# --- MediaPipe ---
mp_hands = mp.solutions.hands
webcam_hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- Settings ---
STABILITY_SECONDS = 1.2    # seconds to hold before adding letter
CONFIDENCE_THRESH = 0.65   # minimum confidence threshold
BUFFER_SIZE       = 20     # prediction buffer size for smoothing

# --- State ---
pred_buffer     = deque(maxlen=BUFFER_SIZE)
current_word    = ""
last_prediction = ""
stable_start    = None
last_added_char = ""
last_added_time = 0
cooldown        = 3.0      # seconds between adding the same letter twice

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\nWord Builder ready!")
print("   Hold a sign for 1 second to add it to the word")
print("   [SPACE] Space  [B] Backspace  [C] Clear  [Q] Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame     = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = webcam_hands.process(rgb_frame)

    predicted_char = ""
    confidence     = 0.0
    stability_prog = 0.0
    now = time.time()

    if results.multi_hand_landmarks:
        hlm = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

        wrist   = hlm.landmark[0]
        lm_list = []
        for lm in hlm.landmark:
            lm_list.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

        arr   = np.array(lm_list).reshape(1, -1)
        arr_s = scaler.transform(arr)
        raw   = model.predict(arr_s)[0]
        pred_buffer.append(raw)

        counts         = Counter(pred_buffer)
        predicted_char = counts.most_common(1)[0][0]
        confidence     = counts[predicted_char] / len(pred_buffer)

        # Stability timer
        if confidence >= CONFIDENCE_THRESH:
            if predicted_char == last_prediction:
                if stable_start is None:
                    stable_start = now
                elapsed        = now - stable_start
                stability_prog = min(elapsed / STABILITY_SECONDS, 1.0)

                if elapsed >= STABILITY_SECONDS:
                    if not (predicted_char == last_added_char and
                            (now - last_added_time) < cooldown):
                        current_word    += predicted_char
                        last_added_char  = predicted_char
                        last_added_time  = now
                        stable_start     = now
            else:
                last_prediction = predicted_char
                stable_start    = None
                stability_prog  = 0.0
        else:
            stable_start   = None
            stability_prog = 0.0
    else:
        pred_buffer.clear()
        stable_start   = None
        stability_prog = 0.0
        predicted_char = ""
        confidence     = 0.0

    # ─── Draw UI ──────────────────────────────────────────────

    # Bottom background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, H-160), (W, H), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Left background
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, 0), (310, 300), (10, 10, 10), -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

    # Current sign
    char_display = predicted_char if predicted_char else "-"
    cv2.putText(frame, char_display, (15, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 4.5, (50, 220, 220), 6, cv2.LINE_AA)

    # Confidence bar
    bar_w     = 270
    conf_fill = int(bar_w * confidence)
    conf_color = (80, 220, 80) if confidence >= 0.8 else (50, 200, 220) if confidence >= 0.6 else (80, 80, 220)
    cv2.rectangle(frame, (15, 145), (15+bar_w, 165), (50, 50, 50), -1)
    cv2.rectangle(frame, (15, 145), (15+conf_fill, 165), conf_color, -1)
    cv2.putText(frame, f"Confidence: {confidence:.0%}", (15, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # Stability bar
    stab_fill = int(bar_w * stability_prog)
    cv2.rectangle(frame, (15, 185), (15+bar_w, 205), (50, 50, 50), -1)
    cv2.rectangle(frame, (15, 185), (15+stab_fill, 205), (220, 160, 50), -1)
    cv2.putText(frame, f"Stability: {stability_prog:.0%}", (15, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    if stability_prog >= 1.0:
        cv2.putText(frame, f"+ '{predicted_char}'", (170, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 255, 80), 2, cv2.LINE_AA)

    # Word Builder
    cv2.putText(frame, "Word:", (15, H-120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
    display = current_word if current_word else "..."
    if len(display) > 30:
        display = display[-30:]
    cv2.putText(frame, display, (15, H-65),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (240, 240, 240), 2, cv2.LINE_AA)

    cv2.putText(frame, "[SPACE] Space  [B] Backspace  [C] Clear  [Q] Quit",
                (15, H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

    cv2.imshow("Sign Language — Word Builder v2", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or key == 27:
        break
    elif key == ord('c') or key == ord('C'):
        current_word = ""
        stable_start = None
    elif key == ord('b') or key == ord('B'):
        if current_word:
            current_word = current_word[:-1]
    elif key == ord(' '):
        current_word += " "

cap.release()
cv2.destroyAllWindows()
if current_word.strip():
    print(f"\nLast word: '{current_word.strip()}'")
