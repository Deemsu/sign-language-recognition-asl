"""
4_Real_Time_v2.py — Enhanced Real-Time Sign Recognition
Features:
- Confidence score with progress bar
- Integrated Word Builder with Backspace
- Stability timer (letter only added after stable hold)
- Supports any model (MLP or Random Forest)
- Clean UI with additional info panels
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from collections import deque, Counter
import time

# --- Load model ---
MODEL_PATH = "sign_model.pkl"
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model '{MODEL_PATH}' not found. Run 3_Train_v2.py first.")
    exit()

model_data = joblib.load(MODEL_PATH)
model  = model_data["model"]
scaler = model_data["scaler"]
labels = model_data.get("labels", model.classes_.tolist())

print(f"Model loaded : {model_data.get('model_name', 'Unknown')}")
print(f"Signs ({len(labels)}): {labels}")

# --- MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# --- Settings ---
BUFFER_SIZE       = 15    # prediction buffer size for smoothing
STABILITY_FRAMES  = 20    # frames to hold before adding to word
CONFIDENCE_THRESH = 0.70  # minimum confidence threshold

# --- State ---
pred_buffer  = deque(maxlen=BUFFER_SIZE)
stable_count = 0
last_added   = ""
current_word = ""
word_history = []
fps_counter  = deque(maxlen=30)
last_time    = time.time()

# --- Colors ---
C_GREEN  = (80, 220, 100)
C_YELLOW = (50, 220, 220)
C_RED    = (80, 80, 240)
C_BLUE   = (220, 160, 50)
C_WHITE  = (240, 240, 240)
C_DARK   = (20, 20, 20)
C_GRAY   = (120, 120, 120)

def draw_confidence_bar(frame, x, y, w, h, value, label="", color=(80, 220, 100)):
    """Draw a labeled progress bar."""
    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 50), -1)
    fill = int(w * value)
    cv2.rectangle(frame, (x, y), (x+fill, y+h), color, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)
    if label:
        cv2.putText(frame, f"{label}: {value:.0%}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1, cv2.LINE_AA)

def get_confidence(pred_label):
    """Calculate confidence from the current prediction buffer."""
    if not pred_buffer:
        return 0.0
    counts = Counter(pred_buffer)
    return counts[pred_label] / len(pred_buffer)

def overlay_panel(frame, x, y, w, h, alpha=0.6):
    """Draw a semi-transparent dark panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

# --- Camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\nReady! Controls:")
print("   SPACE : add space to word")
print("   B     : backspace (delete last letter)")
print("   C     : clear word")
print("   S     : save word to history")
print("   ESC   : quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    now   = time.time()
    fps_counter.append(1.0 / max(now - last_time, 1e-9))
    last_time = now
    fps = np.mean(fps_counter)

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction    = "---"
    confidence    = 0.0
    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True
        hlm = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS,
                               mp_style.get_default_hand_landmarks_style(),
                               mp_style.get_default_hand_connections_style())

        wrist   = hlm.landmark[0]
        lm_list = []
        for lm in hlm.landmark:
            lm_list.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

        arr      = np.array(lm_list).reshape(1, -1)
        arr_s    = scaler.transform(arr)
        raw_pred = model.predict(arr_s)[0]
        pred_buffer.append(raw_pred)

        prediction = max(set(pred_buffer), key=lambda p: Counter(pred_buffer)[p])
        confidence = get_confidence(prediction)

        if confidence >= CONFIDENCE_THRESH:
            if prediction == last_added:
                stable_count += 1
            else:
                stable_count = 0
                last_added   = prediction

            if stable_count == STABILITY_FRAMES:
                current_word += prediction
                stable_count  = 0
        else:
            stable_count = 0
    else:
        if pred_buffer:
            pred_buffer.popleft()
        stable_count = 0
        if not pred_buffer:
            prediction = "---"

    # ─── UI ─────────────────────────────────────────────────────

    # Left panel (recognition info)
    overlay_panel(frame, 0, 0, 320, H)

    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GRAY, 1, cv2.LINE_AA)

    hand_status = "Hand detected" if hand_detected else "No hand"
    hand_color  = C_GREEN if hand_detected else C_RED
    cv2.putText(frame, hand_status, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 1, cv2.LINE_AA)

    cv2.putText(frame, "Sign:", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, prediction, (10, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, C_YELLOW, 4, cv2.LINE_AA)

    conf_color = C_GREEN if confidence >= 0.8 else (C_YELLOW if confidence >= 0.5 else C_RED)
    draw_confidence_bar(frame, 10, 200, 280, 18, confidence, "Confidence", conf_color)

    stab_val = stable_count / STABILITY_FRAMES if STABILITY_FRAMES > 0 else 0
    draw_confidence_bar(frame, 10, 235, 280, 18, stab_val, "Stability", C_BLUE)

    cv2.line(frame, (10, 270), (300, 270), (60, 60, 60), 1)

    # Bottom panel (Word Builder)
    overlay_panel(frame, 0, H-130, W, 130, alpha=0.7)

    cv2.putText(frame, "Word:", (15, H-100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_GRAY, 1, cv2.LINE_AA)

    display_word = current_word if current_word else "_"
    cv2.putText(frame, display_word, (15, H-45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, C_WHITE, 2, cv2.LINE_AA)

    hints = "[SPACE] Space  [B] Backspace  [C] Clear  [S] Save  [ESC] Quit"
    cv2.putText(frame, hints, (15, H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GRAY, 1, cv2.LINE_AA)

    # Right panel — word history
    if word_history:
        overlay_panel(frame, W-250, 0, 250, H, alpha=0.55)
        cv2.putText(frame, "Word History:", (W-240, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GRAY, 1, cv2.LINE_AA)
        for i, w in enumerate(reversed(word_history[-10:])):
            cv2.putText(frame, w, (W-240, 60 + i*35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, C_WHITE, 1, cv2.LINE_AA)

    cv2.imshow("Sign Language Recognition v2", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:   # ESC
        break
    elif key == ord('c') or key == ord('C'):
        current_word = ""
        stable_count = 0
    elif key == ord('b') or key == ord('B'):
        if current_word:
            current_word = current_word[:-1]
    elif key == ord(' '):
        current_word += " "
    elif key == ord('s') or key == ord('S'):
        if current_word.strip():
            word_history.append(current_word.strip())
            print(f"Saved word: '{current_word.strip()}'")
            current_word = ""

cap.release()
cv2.destroyAllWindows()
print(f"\nSaved words: {word_history}")