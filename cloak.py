import cv2
import mediapipe as mp
import time
import numpy as np
import threading
import os
from playsound import playsound

# Function to play sound in background thread (safe)
def play_magic_sound():
    if os.path.exists("magic_whoosh.wav"):  # check if file exists
        threading.Thread(
            target=lambda: playsound("magic_whoosh.wav"), daemon=True
        ).start()
    else:
        print("⚠️ Sound file 'magic_whoosh.wav' not found. Skipping sound.")

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_selfie_segmentation = mp.solutions.selfie_segmentation
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

time.sleep(2)  # Wait so you can move out of frame
ret, background = cap.read()
background = cv2.flip(background, 1)

# States
invisible = False
last_gesture_time = 0
gesture_cooldown = 1.0  # seconds
fade_frames = 20
fade_progress = 0
fading = False
fade_direction = 1  # 1 = fade out, -1 = fade in

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand detection
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            thumb_tip = landmarks[4]   # Thumb tip
            index_tip = landmarks[8]   # Index tip

            # Thumbs up condition
            if (thumb_tip.y < index_tip.y and
                landmarks[12].y > landmarks[9].y and
                landmarks[16].y > landmarks[13].y and
                landmarks[20].y > landmarks[17].y):
                
                current_time = time.time()
                if current_time - last_gesture_time > gesture_cooldown:
                    invisible = not invisible
                    fade_direction = 1 if invisible else -1
                    fade_progress = 0
                    fading = True
                    last_gesture_time = current_time

                    # Play sound safely
                    play_magic_sound()

    # Segmentation
    seg_result = segmentation.process(rgb)
    mask = seg_result.segmentation_mask

    # Smooth mask to avoid patches
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    condition = mask > 0.6  # stronger threshold

    # Normal view
    output_frame = frame.copy()

    if invisible or fading:
        replaced = np.where(condition[..., None], background, frame)

        if fading:
            alpha = fade_progress / fade_frames
            if fade_direction == -1:
                alpha = 1 - alpha

            # Feather edges: blend mask instead of harsh cut
            output_frame = cv2.addWeighted(replaced, alpha, frame, 1 - alpha, 0)

            fade_progress += 1
            if fade_progress > fade_frames:
                fading = False
                output_frame = replaced if invisible else frame
        else:
            output_frame = replaced

    cv2.imshow("🪄 Invisibility Cloak (Smooth + WAV Sound)", output_frame)

    # Exit
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:  # 'q' or ESC


        break

cap.release()
cv2.destroyAllWindows()