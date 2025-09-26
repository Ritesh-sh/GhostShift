import cv2
import mediapipe as mp
import time
import numpy as np
import threading
import os
from playsound import playsound

# Function to play sound in background thread (safe)
def play_magic_sound():
    if os.path.exists("ramayan_notification.mp3"):  # check if file exists
        threading.Thread(
            target=lambda: playsound("ramayan_notification.mp3"), daemon=True
        ).start()
    else:
        print("⚠️ Sound file 'ramayan_notification.mp3' not found. Skipping sound.")

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

    # Segmentation with improved edge detection and expanded coverage
    seg_result = segmentation.process(rgb)
    mask = seg_result.segmentation_mask

    # Improved mask processing for better edge detection and expanded coverage
    # Apply morphological operations to clean up and expand the mask
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((7, 7), np.uint8)  # Larger kernel for expansion

    # Initial cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Expand the mask to cover more body area
    mask = cv2.dilate(mask, kernel_large, iterations=2)  # Dilate to expand coverage

    # Additional smoothing after expansion
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Lower threshold to include more area around the person
    mask_mean = np.mean(mask)
    threshold = max(0.2, min(0.6, mask_mean * 0.6))  # Lower threshold for more coverage
    condition = mask > threshold

    # Apply edge refinement using Canny edge detection on the mask
    mask_uint8 = (mask * 255).astype(np.uint8)
    edges = cv2.Canny(mask_uint8, 50, 150)
    edges_dilated = cv2.dilate(edges, kernel_small, iterations=1)

    # Refine mask by removing noise and smoothing edges
    refined_mask = cv2.GaussianBlur(mask, (7, 7), 0)  # Smaller blur for sharper edges
    condition = refined_mask > threshold

    # Additional smoothing for final mask
    condition = cv2.medianBlur(condition.astype(np.uint8), 5).astype(bool)

    # Normal view
    output_frame = frame.copy()

    if invisible or fading:
        replaced = np.where(condition[..., None], background, frame)

        if fading:
            alpha = fade_progress / fade_frames
            if fade_direction == -1:
                alpha = 1 - alpha

            # Improved feathering: use distance transform for better edge blending
            mask_binary = condition.astype(np.uint8) * 255
            dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
            dist_transform = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)

            # Create soft edges using distance transform
            soft_mask = np.where(dist_transform < 0.1, alpha * dist_transform * 10, alpha)
            soft_mask = np.clip(soft_mask, 0, 1)

            # Apply soft mask for smoother blending using numpy array operations
            soft_mask_3d = soft_mask[..., None]  # Add channel dimension for broadcasting
            output_frame = (soft_mask_3d * replaced + (1 - soft_mask_3d) * frame).astype(np.uint8)

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