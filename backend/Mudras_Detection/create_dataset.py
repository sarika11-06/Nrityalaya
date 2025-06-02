import cv2
import mediapipe as mp
import os
import pickle

# MediaPipe hand setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)  # ✅ Allow 2 hands
mp_drawing = mp.solutions.drawing_utils

data = []
labels = []

# Path to dataset directory (adjust if needed)
dataset_path = './data'

# Loop through each gesture directory
for dir_ in os.listdir(dataset_path):
    dir_path = os.path.join(dataset_path, dir_)
    if not os.path.isdir(dir_path):
        continue

    for img_path in os.listdir(dir_path):
        image = cv2.imread(os.path.join(dir_path, img_path))
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            all_landmarks = results.multi_hand_landmarks
            data_aux = []
            all_x, all_y = [], []

            # Collect all x and y for normalization
            for hand_landmarks in all_landmarks:
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                all_x.extend(x_)
                all_y.extend(y_)

            min_x = min(all_x)
            min_y = min(all_y)

            # Normalize and append features
            for hand_landmarks in all_landmarks:
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

            # Pad with 0s if only one hand detected
            if len(all_landmarks) == 1:
                data_aux.extend([0] * 42)  # 21 landmarks × 2 = 42 features per hand

            if len(data_aux) == 84:
                data.append(data_aux)
                labels.append(dir_)  # Keep string labels like 'Pataka', 'Alapadma', etc.
            else:
                print(f"⚠️ Skipped {img_path} in '{dir_}' due to incomplete landmarks.")

# Save dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("✅ Dataset saved as data.pickle")
