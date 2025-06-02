import cv2
import mediapipe as mp
import os
import pickle

# Setup MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)  # Static mode for image dataset
mp_drawing = mp.solutions.drawing_utils

data = []
labels = []

# Path to dataset directory (adjust if needed)
dataset_path = './data'  # Example: data/araimandi/, data/nataraja/

# Loop through each pose directory
for dir_ in os.listdir(dataset_path):
    dir_path = os.path.join(dataset_path, dir_)
    if not os.path.isdir(dir_path):
        continue

    print(f"📁 Processing '{dir_}'...")
    for img_file in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_file)
        image = cv2.imread(img_path)

        if image is None:
            print(f"❌ Could not read {img_file}, skipping.")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            data_aux = []

            # Collect all x and y for normalization
            all_x = [lm.x for lm in landmarks]
            all_y = [lm.y for lm in landmarks]
            min_x = min(all_x)
            min_y = min(all_y)

            for lm in landmarks:
                data_aux.append(lm.x - min_x)
                data_aux.append(lm.y - min_y)

            if len(data_aux) == 66:  # 33 keypoints × 2 (x, y)
                data.append(data_aux)
                labels.append(dir_)
            else:
                print(f"⚠️ Skipped {img_file} in '{dir_}' due to incomplete landmarks.")
        else:
            print(f"⚠️ No pose detected in {img_file}, skipping.")

# Save dataset to pickle file
with open('pose_data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("\n✅ Dataset saved as 'pose_data.pickle'")
