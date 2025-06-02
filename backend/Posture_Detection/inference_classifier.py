import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model and label encoder
model = load_model('cnn_model.h5')
with open('label_encoder.pickle', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    data_aux = []

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Normalize using min_x and min_y across all pose landmarks
        x_vals = [lm.x for lm in landmarks]
        y_vals = [lm.y for lm in landmarks]
        min_x = min(x_vals)
        min_y = min(y_vals)

        for lm in landmarks:
            data_aux.append(lm.x - min_x)
            data_aux.append(lm.y - min_y)

        if len(data_aux) == 66:  # 33 landmarks * 2 (x, y)
            input_array = np.array([data_aux])
            prediction = model.predict(input_array)[0]
            predicted_index = np.argmax(prediction)
            predicted_label = label_encoder.inverse_transform([predicted_index])[0]

            # Draw label and pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            x1 = int(min_x * W) - 20
            y1 = int(min_y * H) - 20
            cv2.putText(frame, f'🧍 {predicted_label}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            print("⚠️ Incomplete pose landmarks detected.")
    else:
        print("❌ No body detected.")

    cv2.imshow("Pose Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
