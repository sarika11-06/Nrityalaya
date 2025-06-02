import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model


# Load model and label encoder
model = load_model('cnn_model.h5')
with open('label_encoder.pickle', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,  # Use video mode
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Start webcam feed
cap = cv2.VideoCapture(0)

print("📷 Starting webcam... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame from webcam.")
        break

    H, W, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    data_aux = []
    all_x, all_y = [], []

    if results.multi_hand_landmarks:
        all_landmarks = results.multi_hand_landmarks

        for hand_landmarks in all_landmarks:
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            all_x.extend(x_)
            all_y.extend(y_)

        min_x = min(all_x)
        min_y = min(all_y)

        for hand_landmarks in all_landmarks:
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min_x)
                data_aux.append(lm.y - min_y)

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Pad if only one hand
        if len(all_landmarks) == 1:
            data_aux.extend([0] * 42)

        if len(data_aux) == 84:
            input_array = np.array([data_aux])
            prediction = model.predict(input_array, verbose=0)[0]
            predicted_index = np.argmax(prediction)
            predicted_label = str(label_encoder.inverse_transform([predicted_index])[0])

            print(f"🖐️ Predicted Gesture: {predicted_label}")

            x1 = int(min(all_x) * W) - 10
            y1 = int(min(all_y) * H) - 10
            x2 = int(max(all_x) * W) + 10
            y2 = int(max(all_y) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, predicted_label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            print("⚠️ Not enough landmarks for prediction.")
    else:
        print("🙌 No hands detected.")

    cv2.imshow("Real-Time Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting webcam.")
        break

cap.release()
cv2.destroyAllWindows()

