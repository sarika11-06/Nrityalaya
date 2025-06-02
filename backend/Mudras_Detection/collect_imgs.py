import os
import cv2
import mediapipe as mp

# Setup Directory
HANDS_DIR = './hands'
os.makedirs(HANDS_DIR, exist_ok=True)

number_of_classes = 10
dataset_size = 100

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

for class_id in range(number_of_classes):
    class_dir = os.path.join(HANDS_DIR, str(class_id))
    os.makedirs(class_dir, exist_ok=True)
    print(f'📸 Collecting hand data for class {class_id}')

    # Wait for user to start
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, 'Ready? Press "Q" to start...', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Hand Collection', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
            counter += 1
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f'Capturing ({counter}/{dataset_size})', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Hand Collection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("✅ Hand data collection complete!")
