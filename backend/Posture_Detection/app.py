# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import cv2
from tensorflow.keras.models import load_model
import mediapipe as mp
import base64
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Load trained model and label encoder
model = load_model('cnn_model.h5')
with open('label_encoder.pickle', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    img_bytes = base64.b64decode(data.split(',')[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    data_aux = []
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        x_vals = [lm.x for lm in landmarks]
        y_vals = [lm.y for lm in landmarks]
        min_x = min(x_vals)
        min_y = min(y_vals)

        for lm in landmarks:
            data_aux.append(lm.x - min_x)
            data_aux.append(lm.y - min_y)

        if len(data_aux) == 66:
            input_array = np.array([data_aux])
            prediction = model.predict(input_array)[0]
            predicted_index = np.argmax(prediction)
            predicted_label = label_encoder.inverse_transform([predicted_index])[0]
            return jsonify({'prediction': predicted_label})
        else:
            return jsonify({'error': 'Incomplete pose landmarks'}), 400
    else:
        return jsonify({'error': 'No pose detected'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)