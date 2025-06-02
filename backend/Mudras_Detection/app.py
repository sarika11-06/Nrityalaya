from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Load model and label encoder
model = load_model('cnn_model.h5')
with open('label_encoder.pickle', 'rb') as f:
    label_encoder = pickle.load(f)

# Setup MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

@app.route('/predict', methods=['POST'])
def predict():
    # Get image from request
    if 'image' not in request.json:
        return jsonify({'error': 'No image found in request'}), 400
    
    # Get base64 image data
    image_data = request.json['image']
    
    # Remove data URI prefix if present
    if 'base64,' in image_data:
        image_data = image_data.split('base64,')[1]
    
    try:
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Process with MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # Default response
        response = {
            'mudra': 'No mudra detected',
            'confidence': 0.0,
        }
        
        # Process hand landmarks if detected
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
            
            # Extract and normalize landmarks
            for hand_landmarks in all_landmarks:
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)
            
            # Pad with zeros if only one hand detected
            if len(all_landmarks) == 1:
                data_aux.extend([0] * 42)  # 21 landmarks × 2 features per hand
            
            # Predict only if we have the right number of features
            if len(data_aux) == 84:
                input_data = np.array([data_aux])
                prediction = model.predict(input_data, verbose=0)[0]
                predicted_index = np.argmax(prediction)
                predicted_label = label_encoder.inverse_transform([predicted_index])[0]
                confidence = float(prediction[predicted_index])
                
                response = {
                    'mudra': predicted_label,
                    'confidence': confidence
                }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/mudras', methods=['GET'])
def get_mudras():
    # Return list of all mudra classes the model can recognize
    mudras = label_encoder.classes_.tolist()
    return jsonify({'mudras': mudras})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'message': 'Mudra Recognition API is running',
        'endpoints': {
            '/predict': 'POST - Predict mudra from image',
            '/mudras': 'GET - List all available mudras'
        }
    })

if __name__ == '__main__':
    print("Starting Mudra Recognition API on port 8000...")
    app.run(host='0.0.0.0', port=8000)

    