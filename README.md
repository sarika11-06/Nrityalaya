# Nrityalya - Bharatnatyam E-Learning and Posture and Mudra Detection

This full-stack project aims to detect and classify Bharatanatyam dance postures using computer vision and deep learning models. It includes a Python backend for body landmark extraction and posture classification, and a React frontend for real-time visual feedback.

Features

✅ Capture Bharatanatyam postures via webcam
✅ Extract body landmarks using **Mediapipe**
✅ Create a labeled dataset
✅ Train a **Random Forest** classifier
✅ Run live inference via **FastAPI backend**
✅ View real-time predictions in a **React frontend**

## 1️⃣ Backend Setup
### 📦 Install Dependencies
Make sure you have Python 3.8+ installed.

pip install fastapi uvicorn opencv-python mediapipe numpy scikit-learn pillow

### 📸 Collect Posture Images
python collect_body_imgs.py

Press `Q` to begin capturing images for each posture class.

### 📊 Create Dataset
python create_dataset.py

This will generate a `data.pickle` file with posture landmark features.

### 🧠 Train Classifier
python train_classifier.py

Creates `model.p`, a trained Random Forest model for posture recognition.

### 🚀 Launch FastAPI Server

python app.py


## 2️⃣ Frontend Setup (React)

### ⚙️ Navigate & Install
cd frontend
npm install

### ▶️ Run React App
npm start

This will launch the React app on `http://localhost:`, which connects to the FastAPI backend for live posture classification.

## 🔬 Model Overview

* Uses **Mediapipe** to extract full-body landmarks
* Features: Angles or coordinates of joints
* Output: Posture class (e.g., *Samapadam*, *Aramandi*)

## 🧑‍🎓 Ideal For

* Bharatanatyam students practicing postures
* Dance instructors offering online feedback
* E-learning platforms to add posture correctness


## 💡 Credits
Built using:
* [OpenCV](https://opencv.org/)
* [Mediapipe](https://google.github.io/mediapipe/)
* [FastAPI](https://fastapi.tiangolo.com/)
* [ReactJS](https://reactjs.org/)

