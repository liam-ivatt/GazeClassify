import pickle
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import mediapipe as mp
from GazeClassify import relative_position, landmarks

# Load prediction model once
with open('../prediction_models/dtree.pkl', 'rb') as file:
    model = pickle.load(file)

# Load MediaPipe landmarker once
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="../landmark_model/face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE
)

landmarker = FaceLandmarker.create_from_options(options)

def extract_features(mp_image):
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    face_landmarks = result.face_landmarks[0]

    r_rx, r_ry = relative_position(
        face_landmarks[landmarks[0]],
        face_landmarks[landmarks[1]],
        face_landmarks[landmarks[2]],
        face_landmarks[landmarks[3]],
        face_landmarks[landmarks[4]],
    )

    l_rx, l_ry = relative_position(
        face_landmarks[landmarks[5]],
        face_landmarks[landmarks[6]],
        face_landmarks[landmarks[7]],
        face_landmarks[landmarks[8]],
        face_landmarks[landmarks[9]],
    )

    return [[r_rx, r_ry, l_rx, l_ry]]  # 2D array for sklearn


app = Flask(__name__)
CORS(app)

@app.route('/process_frame', methods=['POST'])
def predict():
    file = request.files.get('frame')
    if file is None:
        return jsonify({'prediction': None})

    img_bytes = file.read()

    # Decode JPEG to NumPy array
    npimg = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Wrap in MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Extract features
    features = extract_features(mp_image)
    if features is None:
        return jsonify({'prediction': None})

    # Predict gaze
    prediction = model.predict(features)[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
