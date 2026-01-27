import os
import mediapipe as mp
import pandas as pd
from GazeClassify import relative_position, landmarks

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="landmark_model/face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE)

with FaceLandmarker.create_from_options(options) as landmarker:

    # Dataset rows
    features = []
    names = []
    labels = []

    for image in os.listdir(f"data/"):

        img = mp.Image.create_from_file(f"data/{image}")
        face_landmarker = landmarker.detect(img)
        face_landmarks = face_landmarker.face_landmarks[0]

        r_rx, r_ry = relative_position(
            face_landmarks[landmarks[0]],
            face_landmarks[landmarks[1]],
            face_landmarks[landmarks[2]],
            face_landmarks[landmarks[3]],
            face_landmarks[landmarks[4]],
            640, 480
        )

        l_rx, l_ry = relative_position(
            face_landmarks[landmarks[5]],
            face_landmarks[landmarks[6]],
            face_landmarks[landmarks[7]],
            face_landmarks[landmarks[8]],
            face_landmarks[landmarks[9]],
            640, 480
        )

        features.append([r_rx, r_ry, l_rx, l_ry])
        labels.append(image.split("_")[1])

    columns = []
    columns.extend([f"right_relative_x", "right_relative_y", "left_relative_x", "left_relative_y"])

    df = pd.DataFrame(features, columns=columns)
    df["label"] = labels

    df.to_csv("dataset.csv", index=False)








