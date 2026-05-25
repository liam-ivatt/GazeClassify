import os
import mediapipe as mp
import pandas as pd
from GazeClassify import relative_position, landmarks, GazeClassifier

def main(image_path):

    classifier = GazeClassifier()

    features = []
    labels = []
    participants = []

    for participant in os.listdir(f"{image_path}"):
        for image in os.listdir(f"{image_path}/{participant}"):
            img = mp.Image.create_from_file(f"{image_path}/{participant}/{image}")
            face_landmarker = classifier.landmarker.detect(img)

            if face_landmarker.face_landmarks:

                face_landmarks = face_landmarker.face_landmarks[0]

                r_rx, r_ry = relative_position(
                    face_landmarks[landmarks[0]],
                    face_landmarks[landmarks[1]],
                    face_landmarks[landmarks[2]],
                    face_landmarks[landmarks[3]],
                    face_landmarks[landmarks[4]]
                )

                l_rx, l_ry = relative_position(
                    face_landmarks[landmarks[5]],
                    face_landmarks[landmarks[6]],
                    face_landmarks[landmarks[7]],
                    face_landmarks[landmarks[8]],
                    face_landmarks[landmarks[9]]
                )

                features.append([r_rx, r_ry, l_rx, l_ry])
                labels.append(image.split("_")[1])
                participants.append(participant)

        columns = ["right_relative_x", "right_relative_y", "left_relative_x", "left_relative_y"]

        df = pd.DataFrame(features, columns=columns)
        df["label"] = labels
        df["participant"] = participants

        df.to_csv(f"data/dataset_test_logo.csv", index=False)