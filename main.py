import cv2
import mediapipe as mp
import numpy as np
from GazeClassify import GazeClassifier, relative_position, landmarks

def main():

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="landmark_model/face_landmarker.task"),
        running_mode=VisionRunningMode.IMAGE)

    while True:
        try:
            selected_model = input("Select landmark_model for gaze prediction: ")
            model = GazeClassifier(selected_model)
            break
        except Exception:
            print(f"{selected_model} is not a valid prediction model.")

    with FaceLandmarker.create_from_options(options) as landmarker:

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect(img)

            if result.face_landmarks:

                face_landmarks = result.face_landmarks[0]
                features = []

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
                predict = model.predict(features)

                cv2.putText(frame, f"Gaze: {predict[0]}", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Gaze Prediction", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


