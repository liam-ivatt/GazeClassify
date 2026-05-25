import cv2
import joblib
import mediapipe as mp

# Iris center landmarks
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473

# Right eye landmarks
RIGHT_EYE_LEFT = 33
RIGHT_EYE_RIGHT = 133
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

# Left eye landmarks
LEFT_EYE_LEFT = 362
LEFT_EYE_RIGHT = 263
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

# Array of all required landmarks
landmarks = [RIGHT_IRIS_CENTER,
             RIGHT_EYE_LEFT,
             RIGHT_EYE_RIGHT,
             RIGHT_EYE_TOP,
             RIGHT_EYE_BOTTOM,
             LEFT_IRIS_CENTER,
             LEFT_EYE_LEFT,
             LEFT_EYE_RIGHT,
             LEFT_EYE_TOP,
             LEFT_EYE_BOTTOM]

class GazeClassifier:

    def __init__(self, model=None, num_faces=None):
        if model:
            self.model = joblib.load(model)

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="C:/Users/User/PycharmProjects/GazeClassify/landmark_model/face_landmarker.task"),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=num_faces if num_faces else 1
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

    def predict(self, features):
        return self.model.predict(features)

    # Returns landmarks, giving an image
    def get_landmarks(self, frame):

        # Convert image to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        return self.landmarker.detect(img)

    def process_frame(self, frame):

        result = self.get_landmarks(frame)

        if not result.face_landmarks:
            return None

        face = result.face_landmarks[0]

        r_rx, r_ry = relative_position(
            face[RIGHT_IRIS_CENTER],
            face[RIGHT_EYE_LEFT],
            face[RIGHT_EYE_RIGHT],
            face[RIGHT_EYE_TOP],
            face[RIGHT_EYE_BOTTOM]
        )

        l_rx, l_ry = relative_position(
            face[LEFT_IRIS_CENTER],
            face[LEFT_EYE_LEFT],
            face[LEFT_EYE_RIGHT],
            face[LEFT_EYE_TOP],
            face[LEFT_EYE_BOTTOM]
        )

        return [[r_rx, r_ry, l_rx, l_ry]]

def relative_position(centre, eye_left, eye_right, eye_top, eye_bottom):

    # left/right
    x = (centre.x - eye_left.x) / (eye_right.x - eye_left.x)

    # top/bottom
    y = (centre.y - eye_top.y) / (eye_bottom.y - eye_top.y)

    return x, y
