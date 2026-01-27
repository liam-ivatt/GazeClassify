import pickle

# Iris center landmarks
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473

RIGHT_EYE_LEFT = 33
RIGHT_EYE_RIGHT = 133
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

LEFT_EYE_LEFT = 362
LEFT_EYE_RIGHT = 263
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

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

    def __init__(self, model):

        model = f"prediction_models/{model}.pkl"
        self.model = pickle.load(open(model, 'rb'))

    def predict(self, features):
        return self.model.predict(features)

def relative_position(centre, eye_left, eye_right, eye_top, eye_bottom, width, height):

    iris_x, iris_y = centre.x * width, centre.y * height
    left_x, right_x = eye_left.x * width, eye_right.x * width
    top_y, bottom_y = eye_top.y * height, eye_bottom.y * height

    r_x = (iris_x - left_x) / (right_x - left_x)
    r_y = (iris_y - top_y) / (bottom_y - top_y)

    return r_x, r_y

