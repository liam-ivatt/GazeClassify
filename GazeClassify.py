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

def relative_position(centre, eye_left, eye_right, eye_top, eye_bottom):

    x = (centre.x - eye_left.x) / (eye_right.x - eye_left.x)
    y = (centre.y - eye_top.y) / (eye_bottom.y - eye_top.y)

    return x, y

