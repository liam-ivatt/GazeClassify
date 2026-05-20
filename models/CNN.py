import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Rescaling
from keras.models import Model, Sequential
from keras.src.callbacks import EarlyStopping
from keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from GazeClassify import GazeClassifier

classifier = GazeClassifier()
RIGHT_EYE = [33, 133, 159, 145]
LEFT_EYE = [362, 263, 386, 374]

def crop_eye(image, landmarks, eye_points):

    height, width, _ = image.shape

    x_landmarks = []
    y_landmarks = []

    for id in eye_points:

        landmark = landmarks[id]

        x_landmarks.append(int(landmark.x * width))
        y_landmarks.append(int(landmark.y * height))

    x_min = max(min(x_landmarks) - 20, 0)
    x_max = min(max(x_landmarks) + 20, width)

    y_min = max(min(y_landmarks) - 10, 0)
    y_max = min(max(y_landmarks) + 10, height)

    return image[y_min:y_max, x_min:x_max]

def preprocess_image(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = classifier.get_landmarks(image)

    if landmarks.face_landmarks is not None:

        left_eye = crop_eye(image, landmarks.face_landmarks[0], LEFT_EYE)
        right_eye = crop_eye(image, landmarks.face_landmarks[0], RIGHT_EYE)

        left_eye = cv2.resize(left_eye, (128, 128))
        right_eye = cv2.resize(right_eye, (128, 128))
        combined_image = np.hstack((left_eye, right_eye))

        return combined_image

    return None

def process_folder(input, output):

    os.makedirs(output, exist_ok=True)

    for file in os.listdir(input):

        image_path = os.path.join(input, file)
        image = cv2.imread(image_path)

        if image is None:
            continue

        processed_image = preprocess_image(image)

        if processed_image is not None:

            save_path = os.path.join(output, file)
            cv2.imwrite(save_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

train_ds = image_dataset_from_directory(
    "C:/Users/User/PycharmProjects/GazeClassify/data/self_test_data/cnn_preprocessing/cnn_train",
    seed=20,
    image_size=(128, 256),
    batch_size=32,
)

val_ds = image_dataset_from_directory(
    "C:/Users/User/PycharmProjects/GazeClassify/data/self_test_data/cnn_preprocessing/cnn_validation",
    image_size=(128, 256),
    batch_size=32,
    shuffle=False,
)

model = Sequential([

    Rescaling(1./255, input_shape=(128, 256, 3)),

    Conv2D(16, 3, padding='same', activation='relu', input_shape=(128, 256, 3)),
    MaxPooling2D(),

    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Flatten(),

    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')

])

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stopping])

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

print(classification_report(y_true, y_pred, target_names=val_ds.class_names))

disp = ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    cmap='Blues',
    colorbar=True,
    display_labels=val_ds.class_names
)

plt.title('Confusion Matrix')
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(history.history['accuracy']))
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()




