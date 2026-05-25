import pandas as pd
import pygame
import cv2
import time
import joblib
from sklearn.model_selection import train_test_split
from GazeClassify import GazeClassifier

# IMPORT MODEL
from models.random_forest import main as rf

# Capture setup
class_names = ["left", "centre", "right"]
results = []
columns = ["right_relative_x", "right_relative_y", "left_relative_x", "left_relative_y", "label"]
classifier = GazeClassifier()

def train_models(dataset, self_training):

    x = dataset.drop(["label"], axis=1).to_numpy()
    y = dataset["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=20
    )

    model = rf(x_train, y_train, x_test, y_test)

    if self_training:
        model_path = "self_training/custom_model.pkl"
        joblib.dump(model, model_path)
    else:
        model_path = "calibrated_model/custom_model.pkl"
        joblib.dump(model, model_path)
    return model_path

def capture_images(capture, class_names, pos_id, pos):

    screen = pygame.display.get_surface()

    # Capture images
    for id in range(0, 100):

        screen.fill("white")
        pygame.draw.circle(screen, "green", pos, 40)
        pygame.display.flip()

        ret, frame = capture.read()

        if ret:
            processed_image = classifier.process_frame(frame)
            if processed_image:
                processed_image = processed_image[0]
                processed_image.append(class_names[pos_id])
                results.append(processed_image)

    return None

def model_trainer(capture):

    screen = pygame.display.get_surface()

    start_pos = pygame.Vector2(screen.get_width() / 6, screen.get_height() / 2)
    centre_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
    end_pos = pygame.Vector2(screen.get_width() * 5 / 6, screen.get_height() / 2)
    positions = [start_pos, centre_pos, end_pos]

    font = pygame.font.SysFont(None, 60)
    intro_text = font.render("Prepare for calibration", True, "black")

    # Show intro text for 5 seconds
    screen.fill("white")
    screen.blit(intro_text, (screen.get_width() / 2 - intro_text.get_width() / 2,
                             screen.get_height() / 2 - intro_text.get_height() / 2))
    pygame.display.flip()
    pygame.time.wait(3000)

    for pos_id, pos in enumerate(positions):

        screen.fill("white")
        pygame.draw.circle(screen, "red", pos, 40)
        pygame.display.flip()

        time.sleep(1)

        capture_images(capture, class_names, pos_id, pos)

        screen.fill("white")
        pygame.draw.circle(screen, "red", pos, 40)
        pygame.display.flip()

        pygame.time.wait(3000)

    return pd.DataFrame(results, columns=columns)

