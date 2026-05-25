import joblib
import random
import sys
import cv2
import pandas as pd
import pygame
from GazeClassify import GazeClassifier

clock = pygame.time.Clock()
capture = cv2.VideoCapture(0)
gaze_map = {
    0:"centre",
    1:"left",
    2:"right",
}

if not capture.isOpened():
    print("Could not open camera")
    pygame.quit()

def run_experiment(model_path=None):

    if model_path is None:
        classifier = GazeClassifier("prediction_models/rf.mdl")
    else:
        classifier = GazeClassifier(model_path)

    pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    screen = pygame.display.get_surface()
    running = True
    results = []
    frame_count = 1

    while running and frame_count < 11:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        screen.fill("white")

        pygame.display.flip()
        pygame.time.wait(1000)

        shape_location = random.choice(["left", "centre", "right"])

        if shape_location == "left":
            pygame.draw.rect(screen,(0, 0, 0),[screen.get_width() / 4 - 50, screen.get_height() / 2 - 50, 100, 100])
        elif shape_location == "right":
            pygame.draw.rect(screen,(0, 0, 0),[screen.get_width() * 3 / 4 - 50, screen.get_height() / 2 - 50, 100, 100])
        else:
            pygame.draw.rect(screen, (0, 0, 0),[screen.get_width() / 2 - 50, screen.get_height() / 2 - 50, 100, 100])

        pygame.display.flip()
        pygame.time.wait(1000)

        ret, frame = capture.read()
        if ret:
            processed_frame = classifier.process_frame(frame)

            if processed_frame is not None:
                result = classifier.predict(processed_frame)
                result = result[0]
                pred_label = result if isinstance(result, str) else gaze_map[int(result)]
                results.append([frame_count, shape_location, pred_label])
            frame_count += 1
        pygame.time.wait(1000)

        clock.tick(60)

    df = pd.DataFrame(results, columns=['frame', 'true_region', 'prediction'])
    accuracy = (df['true_region'] == df['prediction']).mean() * 100
    df["accuracy"] = accuracy
    df.to_csv("results/results.csv")


