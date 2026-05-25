import cv2
import pandas as pd
import pygame
import sys
from GazeClassify import GazeClassifier
from experiment import run_experiment
from calibration import model_trainer, train_models

pygame.init()
clock = pygame.time.Clock()

BUTTON_WIDTH = 320
BUTTON_HEIGHT = 70

screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("IvattGaze - Third Year Project")

# Colors
BG = (30, 30, 30)
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
BLACK = (0, 0, 0)

font = pygame.font.SysFont("Arial", 40)
clock.tick(60)

# OpenCV setup
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Could not open camera")
    pygame.quit()
    exit()

# Render a message to the GUI, usually loading screens etc
def show_message(message, time=1000):

    screen.fill(WHITE)
    text = font.render(message, True, BLACK)

    text_rect = text.get_rect(center=(screen.get_width() / 2, screen.get_height() / 2))
    screen.blit(text, text_rect)
    pygame.display.update()

    pygame.time.wait(time)

# Add camera to screen
def get_camera(x_offset=0, y_offset=0):
    ret, frame = capture.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.swapaxes(0, 1)
        webcam_surface = pygame.surfarray.make_surface(frame)
        webcam_surface = pygame.transform.scale(webcam_surface, (640, 480))
        screen.blit(webcam_surface, (x_offset, y_offset))

def fullscreen():
    return pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

def draw_button(rect, text, hovered):

    color = GREY if hovered else BLACK
    pygame.draw.rect(screen, color, rect, width=3, border_radius=8)

    label = font.render(text, True, color)
    label_rect = label.get_rect(center=rect.center)
    screen.blit(label, label_rect)

def experiment_menu():

    general_model_button = pygame.Rect((screen.get_width() - BUTTON_WIDTH) // 2, screen.get_height() // 3, BUTTON_WIDTH, BUTTON_HEIGHT)
    train_own_model_button = pygame.Rect((screen.get_width() - BUTTON_WIDTH) // 2, screen.get_height() // 3 + 100, BUTTON_WIDTH, BUTTON_HEIGHT)
    calibrate_button = pygame.Rect((screen.get_width() - BUTTON_WIDTH) // 2, screen.get_height() // 3 + 200, BUTTON_WIDTH, BUTTON_HEIGHT)
    sandbox_button = pygame.Rect((screen.get_width() - BUTTON_WIDTH) // 2, screen.get_height() // 3 + 300, BUTTON_WIDTH, BUTTON_HEIGHT)
    back_button = pygame.Rect((screen.get_width() - BUTTON_WIDTH) // 2, screen.get_height() // 3 + 400, BUTTON_WIDTH, BUTTON_HEIGHT)

    while True:
        screen.fill(WHITE)
        mouse = pygame.mouse.get_pos()

        # Title
        title = font.render("IvattGaze Experiment Menu", True, BLACK)
        title_rect = title.get_rect(center=(screen.get_width() // 2, screen.get_height() // 4))
        screen.blit(title, title_rect)

        # Buttons
        draw_button(general_model_button, "Run Experiment", general_model_button.collidepoint(mouse))
        draw_button(calibrate_button, "Calibrate", calibrate_button.collidepoint(mouse))
        draw_button(train_own_model_button, "Train Own Model", train_own_model_button.collidepoint(mouse))
        draw_button(sandbox_button, "Sandbox", sandbox_button.collidepoint(mouse))
        draw_button(back_button, "Quit", back_button.collidepoint(mouse))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:

                if general_model_button.collidepoint(mouse):
                    start_experiment()

                if calibrate_button.collidepoint(mouse):
                    start_calibration()

                if train_own_model_button.collidepoint(mouse):
                    start_self_training()

                if sandbox_button.collidepoint(mouse):
                    start_sandbox()

                if back_button.collidepoint(mouse):
                    return

        pygame.display.update()

def start_experiment():

    fullscreen()
    screen = pygame.display.get_surface()
    width, height = screen.get_width(), screen.get_height()

    right_col_x = (width * 3//4) - 50

    start_button = pygame.Rect(right_col_x - BUTTON_WIDTH // 2, height // 2, BUTTON_WIDTH, BUTTON_HEIGHT)
    exit_button  = pygame.Rect(right_col_x - BUTTON_WIDTH // 2, height // 2 + 100, BUTTON_WIDTH, BUTTON_HEIGHT)

    running = True

    while running:

        screen.fill(WHITE)
        get_camera(x_offset=50, y_offset=(height // 2) - 240)

        mouse = pygame.mouse.get_pos()

        title = font.render("Please ensure your camera is working before starting", True, BLACK)
        screen.blit(title, title.get_rect(center=(screen.get_width() // 4, height * 4//5)))

        title = font.render("Press start when you are ready to run the experiment", True, BLACK)
        screen.blit(title, title.get_rect(center=(right_col_x, height // 3)))

        title = font.render("Press q to exit during any point of the process", True, BLACK)
        screen.blit(title, title.get_rect(center=(right_col_x, height // 3 + 50)))

        # Draw buttons
        draw_button(start_button, "Start Experiment", start_button.collidepoint(mouse))
        draw_button(exit_button, "Exit", exit_button.collidepoint(mouse))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if start_button.collidepoint(mouse):
                    show_message("Experiment beginning shortly")
                    run_experiment()
                    show_results()
                    return
                if exit_button.collidepoint(mouse):
                    return

        pygame.display.update()

def start_self_training():

    fullscreen()
    screen = pygame.display.get_surface()

    width, height = screen.get_width(), screen.get_height()
    right_col_x = (width * 3//4) - 50

    start_button = pygame.Rect(right_col_x - BUTTON_WIDTH // 2, height // 2, BUTTON_WIDTH, BUTTON_HEIGHT)
    exit_button  = pygame.Rect(right_col_x - BUTTON_WIDTH // 2, height // 2 + 100, BUTTON_WIDTH, BUTTON_HEIGHT)

    running = True

    while running:

        screen.fill(WHITE)
        get_camera(x_offset=50, y_offset=(height // 2) - 240)

        mouse = pygame.mouse.get_pos()

        title = font.render("Please ensure your camera is working before starting", True, BLACK)
        screen.blit(title, title.get_rect(center=(screen.get_width() // 4, height * 4//5)))

        title = font.render("Press start when you are ready to train your own model.", True, BLACK)
        screen.blit(title, title.get_rect(center=(right_col_x, height // 3)))

        title = font.render("Press q to exit during any point of the process.", True, BLACK)
        screen.blit(title, title.get_rect(center=(right_col_x, height // 3 + 50)))

        # Draw buttons
        draw_button(start_button, "Start Training", start_button.collidepoint(mouse))
        draw_button(exit_button, "Exit", exit_button.collidepoint(mouse))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if start_button.collidepoint(mouse):
                    data = model_trainer()
                    show_message("Training models...")
                    model_path = train_models(data, True)
                    show_message("Experiment beginning shortly")
                    run_experiment(model_path)
                    show_results()
                    return

                if exit_button.collidepoint(mouse):
                    return

        pygame.display.update()

def start_calibration():

    fullscreen()
    screen = pygame.display.get_surface()

    width, height = screen.get_width(), screen.get_height()
    right_col_x = (width * 3//4) - 50

    start_button = pygame.Rect(right_col_x - BUTTON_WIDTH // 2, height // 2, BUTTON_WIDTH, BUTTON_HEIGHT)
    exit_button  = pygame.Rect(right_col_x - BUTTON_WIDTH // 2, height // 2 + 100, BUTTON_WIDTH, BUTTON_HEIGHT)

    running = True

    while running:

        screen.fill(WHITE)
        get_camera(x_offset=50, y_offset=(height // 2) - 240)

        mouse = pygame.mouse.get_pos()

        title = font.render("Please ensure your camera is working before starting", True, BLACK)
        screen.blit(title, title.get_rect(center=(screen.get_width() // 4, height * 4//5)))

        title = font.render("Press start when you are ready to calibrate", True, BLACK)
        screen.blit(title, title.get_rect(center=(right_col_x, height // 3)))

        title = font.render("Press q to exit during any point of the process.", True, BLACK)
        screen.blit(title, title.get_rect(center=(right_col_x, height // 3 + 50)))

        # Draw buttons
        draw_button(start_button, "Start Training", start_button.collidepoint(mouse))
        draw_button(exit_button, "Exit", exit_button.collidepoint(mouse))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if start_button.collidepoint(mouse):
                    data = model_trainer()
                    dataset = pd.concat([data, pd.read_csv("data/dataset_test_logo.csv").drop(columns=["participant"])], ignore_index=True)
                    show_message("Training model...")
                    model_path = train_models(dataset, False)
                    show_message("Experiment beginning shortly")
                    run_experiment(model_path)
                    show_results()
                    return

                if exit_button.collidepoint(mouse):
                    return

        pygame.display.update()

def start_sandbox():

    fullscreen()
    screen = pygame.display.get_surface()
    width, height = screen.get_width(), screen.get_height()
    classifier = GazeClassifier("prediction_models/rf.mdl")

    show_landmarks  = False
    show_prediction = False

    gaze_map = {0: "centre", 1: "left", 2: "right"}

    camera_width, camera_height = 640*1.4, 480*1.4
    camera_x = width - camera_width - 150
    camera_y = (height - camera_height) // 2
    left_col_x = width // 5

    landmark_button = pygame.Rect(left_col_x - BUTTON_WIDTH // 2, height // 2 - 120, BUTTON_WIDTH, BUTTON_HEIGHT)
    prediction_button = pygame.Rect(left_col_x - BUTTON_WIDTH // 2, height // 2 - 120 + 100, BUTTON_WIDTH, BUTTON_HEIGHT)
    quit_button = pygame.Rect(left_col_x - BUTTON_WIDTH // 2, height // 2 - 120 + 200, BUTTON_WIDTH, BUTTON_HEIGHT)

    while True:
        ret, frame = capture.read()
        if not ret:
            continue

        screen.fill(WHITE)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        frame_surface = pygame.transform.scale(frame_surface, (camera_width, camera_height))
        screen.blit(frame_surface, (camera_x, camera_y))

        frame_result = classifier.get_landmarks(frame_rgb)
        if frame_result.face_landmarks:
            face = frame_result.face_landmarks[0]

            if show_landmarks:
                for point in face:
                    px = int(point.x * camera_width) + camera_x
                    py = int(point.y * camera_height) + camera_y
                    pygame.draw.circle(screen, (0, 200, 255), (px, py), 2)

            if show_prediction:
                features = classifier.process_frame(frame)
                if features:
                    pred = classifier.predict(features)[0]
                    pred_text = font.render(f"Gaze: {gaze_map[pred]}", True, BLACK)
                    screen.blit(pred_text, pred_text.get_rect(center=(camera_x + camera_width // 2, camera_y + camera_height + 40)))

        title = font.render("Sandbox", True, BLACK)
        screen.blit(title, title.get_rect(center=(left_col_x, height // 4)))

        mouse = pygame.mouse.get_pos()
        draw_button(landmark_button,"Landmarks", landmark_button.collidepoint(mouse))
        draw_button(prediction_button,"Prediction", prediction_button.collidepoint(mouse))
        draw_button(quit_button,"Quit", quit_button.collidepoint(mouse))

        if show_landmarks:
            pygame.draw.rect(screen, (0, 180, 80), landmark_button, width=3, border_radius=8)
        if show_prediction:
            pygame.draw.rect(screen, (0, 180, 80), prediction_button, width=3, border_radius=8)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return
            if event.type == pygame.MOUSEBUTTONDOWN:
                if landmark_button.collidepoint(mouse):
                    show_landmarks = not show_landmarks
                if prediction_button.collidepoint(mouse):
                    show_prediction = not show_prediction
                if quit_button.collidepoint(mouse):
                    return

def show_results():
    fullscreen()
    screen = pygame.display.get_surface()
    width, height = screen.get_width(), screen.get_height()

    df = pd.read_csv("results/results.csv")
    last_accuracy = df['accuracy'].iloc[-1]

    quit_button = pygame.Rect((width - BUTTON_WIDTH) // 2, height // 2 + 100, BUTTON_WIDTH, BUTTON_HEIGHT)

    while True:
        screen.fill(WHITE)
        mouse = pygame.mouse.get_pos()

        title = font.render("Experiment Complete", True, BLACK)
        screen.blit(title, title.get_rect(center=(width // 2, height // 3)))

        acc = font.render(f"Accuracy: {last_accuracy:.1f}%", True, BLACK)
        screen.blit(acc, acc.get_rect(center=(width // 2, height // 2)))

        draw_button(quit_button, "Back to Menu", quit_button.collidepoint(mouse))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if quit_button.collidepoint(mouse):
                    pygame.quit()
                    sys.exit()

        pygame.display.update()

experiment_menu()