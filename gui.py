import cv2
import pandas as pd
import pygame
import sys

from experiment import run_experiment
from calibration import model_trainer, train_models

pygame.init()
clock = pygame.time.Clock()

BUTTON_WIDTH = 320
BUTTON_HEIGHT = 70

screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("IvattGaze")

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

def show_message(message, time=1000):

    screen.fill(WHITE)
    text = font.render(message, True, BLACK)

    text_rect = text.get_rect(center=(screen.get_width() / 2, screen.get_height() / 2))
    screen.blit(text, text_rect)
    pygame.display.update()

    pygame.time.wait(time)

def get_camera():

    ret, frame = capture.read()

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.swapaxes(0, 1)
        webcam_surface = pygame.surfarray.make_surface(frame)

        webcam_surface = pygame.transform.scale(webcam_surface, (640, 480))
        screen.blit(webcam_surface, (screen.get_width() // 2 - 240, screen.get_height() // 2 + 200))

def fullscreen():
    return pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

def draw_button(rect, text, hovered):
    # Outline only
    color = GREY if hovered else BLACK
    pygame.draw.rect(screen, color, rect, width=3, border_radius=8)

    label = font.render(text, True, color)
    label_rect = label.get_rect(center=rect.center)
    screen.blit(label, label_rect)

def start_menu():
    experiment_button = pygame.Rect((screen.get_width() - BUTTON_WIDTH) // 2, screen.get_height() // 2 , BUTTON_WIDTH, BUTTON_HEIGHT)
    quit_button = pygame.Rect((screen.get_width() - BUTTON_WIDTH) // 2, screen.get_height() // 2 + 100, BUTTON_WIDTH, BUTTON_HEIGHT)

    while True:
        screen.fill(WHITE)

        mouse = pygame.mouse.get_pos()

        # Draw buttons
        draw_button(experiment_button, "Run Experiment", experiment_button.collidepoint(mouse))
        draw_button(quit_button, "Quit", quit_button.collidepoint(mouse))

        # Title
        title = font.render("IvattGaze", True, BLACK)
        title_rect = title.get_rect(center=(screen.get_width() // 2, screen.get_height() // 3))
        screen.blit(title, title_rect)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if experiment_button.collidepoint(mouse):
                    experiment_menu()

                if quit_button.collidepoint(mouse):
                    pygame.quit()
                    sys.exit()

        pygame.display.update()

def experiment_menu():

    general_model_button = pygame.Rect((screen.get_width() - BUTTON_WIDTH) // 2, screen.get_height() // 2, BUTTON_WIDTH, BUTTON_HEIGHT)
    train_own_model_button = pygame.Rect((screen.get_width() - BUTTON_WIDTH) // 2, screen.get_height() // 2 + 100, BUTTON_WIDTH, BUTTON_HEIGHT)
    calibrate_button = pygame.Rect((screen.get_width() - BUTTON_WIDTH) // 2, screen.get_height() // 2 + 200, BUTTON_WIDTH, BUTTON_HEIGHT)
    back_button = pygame.Rect((screen.get_width() - BUTTON_WIDTH) // 2, screen.get_height() // 2 + 300, BUTTON_WIDTH, BUTTON_HEIGHT)

    while True:
        screen.fill(WHITE)
        mouse = pygame.mouse.get_pos()

        # Title
        title = font.render("IvattGaze Experiment Menu", True, BLACK)
        title_rect = title.get_rect(center=(screen.get_width() // 2, screen.get_height() // 3))
        screen.blit(title, title_rect)

        # Buttons
        draw_button(general_model_button, "Run Experiment", general_model_button.collidepoint(mouse))
        draw_button(calibrate_button, "Calibrate", calibrate_button.collidepoint(mouse))
        draw_button(train_own_model_button, "Train Own Model", train_own_model_button.collidepoint(mouse))
        draw_button(back_button, "Back", back_button.collidepoint(mouse))

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

                if back_button.collidepoint(mouse):
                    return

        pygame.display.update()

def start_experiment():

    fullscreen()
    screen = pygame.display.get_surface()

    start_button = pygame.Rect((screen.get_width() - BUTTON_WIDTH) // 2, screen.get_height() // 2, BUTTON_WIDTH, BUTTON_HEIGHT)
    exit_button = pygame.Rect((screen.get_width() - BUTTON_WIDTH) // 2, screen.get_height() // 2 + 100, BUTTON_WIDTH, BUTTON_HEIGHT)

    running = True

    while running:

        screen.fill(WHITE)
        get_camera()

        mouse = pygame.mouse.get_pos()

        # Draw text
        title = font.render("Press start when you are ready to run the experiment.", True, BLACK)
        title_rect = title.get_rect(center=(screen.get_width() // 2, screen.get_height() // 4))
        screen.blit(title, title_rect)

        title = font.render("Press q to exit during any point of the process.", True, BLACK)
        title_rect = title.get_rect(center=(screen.get_width() // 2, screen.get_height() // 3))
        screen.blit(title, title_rect)

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

                if exit_button.collidepoint(mouse):
                    return

        pygame.display.update()

def start_self_training():

    fullscreen()
    screen = pygame.display.get_surface()

    start_button = pygame.Rect((screen.get_width() - BUTTON_WIDTH) // 2, screen.get_height() // 2, BUTTON_WIDTH, BUTTON_HEIGHT)
    exit_button = pygame.Rect((screen.get_width() - BUTTON_WIDTH) // 2, screen.get_height() // 2 + 100, BUTTON_WIDTH, BUTTON_HEIGHT)

    running = True

    while running:

        screen.fill(WHITE)
        get_camera()

        mouse = pygame.mouse.get_pos()

        # Draw text
        title = font.render("Press start when you are ready to train your own model.", True, BLACK)
        title_rect = title.get_rect(center=(screen.get_width() // 2, screen.get_height() // 4))
        screen.blit(title, title_rect)

        title = font.render("Press q to exit during any point of the process.", True, BLACK)
        title_rect = title.get_rect(center=(screen.get_width() // 2, screen.get_height() // 3))
        screen.blit(title, title_rect)

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

                if exit_button.collidepoint(mouse):
                    return

        pygame.display.update()

def start_calibration():

    fullscreen()
    screen = pygame.display.get_surface()

    start_button = pygame.Rect((screen.get_width() - BUTTON_WIDTH) // 2, screen.get_height() // 2, BUTTON_WIDTH, BUTTON_HEIGHT)
    exit_button = pygame.Rect((screen.get_width() - BUTTON_WIDTH) // 2, screen.get_height() // 2 + 100, BUTTON_WIDTH, BUTTON_HEIGHT)

    running = True

    while running:

        screen.fill(WHITE)
        get_camera()

        mouse = pygame.mouse.get_pos()

        # Draw text
        title = font.render("Press start when you are ready to calibrate the generalised model..", True, BLACK)
        title_rect = title.get_rect(center=(screen.get_width() // 2, screen.get_height() // 4))
        screen.blit(title, title_rect)

        title = font.render("Press q to exit during any point of the process.", True, BLACK)
        title_rect = title.get_rect(center=(screen.get_width() // 2, screen.get_height() // 3))
        screen.blit(title, title_rect)

        # Draw buttons
        draw_button(start_button, "Start Training", start_button.collidepoint(mouse))
        draw_button(exit_button, "Exit", exit_button.collidepoint(mouse))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if start_button.collidepoint(mouse):
                    data = model_trainer()
                    dataset = pd.concat([data, pd.read_csv("data/dataset.csv")], ignore_index=True)

                    show_message("Training model...")

                    model_path = train_models(dataset, False)

                    show_message("Experiment beginning shortly")

                    run_experiment(model_path)

                if exit_button.collidepoint(mouse):
                    return

        pygame.display.update()


start_menu()