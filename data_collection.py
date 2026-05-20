import pygame
import cv2
import time

def main():

    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080))
    screen.fill("white")
    running = True

    # OpenCV setup
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Could not open camera")
        pygame.quit()
        exit()

    # Capture setup
    images_per_point = 100
    class_names = ["left", "centre", "right"]

    def capture_images(images_per_point, class_names):

        # Capture images
        for img_id in range(900, 1000):

            screen.fill("black")
            pygame.draw.circle(screen, "green", pos, 40)
            pygame.display.flip()

            ret, frame = capture.read()

            if ret:
                filename = f"data/data_main/point_{class_names[pos_id]}_{img_id}.png"
                cv2.imwrite(filename, frame)
                print(f"Captured {filename}")

            if not running:
                return False

        return None

    start_pos = pygame.Vector2(screen.get_width() / 6, screen.get_height() / 2)
    centre_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
    end_pos = pygame.Vector2(screen.get_width() * 5 / 6, screen.get_height() / 2)
    positions = [start_pos, centre_pos, end_pos]

    font = pygame.font.SysFont(None, 60)
    intro_text = font.render("Setting up calibration.", True, "black")

    # Show intro text for 5 seconds
    screen.fill("white")
    screen.blit(intro_text, (screen.get_width() / 2 - intro_text.get_width() / 2,
                             screen.get_height() / 2 - intro_text.get_height() / 2))
    pygame.display.flip()
    pygame.time.wait(2000)

    if not running:
        return False

    for pos_id, pos in enumerate(positions):
        # Draw circle
        screen.fill("black")
        pygame.draw.circle(screen, "red", pos, 40)
        pygame.display.flip()

        time.sleep(1)

        capture_images(images_per_point, class_names)

        screen.fill("black")
        pygame.draw.circle(screen, "red", pos, 40)
        pygame.display.flip()

        pygame.time.wait(3000)

    capture.release()
    pygame.quit()

if __name__ == "__main__":
    main()

