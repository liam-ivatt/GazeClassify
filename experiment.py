import math
import random
import sys

import pygame

pygame.init()

screen = pygame.display.set_mode((1920, 1080))
clock = pygame.time.Clock()
running = True

def draw_divider():
    pygame.draw.rect(screen, (0, 0, 0), [screen.get_width() / 2, 0, 1, screen.get_height()])

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    # Draw blank screen with divider
    screen.fill("white")
    draw_divider()
    pygame.display.flip()
    pygame.time.wait(1000)

    shape_location = random.randrange(1, 10)

    if shape_location < 5:
        pygame.draw.rect(screen, (0, 0, 0), [screen.get_width() / 4, screen.get_height() / 2, 100, 100])
    else:
        pygame.draw.rect(screen, (0, 0, 0), [screen.get_width() * 3 / 4, screen.get_height() / 2, 100, 100])

    pygame.display.flip()
    pygame.time.wait(1000)

    clock.tick(60)

pygame.quit()
sys.exit()