from info_ import *
import pygame

def build_Nikolskaya(screen):
    screen.fill(FONT)
    pygame.display.update()
    # left wall
    r_left = pygame.Rect(0, 80, 5, 500)
    pygame.draw.rect(screen, BLACK, r_left, 0)
    # right wall
    r_right = pygame.Rect(795, 80, 5, 500)
    pygame.draw.rect(screen, BLACK, r_right, 0)
    # up
    r_up1 = pygame.Rect(50, 80, 325, 5)
    pygame.draw.rect(screen, BLACK, r_up1, 0)
    r_up2 = pygame.Rect(425, 80, 325, 5)
    pygame.draw.rect(screen, BLACK, r_up2, 0)
    # down
    r_down = pygame.Rect(0, 580, 800, 5)
    pygame.draw.rect(screen, BLACK, r_down, 0)

    # 1st wall
    r_1 = pygame.Rect(98, 180, 40, 300)
    pygame.draw.rect(screen, BLACK, r_1, 0)
    # 6st wall
    r_6 = pygame.Rect(662, 180, 40, 300)
    pygame.draw.rect(screen, BLACK, r_6, 0)
    # 2st wall
    r_2 = pygame.Rect(268, 80, 70, 190)
    pygame.draw.rect(screen, BLACK, r_2, 0)
    # 4st wall
    r_4 = pygame.Rect(462, 80, 70, 190)
    pygame.draw.rect(screen, BLACK, r_4, 0)
    # 3st wall
    r_3 = pygame.Rect(268, 353, 70, 100)
    pygame.draw.rect(screen, BLACK, r_3, 0)
    # 5st wall
    r_5 = pygame.Rect(462, 353, 70, 100)
    pygame.draw.rect(screen, BLACK, r_5, 0)

    # entrance
    pygame.draw.line(screen, BLACK, [395, 110], [425, 80], 5)
    # exit
    pygame.draw.line(screen, BLACK, [0, 80], [30, 50], 5)
    pygame.draw.line(screen, BLACK, [770, 50], [800, 80], 5)

    korm_1 = pygame.Rect(185, 580, 50, 20)
    pygame.draw.rect(screen, GREEN, korm_1, 0)
    korm_2 = pygame.Rect(563, 580, 50, 20)
    pygame.draw.rect(screen, GREEN, korm_2, 0)

    not_korm_1 = pygame.Rect(185, 60, 50, 20)
    pygame.draw.rect(screen, GRAY, not_korm_1, 0)
    not_korm_2 = pygame.Rect(563, 60, 50, 20)
    pygame.draw.rect(screen, GRAY, not_korm_2, 0)


def build_Berezhnoy(screen):
    screen.fill(BLACK)
    pygame.display.update()
    # rects
    for i in range(4):
        r1 = pygame.Rect(150 + i * 150, 0, 50, 600)
        pygame.draw.rect(screen, BLUE, r1, 0)
        r2 = pygame.Rect(100, 50 + i * 150, 600, 50)
        pygame.draw.rect(screen, BLUE, r2, 0)

    # squares
    for i in range(4):
        for j in range(4):
            sq = pygame.Rect(125 + i * 150, 25 + j * 150, 100, 100)
            pygame.draw.rect(screen, BLUE, sq, 0)

    # enter
    enter = pygame.Rect(125 + 150, 25 + 150, 100, 100)
    pygame.draw.rect(screen, GREEN, enter, 10)

    # korm
    korm = pygame.Rect(125 + 3 * 150, 25 + 3 * 150, 100, 100)
    pygame.draw.rect(screen, GRAY, korm, 10)


def build_Chelnok(screen):
    screen.fill(BLACK)
    pygame.display.update()

    # up
    r_left = pygame.Rect(37, 6, 726, 66)
    pygame.draw.rect(screen, BLUE, r_left, 0)

    # left walls
    r_right = pygame.Rect(37, 6, 66, 330)
    pygame.draw.rect(screen, BLUE, r_right, 0)
    r_right_2 = pygame.Rect(37, 402, 66, 198)
    pygame.draw.rect(screen, BLUE, r_right_2, 0)

    # right walls
    r_up1 = pygame.Rect(697, 6, 66, 330)
    pygame.draw.rect(screen, BLUE, r_up1, 0)
    r_up2 = pygame.Rect(697, 402, 66, 198)
    pygame.draw.rect(screen, BLUE, r_up2, 0)

    # down
    r_down = pygame.Rect(37, 534, 726, 66)
    pygame.draw.rect(screen, BLUE, r_down, 0)

    for i in range(4):
        for j in range(3):
            sq = pygame.Rect(169 + i * 132, 138 + 132 * j, 66, 66)
            pygame.draw.rect(screen, BLUE, sq, 0)

    # entrance
    pygame.draw.line(screen, BLUE, [367, 204], [397, 234], 5)
    # exit
    pygame.draw.line(screen, BLUE, [103, 138], [133, 108], 5)
    pygame.draw.line(screen, BLUE, [664, 108], [694, 138], 5)

    # korm
    korm_1 = pygame.Rect(730, 336, 33, 66)
    pygame.draw.rect(screen, GREEN, korm_1, 0)
    korm_2 = pygame.Rect(37, 336, 33, 66)
    pygame.draw.rect(screen, GREEN, korm_2, 0)

    # not korm
    not_korm_1 = pygame.Rect(235, 138, 66, 66)
    pygame.draw.rect(screen, GRAY, not_korm_1, 0)
    not_korm_2 = pygame.Rect(235, 534, 66, 66)
    pygame.draw.rect(screen, GRAY, not_korm_2, 0)
    not_korm_3 = pygame.Rect(499, 138, 66, 66)
    pygame.draw.rect(screen, GRAY, not_korm_3, 0)
    not_korm_4 = pygame.Rect(499, 534, 66, 66)
    pygame.draw.rect(screen, GRAY, not_korm_4, 0)
