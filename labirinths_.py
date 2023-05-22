from info_ import *
from build_ import *
from path_ import *
import pygame
import sys


def labirinth_Nikolskaya(screen, font, clock, tmp_df_mouse, tmp_df_fish, tmp_df_monkey, tmp_df_person):
    build_Nikolskaya(screen)

    user_text = ''
    input_rect = pygame.Rect(300, 10, 200, 30)
    color_active = pygame.Color('lightskyblue3')
    color_passive = pygame.Color(GRAY)
    color = color_passive
    active = False
    iter_ = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if input_rect.collidepoint(event.pos):
                    active = True
                else:
                    active = False
            if event.type == pygame.KEYDOWN:
                if event.type == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]

                if event.key == pygame.K_RETURN:
                    build_Nikolskaya(screen)

                    type_animal = user_text.split()[0]
                    if type_animal == 'mouse':
                        l = len(((tmp_df_mouse[tmp_df_mouse['input'] == user_text].sort_values(by='Lev')[
                                    'path'])))
                        if l > iter_:
                            input_path_Nikolskaya = (''.join(tmp_df_mouse[tmp_df_mouse['input'] == user_text].sort_values(by='Lev')[
                                    'path'].iloc[iter_].split()))
                            input_path_Nikolskaya_second = (''.join(tmp_df_mouse[tmp_df_mouse['input'] == user_text].sort_values(by='Lev')[
                                    'generated'].iloc[iter_].split()))
                        else:
                            input_path_Nikolskaya = (''.join(tmp_df_mouse[tmp_df_mouse['input'] == user_text].sort_values(by='Lev')[
                                    'path'].iloc[l - 1].split()))
                            input_path_Nikolskaya_second = (''.join(tmp_df_mouse[tmp_df_mouse['input'] == user_text].sort_values(by='Lev')[
                                    'generated'].iloc[l - 1].split()))
                    elif type_animal == 'fish':
                        l = len(((tmp_df_fish[tmp_df_fish['input'] == user_text].sort_values(by='Lev')[
                                    'path'])))
                        if l > iter_:
                            input_path_Nikolskaya = (''.join(tmp_df_fish[tmp_df_fish['input'] == user_text].sort_values(by='Lev')[
                                    'path'].iloc[iter_].split()))
                            input_path_Nikolskaya_second = (''.join(tmp_df_fish[tmp_df_fish['input'] == user_text].sort_values(by='Lev')[
                                    'generated'].iloc[iter_].split()))
                        else:
                            input_path_Nikolskaya = (''.join(tmp_df_fish[tmp_df_fish['input'] == user_text].sort_values(by='Lev')[
                                    'path'].iloc[l - 1].split()))
                            input_path_Nikolskaya_second = (''.join(tmp_df_fish[tmp_df_fish['input'] == user_text].sort_values(by='Lev')[
                                    'generated'].iloc[l - 1].split()))
                    elif type_animal == 'monkey':
                        l = len(((tmp_df_monkey[tmp_df_monkey['input'] == user_text].sort_values(by='Lev')[
                                    'path'])))
                        if l > iter_:
                            input_path_Nikolskaya = (''.join(tmp_df_monkey[tmp_df_monkey['input'] == user_text].sort_values(by='Lev')[
                                    'path'].iloc[iter_].split()))
                            input_path_Nikolskaya_second = (''.join(tmp_df_monkey[tmp_df_monkey['input'] == user_text].sort_values(by='Lev')[
                                    'generated'].iloc[iter_].split()))
                        else:
                            input_path_Nikolskaya = (''.join(tmp_df_monkey[tmp_df_monkey['input'] == user_text].sort_values(by='Lev')[
                                    'path'].iloc[l - 1].split()))
                            input_path_Nikolskaya_second = (''.join(tmp_df_monkey[tmp_df_monkey['input'] == user_text].sort_values(by='Lev')[
                                    'generated'].iloc[l - 1].split()))
                    else:
                        l = len(((tmp_df_person[tmp_df_person['input'] == user_text].sort_values(by='Lev')[
                                    'path'])))
                        if l > iter_:
                            input_path_Nikolskaya = (''.join(tmp_df_person[tmp_df_person['input'] == user_text].sort_values(by='Lev')[
                                    'path'].iloc[iter_].split()))
                            input_path_Nikolskaya_second = (''.join(tmp_df_person[tmp_df_person['input'] == user_text].sort_values(by='Lev')[
                                    'generated'].iloc[iter_].split()))
                        else:
                            input_path_Nikolskaya = (''.join(tmp_df_person[tmp_df_person['input'] == user_text].sort_values(by='Lev')[
                                    'path'].iloc[l - 1].split()))
                            input_path_Nikolskaya_second = (''.join(tmp_df_person[tmp_df_person['input'] == user_text].sort_values(by='Lev')[
                                    'generated'].iloc[l - 1].split()))

                    iter_ += 1

                    input_ = font.render('real:' + input_path_Nikolskaya, True, RED)
                    screen.blit(input_, (0, 0))

                    input_second = font.render('generated:' + input_path_Nikolskaya_second, True, GREEN)
                    screen.blit(input_second, (0, 20))

                    path_res_Nikolskaya = path_Nikolskaya(input_path_Nikolskaya, maze_Nikolskaya)
                    path_res_Nikolskaya_second = path_Nikolskaya(input_path_Nikolskaya_second, maze_Nikolskaya)

                    max_ = max(len(path_res_Nikolskaya), len(path_res_Nikolskaya_second))
                    for now in range(max_):
                        if now < len(path_res_Nikolskaya) and now < len(path_res_Nikolskaya_second):
                            max_2 = max(len(path_res_Nikolskaya[now]), len(path_res_Nikolskaya_second[now]))
                        elif now < len(path_res_Nikolskaya):
                            max_2 = len(path_res_Nikolskaya[now])
                        else:
                            max_2 = len(path_res_Nikolskaya_second[now])
                        for i in range(max_2 - 1):
                            if now < len(path_res_Nikolskaya) and i < len(path_res_Nikolskaya[now]) - 1:
                                x_start, y_start = coordinates_all_Nikolskaya[path_res_Nikolskaya[now][i]]
                                x_finish, y_finish = coordinates_all_Nikolskaya[path_res_Nikolskaya[now][i + 1]]

                            if now < len(path_res_Nikolskaya_second) and i < len(path_res_Nikolskaya_second[now]) - 1:
                                x_start_second, y_start_second = coordinates_all_Nikolskaya[
                                    path_res_Nikolskaya_second[now][i]]
                                x_finish_second, y_finish_second = coordinates_all_Nikolskaya[
                                    path_res_Nikolskaya_second[now][i + 1]]

                            k = 0
                            while (y_start != y_finish or x_start != x_finish) or (
                                    y_start_second != y_finish_second or x_start_second != x_finish_second):
                                do_y_start = ((y_start < y_finish) - (y_start > y_finish))
                                do_x_start = ((x_start < x_finish) - (x_start > x_finish))

                                do_y_start_second = (
                                        (y_start_second < y_finish_second) - (y_start_second > y_finish_second))
                                do_x_start_second = (
                                        (x_start_second < x_finish_second) - (x_start_second > x_finish_second))

                                pygame.draw.circle(screen, RED, (x_start + 5, y_start + 5), 5)
                                pygame.draw.circle(screen, GREEN, (x_start_second - 5, y_start_second - 5), 5)
                                pygame.display.update()
                                pygame.draw.circle(screen, FONT, (x_start + 5, y_start + 5), 5)
                                pygame.draw.circle(screen, FONT, (x_start_second - 5, y_start_second - 5), 5)

                                y_start += do_y_start
                                x_start += do_x_start

                                y_start_second += do_y_start_second
                                x_start_second += do_x_start_second
                                k += 1

                                pygame.draw.line(screen, RED, (
                                    x_start - k * do_x_start + 5, y_start - k * do_y_start + 5),
                                                 (x_start + 5, y_start + 5), 3)

                                pygame.draw.line(screen, GREEN, (
                                    x_start_second - k * do_x_start_second - 5,
                                    y_start_second - k * do_y_start_second - 5),
                                                 (x_start_second - 5, y_start_second - 5), 3)

                                clock.tick(100)
                else:
                    if iter_ != 0:
                        user_text = ''
                    iter_ = 0
                    user_text += event.unicode

        if active:
            color = color_active
        else:
            color = color_passive

        pygame.draw.rect(screen, color, input_rect)
        text_surface = font.render(user_text, True, (255, 255, 255))
        screen.blit(text_surface, (input_rect.x + 5, input_rect.y + 5))

        pygame.display.flip()


def labirinth_Berezhnoy(screen, clock, df_Berezhnoy):
    build_Berezhnoy(screen)

    now = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    build_Berezhnoy(screen)
                    if len(df_Berezhnoy) > now:
                        pth_ = path_Berezhnoy(df_Berezhnoy.path[now], maze_Berezhnoy)
                        for i in range(len(pth_)):
                            for j in range(len(pth_[i]) - 1):
                                k = 0
                                x_start, y_start = coordinates_Berezhnoy[pth_[i][j]]
                                x_finish, y_finish = coordinates_Berezhnoy[pth_[i][j + 1]]
                                while y_start != y_finish or x_start != x_finish:
                                    do_y_start = ((y_start < y_finish) - (y_start > y_finish))
                                    do_x_start = ((x_start < x_finish) - (x_start > x_finish))

                                    pygame.draw.circle(screen, RED, (x_start, y_start), 5)
                                    pygame.display.update()
                                    pygame.draw.circle(screen, BLUE, (x_start, y_start), 5)

                                    y_start += do_y_start
                                    x_start += do_x_start
                                    k += 1

                                    pygame.draw.line(screen, RED, (
                                        x_start - k * do_x_start, y_start - k * do_y_start),
                                                     (x_start, y_start), 3)

                                    clock.tick(100)
                    now += 1
        pygame.display.flip()


def labirinth_Chelnok(screen, clock, df_Chelnok):
    build_Chelnok(screen)

    now = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    build_Chelnok(screen)
                    if now < len(df_Chelnok):
                        pth_ = path_Chelnok(df_Chelnok.path[now], maze_Chelnok)
                        for i in range(len(pth_)):
                            for j in range(len(pth_[i]) - 1):
                                k = 0
                                x_start, y_start = coordinates_all_Chelnok[pth_[i][j]]
                                x_finish, y_finish = coordinates_all_Chelnok[pth_[i][j + 1]]
                                while y_start != y_finish or x_start != x_finish:

                                    do_y_start = ((y_start < y_finish) - (y_start > y_finish))
                                    do_x_start = ((x_start < x_finish) - (x_start > x_finish))

                                    pygame.draw.circle(screen, RED, (x_start, y_start), 5)
                                    pygame.display.update()
                                    pygame.draw.circle(screen, BLACK, (x_start, y_start), 5)

                                    y_start += do_y_start
                                    x_start += do_x_start
                                    k += 1

                                    pygame.draw.line(screen, RED, (
                                        x_start - k * do_x_start, y_start - k * do_y_start),
                                                     (x_start, y_start), 3)

                                    clock.tick(100)
                    now += 1
        pygame.display.flip()
