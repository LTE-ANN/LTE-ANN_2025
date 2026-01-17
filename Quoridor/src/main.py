import sys
import pygame
from src.game_implement import Player, BlockManager
from src.ai_implement.policy_net import *
from configs import *
import time

pygame.init()
state = "Start"

surface = pygame.display.set_mode((SURFACE_WIDTH * 2, SURFACE_HEIGHT))

left_surface = surface.subsurface((0, 0, SURFACE_WIDTH, SURFACE_HEIGHT))
right_surface = surface.subsurface((SURFACE_WIDTH, 0, SURFACE_WIDTH, SURFACE_HEIGHT))
bm1 = BlockManager.BlockManager(N)
bm2 = BlockManager.BlockManager(N)

player1 = Player.Player(N // 2, 0, 0)
player2 = Player.Player(N // 2, 0, 1)

title_font = pygame.font.Font(title_font_path, title_size)
normal_font = pygame.font.Font(normal_font_path, normal_size)

net = SimpleActorCritic(A=2 * (N - 1) * (N - 1)).to(device_gpu)
policy = MaskedPolicy(net)
ai_policy = load_agent(policy, ai_path, device=device_gpu)
ai_last_action = time.time()

FPSCLOCK = pygame.time.Clock()

block_x, block_y, block_direction = 0, 0, 0
last_x, last_y = None, None
state = "Playing"

while True:
    if state == "Playing":
        if player1.terminal():
            state = "Win"
        elif player2.terminal():
            state = "Loose"

        order = 0

        surface.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    block_x = min(block_x+1, N-2)
                elif event.key == pygame.K_LEFT:
                    block_x = max(block_x-1, 0)
                elif event.key == pygame.K_UP:
                    block_y = max(block_y-1, 0)
                elif event.key == pygame.K_DOWN:
                    block_y = min(block_y+1, N-2)
                elif event.key == pygame.K_SPACE:
                    block_direction = 1-block_direction
                elif event.key == pygame.K_RETURN:
                    order = 1
                elif event.key == pygame.K_w:
                    player1.move_input('w')
                elif event.key == pygame.K_s:
                    player1.move_input('s')
                elif event.key == pygame.K_a:
                    player1.move_input('a')
                elif event.key == pygame.K_d:
                    player1.move_input('d')
        if(time.time() - ai_last_action >=cool_time):
            p1_vec = player2.get_obs()
            p2_vec = player1.get_obs()
            b1_list = bm2.get_obs()
            b2_list = bm1.get_obs()

            block_list = b1_list + b2_list
            cnn_obs = np.array(block_list, dtype = np.int32)
            fc_obs = np.concatenate([p1_vec, p2_vec]).astype(np.float32)

            raw_move_mask = Player.get_move_mask(player2, bm2)
            move_mask = np.array(raw_move_mask, dtype=bool)

            raw_place_mask = bm1.get_place_mask()
            place_mask = np.array(raw_place_mask, dtype=bool)

            move_idx, place_idx = predict_single_step(
                policy = ai_policy,
                cnn_input = cnn_obs,
                fc_input = fc_obs,
                move_mask = move_mask,
                place_mask = place_mask,
                deterministic = False,
                device = device_gpu,
                x = player2.get_pos()[0],
                y = player2.get_pos()[1],
                px = last_x,
                py = last_y,
                bm = bm2,
                N = N
            )

            last_x, last_y = player2.get_pos()
            player2.move_input(move_idx)
            bm1.place_block_for_ai(place_idx)
            ai_last_action = time.time()


        player1.move(bm1)
        player2.move(bm2)

        if bm2.get_obs_specific(block_x, block_y, block_direction) == 0:
            if order == 1:
                bm2.update_block_state(block_x, block_y, block_direction, 1)
            else:
                bm2.update_block_state(block_x, block_y, block_direction, 2)


        bm1.blit(left_surface)
        player1.blit(left_surface)

        bm2.blit(right_surface)
        player2.blit(right_surface)

        pygame.draw.line(surface, (60, 60, 60), (SURFACE_WIDTH, 0), (SURFACE_WIDTH, SURFACE_HEIGHT), 2)

        pygame.display.update()

        if order != 1 and bm1.get_obs_specific(block_x, block_y, block_direction) == 0:
            bm1.update_block_state(block_x, block_y, block_direction, 0)
        if order != 1 and bm2.get_obs_specific(block_x, block_y, block_direction) == 0:
            bm2.update_block_state(block_x, block_y, block_direction, 0)

        FPSCLOCK.tick(30)

    elif state == "Win":
        title_surf = title_font.render("YOU WON!!!", False, (255, 240, 80))
        title_rect = title_surf.get_rect(center=(SURFACE_WIDTH, SURFACE_HEIGHT // 2 - 60))

        info_surf = normal_font.render("press any 'x' to exit", False, (220, 220, 220))
        info_rect = info_surf.get_rect(center=(SURFACE_WIDTH, SURFACE_HEIGHT // 2 + 40))

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_x:
                        pygame.quit()
                        sys.exit()

            surface.fill((15, 15, 15))
            surface.blit(title_surf, title_rect)
            surface.blit(info_surf, info_rect)
            pygame.display.flip()

    elif state == "Loose":
        title_surf = title_font.render("YOU LOST...", False, (255, 240, 80))
        title_rect = title_surf.get_rect(center=(SURFACE_WIDTH, SURFACE_HEIGHT // 2 - 60))

        info_surf = normal_font.render("press 'x' to restart", False, (220, 220, 220))
        info_rect = info_surf.get_rect(center=(SURFACE_WIDTH, SURFACE_HEIGHT // 2 + 40))

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_x:
                        state = "Playing"
                        break

            surface.fill((15, 15, 15))
            surface.blit(title_surf, title_rect)
            surface.blit(info_surf, info_rect)
            pygame.display.flip()

    elif state == "Start":
        title_surf = title_font.render("LTE-ANN", False, (255, 240, 80))
        title_rect = title_surf.get_rect(center=(SURFACE_WIDTH, SURFACE_HEIGHT // 2 - 60))

        info_surf = normal_font.render("press 'x' to restart", False, (220, 220, 220))
        info_rect = info_surf.get_rect(center=(SURFACE_WIDTH, SURFACE_HEIGHT // 2 + 40))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:
                    state = "Playing"
                    break

            surface.fill((15, 15, 15))
            surface.blit(title_surf, title_rect)
            surface.blit(info_surf, info_rect)
            pygame.display.flip()

pygame.quit()
sys.exit(0)
