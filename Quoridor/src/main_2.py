import sys
import pygame
from src.game_implement import Player, BlockManager
from src.game_implement.Player import Player_for_ai
from src.game_implement.BlockManager import BlockManager_for_ai
from src.ai_implement.policy_net import *
from src.ai_implement.game_adapter import GameAdapter
from src.ai_implement.mcts import HumanInput, mcts_choose_action_for_p1, apply_joint_actions
from src.ai_implement.EncoderANDDecoder import player_decoder, player_encoder, block_encoder
from configs import *
import time

pygame.init()

surface = pygame.display.set_mode((SURFACE_WIDTH * 2, SURFACE_HEIGHT))

left_surface = surface.subsurface((0, 0, SURFACE_WIDTH, SURFACE_HEIGHT))
right_surface = surface.subsurface((SURFACE_WIDTH, 0, SURFACE_WIDTH, SURFACE_HEIGHT))

bm1 = BlockManager.BlockManager(N, cool_time = cool_time)
bm2 = BlockManager.BlockManager(N, cool_time = cool_time)
player1 = Player.Player(N // 2, 0, cool_time = cool_time)
player2 = Player.Player(N // 2, 0, cool_time = cool_time)

net = SimpleActorCritic(A=2*(N-1)*(N-1)).to(device_gpu)
policy = MaskedPolicy(net)
ai_policy = load_agent(policy, ai_path, device=device_gpu)
ai_last_action = time.time() - cool_time


env = GameAdapter(
    N=N,
    player_1=Player_for_ai((N+1)//2, 0, N),
    player_2=Player_for_ai((N+1)//2, 0, N),
    block_manager_1=BlockManager_for_ai(N),
    block_manager_2=BlockManager_for_ai(N),
    get_move_mask_fn=Player.get_move_mask,
    player_decoder_fn=player_decoder,
    max_steps=300,
)

obs, info = env.reset()
ai_policy.net.eval()

FPSCLOCK = pygame.time.Clock()
block_x, block_y, block_direction = 0, 0, 0
a2 = [None, None]
while True:
    surface.fill((0, 0, 0))
    order = 0

    if player1.terminal() or player2.terminal():
        print("Game Over")
        break

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                block_x = min(block_x + 1, N - 2)
            elif event.key == pygame.K_LEFT:
                block_x = max(block_x - 1, 0)
            elif event.key == pygame.K_UP:
                block_y = max(block_y - 1, 0)
            elif event.key == pygame.K_DOWN:
                block_y = min(block_y + 1, N - 2)
            elif event.key == pygame.K_SPACE:
                block_direction = 1 - block_direction
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

    a2[0] = player2.move(bm2)
    if bm2.get_obs_specific(block_x, block_y, block_direction) == 0:
        if order == 1:
            a2[1] = bm2.update_block_state(block_x, block_y, block_direction, 1)###
        else:
            bm2.update_block_state(block_x, block_y, block_direction, 2)
    if (time.time() - ai_last_action >= cool_time):
        a1 = mcts_choose_action_for_p1(env, policy)
        a2[0] = player2.move(bm2)###

        player2.move_input(a1[0])
        bm1.place_block_for_ai(a1[1])
        r1, terminated, truncated = apply_joint_actions(env, a1, tuple(a2))
        ai_last_action = time.time()
        print(a1)
        a2 = [None, None]


    bm1.blit(left_surface)
    player1.blit(left_surface)

    bm2.blit(right_surface)
    player2.blit(right_surface)

    if order != 1 and bm2.get_obs_specific(block_x, block_y, block_direction) == 0:
        bm2.update_block_state(block_x, block_y, block_direction, 0)

    pygame.draw.line(surface, (60, 60, 60), (SURFACE_WIDTH, 0), (SURFACE_WIDTH, SURFACE_HEIGHT), 2)

    pygame.display.update()

    FPSCLOCK.tick(30)

