import sys
import pygame
from src.game_implement import Player, BlockManager
from src.configs import *

pygame.init()

surface=pygame.display.set_mode((SURFACE_WIDTH, SURFACE_HEIGHT))
blocks = BlockManager.BlockManager(N)
player = Player.Player(N // 2, 0)
FPSCLOCK = pygame.time.Clock()

block_x, block_y, block_direction = 0, 0, 0
while True:
    if player.terminal():
        print("Game Over")
        break

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
                player.move_input('w')
            elif event.key == pygame.K_s:
                player.move_input('s')
            elif event.key == pygame.K_a:
                player.move_input('a')
            elif event.key == pygame.K_d:
                player.move_input('d')
    player.move(blocks)


    if(blocks.get_obs_specific(block_x, block_y, block_direction) == 0):
        if(order == 1):
            blocks.update_block_state(block_x, block_y, block_direction, 1)
        else:
            blocks.update_block_state(block_x, block_y, block_direction, 2)

    blocks.blit(surface)
    player.blit(surface)
    pygame.display.update()

    if(order != 1 and blocks.get_obs_specific(block_x, block_y, block_direction) == 0):
        blocks.update_block_state(block_x, block_y, block_direction, 0)
    FPSCLOCK.tick(30)

pygame.quit()
sys.exit(0)
