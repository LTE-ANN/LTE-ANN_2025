import pygame
import time
from src.ai_implement.EncoderANDDecoder import player_gamestate_decoder, player_decoder, player_encoder
import numpy as np
from src.configs import *
from src.game_implement.Timer import Timer

#말에 대한 클래스
class Player:
    def __init__(self, x, y, type, cool_time = 1):
        self.x = x #시작 x 좌표
        self.y = y #시작 y 좌표
        self.N = N
        self.cool = cool_time #움직이기 쿨타임
        self.last_time = time.time()-1 #움직인 이후 걸린 시간
        self.last_input = None #마지막으로 입력 받은 값

        self.image = pygame.image.load(player_image_path[type])
        self.image = pygame.transform.scale(self.image, (RADIUS*2, RADIUS*2))
        self.show_x = WIDTH + X_OFFSET + (WIDTH + HEIGHT) * self.x + (HEIGHT-2*RADIUS)/2
        self.show_y = WIDTH + Y_OFFSET + (WIDTH + HEIGHT) * self.y + (HEIGHT-2*RADIUS)/2

        self.timer = Timer(player_timer_image_dir, 9, "player_timer", TIMER_SIZE, TIMER_SIZE, SURFACE_WIDTH - TIMER_SIZE * 1.2,
                           SURFACE_HEIGHT - TIMER_SIZE * 1.2)

    def move_input(self, key):
        '''
        key를 입력받은 것을 저장(저장할 뿐 이동하지 않음)
        :param key: 'w' or 'a' or 's' or 'd'
        :return: None
        '''
        if(type(key)== int):
            key = player_decoder(key)
        self.last_input = key#last input 받기

    def terminal(self):
        '''
        종료 여부를 확인하는 함수로 추정(맨 아랫 줄인지 확인해줌)
        :return: 현재 위치가 맨 아랫줄인지 여부(boolean)
        '''
        return self.y == self.N-1

    def move(self, bm):
        '''
        last_input을 바탕으로 이동해줌
        :param bm: 이동 가능 여부를 판단해줄 BoardManager
        :return: None
        '''
        if(time.time() - self.last_time < self.cool or self.last_input ==None):
            return None
        if (not bm.move_allow(self.x, self.y, self.last_input)):
            return None
        if self.last_input == 'w':#위로 움직이기
            self.y-=1
        elif self.last_input == 'a':#왼쪽으로 움직이기
            self.x -=1
        elif self.last_input == 'd':#오른쪽으로 움직이기
            self.x +=1
        elif self.last_input == 's':#아래로 움직이기
            self.y +=1
        else:
            return None
        self.last_time = time.time()
        tmp = self.last_input
        self.last_input = None
        return player_encoder(tmp)

    def blit(self, surface):
        '''
        현재 상태를 surface에 보이게 해둠
        :param surface: 표시할 surface
        :return: None
        '''
        self.show_x = WIDTH + X_OFFSET + (WIDTH + HEIGHT) * self.x + (HEIGHT - 2 * RADIUS) / 2
        self.show_y = WIDTH + Y_OFFSET + (WIDTH + HEIGHT) * self.y + (HEIGHT - 2 * RADIUS) / 2
        surface.blit(self.image, (self.show_x, self.show_y))
        self.timer.blit(surface, time.time()-self.last_time, self.cool)

    def get_pos(self):
        return self.x, self.y

    def get_obs(self):
        one_hot = np.zeros(self.N * self.N, dtype=int)
        index = self.x * self.N + self.y
        one_hot[index] = 1
        return one_hot

    def set_gamestate(self, n):
        self.x, self.y = player_gamestate_decoder(n, self.N)

    def reset(self):
        self.x = (self.N + 1) // 2
        self.y = 0

class Player_for_ai:
    def __init__(self, x, y, N):
        self.x = x  # 시작 x 좌표
        self.y = y  # 시작 y 좌표
        self.N = N

    def terminal(self):
        return self.y == self.N - 1

    def move(self, direction, block_manager):  # 움직이는 함수
        if direction == 'w':  # 위로 움직이기
            self.y -= 1
        elif direction == 'a':  # 왼쪽으로 움직이기
            self.x -= 1
        elif direction == 'd':  # 오른쪽으로 움직이기
            self.x += 1
        elif direction == 's':  # 아래로 움직이기
            self.y += 1
        else:
            raise NotAllowMoveException

    def get_pos(self):
        return self.x, self.y

    def get_obs(self):
        one_hot = np.zeros(self.N * self.N, dtype=int)
        index = self.x * self.N + self.y
        one_hot[index] = 1
        return one_hot

    def set_gamestate(self, n):
        self.x, self.y = player_gamestate_decoder(n, self.N)

    def reset(self):
        self.x = (self.N + 1) // 2
        self.y = 0


def get_move_mask(player, block_manager):
    x,y = player.get_pos()
    return torch.tensor([block_manager.move_allow_w(x, y), block_manager.move_allow_a(x, y), block_manager.move_allow_s(x, y), block_manager.move_allow_d(x, y)]).unsqueeze(0)
