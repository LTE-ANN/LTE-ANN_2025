import pygame
from src.configs import *

class Block():
    def __init__(self, x, y, occupied, width, height, show_x, show_y):
        self.x = x
        self.y = y
        self._occupied = occupied
        self.width=width
        self.height=height
        self.state=0 # 0:안 보임 1: 그냥 보임(완전히 놓아진 상태) 2: 반투명하게 보임(놓기 직전 상태)
        self.show_x=show_x
        self.show_y=show_y

        self.image=pygame.image.load(block_image_path)
        self.image=pygame.transform.scale(self.image,(self.width,self.height))

        # self.rect=self.images.get_rect()
        # self.rect=self.rect.move((self.x, self.y))

    def occupy(self):
        self.state = 1
        self._occupied = 1

    def update_state(self, state):
        self.state = state
        if state==1:
            self._occupied=1
        else:
            self._occupied=0

    def blit(self, surface):
        alpha=0
        if self.state==0:
            alpha=0
        elif self.state==1:
            alpha=255
        else:
            alpha=123
        self.image.set_alpha(alpha)
        surface.blit(self.image, (self.show_x, self.show_y))
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y

    @property
    def occupied(self):
        return self._occupied

    def set_x(self, x):
        self.x = x
    def set_y(self, y):
        self.y = y

    @occupied.setter
    def occupied(self, occupied):
        if occupied != 0 and occupied != 1:
            raise OccupiedException
        self._occupied = occupied


class ColBlock(Block):
    def __init__(self, x, y, occupied, width=WIDTH, height=2*HEIGHT + WIDTH):
        show_x = X_OFFSET + (x+1)*(WIDTH+HEIGHT)
        show_y = Y_OFFSET + y*(WIDTH+HEIGHT) + WIDTH
        super().__init__(x, y, occupied, width, height, show_x, show_y)
        self.axis=1 #가로방향


class RowBlock(Block):
    def __init__(self, x, y, occupied, width=2*HEIGHT + WIDTH, height=WIDTH):
        show_x = X_OFFSET + x*(WIDTH+HEIGHT) + WIDTH
        show_y = Y_OFFSET + (y+1)*(WIDTH+HEIGHT)
        super().__init__(x, y, occupied, width, height, show_x, show_y)
        self.axis=0 #세로방향

class Block_for_ai:
    def __init__(self, x, y, occupied):
        self.x = x
        self.y = y
        self._occupied = 0  # Initialize with a default private attribute
        self.occupied = occupied  # Use the setter for initial validation

    def occupy(self):
        self.occupied = 1

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    @property
    def occupied(self):
        return self._occupied  # Access the private attribute

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    @occupied.setter
    def occupied(self, occupied):
        if occupied != 0 and occupied != 1:
            raise OccupiedException
        self._occupied = occupied  # Set the private attribute

class RowBlock_for_ai(Block_for_ai):
    def __init__(self, x, y, occupied):
        super().__init__(x, y, occupied)

class ColBlock_for_ai(Block_for_ai):
    def __init__(self, x, y, occupied):
        super().__init__(x, y, occupied)