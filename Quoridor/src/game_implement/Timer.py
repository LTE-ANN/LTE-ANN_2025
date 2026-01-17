import os
import pygame

class Timer(object):
    def __init__(self, image_dir, image_num, filename, width, height, show_x, show_y):
        self.image_dir = image_dir
        self.image_num = image_num
        self.width = width
        self.height = height
        self.show_x = show_x
        self.show_y = show_y
        self.images = [pygame.transform.scale(pygame.image.load(os.path.join(image_dir, filename+f'_{i}.png')), (self.width, self.height))
                       for i in range(image_num)]
    def blit(self, surface, time_delta, cool):
        index = 0
        if(time_delta > cool):
            index = self.image_num - 1
        elif(time_delta < 0):
            index = 0
        else:
            index = int(time_delta*(self.image_num-1)/cool)
        surface.blit(self.images[index], (self.show_x, self.show_y))


