import os
import torch

device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WIDTH = 5
HEIGHT = 50
RADIUS = 20
X_OFFSET = 82.5
Y_OFFSET = 100
N = 8

cool_time = 1

TIMER_SIZE = 100
SURFACE_WIDTH = 600
SURFACE_HEIGHT = 800

area_image_path = os.path.join(os.path.dirname(__file__), '../images/area.png')
block_image_path = os.path.join(os.path.dirname(__file__), '../images/block.png')
player_image_path = [os.path.join(os.path.dirname(__file__), '../images/player.png'), os.path.join(os.path.dirname(__file__), '../images/ai.png')]
ai_path = "./ai_implement/policy_v4.pt"
title_font_path = os.path.join(os.path.dirname(__file__), '../fonts/DOSPilgi.ttf')
normal_font_path = os.path.join(os.path.dirname(__file__), '../fonts/Sam3KRFont.ttf')
title_size = 108
normal_size = 36

block_timer_image_dir = os.path.join(os.path.dirname(__file__), '../images/timers/block')
player_timer_image_dir = os.path.join(os.path.dirname(__file__), '../images/timers/player')

class OccupiedException(Exception):
    def __init__(self, message="0, 1 is only allowed for occupied"):
        self.message = message
        super().__init__(self.message)

class NotAllowPosException(Exception):
    def __init__(self, message="position not allowed"):
        self.message = message
        super().__init__(self.message)

class RowColException(Exception):
    def __init__(self, message="wrong direction. you only can put row or col"):
        self.message = message
        super().__init__(self.message)

class NotAllowedToPlace(Exception):
    def __init__(self, message="you are not allowed to place it"):
        self.message = message
        super().__init__(self.message)

class NotAllowMoveException(Exception):
  def __init__(self, message="move not allowed"):
      self.message = message
      super().__init__(self.message)