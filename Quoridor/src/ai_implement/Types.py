#MCTS에서 저장할 정보들에 대한 규격을 정의하는 파일
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class StepInfo:
    # masks used to choose THIS step's actions (important for PPORecord)
    move_mask: torch.Tensor
    place_mask: torch.Tensor
    # opponent action actually applied this step (debug/replay)
    opp_move_action: int
    opp_place_action: int
    # terminals after applying both actions
    terminal_1: bool
    terminal_2: bool
    # reward from both perspectives (you usually train with reward_1 only)
    reward_1: float
    reward_2: float
    # next-step masks (optional convenience)
    next_move_mask: torch.Tensor
    next_place_mask: torch.Tensor

@dataclass(frozen=True)
class ActOutput:
    move_action: torch.Tensor      # (B,)
    place_action: torch.Tensor     # (B,)
    value: torch.Tensor            # (B,1)
    joint_log_prob: torch.Tensor   # (B,)
    joint_entropy: torch.Tensor    # (B,)

@dataclass
class EdgeStats:
    N: int = 0
    W: float = 0.0  # player1 perspective value sum
    P: float = 0.0  # prior