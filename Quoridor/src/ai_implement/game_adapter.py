# game_adapter.py
from __future__ import annotations
import torch
import numpy as np
from typing import Any, Callable, Optional, Tuple, Union

from src.ai_implement.Types import StepInfo


Obs = Tuple[torch.Tensor, torch.Tensor]  # (cnn_input, fc_input)
Action = Union[Tuple[int, int], np.ndarray, torch.Tensor]


def _to_int(x: Any) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    if torch.is_tensor(x):
        return int(x.item())
    if isinstance(x, np.ndarray):
        return int(x.item())
    raise TypeError(f"Action element must be int-like, got {type(x)}")


def _as_bool_tensor(x: Any, device: torch.device) -> torch.Tensor:
    t = x if torch.is_tensor(x) else torch.as_tensor(x)
    return t.to(device=device, dtype=torch.bool)


class RandomMaskedPolicy:
    """
    Fallback opponent policy: uniform random over valid actions (mask==True).
    Replace with your opponent-pool policy later without changing GameAdapter.
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs: Obs, move_mask: torch.Tensor, place_mask: torch.Tensor) -> Tuple[int, int]:
        # masks are expected shape (1,4) and (1,A)
        mm = move_mask[0].detach().cpu().numpy().astype(bool)
        pm = place_mask[0].detach().cpu().numpy().astype(bool)

        move_candidates = np.flatnonzero(mm)
        place_candidates = np.flatnonzero(pm)

        if move_candidates.size == 0:
            raise ValueError("No valid move actions (move_mask all False).")
        if place_candidates.size == 0:
            raise ValueError("No valid place actions (place_mask all False).")

        move = int(self._rng.choice(move_candidates))
        place = int(self._rng.choice(place_candidates))
        return move, place

class GameAdapter:
    """
    PPO-friendly wrapper around your existing game classes.

    Document-aligned invariants:
      - obs is built from get_obs(): block_board_1+block_board_2 (4 channels) and
        concat(player_board_1, player_board_2) for 1p; swapped for 2p.
      - actions are (move_action:int in [0..3], place_action:int in [0..A-1])
      - simultaneous: both players choose actions from the same pre-step state, then both applied
      - termination only from Player.terminal() after applying both actions
      - truncation at max_steps with reward=0 and terminated=False
    """

    def __init__(
        self,
        N: int,
        *,
        player_1: Any,
        player_2: Any,
        block_manager_1: Any,
        block_manager_2: Any,
        get_move_mask_fn: Callable[[Any, Any], Any],
        player_decoder_fn: Callable[[int], Any],
        opponent_policy: Optional[Any] = None,
        max_steps: int = 50,
        device: Union[str, torch.device] = "cpu",
    ):
        self.N = int(N)
        self.A = int(2 * (self.N - 1) * (self.N - 1))
        self.max_steps = int(max_steps)
        self.device = torch.device(device)

        self.p1 = player_1
        self.p2 = player_2
        self.bm1 = block_manager_1
        self.bm2 = block_manager_2

        self.get_move_mask = get_move_mask_fn
        self.player_decoder = player_decoder_fn

        self.opp_policy = opponent_policy if opponent_policy is not None else RandomMaskedPolicy()
        self._t = 0

    def _build_obs(self, perspective: int) -> Obs:
        # Raw boards (document: get_obs())
        p1_board = self.p1.get_obs()
        p2_board = self.p2.get_obs()
        b1 = self.bm1.get_obs()
        b2 = self.bm2.get_obs()

        if perspective == 1:
            block_list = b1 + b2
            player_vec = np.concatenate([p1_board, p2_board])
        elif perspective == 2:
            block_list = b2 + b1
            player_vec = np.concatenate([p2_board, p1_board])
        else:
            raise ValueError("perspective must be 1 or 2")

        # block_list must become (1,4,H,W) for CNN (or at least (1,4,*) consistently)
        block_arr = np.asarray(block_list)
        if block_arr.ndim < 2:
            raise ValueError(
                f"Block boards must be stackable into at least 2D arrays; got shape {block_arr.shape}. "
                "Ensure BlockManager.get_obs() returns a list of 2D boards."
            )
        if block_arr.shape[0] != 4:
            raise ValueError(
                f"Expected 4 block channels from block_board_1+block_board_2, got {block_arr.shape[0]}."
            )

        cnn_input = torch.tensor(block_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        fc_input = torch.tensor(player_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

        return cnn_input, fc_input

    def _masks(self, perspective: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if perspective == 1:
            move_mask = self.get_move_mask(self.p1, self.bm2)
            place_mask = self.bm1.get_place_mask()
        elif perspective == 2:
            move_mask = self.get_move_mask(self.p2, self.bm1)
            place_mask = self.bm2.get_place_mask()
        else:
            raise ValueError("perspective must be 1 or 2")

        move_mask_t = _as_bool_tensor(move_mask, self.device)
        place_mask_t = _as_bool_tensor(place_mask, self.device)

        if move_mask_t.shape != (1, 4):
            raise ValueError(f"move_mask must have shape (1,4), got {tuple(move_mask_t.shape)}")
        if place_mask_t.shape != (1, self.A):
            raise ValueError(f"place_mask must have shape (1,{self.A}), got {tuple(place_mask_t.shape)}")

        return move_mask_t, place_mask_t

    # ===== GameAdapter에 필요한 부분만 수정/보강 =====
    # 이미 동일하게 구현돼 있으면 이 블록은 건드리지 말 것.

    def reset(self, *, seed=None):
        if seed is not None and hasattr(self, "opp_policy") and self.opp_policy is not None:
            # 선택: 상대 정책이 랜덤 시드를 받는다면 여기서 고정
            try:
                self.opp_policy = type(self.opp_policy)(seed=seed)
            except Exception:
                pass

        self._t = 0
        self.p1.reset()
        self.p2.reset()
        self.bm1.reset()
        self.bm2.reset()

        obs1 = self._build_obs(perspective=1)
        move_mask_1, place_mask_1 = self._masks(perspective=1)

        info = {"move_mask": move_mask_1, "place_mask": place_mask_1}
        return obs1, info

    ###
    #step에서 action_1을 정해줘야 함?
    ###
    def step(self, action_1):
        # action_1: (move:int, place:int)
        move_action_1, place_action_1 = int(action_1[0]), int(action_1[1])

        # pre-step obs/masks (동시 의사결정 지점)
        obs2_pre = self._build_obs(perspective=2)

        move_mask_1, place_mask_1 = self._masks(perspective=1)
        move_mask_2, place_mask_2 = self._masks(perspective=2)

        # move는 항상 검증 (전부 False는 버그)
        if not bool(move_mask_1[0, move_action_1].item()):
            raise ValueError(f"Illegal move_action_1={move_action_1}.")

        # place는 "가능할 때만" 검증
        can_place_1 = bool(place_mask_1.any().item())
        if can_place_1 and (not bool(place_mask_1[0, place_action_1].item())):
            raise ValueError(f"Illegal place_action_1={place_action_1}.")

        # opponent action 샘플링은 policy가 no-op 처리했으면 place_action은 아무 값이어도 OK
        opp_move_action_2, opp_place_action_2 = self.opp_policy.act(obs2_pre, move_mask_2, place_mask_2)
        opp_move_action_2, opp_place_action_2 = int(opp_move_action_2), int(opp_place_action_2)

        if not bool(move_mask_2[0, opp_move_action_2].item()):
            raise ValueError(f"Opponent illegal move_action_2={opp_move_action_2}.")

        can_place_2 = bool(place_mask_2.any().item())
        if can_place_2 and (not bool(place_mask_2[0, opp_place_action_2].item())):
            raise ValueError(f"Opponent illegal place_action_2={opp_place_action_2}.")

        # 적용도 "가능할 때만" place 호출
        self.p1.move(self.player_decoder(move_action_1), self.bm2)
        if can_place_1:
            self.bm1.place_block_for_ai(place_action_1)

        self.p2.move(self.player_decoder(opp_move_action_2), self.bm1)
        if can_place_2:
            self.bm2.place_block_for_ai(opp_place_action_2)

        self._t += 1

        # Termination AFTER both applied
        terminal_1 = bool(self.p1.terminal())
        terminal_2 = bool(self.p2.terminal())

        # Truncation by max_steps (reward=0, terminated=False)
        truncated = self._t >= self.max_steps and not (terminal_1 or terminal_2)

        if terminal_1 or terminal_2:
            denom = (1 if terminal_1 else 0) + (1 if terminal_2 else 0)
            # denom is 1 or 2 here
            reward_1 = (1.0 if terminal_1 else 0.0) / float(denom)
            reward_2 = (1.0 if terminal_2 else 0.0) / float(denom)
            terminated = True
        else:
            reward_1 = 0.0
            reward_2 = 0.0
            terminated = False

        if truncated:
            # Document: truncated => both rewards 0, and you should treat "done mask" as False in PPO
            reward_1 = 0.0
            reward_2 = 0.0
            terminated = False

        # Next obs/masks (for convenience; PPORecord should store the pre-step masks above)
        obs1_next = self._build_obs(perspective=1)
        next_move_mask_1, next_place_mask_1 = self._masks(perspective=1)

        info = StepInfo(
            move_mask=move_mask_1,
            place_mask=place_mask_1,
            opp_move_action=opp_move_action_2,
            opp_place_action=opp_place_action_2,
            terminal_1=terminal_1,
            terminal_2=terminal_2,
            reward_1=float(reward_1),
            reward_2=float(reward_2),
            next_move_mask=next_move_mask_1,
            next_place_mask=next_place_mask_1,
        )

        return obs1_next, float(reward_1), bool(terminated), bool(truncated), info