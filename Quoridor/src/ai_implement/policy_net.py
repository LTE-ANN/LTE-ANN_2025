from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical
import copy
import numpy as np

from src.ai_implement.Types import ActOutput
from collections import deque
from src.game_implement import BlockManager
from src.ai_implement.EncoderANDDecoder import player_encoder, player_decoder

Obs = Tuple[torch.Tensor, torch.Tensor]  # (cnn_x, fc_x)


def masked_categorical_strict(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    # 기존 masked_categorical(엄격 버전): move에는 계속 엄격하게 쓰는 게 안전함
    if mask.dtype != torch.bool:
        mask = mask.to(dtype=torch.bool)
    if mask.shape != logits.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} must match logits shape {tuple(logits.shape)}")
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("Action mask has a batch item with no legal actions (all False).")
    masked_logits = logits.masked_fill(~mask, logits.new_tensor(-1e9))
    return Categorical(logits=masked_logits)

class SimpleActorCritic(nn.Module):
    """
    [Drop-in replacement]
    - 기존과 동일: __init__(A), forward(obs)-> (value, move_logits, place_logits)
    - obs = (cnn_x, fc_x)
      cnn_x: (B, 4, H, W)
      fc_x : (B, F)   (F는 Lazy로 처리)
    - move action dim = 4  (너의 환경 정의 유지)
    - place action dim = A
    """

    def __init__(self, A: int):
        super().__init__()
        self.A = int(A)

        # ---- CNN trunk (bigger) ----
        # H,W가 무엇이든 평균풀링으로 고정 크기로 만들기 때문에 안정적
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4)),  # (B,64,4,4)
            nn.Flatten(),                  # (B,64*16)= (B,1024)
        )
        cnn_out_dim = 64 * 4 * 4  # 1024

        # ---- FC trunk (bigger, Lazy로 입력 차원 자동 처리) ----
        self.fc_trunk = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )

        # ---- Fusion trunk (bigger) ----
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + 256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
        )

        # ---- Heads: split move / place + value (reduce interference) ----
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.move_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )

        self.place_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.A),
        )

    def forward(self, obs):
        cnn_x, fc_x = obs  # cnn_x: (B,4,H,W), fc_x: (B,F)

        z_cnn = self.cnn(cnn_x)
        z_fc = self.fc_trunk(fc_x)
        z = torch.cat([z_cnn, z_fc], dim=-1)
        h = self.fusion(z)

        value = self.value_head(h)               # (B,1)
        move_logits = self.move_head(h)          # (B,4)
        place_logits = self.place_head(h)        # (B,A)
        return value, move_logits, place_logits



class MaskedPolicy:
    """
    Thin wrapper that:
      - applies masks to logits
      - samples actions (or argmax if deterministic)
      - computes joint log_prob/entropy = move + place

    This wrapper never creates masks; it only consumes masks supplied by the env / PPORecord.
    """

    def __init__(self, net: SimpleActorCritic):
        self.net = net

    @torch.no_grad()
    def act(self, obs, move_mask, place_mask, *, deterministic: bool = False):
        value, move_logits, place_logits = self.net(obs)

        # move는 여전히 엄격: 전부 False면 버그로 보고 즉시 터뜨리는 게 맞음
        dist_move = masked_categorical_strict(move_logits, move_mask)

        if deterministic:
            move_action = torch.argmax(dist_move.logits, dim=-1)
        else:
            move_action = dist_move.sample()
        move_logp = dist_move.log_prob(move_action)
        move_ent = dist_move.entropy()

        # place는 "전부 False면 no-op" 허용
        place_valid = place_mask.any(dim=-1)  # (B,)
        B = place_logits.shape[0]
        place_action = torch.zeros((B,), device=place_logits.device, dtype=torch.long)
        place_logp = torch.zeros((B,), device=place_logits.device, dtype=torch.float32)
        place_ent = torch.zeros((B,), device=place_logits.device, dtype=torch.float32)

        if place_valid.any():
            idx = torch.nonzero(place_valid, as_tuple=False).view(-1)
            dist_place = Categorical(logits=place_logits[idx].masked_fill(~place_mask[idx], place_logits.new_tensor(-1e9)))
            if deterministic:
                place_action[idx] = torch.argmax(dist_place.logits, dim=-1)
            else:
                place_action[idx] = dist_place.sample()
            place_logp[idx] = dist_place.log_prob(place_action[idx])
            place_ent[idx] = dist_place.entropy()

        joint_logp = move_logp + place_logp
        joint_ent = move_ent + place_ent

        return ActOutput(move_action, place_action, value, joint_logp, joint_ent)

    def evaluate_actions(self, obs, move_mask, place_mask, move_action, place_action):
        value, move_logits, place_logits = self.net(obs)

        dist_move = masked_categorical_strict(move_logits, move_mask)
        move_action = move_action.view(-1).to(device=move_logits.device)
        move_logp = dist_move.log_prob(move_action)
        move_ent = dist_move.entropy()

        # place는 동일 규칙(중요): rollout 때 no-op이면 update 때도 logp=0이 되어야 함
        place_valid = place_mask.any(dim=-1)
        B = place_logits.shape[0]
        place_logp = torch.zeros((B,), device=place_logits.device, dtype=torch.float32)
        place_ent = torch.zeros((B,), device=place_logits.device, dtype=torch.float32)

        if place_valid.any():
            idx = torch.nonzero(place_valid, as_tuple=False).view(-1)
            dist_place = Categorical(logits=place_logits[idx].masked_fill(~place_mask[idx], place_logits.new_tensor(-1e9)))
            pa = place_action.view(-1).to(device=place_logits.device)
            place_logp[idx] = dist_place.log_prob(pa[idx])
            place_ent[idx] = dist_place.entropy()

        joint_logp = move_logp + place_logp
        joint_ent = move_ent + place_ent
        return value, joint_logp, joint_ent

# Obs = Tuple[torch.Tensor, torch.Tensor]  # (cnn_x, fc_x)
#
#
# def masked_categorical_strict(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
#     # 기존 masked_categorical(엄격 버전): move에는 계속 엄격하게 쓰는 게 안전함
#     if mask.dtype != torch.bool:
#         mask = mask.to(dtype=torch.bool)
#     if mask.shape != logits.shape:
#         raise ValueError(f"mask shape {tuple(mask.shape)} must match logits shape {tuple(logits.shape)}")
#     if not torch.all(mask.any(dim=-1)):
#         raise ValueError("Action mask has a batch item with no legal actions (all False).")
#     masked_logits = logits.masked_fill(~mask, logits.new_tensor(-1e9))
#     return Categorical(logits=masked_logits)
#
# class SimpleActorCritic(nn.Module):
#     """
#     Minimal actor-critic network aligned with the document:
#
#       inputs:
#         cnn_x: (B,4,H,W) float32  [block_board_1 + block_board_2]
#         fc_x : (B,F)     float32  [concat(player_board_1, player_board_2)]
#
#       outputs (logits, not probs):
#         value      : (B,1)
#         move_logits: (B,4)
#         place_logits:(B,A)
#     """
#
#     def __init__(
#         self,
#         A: int,
#         *,
#         cnn_embed: int = 128,
#         fc_embed: int = 64,
#         hidden: int = 128,
#     ):
#         super().__init__()
#         self.A = int(A)
#
#         # CNN trunk (keep it small; no BatchNorm to avoid small-batch instability)
#         self.cnn = nn.Sequential(
#             nn.Conv2d(4, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.cnn_proj = nn.Sequential(
#             nn.Flatten(),
#             nn.LazyLinear(cnn_embed),
#             nn.ReLU(inplace=True),
#         )
#
#         # FC side (LazyLinear so you don't hardcode player feature length)
#         self.fc_proj = nn.Sequential(
#             nn.LazyLinear(fc_embed),
#             nn.ReLU(inplace=True),
#         )
#
#         # Shared body
#         self.body = nn.Sequential(
#             nn.Linear(cnn_embed + fc_embed, hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(inplace=True),
#         )
#
#         # Heads
#         self.value_head = nn.Linear(hidden, 1)
#         self.move_head = nn.Linear(hidden, 4)
#         self.place_head = nn.Linear(hidden, self.A)
#
#     def forward(self, obs: Obs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         cnn_x, fc_x = obs
#
#         if cnn_x.ndim != 4 or cnn_x.shape[1] != 4:
#             raise ValueError(f"cnn_x must be (B,4,H,W). Got {tuple(cnn_x.shape)}")
#         if fc_x.ndim != 2:
#             raise ValueError(f"fc_x must be (B,F). Got {tuple(fc_x.shape)}")
#         if cnn_x.shape[0] != fc_x.shape[0]:
#             raise ValueError("Batch size mismatch between cnn_x and fc_x.")
#
#         z_cnn = self.cnn_proj(self.cnn(cnn_x))
#         z_fc = self.fc_proj(fc_x)
#         z = torch.cat([z_cnn, z_fc], dim=-1)
#         h = self.body(z)
#
#         value = self.value_head(h)          # (B,1)
#         move_logits = self.move_head(h)     # (B,4)
#         place_logits = self.place_head(h)   # (B,A)
#
#         return value, move_logits, place_logits
#
#
# class MaskedPolicy:
#     """
#     Thin wrapper that:
#       - applies masks to logits
#       - samples actions (or argmax if deterministic)
#       - computes joint log_prob/entropy = move + place
#
#     This wrapper never creates masks; it only consumes masks supplied by the env / PPORecord.
#     """
#
#     def __init__(self, net: SimpleActorCritic):
#         self.net = net
#
#     @torch.no_grad()
#     def act(self, obs, move_mask, place_mask, *, deterministic: bool = False):
#         value, move_logits, place_logits = self.net(obs)
#
#         # move는 여전히 엄격: 전부 False면 버그로 보고 즉시 터뜨리는 게 맞음
#         dist_move = masked_categorical_strict(move_logits, move_mask)
#
#         if deterministic:
#             move_action = torch.argmax(dist_move.logits, dim=-1)
#         else:
#             move_action = dist_move.sample()
#         move_logp = dist_move.log_prob(move_action)
#         move_ent = dist_move.entropy()
#
#         # place는 "전부 False면 no-op" 허용
#         place_valid = place_mask.any(dim=-1)  # (B,)
#         B = place_logits.shape[0]
#         place_action = torch.zeros((B,), device=place_logits.device, dtype=torch.long)
#         place_logp = torch.zeros((B,), device=place_logits.device, dtype=torch.float32)
#         place_ent = torch.zeros((B,), device=place_logits.device, dtype=torch.float32)
#
#         if place_valid.any():
#             idx = torch.nonzero(place_valid, as_tuple=False).view(-1)
#             dist_place = Categorical(logits=place_logits[idx].masked_fill(~place_mask[idx], place_logits.new_tensor(-1e9)))
#             if deterministic:
#                 place_action[idx] = torch.argmax(dist_place.logits, dim=-1)
#             else:
#                 place_action[idx] = dist_place.sample()
#             place_logp[idx] = dist_place.log_prob(place_action[idx])
#             place_ent[idx] = dist_place.entropy()
#
#         joint_logp = move_logp + place_logp
#         joint_ent = move_ent + place_ent
#
#         return ActOutput(move_action, place_action, value, joint_logp, joint_ent)
#
#     def evaluate_actions(self, obs, move_mask, place_mask, move_action, place_action):
#         value, move_logits, place_logits = self.net(obs)
#
#         dist_move = masked_categorical_strict(move_logits, move_mask)
#         move_action = move_action.view(-1).to(device=move_logits.device)
#         move_logp = dist_move.log_prob(move_action)
#         move_ent = dist_move.entropy()
#
#         # place는 동일 규칙(중요): rollout 때 no-op이면 update 때도 logp=0이 되어야 함
#         place_valid = place_mask.any(dim=-1)
#         B = place_logits.shape[0]
#         place_logp = torch.zeros((B,), device=place_logits.device, dtype=torch.float32)
#         place_ent = torch.zeros((B,), device=place_logits.device, dtype=torch.float32)
#
#         if place_valid.any():
#             idx = torch.nonzero(place_valid, as_tuple=False).view(-1)
#             dist_place = Categorical(logits=place_logits[idx].masked_fill(~place_mask[idx], place_logits.new_tensor(-1e9)))
#             pa = place_action.view(-1).to(device=place_logits.device)
#             place_logp[idx] = dist_place.log_prob(pa[idx])
#             place_ent[idx] = dist_place.entropy()
#
#         joint_logp = move_logp + place_logp
#         joint_ent = move_ent + place_ent
#         return value, joint_logp, joint_ent

def to_device_obs(obs, device):
    cnn_x, fc_x = obs
    return (cnn_x.to(device, non_blocking=True), fc_x.to(device, non_blocking=True))

def to_device_mask(move_mask, place_mask, device):
    return (move_mask.to(device, non_blocking=True), place_mask.to(device, non_blocking=True))

def load_agent(policy, path: str, device: str | torch.device = "cpu"):
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["policy_state"]

    # 구조 충돌/생성자 인자 충돌을 피하려고 deepcopy 사용(가장 안전)
    net = copy.deepcopy(policy.net)
    net.load_state_dict(sd, strict=True)
    net.to(torch.device(device))
    net.eval()
    return type(policy)(net)


@torch.no_grad()
def predict_single_step(
        policy,
        cnn_input,
        fc_input,
        move_mask,
        place_mask,
        device,
        deterministic: bool = True,
        N=8,
        bm : BlockManager = None,
        x: int = None,
        y: int = None,
        px: int = None,
        py: int = None,
) -> tuple[int, int]:
    t_cnn = torch.as_tensor(cnn_input, dtype=torch.float32, device=device).unsqueeze(0)
    t_fc = torch.as_tensor(fc_input, dtype=torch.float32, device=device).unsqueeze(0)

    # Masks: (N,) -> (1, N), Bool 타입 강제
    t_move = torch.as_tensor(move_mask, device=device).bool()
    t_place = torch.as_tensor(place_mask, device=device).bool()

    output = policy.act((t_cnn, t_fc), t_move, t_place, deterministic=deterministic)

    dir_dict={'w':(0, -1), 'a': (-1, 0), 's': (0, 1), 'd': (1, 0)}
    nx, ny = (x+dir_dict[player_decoder(output.move_action.item())][0],
              y+dir_dict[player_decoder(output.move_action.item())][1])



    if px==nx and py==ny:
        depth = [[-1 for i in range(N)] for j in range(N)]
        dq = deque()
        for i in range(N):
            depth[i][N-1] = 0
            dq.append((i, N-1))
        while(len(dq) !=0):
            cur = dq.popleft()
            for direction  in ['w', 'a', 's', 'd']:
                next = (cur[0] + dir_dict[direction][0], cur[1] + dir_dict[direction][1])
                if next[0]< 0 or next[0] >=N or next[1]<0 or next[1]>=N:
                    continue
                if (not bm.move_allow(cur[0], cur[1], direction)) or (depth[next[0]][next[1]] != -1):
                    continue
                dq.append((next[0], next[1]))
                depth[next[0]][next[1]] = depth[cur[0]][cur[1]] + 1
            if(depth[x][y]!=-1):
                break
        for direction in ['w', 'a', 's', 'd']:
            nx, ny = (x + dir_dict[direction][0], y + dir_dict[direction][1])
            if bm.move_allow(x, y, direction) and  depth[nx][ny] + 1 == depth[x][y]:
                return player_encoder(direction), output.place_action.item()

    return output.move_action.item(), output.place_action.item()