import copy
import math
import random
from typing import Any, Dict, Optional, Tuple
import torch

from src.ai_implement.Types import EdgeStats

# ------------------------------------------------------------
# 1) "동시 행동"을 외부에서 직접 적용하는 helper
#    (GameAdapter.step(action_1)은 opp_policy를 내부에서 호출하므로, MCTS에는 부적합)
# ------------------------------------------------------------
def apply_joint_actions(
    env: Any,
    a1: Tuple[int, int],
    a2: Tuple[Optional[int], Optional[int]],
) -> Tuple[float, bool, bool]:
    """
    a1: (move1:int, place1:int)  # AI는 None 없음
    a2: (move2:Optional[int], place2:Optional[int])  # 인간은 None 가능

    규칙:
    - move가 None이면 move 적용/검증을 하지 않음
    - place가 None이면 place 적용/검증을 하지 않음
    - place_mask가 전부 False면 place는 스킵(no-op) (기존 규칙 유지)
    - terminal reward / truncated 처리 규칙은 그대로 유지
    """
    move1, place1 = int(a1[0]), int(a1[1])
    move2, place2 = a2[0], a2[1]

    # pre-step masks
    move_mask_1, place_mask_1 = env._masks(perspective=1)
    move_mask_2, place_mask_2 = env._masks(perspective=2)

    # --- 1p(move)은 항상 있어야 함 ---
    if not bool(move_mask_1[0, move1].item()):
        raise ValueError(f"Illegal move_action_1={move1}")

    # --- 2p(move)은 None이면 스킵, 아니면 합법 검증 ---
    if move2 is not None:
        move2 = int(move2)
        if not bool(move_mask_2[0, move2].item()):
            raise ValueError(f"Illegal move_action_2={move2}")

    # place는 "가능할 때만" 검증/적용 + None이면 스킵
    can_place_1 = bool(place_mask_1.any().item())
    can_place_2 = bool(place_mask_2.any().item())

    if place2 is not None:
        place2 = int(place2)

    if can_place_1 and (not bool(place_mask_1[0, place1].item())):
        raise ValueError(f"Illegal place_action_1={place1}")

    if (place2 is not None) and can_place_2 and (not bool(place_mask_2[0, place2].item())):
        raise ValueError(f"Illegal place_action_2={place2}")

    # ---- 동시 적용 (단, 2p는 None이면 해당 적용 생략) ----
    env.p1.move(env.player_decoder(move1), env.bm2)
    if can_place_1:
        env.bm1.place_block_for_ai(place1)

    if move2 is not None:
        env.p2.move(env.player_decoder(move2), env.bm1)
    if can_place_2 and (place2 is not None):
        env.bm2.place_block_for_ai(place2)

    env._t += 1

    terminal_1 = bool(env.p1.terminal())
    terminal_2 = bool(env.p2.terminal())

    if terminal_1 or terminal_2:
        denom = (1 if terminal_1 else 0) + (1 if terminal_2 else 0)
        reward_1 = (1.0 if terminal_1 else 0.0) / float(denom)
        terminated = True
        truncated = False
    else:
        reward_1 = 0.0
        terminated = False
        truncated = (env._t >= env.max_steps)

    # 문서 규칙: truncated면 reward=0, terminated=False
    if truncated:
        reward_1 = 0.0
        terminated = False

    return float(reward_1), bool(terminated), bool(truncated)

# ------------------------------------------------------------
# 2) 정책 네트워크에서 "마스크된 상위 K개 행동" 추출 (branch별: move / place)
#    - action space가 커서 MCTS branching 폭을 제한하는 게 중요함
# ------------------------------------------------------------
@torch.no_grad()
def topk_actions_from_policy(policy, obs, move_mask, place_mask, k_move=4, k_place=32, deterministic_priors=False):
    """
    반환:
      moves: [(move_id, prior_prob), ...]
      places: [(place_id, prior_prob), ...]  (place 불가면 빈 리스트)
    """
    value, move_logits, place_logits = policy.net(obs)
    move_logits = move_logits.clone()
    place_logits = place_logits.clone()

    # move: mask 적용
    move_logits[~move_mask] = -1e9
    move_probs = torch.softmax(move_logits, dim=-1).view(-1)  # (4,)

    # place: mask 적용 (가능할 때만)
    can_place = bool(place_mask.any().item())
    if can_place:
        place_logits[~place_mask] = -1e9
        place_probs = torch.softmax(place_logits, dim=-1).view(-1)  # (A,)
    else:
        place_probs = None

    # move 후보는 최대 4개라 그냥 전부 가져도 됨
    legal_moves = torch.nonzero(move_mask.view(-1), as_tuple=False).view(-1)
    moves = [(int(i.item()), float(move_probs[i].item())) for i in legal_moves]
    moves.sort(key=lambda x: x[1], reverse=True)
    moves = moves[: min(k_move, len(moves))]

    # place 후보는 top-k만
    places = []
    if can_place:
        legal_places = torch.nonzero(place_mask.view(-1), as_tuple=False).view(-1)
        if len(legal_places) > 0:
            # legal 중에서 top-k
            probs_legal = place_probs[legal_places]
            k = min(k_place, probs_legal.numel())
            topv, topidx = torch.topk(probs_legal, k=k, largest=True, sorted=True)
            for j in range(k):
                pid = int(legal_places[topidx[j]].item())
                pr = float(topv[j].item())
                places.append((pid, pr))
    return moves, places


@torch.no_grad()
def value_from_policy(policy, obs) -> float:
    """
    value head는 (0~1 기대 보상)에 가까울 거라고 가정하고 clip.
    """
    v, _, _ = policy.net(obs)
    v = float(v.view(-1)[0].item())
    return max(0.0, min(1.0, v))


# ------------------------------------------------------------
# 3) MCTS (PUCT) with "half-step" turn model
#    - to_play=1: 1p가 행동 선택
#    - to_play=2: 2p가 행동 선택
#    - pending_a1이 있으면 to_play=2에서 a2를 뽑아 joint 적용 후 다음 상태로 감
# ------------------------------------------------------------
class MCTSNode:
    __slots__ = ("state_env", "to_play", "pending_a1", "children", "expanded", "terminal", "terminal_value")

    def __init__(self, state_env: Any, to_play: int, pending_a1: Optional[Tuple[int, int]]):
        self.state_env = state_env
        self.to_play = int(to_play)             # 1 or 2
        self.pending_a1 = pending_a1            # None or (move, place) chosen by 1p in this ply
        self.children: Dict[Tuple[int, int], Tuple["MCTSNode", EdgeStats]] = {}
        self.expanded = False
        self.terminal = False
        self.terminal_value = 0.0              # from player1 perspective


def puct_select(node: MCTSNode, c_puct: float) -> Tuple[Tuple[int, int], MCTSNode, EdgeStats]:
    """
    node.to_play==1이면 maximize, ==2이면 minimize (player1 value 기준)
    """
    # total visits
    total_N = 1 + sum(es.N for _, es in node.children.values())

    best_a = None
    best_child = None
    best_es = None
    best_score = None

    for a, (child, es) in node.children.items():
        Q = (es.W / es.N) if es.N > 0 else 0.0
        U = c_puct * es.P * math.sqrt(total_N) / (1 + es.N)
        score = (Q + U)

        # player2 turn: minimize player1 value -> pick smallest (Q - U) 느낌이지만,
        # 탐색 보존 위해 같은 U를 더한 값에서 minimize로 처리
        if node.to_play == 2:
            score = (-Q + U)  # minimize Q <=> maximize -Q

        if best_score is None or score > best_score:
            best_score = score
            best_a = a
            best_child = child
            best_es = es

    return best_a, best_child, best_es


def normalize_priors(pairs):
    s = sum(p for _, p in pairs)
    if s <= 0:
        n = len(pairs)
        return [(a, 1.0 / n) for a, _ in pairs] if n > 0 else []
    return [(a, p / s) for a, p in pairs]

def expand_node(node: MCTSNode, policy, k_place: int = 32):
    env = node.state_env

    if node.to_play == 1:
        # --- 1p 후보: None 없이 기존과 동일 ---
        obs = env._build_obs(perspective=1)
        move_mask, place_mask = env._masks(perspective=1)
        moves, places = topk_actions_from_policy(policy, obs, move_mask, place_mask, k_move=4, k_place=k_place)

        if len(places) == 0:
            # place 불가면 place_id는 0으로 고정(기존 no-op 규칙)
            a_list = [((m, 0), p) for (m, p) in moves]
        else:
            moves_n = normalize_priors([(m, p) for (m, p) in moves])
            places_n = normalize_priors([(p, q) for (p, q) in places])
            a_list = []
            for m, pm in moves_n:
                for pl, ppl in places_n:
                    a_list.append(((m, pl), pm * ppl))
            a_list.sort(key=lambda x: x[1], reverse=True)
            a_list = a_list[: max(16, k_place)]

        a_list = normalize_priors(a_list)
        for a1, prior in a_list:
            child_env = copy.deepcopy(env)  # 아직 joint 적용 안 함
            child = MCTSNode(child_env, to_play=2, pending_a1=a1)
            node.children[a1] = (child, EdgeStats(N=0, W=0.0, P=float(prior)))

    else:
        # --- 2p 후보: None(스킵) 포함 ---
        if node.pending_a1 is None:
            raise RuntimeError("to_play=2 node must have pending_a1")

        obs = env._build_obs(perspective=2)
        move_mask, place_mask = env._masks(perspective=2)
        moves, places = topk_actions_from_policy(policy, obs, move_mask, place_mask, k_move=4, k_place=k_place)

        # 스킵 prior(너무 크면 “아무것도 안 함”에 갇힘). 보통 0.05~0.20 범위가 무난함.
        SKIP_MOVE_PRIOR = 0.10
        SKIP_PLACE_PRIOR = 0.10

        # move 후보: (move_id, prior) + (None, prior)
        move_candidates = [(m, p) for (m, p) in moves]
        move_candidates.append((None, SKIP_MOVE_PRIOR))
        move_candidates = normalize_priors(move_candidates)

        # place 후보: 가능하면 top-k + None, 불가능하면 None만
        can_place = bool(place_mask.any().item())
        if not can_place:
            place_candidates = [(None, 1.0)]
        else:
            place_candidates = [(p, q) for (p, q) in places]
            place_candidates.append((None, SKIP_PLACE_PRIOR))
            place_candidates = normalize_priors(place_candidates)

        # factorize 결합: prior = P(move)*P(place)
        a_list = []
        for mv, pm in move_candidates:
            for pl, pp in place_candidates:
                a_list.append(((mv, pl), pm * pp))

        a_list.sort(key=lambda x: x[1], reverse=True)
        # 폭발 방지: 상위만 유지
        a_list = a_list[: max(16, k_place)]
        a_list = normalize_priors(a_list)

        for a2, prior in a_list:
            child_env = copy.deepcopy(env)
            r1, terminated, truncated = apply_joint_actions(child_env, node.pending_a1, a2)

            child = MCTSNode(child_env, to_play=1, pending_a1=None)
            if terminated or truncated:
                child.terminal = True
                child.terminal_value = r1  # player1 perspective

            node.children[a2] = (child, EdgeStats(N=0, W=0.0, P=float(prior)))

    node.expanded = True


def mcts_simulate(root: MCTSNode, policy, c_puct: float, k_place: int):
    path = []  # (node, action, edge_stats)
    node = root

    # selection
    while node.expanded and (not node.terminal) and len(node.children) > 0:
        a, child, es = puct_select(node, c_puct)
        path.append((node, a, es))
        node = child

    # evaluation / expansion
    if node.terminal:
        leaf_value = float(node.terminal_value)
    else:
        if not node.expanded:
            expand_node(node, policy, k_place=k_place)
        # leaf value는 "현재 상태"에서 player1 perspective value
        obs1 = node.state_env._build_obs(perspective=1)
        leaf_value = value_from_policy(policy, obs1)

    # backup (player1 value 기준으로 W 누적)
    for n, a, es in path:
        es.N += 1
        es.W += leaf_value


@torch.no_grad()
def mcts_choose_action_for_p1(env, policy, sims: int = 10, c_puct: float = 1.5, k_place: int = 32) -> Tuple[int, int]:
    """
    현재 env 상태에서 1p 행동을 MCTS로 선택.
    반환: (move_id, place_id)
    """
    root_env = copy.deepcopy(env)
    root = MCTSNode(root_env, to_play=1, pending_a1=None)
    expand_node(root, policy, k_place=k_place)

    for _ in range(int(sims)):
        mcts_simulate(root, policy, c_puct=c_puct, k_place=k_place)

    # root에서는 child key가 a1이고, visit이 가장 큰 a1 선택
    best_a = None
    best_n = -1
    for a1, (child, es) in root.children.items():
        if es.N > best_n:
            best_n = es.N
            best_a = a1

    if best_a is None:
        # fallback: 정책대로 한 번 뽑기
        obs = env._build_obs(perspective=1)
        mm, pm = env._masks(perspective=1)
        out = policy.act((obs[0], obs[1]) if isinstance(obs, tuple) else obs, mm, pm, deterministic=True)
        return int(out.move_action.item()), int(out.place_action.item())

    return int(best_a[0]), int(best_a[1])


# ------------------------------------------------------------
# 4) PvP(인간 vs AI+MCTS) 플레이 루프
#    - "동시 행동 규칙"은 joint로 적용하지만, 입력/진행은 1p->2p 순서로 받음(너가 말한 가정)
# ------------------------------------------------------------
class HumanInput:
    """
    규칙:
    - 정수 입력
    - -1 입력 시 None (스킵)
    """

    def _ask_int_or_none(self, prompt: str):
        while True:
            s = input(prompt).strip()
            try:
                v = int(s)
                if v == -1:
                    return None
                return v
            except Exception:
                print("정수 또는 -1을 입력해줘 (-1 = 스킵).")

    def choose(self, env, perspective: int):
        """
        반환:
          (move_id | None, place_id | None)
        """
        move_mask, place_mask = env._masks(perspective=perspective)

        # ---- move 선택 ----
        legal_moves = torch.nonzero(move_mask.view(-1), as_tuple=False).view(-1).tolist()
        print(f"[HUMAN p{perspective}] legal move ids: {legal_moves}")
        print(f"[HUMAN p{perspective}] move = -1 로 입력하면 move 스킵")

        while True:
            m = self._ask_int_or_none(f"[HUMAN p{perspective}] move id = ")
            if m is None:
                break
            if m in legal_moves:
                break
            print("불가능한 move id야. 다시.")

        # ---- place 선택 ----
        can_place = bool(place_mask.any().item())
        if not can_place:
            print(f"[HUMAN p{perspective}] place not available -> 자동 스킵")
            return m, None

        legal_places = torch.nonzero(place_mask.view(-1), as_tuple=False).view(-1).tolist()
        print(f"[HUMAN p{perspective}] legal place ids: {legal_places}")
        print(f"[HUMAN p{perspective}] place = -1 로 입력하면 place 스킵")

        while True:
            p = self._ask_int_or_none(f"[HUMAN p{perspective}] place id = ")
            if p is None:
                break
            if p in legal_places:
                break
            print("불가능한 place id야. 다시.")

        return m, p