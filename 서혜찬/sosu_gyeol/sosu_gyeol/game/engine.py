import time
import uuid
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Literal

from .prime import is_prime

DigitPos3 = Literal["hundreds", "tens"]
PlayerId = Literal["p1", "p1b", "p2"]


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class Game:
    game_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    round_no: int = 1
    phase: str = "idle"

    p1_name: str = "player1&player2"
    p2_name: str = "AI"
    p1_score: int = 0
    p2_score: int = 0

    team_members: Dict[str, str] = field(default_factory=lambda: {"p1": "player1", "p1b": "player2"})

    buzz_locked_by: Optional[PlayerId] = None
    history: List[dict] = field(default_factory=list)

    max_rounds: int = 9
    game_over: bool = False
    winner: Optional[Literal["p1", "p2", "draw"]] = None

    turn_first: Literal["p1", "p2"] = "p1"
    host_ones: int = 1

    submissions: Dict[str, Optional[int]] = field(default_factory=lambda: {
        "hundreds": None,
        "tens": None,
        "ones": None,
    })
    revealed_number: Optional[int] = None

    p1_used: set[int] = field(default_factory=set)
    p2_used: set[int] = field(default_factory=set)

    submit_deadline_ms: Optional[int] = None
    buzz_deadline_ms: Optional[int] = None

    def _canon_side(self, player: PlayerId) -> Literal["p1", "p2"]:
        return "p1" if player in ("p1", "p1b") else "p2"

    def _used_set(self, player: PlayerId):
        side = self._canon_side(player)
        return self.p1_used if side == "p1" else self.p2_used

    def _remaining_list(self, side: Literal["p1", "p2"]):
        used = self.p1_used if side == "p1" else self.p2_used
        return [n for n in range(1, 10) if n not in used]

    def new_game(self, p1_name: str, p2_name: str, team_members: Optional[Dict[str, str]] = None):
        self.game_id = str(uuid.uuid4())
        self.round_no = 1
        self.phase = "round_setup"

        self.p1_name = p1_name
        self.p2_name = p2_name
        if team_members:
            self.team_members = {"p1": team_members.get("p1", "player1"), "p1b": team_members.get("p1b", "player2")}
        else:
            self.team_members = {"p1": "player1", "p1b": "player2"}

        self.p1_score = 0
        self.p2_score = 0
        self.turn_first = "p1"

        self.p1_used = set()
        self.p2_used = set()

        self.history = []
        self.buzz_locked_by = None
        self.submit_deadline_ms = None
        self.buzz_deadline_ms = None
        self.game_over = False
        self.winner = None

        self._start_round()

    def _start_round(self):
        if self.round_no > self.max_rounds:
            self.game_over = True
            self.phase = "game_over"
            self.submit_deadline_ms = None
            self.buzz_deadline_ms = None
            return

        self.phase = "p1_submit" if self.turn_first == "p1" else "p2_submit"
        self.host_ones = random.choice([1, 3, 7, 9])
        self.submissions = {"hundreds": None, "tens": None, "ones": self.host_ones}
        self.revealed_number = None

        self.submit_deadline_ms = now_ms() + 5 * 30_000
        self.buzz_deadline_ms = None
        self.buzz_locked_by = None

    def _add_score(self, player: PlayerId, delta: int):
        side = self._canon_side(player)
        if side == "p1":
            self.p1_score += delta
        else:
            self.p2_score += delta

    def _check_timeouts(self):
        if self.phase == "buzz_locked" and self.buzz_deadline_ms is not None:
            if now_ms() > self.buzz_deadline_ms:
                self._handle_answer_timeout()

    def _handle_answer_timeout(self):
        player = self.buzz_locked_by
        if player is None:
            self.phase = "scored"
            self.buzz_deadline_ms = None
            self.round_no += 1
            self.turn_first = "p2" if self.turn_first == "p1" else "p1"
            self._start_round()
            return

        delta = -1
        self._add_score(player, delta)

        self.history.append({
            "round": self.round_no,
            "number": self.revealed_number,
            "buzz_by": player,
            "answer_is_prime": None,
            "correct": False,
            "timeout": True,
            "delta": delta,
            "p1_score": self.p1_score,
            "p2_score": self.p2_score,
        })

        self.phase = "scored"
        self.buzz_deadline_ms = None
        self.buzz_locked_by = None
        self._finish_round_and_advance()

    def submit(self, player: PlayerId, pos: DigitPos3, card: int):
        self._check_timeouts()

        if self.phase not in ["p1_submit", "p2_submit"]:
            raise ValueError("지금은 제출 단계가 아님")

        if self.submit_deadline_ms is not None and now_ms() > self.submit_deadline_ms:
            raise ValueError("제출 시간 초과")

        expected_side: Literal["p1", "p2"] = "p1" if self.phase == "p1_submit" else "p2"
        if self._canon_side(player) != expected_side:
            raise ValueError("지금 차례가 아님")

        if pos not in ["hundreds", "tens"]:
            raise ValueError("자리 오류")
        if self.submissions[pos] is not None:
            raise ValueError("그 자리는 이미 제출됨")

        used = self._used_set(player)
        if card in used:
            raise ValueError("이미 사용한 카드")

        self.submissions[pos] = card
        used.add(card)

        if self.submissions["hundreds"] is not None and self.submissions["tens"] is not None:
            self.phase = "ready_reveal"
            self.submit_deadline_ms = None
            self.revealed_number = None
            return

        self.phase = "p2_submit" if expected_side == "p1" else "p1_submit"

    def reveal(self):
        self._check_timeouts()
        if self.phase != "ready_reveal":
            raise ValueError("지금은 공개할 수 없음")

        h = self.submissions["hundreds"]
        t = self.submissions["tens"]
        o = self.submissions["ones"]
        if h is None or t is None or o is None:
            raise ValueError("숫자 구성이 완성되지 않음")

        self.revealed_number = h * 100 + t * 10 + o
        self.phase = "buzz_open"
        self.buzz_deadline_ms = None
        self.buzz_locked_by = None

    def buzz_press(self, player: PlayerId):
        self._check_timeouts()
        if self.phase != "buzz_open":
            raise ValueError("지금은 버저를 누를 수 없음")
        self.buzz_locked_by = player
        self.phase = "buzz_locked"
        self.buzz_deadline_ms = now_ms() + 5_000

    def answer(self, player: PlayerId, answer_is_prime: bool):
        self._check_timeouts()
        if self.phase != "buzz_locked":
            raise ValueError("지금은 답할 수 없음")
        if self.buzz_deadline_ms is not None and now_ms() > self.buzz_deadline_ms:
            raise ValueError("답변 시간 초과")
        if self.revealed_number is None:
            raise ValueError("공개된 숫자가 없음")
        if self.buzz_locked_by is None:
            raise ValueError("버저 선점자가 없음")
        if self.buzz_locked_by != player:
            raise ValueError("선점자만 답할 수 있음")

        correct = (is_prime(self.revealed_number) == answer_is_prime)
        delta = 1 if correct else -2
        self._add_score(player, delta)

        self.history.append({
            "round": self.round_no,
            "number": self.revealed_number,
            "buzz_by": player,
            "answer_is_prime": answer_is_prime,
            "correct": correct,
            "delta": delta,
            "p1_score": self.p1_score,
            "p2_score": self.p2_score,
        })

        self.phase = "scored"
        self.buzz_locked_by = None
        self.buzz_deadline_ms = None
        self._finish_round_and_advance()

    def public_state(self) -> dict:
        self._check_timeouts()
        return {
            "game_id": self.game_id,
            "round_no": self.round_no,
            "phase": self.phase,
            "p1_name": self.p1_name,
            "p2_name": self.p2_name,
            "p1_score": self.p1_score,
            "p2_score": self.p2_score,
            "turn_first": self.turn_first,
            "host_ones": self.host_ones,
            "submissions": self.submissions,
            "revealed_number": self.revealed_number,
            "p1_remaining": sorted(self._remaining_list("p1")),
            "p2_remaining": sorted(self._remaining_list("p2")),
            "submit_deadline_ms": self.submit_deadline_ms,
            "buzz_deadline_ms": self.buzz_deadline_ms,
            "buzz_locked_by": self.buzz_locked_by,
            "history": self.history[-20:],
            "max_rounds": self.max_rounds,
            "game_over": self.game_over,
            "winner": self.winner,
            "team_members": self.team_members,
        }

    def _finish_round_and_advance(self):
        if self.round_no >= self.max_rounds:
            self.game_over = True
            self.phase = "game_over"
            self.submit_deadline_ms = None
            self.buzz_deadline_ms = None
            self.buzz_locked_by = None

            if self.p1_score > self.p2_score:
                self.winner = "p1"
            elif self.p2_score > self.p1_score:
                self.winner = "p2"
            else:
                self.winner = "draw"
            return

        self.round_no += 1
        self.turn_first = "p2" if self.turn_first == "p1" else "p1"

        self.buzz_locked_by = None
        self.submit_deadline_ms = None
        self.buzz_deadline_ms = None
        self.revealed_number = None

        self._start_round()
