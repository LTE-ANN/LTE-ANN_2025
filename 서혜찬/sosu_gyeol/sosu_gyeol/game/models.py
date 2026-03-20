from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, List

DigitPos3 = Literal["hundreds", "tens"]  # 1~8 라운드에서만 사용
PlayerId = Literal["p1", "p1b", "p2"]

class NewGameRequest(BaseModel):
    p1_name: str = "P1"
    p2_name: str = "P2"

class SubmitRequest(BaseModel):
    player: PlayerId
    pos: DigitPos3
    card: int = Field(ge=1, le=9)

class BuzzRequest(BaseModel):
    player: PlayerId
    answer_is_prime: bool

class GamePublicState(BaseModel):
    game_id: str
    round_no: int
    phase: str

    p1_name: str
    p2_name: str
    p1_score: int
    p2_score: int

    turn_first: Literal["p1", "p2"]
    host_ones: int

    submissions: Dict[str, Optional[int]]  # hundreds/tens/ones (+ reveal 시)
    revealed_number: Optional[int]

    p1_remaining: List[int]
    p2_remaining: List[int]

    submit_deadline_ms: Optional[int]
    buzz_deadline_ms: Optional[int]
    buzz_locked_by: Optional[str]
    history: List[dict]

    max_rounds: int = 9
    game_over: bool = False
    winner: Optional[Literal["p1", "p2", "draw"]] = None
    team_members: Dict[str, str] = {}

class BuzzPressRequest(BaseModel):
    player: PlayerId

class AnswerRequest(BaseModel):
    player: PlayerId
    answer_is_prime: bool
