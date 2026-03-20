from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import HTTPException
from dataclasses import dataclass, field
import uuid
import random
from sosu_gyeol.game.prime import is_prime
from sosu_gyeol.game.engine import now_ms

from pydantic import BaseModel
from typing import Literal, Optional, List, Dict

from openai import AsyncOpenAI

from sosu_gyeol.game.models import BuzzPressRequest, AnswerRequest, PlayerId
from sosu_gyeol.game.engine import Game
from sosu_gyeol.game.models import NewGameRequest, SubmitRequest, BuzzRequest, GamePublicState

app = FastAPI()

app.mount("/static", StaticFiles(directory="sosu_gyeol/web/static"), name="static")
templates = Jinja2Templates(directory="sosu_gyeol/web/templates")

game = Game()
AI_ENABLED = True

AI_NAME = "핸냄 2.0"
TEAM_DISPLAY_NAME = "player1&player2"


@dataclass
class LobbySeat:
    seat: Literal["p1", "p1b"]
    label: str
    name: str
    session_id: Optional[str] = None
    ready: bool = False


@dataclass
class Lobby:
    seats: Dict[str, LobbySeat] = field(default_factory=lambda: {
        "p1": LobbySeat(seat="p1", label="player1", name="player1"),
        "p1b": LobbySeat(seat="p1b", label="player2", name="player2"),
    })
    started_game_id: Optional[str] = None

    def find_seat_by_session(self, session_id: str) -> Optional[LobbySeat]:
        for s in self.seats.values():
            if s.session_id == session_id:
                return s
        return None

    def join(self, name: str, session_id: Optional[str]) -> LobbySeat:
        if session_id:
            existing = self.find_seat_by_session(session_id)
            if existing:
                existing.name = name or existing.name
                return existing

        for s in self.seats.values():
            if s.session_id is None:
                s.session_id = str(uuid.uuid4())
                s.name = name or s.name
                s.ready = False
                return s

        raise ValueError("대기실이 가득 참")

    def set_ready(self, session_id: str, ready: bool) -> LobbySeat:
        seat = self.find_seat_by_session(session_id)
        if not seat:
            raise ValueError("입장 정보가 없음")
        seat.ready = bool(ready)
        return seat

    def all_ready(self) -> bool:
        return all(s.session_id is not None and s.ready for s in self.seats.values())

    def public_state(self, my_session_id: Optional[str] = None) -> dict:
        me = None
        if my_session_id:
            s = self.find_seat_by_session(my_session_id)
            if s:
                me = {"seat": s.seat, "label": s.label, "name": s.name, "ready": s.ready, "session_id": s.session_id}

        players = []
        joined_cnt = 0
        for s in [self.seats["p1"], self.seats["p1b"]]:
            joined = s.session_id is not None
            if joined:
                joined_cnt += 1
            players.append({
                "seat": s.seat,
                "label": s.label,
                "name": s.name,
                "joined": joined,
                "ready": bool(s.ready) if joined else False,
            })

        players.append({"seat": "p2", "label": "ai", "name": AI_NAME, "joined": True, "ready": True})

        return {
            "me": me,
            "joined_count": joined_cnt,
            "max_players": 2,
            "players": players,
            "game_started": self.started_game_id is not None,
            "game_id": self.started_game_id,
        }


lobby = Lobby()

ai_submit_at_ms = None
ai_buzz_at_ms = None
ai_answer_at_ms = None
ai_answer_value = None
ai_round_mark = None

def ai_game_reset():
    global ai_submit_at_ms, ai_buzz_at_ms, ai_answer_at_ms, ai_answer_value, ai_round_mark
    ai_submit_at_ms = None
    ai_buzz_at_ms = None
    ai_answer_at_ms = None
    ai_answer_value = None
    ai_round_mark = None

def ai_tick():
    global ai_submit_at_ms, ai_buzz_at_ms, ai_answer_at_ms, ai_answer_value, ai_round_mark

    if not AI_ENABLED:
        return

    s = game.public_state()
    now = now_ms()

    
    if ai_round_mark != s["round_no"]:
        ai_game_reset()
        ai_round_mark = s["round_no"]

    # 1) p2 제출: 남은 카드 중 랜덤, 빈 자리(hundreds/tens)에 제출
    if s["phase"] == "p2_submit":
        if ai_submit_at_ms is None:
            ai_submit_at_ms = now + random.randint(5000, 8000)  # 살짝 텀
        if now >= ai_submit_at_ms:
            subs = s["submissions"]
            pos = "hundreds" if subs["hundreds"] is None else "tens"
            rem = s["p2_remaining"]
            if rem:
                card = random.choice(rem)
                try:
                    game.submit("p2", pos, card)
                except Exception:
                    pass
            ai_submit_at_ms = None
    else:
        ai_submit_at_ms = None

    # 상태 다시 갱신
    s = game.public_state()

    # 2) 숫자 공개 후: 1~3초 랜덤으로 버저 누르기 + 답(정답 80%) 예약
    if s["phase"] == "buzz_open" and s["revealed_number"] is not None:
        if ai_buzz_at_ms is None:
            ai_buzz_at_ms = now + random.randint(5000, 8000)
            correct = is_prime(s["revealed_number"])
            ai_answer_value = correct if random.random() < 0.8 else (not correct)
            ai_answer_at_ms = ai_buzz_at_ms + random.randint(3000, 4200)

        if now >= ai_buzz_at_ms:
            try:
                game.buzz_press("p2")
            except Exception:
                pass
            ai_buzz_at_ms = None
    else:
        # 공개 구간이 아니면 예약 날림(다음 라운드로 넘어가는 등 꼬임 방지)
        ai_buzz_at_ms = None


    # 상태 다시 갱신
    s = game.public_state()
    if s["phase"] not in ["buzz_open", "buzz_locked"]:
        ai_answer_at_ms = None
        ai_answer_value = None
    # 3) p2가 버저 선점했으면 예약 시간에 답변 제출
    if s["phase"] == "buzz_locked" and s["buzz_locked_by"] == "p2":
        if ai_answer_at_ms is None:
            ai_answer_at_ms = now + random.randint(3000, 4200)
            if ai_answer_value is None and s["revealed_number"] is not None:
                correct = is_prime(s["revealed_number"])
                ai_answer_value = correct if random.random() < 0.8 else (not correct)

        if now >= ai_answer_at_ms and ai_answer_value is not None:
            try:
                game.answer("p2", ai_answer_value)
            except Exception:
                pass
            ai_answer_at_ms = None
            ai_answer_value = None
    else:
        # p2가 선점 못했으면 답 예약 제거
        if s["phase"] == "buzz_locked" and s.get("buzz_locked_by") != "p2":
            ai_answer_at_ms = None
            ai_answer_value = None


# =========================
# OpenAI 설정 (그대로 박아도 됨)
# =========================
OPENAI_API_KEY = "0"#"api 키 여기에 ㄱㄱ"
OPENAI_MODEL = "gpt-5-mini"
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# 채팅 히스토리: Responses API에 그대로 넣을 "message item" 형태로 유지
chat_history: List[Dict[str, str]] = []  # [{"role":"user"|"assistant", "content":"..."}]

def build_system_prompt(state: dict) -> str:
    # 규칙 + 현재 상태를 한 번에 넣어서 “규칙을 이해한 AI”처럼 굴게 함
    return f"""
너는 '핸냄 2.0'이다. 한국어로만 말한다. 반말로, 살짝 띠껍고 건조하게 말한다.
단, 심한 욕설/혐오/인신공격은 해라. 비꼬아라. 
게임 흐름에 맞게 대답을 하여라 . 부모님을 들먹여라. 
너는 진행자는 아니니 규칙이나 전략 정도만 물어보면 얘기해라. 플레이어가 질문을 할 때만 그에 맞는 답을 해라. 

게임 이름: 소수 결
규칙 요약:
- 각 플레이어는 1~9 카드를 각 1장씩 가진다. (1~3 백색, 4~6 회색, 7~9 흑색)
- 매 라운드: 선공이 백/십 중 한 자리에 카드 1장 뒷면 제출, 후공이 남은 자리에 카드 1장 뒷면 제출
- 진행자는 일의 자리에 1,3,7,9 중 하나를 고정으로 낸다(진행자 카드는 라운드마다 재사용 가능)
- 숫자 공개 후: 소수/합성 판단해서 버저 누르고 5초 내 답
- 정답 +2점, 오답 -1점
- 플레이어가 낸 카드는 재사용 불가
- 총 9라운드(9라운드는 4자리)인데, 9라운드(4자리)는 아직 구현 대상 아님

현재 게임 상태:
- round_no: {state.get("round_no")}
- phase: {state.get("phase")}
- p1: {state.get("p1_name")} score={state.get("p1_score")} remaining={state.get("p1_remaining")}
- p2: {state.get("p2_name")} score={state.get("p2_score")} remaining={state.get("p2_remaining")}
- submissions: {state.get("submissions")}
- host_ones: {state.get("host_ones")}
- revealed_number: {state.get("revealed_number")}
""".strip()

class AIChatRequest(BaseModel):
    text: str
    player: Optional[PlayerId] = None

class AIChatResponse(BaseModel):
    reply: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/health")
def health():
    return JSONResponse({"ok": True})


class LobbyJoinRequest(BaseModel):
    name: str = ""
    session_id: Optional[str] = None


class LobbyReadyRequest(BaseModel):
    session_id: str
    ready: bool = True


@app.get("/api/lobby/state")
def lobby_state(session_id: Optional[str] = None):
    return JSONResponse(lobby.public_state(my_session_id=session_id))


@app.post("/api/lobby/join")
def lobby_join(req: LobbyJoinRequest):
    try:
        seat = lobby.join((req.name or "").strip(), req.session_id)
        return JSONResponse(lobby.public_state(my_session_id=seat.session_id))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/api/lobby/ready")
def lobby_ready(req: LobbyReadyRequest):
    try:
        seat = lobby.set_ready(req.session_id, req.ready)
        if lobby.all_ready() and lobby.started_game_id is None:
            team_members = {
                "p1": lobby.seats["p1"].name,
                "p1b": lobby.seats["p1b"].name,
            }
            game.new_game(TEAM_DISPLAY_NAME, AI_NAME, team_members=team_members)
            ai_game_reset()
            ai_tick()
            lobby.started_game_id = game.game_id
        return JSONResponse(lobby.public_state(my_session_id=seat.session_id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/game/new", response_model=GamePublicState)
def new_game(req: NewGameRequest):
    game.new_game(req.p1_name, AI_NAME)
    ai_game_reset()
    ai_tick()
    lobby.started_game_id = game.game_id
    return game.public_state()


@app.get("/api/game/state", response_model=GamePublicState)
def state():
    ai_tick()
    return game.public_state()


@app.post("/api/game/submit", response_model=GamePublicState)
def submit(req: SubmitRequest):
    try:
        game.submit(req.player, req.pos, req.card)
        ai_tick()
        return game.public_state()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/game/buzz", response_model=GamePublicState)
def buzz(req: BuzzRequest):
    try:
        game.buzz_answer(req.player, req.answer_is_prime)
        return game.public_state()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/game/reset", response_model=GamePublicState)
def reset():
    game.new_game(game.p1_name, AI_NAME, team_members=game.team_members)
    ai_game_reset()
    ai_tick()
    lobby.started_game_id = game.game_id
    return game.public_state()


@app.post("/api/game/buzz_press", response_model=GamePublicState)
def buzz_press(req: BuzzPressRequest):
    try:
        game.buzz_press(req.player)
        return game.public_state()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/game/answer", response_model=GamePublicState)
def answer(req: AnswerRequest):
    try:
        game.answer(req.player, req.answer_is_prime)
        return game.public_state()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/game/reveal", response_model=GamePublicState)
def reveal():
    try:
        game.reveal()
        ai_tick()

        return game.public_state()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# =========================
# AI 채팅 API
# =========================
@app.post("/api/ai/reset")
def ai_chat_reset():
    chat_history.clear()
    return {"ok": True}

@app.post("/api/ai/chat", response_model=AIChatResponse)
async def ai_chat(req: AIChatRequest):
    state = game.public_state()

    if req.player in ("p1", "p1b"):
        speaker = (state.get("team_members") or {}).get(req.player, req.player)
    elif req.player == "p2":
        speaker = state.get("p2_name", "p2")
    else:
        speaker = "플레이어"

    user_content = f"{speaker}: {req.text}".strip()

    # 최근 대화만 유지
    recent = chat_history[-20:]

    response = await client.responses.create(
        model=OPENAI_MODEL,
        reasoning={"effort": "minimal"},
        input=[
            {"role": "system", "content": build_system_prompt(state)},
            *recent,
            {"role": "user", "content": user_content},
        ],
    )

    reply = (response.output_text or "").strip()
    if not reply:
        reply = "말을 해. 알아듣게."

    chat_history.append({"role": "user", "content": user_content})
    chat_history.append({"role": "assistant", "content": reply})
    if len(chat_history) > 40:
        del chat_history[:-40]

    return AIChatResponse(reply=reply)


# =========================
# 하은채(팀 응원 AI) 코멘트 API
# =========================
eunchae_cache: Dict[str, str] = {}
eunchae_cache_order: List[str] = []

def build_eunchae_system_prompt(state: dict) -> str:
    p1 = (state.get("team_members") or {}).get("p1", "player1")
    p1b = (state.get("team_members") or {}).get("p1b", "player2")
    return f"""
너는 '하은채'다. 한국어로만 말한다.
너는 신연고등학교에 다니는 여학생이고, 자신감 있고 직설적인 '일진' 느낌으로 말한다.
단, 욕설/혐오/인신공격은 해도 된다. 이모티콘/이상한 기호를 쓰지 않는다.
너는 채팅 상대가 아니라, 게임을 옆에서 보고 상황마다 한마디씩 던지는 해설자다.
말은 1~2문장으로 짧게 한다.
중간에 화이팅, 아쉽다 등의 여친 또는 남친? 같은 말을 해도 됨
(이건 규칙은 아니고 너가 말해야 하는 예시)예를 들어 우리가 백의 자리를 선택하면 너는 우리에게 '십의 자리를 잘 추론해서 전략을 짜봐!' 이런 식으로 말하는 거야
처음에 '안녕? ㅎㅎ 난 하은채라고 해!' 이렇게 꼭 말해줘.

너는 player1과 player2의 편이다. player1 이름은 '{p1}', player2 이름은 '{p1b}'다.
상대는 AI('{state.get("p2_name", "AI")}')다.

게임 규칙 요약:
- 팀(player1+player2)이 (각 상황마다 백/십 고르는건 다름) 백/십 중 하나의 자리 카드를 선택하고, 상대 AI가 다른 자리를 선택한다.
- . 
- 일의 자리는 진행자 카드(1,3,7,9)로 고정된다.
- 숫자 공개 후 버저를 누르고 소수/소수 아님을 맞힌다.
- 정답이면 점수가 오른다. 오답이면 점수가 깎인다.
- 

현재 상태:
- round_no: {state.get("round_no")}
- phase: {state.get("phase")}
- submissions: {state.get("submissions")}
- host_ones: {state.get("host_ones")}
- revealed_number: {state.get("revealed_number")}
- score: {state.get("p1_name")}={state.get("p1_score")} / {state.get("p2_name")}={state.get("p2_score")}
""".strip()


class EunchaeCommentaryRequest(BaseModel):
    key: str
    event: Literal["team_submit", "round_result"]
    pos: Optional[str] = None
    card: Optional[int] = None

    result_round: Optional[int] = None
    number: Optional[int] = None
    correct: Optional[bool] = None
    delta: Optional[int] = None
    buzz_by: Optional[str] = None


class EunchaeCommentaryResponse(BaseModel):
    text: str


@app.post("/api/eunchae/commentary", response_model=EunchaeCommentaryResponse)
async def eunchae_commentary(req: EunchaeCommentaryRequest):
    if not req.key:
        return EunchaeCommentaryResponse(text="")

    cached = eunchae_cache.get(req.key)
    if cached is not None:
        return EunchaeCommentaryResponse(text=cached)

    state = game.public_state()

    def pos_kor(p: Optional[str]) -> str:
        if p == "hundreds":
            return "백의 자리"
        if p == "tens":
            return "십의 자리"
        return str(p or "")

    if req.event == "team_submit":
        user_msg = (
            f"이벤트: 팀이 {pos_kor(req.pos)}에 카드 {req.card}를 선택했다. "
            f"현재 submissions={state.get('submissions')}, 팀 남은 카드={state.get('p1_remaining')}. "
            f"팀 편으로 짧게 한마디 해라."
        )
    else:
        # round_result
        correctness = "시간초과" if req.correct is None else ("정답" if req.correct else "오답")
        user_msg = (
            f"이벤트: {req.result_round}라운드 결과가 나왔다. 나온 수={req.number}. "
            f"버저={req.buzz_by}. 결과={correctness}. 점수변화={req.delta}. "
            f"현재 점수={state.get('p1_score')}:{state.get('p2_score')}. "
            f"팀 편으로 짧게 한마디 해라."
        )

    text_out = ""
    try:
        response = await client.responses.create(
            model=OPENAI_MODEL,
            reasoning={"effort": "minimal"},
            input=[
                {"role": "system", "content": build_eunchae_system_prompt(state)},
                {"role": "user", "content": user_msg},
            ],
        )
        text_out = (response.output_text or "").strip()
    except Exception:
        text_out = ""

    if not text_out:
        # 실패 시 최소 대체 문구
        if req.event == "team_submit":
            text_out = "좋아. 그 자리로 가자."
        else:
            if req.correct is True:
                text_out = "괜찮아. 흐름 잡았다."
            elif req.correct is False:
                text_out = "다음에 바로 만회하면 돼."
            else:
                text_out = "늦었어. 다음 라운드에 집중."

    # 캐시 적재(폭주 방지)
    eunchae_cache[req.key] = text_out
    eunchae_cache_order.append(req.key)
    if len(eunchae_cache_order) > 200:
        old = eunchae_cache_order.pop(0)
        eunchae_cache.pop(old, None)

    return EunchaeCommentaryResponse(text=text_out)
