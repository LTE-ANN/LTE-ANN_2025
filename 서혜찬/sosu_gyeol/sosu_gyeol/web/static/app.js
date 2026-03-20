const el = (id) => document.getElementById(id);

async function apiGet(path) {
  const r = await fetch(path);
  const t = await r.text();
  let data;
  try { data = JSON.parse(t); } catch { throw new Error(t); }
  if (!r.ok) throw new Error(data.detail || "요청 실패");
  return data;
}

async function apiPost(path, bodyObj) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(bodyObj),
  });
  const t = await r.text();
  let data;
  try { data = JSON.parse(t); } catch { throw new Error(t); }
  if (!r.ok) throw new Error(data.detail || "요청 실패");
  return data;
}

/* =========================
   하은채(팀 응원 AI) 말풍선
   ========================= */
let eunchaeLastKey = null;

function setEunchaeBubble(text) {
  const b = document.getElementById("eunchaeBubble");
  if (!b) return;
  b.textContent = (text || "").trim();
  b.classList.remove("pop");
  void b.offsetWidth;
  b.classList.add("pop");
}

async function requestEunchae(payload) {
  try {
    if (!payload || !payload.key) return;
    if (payload.key === eunchaeLastKey) return; // 프론트 중복 방지
    eunchaeLastKey = payload.key;

    const resp = await apiPost("/api/eunchae/commentary", payload);
    if (resp && typeof resp.text === "string" && resp.text.trim()) {
      setEunchaeBubble(resp.text.trim());
    }
  } catch (e) {
    // 실패해도 게임 진행은 그대로
  }
}

function changedSubmission(prev, cur) {
  const a = prev?.submissions || {};
  const b = cur?.submissions || {};
  if (a.hundreds == null && b.hundreds != null) return { pos: "hundreds", card: b.hundreds };
  if (a.tens == null && b.tens != null) return { pos: "tens", card: b.tens };
  return null;
}

function isTypingTarget(target) {
  if (!target) return false;
  const tag = (target.tagName || "").toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select";
}

/* =========================
   UI를 통째로 다시 그림 (index.html 건드릴 필요 없음)
   ========================= */
function buildUI() {
  document.body.innerHTML = `
    
<div id="eunchaePanel" class="eunchae-panel">
  <img class="eunchae-avatar" src="/static/eunchae.png" alt="하은채" />
  <div id="eunchaeBubble" class="eunchae-bubble">좋아. 시작해</div>
</div>

<div class="app-container">
      <header class="game-header">
        <h1>소수 결</h1>
        <div class="subtitle">세 자리 수가 소수인지 빠르게 맞춰라</div>
      </header>

      <section id="screenLobby" class="screen active">
        <div class="card">
          <h2>대기실</h2>

          <div class="form-row">
            <input id="lobbyName" class="input" placeholder="이름 입력" />
          </div>

          <div class="btn-row">
            <button id="btnJoin" class="btn btn-primary">게임 입장하기</button>
          </div>

          <div id="joinStatus" class="join-status"></div>

          <div class="player-count">
            <div class="count-label">대기 중 플레이어</div>
            <div id="playerCount" class="count-number">0 / 2</div>
          </div>

          <div id="playerList" class="player-list"></div>

          <div class="btn-row">
            <button id="btnReady" class="btn btn-success" disabled>준비 완료</button>
          </div>

          <div class="hint-text">
            2인 협동: player1&player2 vs 핸냄 2.0<br/>
            두 명 모두 준비 완료를 누르면 시작
          </div>
        </div>
      </section>

      <section id="screenGame" class="screen">
        <div class="card">
          <h2>메인 게임</h2>

          <div class="game-top">
            <div class="badges">
              <div id="phaseBadge" class="badge">idle</div>
              <div id="roundBadge" class="badge"></div>
              <div id="timeLine" class="badge"></div>
            </div>
            <div id="scoreLine" class="scoreline"></div>
          </div>

          <div class="big-number">
            <div id="statusLine" class="label"></div>

            <div id="slots" class="slots"></div>

            <div id="bigNumberValue" class="value">---</div>
            <div id="numberLine" class="sub"></div>
            <div class="sub">버저: Space, 답: q(소수) w(소수 아님)</div>
            <div class="sub" style="margin-top:8px;">
              <span style="display:inline-flex;align-items:center;gap:6px;margin-right:14px;">
                <span class="card-back back-white" style="width:14px;height:18px;border-radius:4px;display:inline-block;"></span>
                흰색 카드: 1~3
              </span>
              <span style="display:inline-flex;align-items:center;gap:6px;margin-right:14px;">
                <span class="card-back back-gray" style="width:14px;height:18px;border-radius:4px;display:inline-block;"></span>
                회색 카드: 4~6
              </span>
              <span style="display:inline-flex;align-items:center;gap:6px;">
                <span class="card-back back-black" style="width:14px;height:18px;border-radius:4px;display:inline-block;"></span>
                검은색 카드: 7~9
              </span>
            </div>
          </div>

          <div id="eventOverlay" class="event-overlay">
            <div id="eventBox" class="event-box event-buzz">
              <div id="eventRound" style="font-size:12px; opacity:0.85; margin-bottom:6px; white-space:pre-line;"></div>
              <div id="eventTitle" class="event-title"></div>
              <div id="eventSub" class="event-sub"></div>
            </div>
          </div>

          <div class="game-grid">
            <div class="panel">
              <h3>제출</h3>
              <div id="posHint" class="small-text"></div>
              <div class="row" style="margin-top:10px;">
                <button id="btnReset" class="btn btn-ghost" style="height:44px;">리셋</button>
              </div>
              <div id="cards" class="cards-grid"></div>
              <div id="msg" class="msg"></div>
            </div>

            <div class="panel">
              <h3>버저</h3>
              <div class="row" style="margin-bottom:10px;">
                <button id="btnBuzz" class="btn btn-ghost" style="height:44px;">버저</button>
              </div>
              <div class="row">
                <button id="btnPrime" class="btn btn-success" style="height:44px;">소수</button>
                <button id="btnNotPrime" class="btn btn-ghost" style="height:44px;">소수 아님</button>
              </div>
              <div id="buzzLine" class="msg"></div>
            </div>
          </div>

          <pre id="log" class="log"></pre>

          <div class="game-footer">
            <button id="btnBackLobby" class="btn btn-ghost">대기실로</button>
            <button id="btnRefresh" class="btn btn-ghost">새로고침</button>
          </div>
        </div>
      </section>
    </div>
  `;
}

/* =========================
   대기실 로직
   ========================= */
let joined = false;
let ready = false;
let myName = "";
let mySeat = null; // "p1" | "p1b"
let sessionId = null;
let lobbyCache = null;
let lobbyPollTimer = null;
let stateCache = null;

function setJoinStatus(text, kind) {
  const box = el("joinStatus");
  if (!box) return;
  box.textContent = text || "";
  box.className = "join-status show " + (kind || "");
}

function renderLobbyList(lobbyState) {
  if (lobbyState) lobbyCache = lobbyState;

  const list = el("playerList");
  const count = el("playerCount");
  if (!list || !count) return;

  const lobby = lobbyCache;
  if (!lobby) {
    count.textContent = "0 / 2";
    list.innerHTML = "";
    const btnReady = el("btnReady");
    if (btnReady) btnReady.disabled = true;
    return;
  }

  count.textContent = `${lobby.joined_count || 0} / 2`;
  list.innerHTML = "";

  (lobby.players || []).forEach((p) => {
    const div = document.createElement("div");
    div.className = "player-item" + (p.ready ? " ready" : "");

    const left = document.createElement("div");
    left.className = "player-name";
    const isMe = mySeat && p.seat === mySeat;
    const base = p.seat === "p2" ? (p.name || "ai") : (p.label || p.seat);
    const nick = p.seat !== "p2" && p.name && p.name !== p.label ? ` (${p.name})` : "";
    left.textContent = isMe ? `${base}${nick} (나)` : `${base}${nick}`;

    const right = document.createElement("div");
    right.className = "player-status";
    right.textContent = p.ready ? "준비 완료" : "대기 중";

    div.appendChild(left);
    div.appendChild(right);
    list.appendChild(div);
  });

  const btnReady = el("btnReady");
  if (btnReady) btnReady.disabled = !joined || ready;
}

function showScreen(which) {
  const lobby = el("screenLobby");
  const game = el("screenGame");
  if (!lobby || !game) return;

  lobby.classList.toggle("active", which === "lobby");
  game.classList.toggle("active", which === "game");
}

/* =========================
   게임 UI
   ========================= */
function setMsg(t) {
  const m = el("msg");
  if (!m) return;
  m.textContent = t || "";
}

function safeText(id, t) {
  const n = el(id);
  if (!n) return;
  n.textContent = t || "";
}

function renderHistory(s) {
  const log = el("log");
  if (!log) return;
  const hist = s.history || [];
  if (hist.length === 0) {
    log.textContent = "";
    return;
  }
  const lines = hist.map(h => {
    const corr = h.correct ? "O" : "X";
    return `R${h.round} | ${h.number} | buzz:${h.buzz_by} | ans:${h.answer_is_prime} | ${corr} | delta:${h.delta} | score P1:${h.p1_score} P2:${h.p2_score}`;
  });
  log.textContent = lines.join("\n");
}

function renderCards() {
  const cardsDiv = el("cards");
  if (!cardsDiv) return;

  cardsDiv.innerHTML = "";
  if (!stateCache) return;

  const remaining = stateCache.p1_remaining || [];
  remaining.forEach((n) => {
    const b = document.createElement("button");
    b.className = "cardbtn";
    b.textContent = String(n);
    b.addEventListener("click", () => submitCard(n));
    cardsDiv.appendChild(b);
  });

  if (remaining.length === 0) {
    const info = document.createElement("div");
    info.className = "small-text";
    info.textContent = "남은 카드 없음";
    cardsDiv.appendChild(info);
  }
}

function updateUIByPhase(s) {
  const phase = s.phase;

  const inSubmit = (phase === "p1_submit");
  const inReadyReveal = (phase === "ready_reveal");
  const inBuzzOpen = (phase === "buzz_open");
  const inBuzzLocked = (phase === "buzz_locked");

  document.querySelectorAll(".cardbtn").forEach((b) => {
    b.disabled = !inSubmit;
  });

  const btnBuzz = el("btnBuzz");
  const btnPrime = el("btnPrime");
  const btnNotPrime = el("btnNotPrime");

  if (btnBuzz) btnBuzz.disabled = !inBuzzOpen;

  const canAnswer = inBuzzLocked && mySeat && (s.buzz_locked_by === mySeat);
  if (btnPrime) btnPrime.disabled = !canAnswer;
  if (btnNotPrime) btnNotPrime.disabled = !canAnswer;

  if (inReadyReveal) safeText("buzzLine", "자동 공개 대기");
  else if (inBuzzOpen) safeText("buzzLine", "Space로 버저");
  else if (inBuzzLocked) safeText("buzzLine", `버저 선점: ${playerNameById(s, s.buzz_locked_by)}`);
  else safeText("buzzLine", "");
}

/* =========================
   오버레이/슬롯/자동공개 상태
   ========================= */
let prevState = null;
let lastHistLen = 0;
let overlayTimer = null;

let autoRevealAtMs = null;
let autoRevealFired = false;

function posKor(pos) {
  if (pos === "hundreds") return "백의 자리";
  if (pos === "tens") return "십의 자리";
  return pos;
}

function playerNameById(s, pid) {
  const tm = s.team_members || {};
  if (pid === "p1") return tm.p1 || "player1";
  if (pid === "p1b") return tm.p1b || "player2";
  if (pid === "p2") return s.p2_name || "ai";
  return String(pid || "");
}

function digitBackClass(d) {
  if (d == null) return "";
  if (d <= 3) return "back-white";
  if (d <= 6) return "back-gray";
  return "back-black";
}

function renderSlots(s) {
  const wrap = el("slots");
  if (!wrap) return;

  const H = s.submissions?.hundreds ?? null;
  const T = s.submissions?.tens ?? null;
  const O = s.host_ones ?? null;

  const items = [
    { v: H, label: "H" },
    { v: T, label: "T" },
    { v: O, label: "O" },
  ];

  wrap.innerHTML = "";

  items.forEach((it) => {
    const slot = document.createElement("div");
    slot.className = "slot" + (it.v == null ? " empty" : "");

    if (it.v != null) {
      const back = document.createElement("div");
      back.className = "card-back " + digitBackClass(it.v);
      slot.appendChild(back);
    }
    wrap.appendChild(slot);
  });
}

function showOverlay(title, sub, kind, durationMs = 2000, topExtra = "") {
  const overlay = el("eventOverlay");
  const box = el("eventBox");
  const r = el("eventRound");
  const t = el("eventTitle");
  const s = el("eventSub");
  if (!overlay || !box || !r || !t || !s) return;

  const rn = stateCache?.round_no;
  const mx = stateCache?.max_rounds;
  let roundText = "";
  if (rn && mx) roundText = `${rn} / ${mx} 라운드`;
  else if (rn) roundText = `${rn}라운드`;

  r.textContent = topExtra ? `${roundText}
${topExtra}` : roundText;

  t.textContent = title || "";
  s.textContent = sub || "";

  box.classList.remove("event-success", "event-danger", "event-buzz");
  if (kind === "success") box.classList.add("event-success");
  else if (kind === "danger") box.classList.add("event-danger");
  else box.classList.add("event-buzz");

  overlay.classList.add("show");

  if (overlayTimer) clearTimeout(overlayTimer);
  overlayTimer = setTimeout(() => {
    overlay.classList.remove("show");
  }, durationMs);
}

function smallestDivisor(n) {
  n = Number(n);
  if (!Number.isFinite(n)) return null;
  if (n <= 1) return 1;
  if (n % 2 === 0) return 2;
  const r = Math.floor(Math.sqrt(n));
  for (let d = 3; d <= r; d += 2) {
    if (n % d === 0) return d;
  }
  return null; // 소수
}

function numberExplain(n) {
  if (n == null) return "";
  n = Number(n);
  if (!Number.isFinite(n)) return "";
  if (n <= 1) return `${n}는 소수가 아닙니다`;
  const d = smallestDivisor(n);
  if (d == null) return `${n}는 '소수'입니다`;
  return `${n}는 '${d}'의 배수입니다`;
}

function formatPointDelta(delta) {
  const d = Number(delta || 0);
  if (d > 0) return `${d}포인트 획득`;
  if (d < 0) return `${Math.abs(d)}포인트 감점`;
  return `0포인트`;
}

let roundBadgeTimer = null;

function showRoundBadge(text, durationMs = 5000) {
  const rb = el("roundBadge");
  if (!rb) return;

  rb.textContent = text || "";

  if (roundBadgeTimer) clearTimeout(roundBadgeTimer);
  roundBadgeTimer = setTimeout(() => {
    // 시간이 지나면 비워서 다시 기본 UI만 보이게
    const rb2 = el("roundBadge");
    if (rb2) rb2.textContent = "";
  }, durationMs);
}
/* =========================
   상태 렌더
   ========================= */
function showState(s) {
  stateCache = s;

  const posHint = el("posHint");
  // 턴/자리 안내
  if (posHint) {
    const subs = s.submissions || {};
    const need = (subs.hundreds == null) ? "hundreds" : (subs.tens == null ? "tens" : null);
    if (s.phase === "p1_submit") {
      posHint.textContent = `너 차례: ${posKor(need)} 카드를 선택해`;

    } else if (s.phase === "p2_submit") {
      posHint.textContent = `${s.p2_name} 차례: ${posKor(need)} 선택 중`;
    } else if (s.phase === "ready_reveal") {
      posHint.textContent = `선택 완료: 곧 숫자 공개`;
    } else if (s.phase === "buzz_open") {
      posHint.textContent = `숫자 공개됨: Space로 버저`;
    } else if (s.phase === "buzz_locked") {
      const who = playerNameById(s, s.buzz_locked_by);
      posHint.textContent = `버저 선점: ${who}`;
    } else {
      posHint.textContent = "";
    }
  }

  // 슬롯(카드 뒷면) 표시
  renderSlots(s);

  // 라운드 시작 표시는 오버레이가 아니라 배지로
  if (!prevState) {
    showRoundBadge(`!!${s.round_no}라운드 시작!!`, 5000);
  } else if (prevState.round_no !== s.round_no) {
    showRoundBadge(`!!${s.round_no}라운드 시작!!`, 5000);
  }
  // 게임 종료면: 최종 결과만 보여주고, 다른 오버레이/자동공개/라운드결과 오버레이는 더 이상 실행하지 않음
  const isGameOver = (s.phase === "game_over" || s.phase === "finished_round9");
  if (isGameOver) {
    // winner가 안 내려올 수도 있으니 점수로 보정
    const winner =
      (s.winner === "p1" || s.winner === "p2" || s.winner === "draw")
        ? s.winner
        : (s.p1_score > s.p2_score) ? "p1"
        : (s.p2_score > s.p1_score) ? "p2"
        : "draw";

    const result =
      winner === "p1" ? "승리" :
      winner === "p2" ? "패배" :
      "무승부";

    if (!prevState || (prevState.phase !== "game_over" && prevState.phase !== "finished_round9")) {
      showRoundBadge(`게임 종료`, 8000);
      showOverlay(
        `게임 종료: ${result}`,
        `최종 점수 ${s.p1_name} ${s.p1_score} : ${s.p2_score} ${s.p2_name}`,
        (winner === "p1") ? "success" : (winner === "p2") ? "danger" : "buzz",
        8000
      );
      stopPolling(); // 덮어쓰기 방지: 더 이상 폴링/틱으로 showState 재호출 안 함
    }

    safeText("timeLine", "종료");
    lastHistLen = (s.history || []).length;
    prevState = s;
    return;
  }



  // 자동 공개 카운트다운: ready_reveal 진입 순간에 시작
  if (prevState && prevState.phase !== "ready_reveal" && s.phase === "ready_reveal") {
    autoRevealAtMs = Date.now() + 3000;
    autoRevealFired = false;
    showOverlay(`3초 뒤 숫자 공개`, "", "buzz");
  }

  // ready_reveal 상태에서 남은 초 표시 + 시간이 되면 자동 공개
  if (s.phase === "ready_reveal" && autoRevealAtMs) {
    const remainMs = autoRevealAtMs - Date.now();
    const remainSec = Math.max(0, Math.ceil(remainMs / 1000));
    showOverlay(`${remainSec}초 뒤 숫자 공개`, "", "buzz");

    if (remainMs <= 0 && !autoRevealFired) {
      autoRevealFired = true;
      revealNumber();
    }
  }

  // ready_reveal 아니면 자동 공개 상태 리셋
  if (s.phase !== "ready_reveal") {
    autoRevealAtMs = null;
    autoRevealFired = false;
  }


// 팀(p1/p1b) 숫자 선택 순간
if (prevState && prevState.phase === "p1_submit" && s.phase !== "p1_submit") {
  const ch = changedSubmission(prevState, s);
  if (ch) {
    const key = `ts:${s.game_id}:${s.round_no}:${ch.pos}:${ch.card}`;
    requestEunchae({
      key,
      event: "team_submit",
      pos: ch.pos,
      card: ch.card,
    });
  }
}

  // AI 숫자 선택 완료
  if (prevState && prevState.phase === "p2_submit" && s.phase !== "p2_submit") {
    showOverlay(`[${s.p2_name}] 숫자 선택 완료!`, "", "buzz");
  }

  // 버저 선점 순간
  if (prevState && !prevState.buzz_locked_by && s.buzz_locked_by) {
    const who = playerNameById(s, s.buzz_locked_by);
    showOverlay(`[${who}] 버저!`, "", "buzz");
  }

  safeText("phaseBadge", s.phase);
  const mx = s.max_rounds ?? 9;
  const roleText = mySeat ? ` | 나: ${mySeat === "p1" ? "player1" : "player2"}` : "";
  safeText("statusLine", `round: ${s.round_no}/${mx} | phase: ${s.phase} | turn_first: ${s.turn_first}${roleText}`);

  safeText("scoreLine", `${s.p1_name}(${s.p1_score}) vs ${s.p2_name}(${s.p2_score})`);

  const hFilled = (s.submissions && s.submissions.hundreds != null);
  const tFilled = (s.submissions && s.submissions.tens != null);
  safeText(
    "numberLine",
    `백의 자리=${hFilled ? "선택됨" : "?"}, 십의 자리=${tFilled ? "선택됨" : "?"} | 공개=${(s.revealed_number ?? "---")}`
  );

  safeText("bigNumberValue", (s.revealed_number != null) ? String(s.revealed_number) : "---");

  // 타이머 뱃지
  const now = Date.now();
  let left = "";
  if (s.phase === "p1_submit" || s.phase === "p2_submit") {
    if (s.submit_deadline_ms) {
      const sec = Math.max(0, Math.floor((s.submit_deadline_ms - now) / 1000));
      left = `제출 남은시간: ${sec}초`;
    }
  } else if (s.phase === "buzz_locked") {
    if (s.buzz_deadline_ms) {
      const sec = Math.max(0, Math.floor((s.buzz_deadline_ms - now) / 1000));
      left = `답변 남은시간: ${sec}초`;
    }
  } else if (s.phase === "ready_reveal") {
    left = "공개 대기";
  } else if (s.phase === "buzz_open") {
    left = "버저 대기";
  }
  safeText("timeLine", left);

  renderHistory(s);
  renderCards();
  updateUIByPhase(s);

  // 라운드 결과(정답/오답/시간초과)
const hist = s.history || [];
if (hist.length > lastHistLen) {
  const last = hist[hist.length - 1] || {};
  const who = playerNameById(s, last.buzz_by);
  const deltaText = formatPointDelta(last.delta);

  const numForExplain = (last.number != null) ? last.number : s.revealed_number;
  const extra = numberExplain(numForExplain);

  if (last.answer_is_prime == null) {
    showOverlay(`[${who}] 시간 초과!`, deltaText, "danger", 2200, extra);
  } else if (last.correct) {
    showOverlay(`[${who}] 정답!`, deltaText, "success", 2200, extra);
  } else {
    showOverlay(`[${who}] 오답!`, deltaText, "danger", 2200, extra);
  }

  const key = `res:${s.game_id}:${last.round}:${last.number}:${last.buzz_by}:${last.correct}:${last.delta}:${last.answer_is_prime}`;
  requestEunchae({
    key,
    event: "round_result",
    result_round: last.round,
    number: last.number,
    correct: (last.answer_is_prime == null) ? null : !!last.correct,
    delta: last.delta,
    buzz_by: who,
  });

  lastHistLen = hist.length;
} else if (!prevState) {
  lastHistLen = hist.length;
}

  prevState = s;
}

/* =========================
   API 액션들
   ========================= */
async function submitCard(n) {
  setMsg("");
  try {
    if (!stateCache) return;

    const subs = stateCache.submissions || {};
    let pos = null;
    if (subs.hundreds == null) pos = "hundreds";
    else if (subs.tens == null) pos = "tens";

    if (!pos) {
      setMsg("제출할 자리가 없음");
      return;
    }

    if (!mySeat) {
      setMsg("역할이 없음");
      return;
    }
    const s = await apiPost("/api/game/submit", { player: mySeat, pos, card: n });
    showState(s);
  } catch (e) {
    setMsg(`제출 실패: ${e.message}`);
  }
}

async function revealNumber() {
  setMsg("");
  try {
    const s = await apiPost("/api/game/reveal", {});
    showState(s);
  } catch (e) {
    setMsg(`공개 실패: ${e.message}`);
  }
}

async function buzzPress() {
  setMsg("");
  try {
    if (!mySeat) {
      setMsg("역할이 없음");
      return;
    }
    const s = await apiPost("/api/game/buzz_press", { player: mySeat });
    showState(s);
  } catch (e) {
    setMsg(`버저 실패: ${e.message}`);
  }
}

async function answerPrime(isPrime) {
  setMsg("");
  try {
    if (!mySeat) {
      setMsg("역할이 없음");
      return;
    }
    const s = await apiPost("/api/game/answer", { player: mySeat, answer_is_prime: isPrime });
    showState(s);
  } catch (e) {
    setMsg(`답변 실패: ${e.message}`);
  }
}

async function resetGame() {
  setMsg("");
  try {
    const s = await apiPost("/api/game/reset", {});
    showState(s);
  } catch (e) {
    setMsg(`리셋 실패: ${e.message}`);
  }
}

/* =========================
   폴링
   ========================= */
let refreshing = false;
let pollTimer = null;
let tickTimer = null;

async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    const s = await apiGet("/api/game/state");
    showState(s);
  } catch (e) {
    setMsg(`에러: ${e.message}`);
  } finally {
    refreshing = false;
  }
}

function startPolling() {
  stopPolling();
  pollTimer = setInterval(refresh, 400);
  tickTimer = setInterval(() => {
    if (!stateCache) return;
    showState(stateCache);
  }, 150);
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  if (tickTimer) clearInterval(tickTimer);
  pollTimer = null;
  tickTimer = null;
}

/* =========================
   이벤트 바인딩
   ========================= */
const LS_SESSION = "sosu_lobby_session";
const LS_NAME = "sosu_lobby_name";

let enteredGame = false;

function syncLobbyState(resp) {
  lobbyCache = resp;
  const me = resp?.me || null;
  if (me && me.session_id) {
    sessionId = me.session_id;
    mySeat = me.seat;
    myName = me.name || myName;
    joined = true;
    ready = !!me.ready;
    localStorage.setItem(LS_SESSION, sessionId);
    if (myName) localStorage.setItem(LS_NAME, myName);
  } else {
    joined = false;
    ready = false;
    mySeat = null;
  }
  renderLobbyList(resp);
}

async function refreshLobby() {
  try {
    const q = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
    const resp = await apiGet(`/api/lobby/state${q}`);
    syncLobbyState(resp);

    if (resp.game_started && !enteredGame) {
      await enterGame();
    }
  } catch (e) {
    setJoinStatus(`대기실 에러: ${e.message}`, "error");
  }
}

function startLobbyPolling() {
  stopLobbyPolling();
  lobbyPollTimer = setInterval(refreshLobby, 500);
}

function stopLobbyPolling() {
  if (lobbyPollTimer) clearInterval(lobbyPollTimer);
  lobbyPollTimer = null;
}

async function joinGame() {
  const nameInput = el("lobbyName");
  myName = (nameInput?.value || "").trim() || "player";
  const savedSession = localStorage.getItem(LS_SESSION) || null;

  setJoinStatus("", "");
  try {
    const resp = await apiPost("/api/lobby/join", { name: myName, session_id: savedSession });
    syncLobbyState(resp);
    setJoinStatus(`입장 완료 (${mySeat === "p1" ? "player1" : "player2"})`, "success");
    startLobbyPolling();
  } catch (e) {
    joined = false;
    ready = false;
    mySeat = null;
    setJoinStatus(`입장 실패: ${e.message}`, "error");
    renderLobbyList();
  }
}

async function setReady() {
  if (!sessionId) {
    setJoinStatus("입장부터 해줘", "error");
    return;
  }
  setJoinStatus("", "");
  try {
    const resp = await apiPost("/api/lobby/ready", { session_id: sessionId, ready: true });
    syncLobbyState(resp);
    setJoinStatus("준비 완료", "success");
    if (resp.game_started && !enteredGame) {
      await enterGame();
    }
  } catch (e) {
    setJoinStatus(`준비 실패: ${e.message}`, "error");
  }
}

async function enterGame() {
  enteredGame = true;
  stopLobbyPolling();

  prevState = null;
  lastHistLen = 0;
  autoRevealAtMs = null;
  autoRevealFired = false;

  showScreen("game");
  startPolling();
  await refresh();
}

function backToLobby() {
  enteredGame = true;
  stopPolling();
  showScreen("lobby");
  startLobbyPolling();
  refreshLobby();
}

/* 키보드: Space 버저, q/w 답 */
document.addEventListener("keydown", (e) => {
  if (isTypingTarget(e.target)) return;
  if (e.repeat) return;

  const gameScreen = el("screenGame");
  const inGame = gameScreen && gameScreen.classList.contains("active");
  if (!inGame) return;

  if (e.code === "Space") {
    e.preventDefault();
    buzzPress();
    return;
  }
  if (e.key === "q" || e.key === "Q") {
    e.preventDefault();
    answerPrime(true);
    return;
  }
  if (e.key === "w" || e.key === "W") {
    e.preventDefault();
    answerPrime(false);
    return;
  }
});

function wireUI() {
  const btnJoin = el("btnJoin");
  const btnReady = el("btnReady");
  const btnBuzz = el("btnBuzz");
  const btnPrime = el("btnPrime");
  const btnNotPrime = el("btnNotPrime");
  const btnReset = el("btnReset");
  const btnBackLobby = el("btnBackLobby");
  const btnRefresh = el("btnRefresh");

  if (btnJoin) btnJoin.addEventListener("click", joinGame);
  if (btnReady) btnReady.addEventListener("click", setReady);

  if (btnBuzz) btnBuzz.addEventListener("click", buzzPress);
  if (btnPrime) btnPrime.addEventListener("click", () => answerPrime(true));
  if (btnNotPrime) btnNotPrime.addEventListener("click", () => answerPrime(false));
  if (btnReset) btnReset.addEventListener("click", resetGame);

  if (btnBackLobby) btnBackLobby.addEventListener("click", backToLobby);
  if (btnRefresh) btnRefresh.addEventListener("click", refresh);

  renderLobbyList();
}

/* 부팅 */
buildUI();
wireUI();

(async () => {
  const nameInput = el("lobbyName");
  const savedName = localStorage.getItem(LS_NAME) || "";
  if (nameInput && savedName) nameInput.value = savedName;

  const savedSession = localStorage.getItem(LS_SESSION) || "";
  if (savedSession) {
    sessionId = savedSession;
    startLobbyPolling();
    await refreshLobby();
  }
})();
