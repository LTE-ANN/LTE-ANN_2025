(() => {
  // ===== DOM =====
  const board = document.getElementById("board");
  const wires = document.getElementById("wires");
  const nodesLayer = document.getElementById("nodesLayer");

  const btnReset = document.getElementById("btnReset");
  const btnHint = document.getElementById("btnHint");
  const btnCommit = document.getElementById("btnCommit");
  const btnAck = document.getElementById("btnAck"); // HTML에 남아있어도 숨김

  const stageBadge = document.getElementById("stageBadge");
  const clearBadge = document.getElementById("clearBadge");

  const targetEq = document.getElementById("targetEq");
  const netEq = document.getElementById("netEq");

  const xSlider = document.getElementById("xSlider");
  const xVal = document.getElementById("xVal");
  const yVal = document.getElementById("yVal");
  const flowLog = document.getElementById("flowLog");

  const plot = document.getElementById("plot");
  const ctx = plot?.getContext?.("2d");

  const tracePill = document.getElementById("tracePill");
  const nodeCountPill = document.getElementById("nodeCountPill");
  const edgeCountPill = document.getElementById("edgeCountPill");

  const toast = document.getElementById("toast");
  const toast1 = document.getElementById("toast1");
  const toast2 = document.getElementById("toast2");

  const agentMsg = document.getElementById("agentMsg");
  const agentSub = document.getElementById("agentSub");

  const logEl = document.getElementById("log");

  if (!board || !wires || !nodesLayer || !btnReset || !btnHint || !btnCommit) {
    console.error("필수 DOM id mismatch");
    return;
  }
  if (btnAck) btnAck.style.display = "none"; // ACK 제거

  // ===== utils =====
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const fmt = (n, d = 3) => (Number.isFinite(n) ? n.toFixed(d) : "NaN");

  let trace = 17;
  let lastToastAt = 0;

  function bumpTrace(delta) {
    trace = clamp(trace + delta, 0, 99);
    if (tracePill) tracePill.textContent = `TRACE: ${trace}%`;
  }

  function showToast(kind, line1, line2) {
    if (!toast) return;
    const now = Date.now();
    if (now - lastToastAt < 180) return;
    lastToastAt = now;

    toast.className = `toast show ${kind || ""}`;
    if (toast1) toast1.textContent = line1 || "";
    if (toast2) toast2.textContent = line2 || "";

    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => toast.classList.remove("show"), 1500);
  }

  function log(msg) {
    if (!logEl) return;
    logEl.textContent = msg + (logEl.textContent ? "\n" + logEl.textContent : "");
  }

  // ===== activations (가중치 전부 1) =====
  const Act = {
    Identity: (x) => x,
    ReLU: (x) => Math.max(0, x),
    Sigmoid: (x) => 1 / (1 + Math.exp(-x)),
    Neg: (x) => -x,
  };

  // ===== stages (연속 layer만 연결 가능하도록 레이어 설계) =====
  // 공통: kind = input | bias | hidden | output
  const STAGES = [
    {
      name: "TUTORIAL / LINK",
      goalEq: "y = x",
      operator: "좋아. 가장 기초부터.\nX(out) → Y(in) 연결.\n연결 = 경로다.",
      hint: "연결한 선을 타고 값이 이동한다.\nY는 들어온 값들을 '합'한 다음 자기 fn을 적용한다.",
      xRange: [-3, 3],
      yRange: [-3.2, 3.2],
      tolerance: 0.02,
      sampleXs: [-3, -2, -1, 0, 1, 2, 3],
      target: (x) => x,
      nodes: [
        { id: "x", name: "X", tag: "IN", fn: "Identity", layer: 0, y: 260, kind: "input" },
        { id: "y", name: "Y", tag: "OUT", fn: "Identity", layer: 1, y: 260, kind: "output" },
      ],
      danger: [],
    },

    {
      name: "TUTORIAL / GATE",
      goalEq: "y = ReLU(x)",
      operator: "다음은 게이트.\nReLU는 음수 흐름을 잘라낸다.\nX → H → Y로 만들어.",
      hint: "ReLU(x)=max(0,x)\n슬라이더를 -2, +2로 움직여서 확인해.",
      xRange: [-3, 3],
      yRange: [-0.5, 3.4],
      tolerance: 0.03,
      sampleXs: [-3, -2, -1, 0, 1, 2, 3],
      target: (x) => Math.max(0, x),
      nodes: [
        { id: "x", name: "X", tag: "IN", fn: "Identity", layer: 0, y: 260, kind: "input" },
        { id: "h", name: "H", tag: "L1", fn: "ReLU", layer: 1, y: 260, kind: "hidden" },
        { id: "y", name: "Y", tag: "OUT", fn: "Identity", layer: 2, y: 260, kind: "output" },
      ],
      danger: [],
    },

    {
      name: "TUTORIAL / MASK",
      goalEq: "y = Sigmoid(x)",
      operator: "마스킹 구역.\nSigmoid는 0~1로 압축한다.\nX → Y로 직결해도 됨.",
      hint: "Sigmoid(x)=1/(1+e^-x)\n큰 양수→1, 큰 음수→0, x=0→0.5",
      xRange: [-6, 6],
      yRange: [-0.08, 1.08],
      tolerance: 0.02,
      sampleXs: [-6, -3, -1, 0, 1, 3, 6],
      target: (x) => 1 / (1 + Math.exp(-x)),
      nodes: [
        { id: "x", name: "X", tag: "IN", fn: "Identity", layer: 0, y: 260, kind: "input" },
        { id: "y", name: "Y", tag: "OUT", fn: "Sigmoid", layer: 1, y: 260, kind: "output" },
      ],
      danger: [],
    },

    {
      name: "TUTORIAL / MERGE",
      goalEq: "y = ReLU(x) + Sigmoid(x)",
      operator:
        "이제 '덧셈'.\n같은 노드로 들어오는 신호는 서로 싸우는 게 아니라 더해진다.\nX→R, X→S 만들고 둘 다 SUM으로 넣어.",
      hint:
        "한 노드로 들어오는 모든 입력은 합산된다.\n즉, 여러 경로를 동시에 연결하면 출력이 합쳐진다.",
      xRange: [-3, 3],
      yRange: [-0.4, 4.2],
      tolerance: 0.05,
      sampleXs: [-3, -2, -1, 0, 1, 2, 3],
      target: (x) => Math.max(0, x) + 1 / (1 + Math.exp(-x)),
      nodes: [
        { id: "x", name: "X", tag: "IN", fn: "Identity", layer: 0, y: 260, kind: "input" },

        { id: "r", name: "R", tag: "L1", fn: "ReLU", layer: 1, y: 170, kind: "hidden" },
        { id: "s", name: "S", tag: "L1", fn: "Sigmoid", layer: 1, y: 350, kind: "hidden" },

        { id: "sum", name: "SUM", tag: "L2", fn: "Identity", layer: 2, y: 260, kind: "hidden" },
        { id: "y", name: "Y", tag: "OUT", fn: "Identity", layer: 3, y: 260, kind: "output" },
      ],
      danger: [],
    },

    // ===== 실전(난이도 완만) =====

    {
      name: "FIELD / ABS",
      goalEq: "y = |x| = ReLU(x) + ReLU(-x)",
      operator:
        "실전.\n절대값은 두 경로 합으로 만든다.\nNEG로 부호를 뒤집고 ReLU를 한 번 더 태워.",
      hint:
        "|x| = ReLU(x) + ReLU(-x)\nNEG는 입력 부호만 뒤집는다.\n(연결은 항상 다음 layer로만!)",
      xRange: [-3, 3],
      yRange: [-0.4, 3.4],
      tolerance: 0.06,
      sampleXs: [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3],
      target: (x) => Math.abs(x),
      nodes: [
        { id: "x", name: "X", tag: "IN", fn: "Identity", layer: 0, y: 260, kind: "input" },

        { id: "rp", name: "P", tag: "L1", fn: "ReLU", layer: 1, y: 170, kind: "hidden" },
        { id: "negx", name: "NEG", tag: "L1", fn: "Neg", layer: 1, y: 360, kind: "hidden" },

        { id: "pp", name: "P'", tag: "L2", fn: "Identity", layer: 2, y: 170, kind: "hidden" },
        { id: "rn", name: "N", tag: "L2", fn: "ReLU", layer: 2, y: 360, kind: "hidden" },

        { id: "sum", name: "SUM", tag: "L3", fn: "Identity", layer: 3, y: 260, kind: "hidden" },
        { id: "y", name: "Y", tag: "OUT", fn: "Identity", layer: 4, y: 260, kind: "output" },

        // decoy (쓰면 오차 올라가게 유도)
        { id: "d1", name: "D1", tag: "DECOY", fn: "Sigmoid", layer: 2, y: 520, kind: "hidden" },
      ],
      danger: ["d1"],
    },

    {
      name: "FIELD / WINDOW (−1~1)",
      goalEq: "y = σ( ReLU(x+1) − ReLU(x−1) )",
      operator:
        "창문 만들기.\nB=1을 더하면 x+1, -B를 더하면 x-1.\n두 ReLU를 빼고 Sigmoid로 살려.",
      hint:
        "x+1: (x relay) + (B relay)\nx-1: (x relay) + (-B)\n빼기는 Neg를 이용해서 더하기로 바꿔라.",
      xRange: [-3, 3],
      yRange: [-0.1, 1.1],
      tolerance: 0.07,
      sampleXs: [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3],
      target: (x) => {
        const relu = (v) => Math.max(0, v);
        const sig = (v) => 1 / (1 + Math.exp(-v));
        return sig(relu(x + 1) - relu(x - 1));
      },
      nodes: [
        { id: "x", name: "X", tag: "IN", fn: "Identity", layer: 0, y: 220, kind: "input" },
        { id: "b", name: "B", tag: "BIAS", fn: "Identity", layer: 0, y: 420, kind: "bias", bias: 1 },

        { id: "x1", name: "X'", tag: "L1", fn: "Identity", layer: 1, y: 220, kind: "hidden" },
        { id: "b1", name: "B'", tag: "L1", fn: "Identity", layer: 1, y: 420, kind: "hidden" },
        { id: "nb", name: "-B", tag: "L1", fn: "Neg", layer: 1, y: 520, kind: "hidden" },

        { id: "xpb", name: "X+1", tag: "L2", fn: "Identity", layer: 2, y: 140, kind: "hidden" },
        { id: "xmb", name: "X-1", tag: "L2", fn: "Identity", layer: 2, y: 320, kind: "hidden" },

        { id: "r1", name: "R1", tag: "L3", fn: "ReLU", layer: 3, y: 140, kind: "hidden" },
        { id: "r2", name: "R2", tag: "L3", fn: "ReLU", layer: 3, y: 320, kind: "hidden" },

        { id: "nr2", name: "-R2", tag: "L4", fn: "Neg", layer: 4, y: 320, kind: "hidden" },
        { id: "diff", name: "Δ", tag: "L5", fn: "Identity", layer: 5, y: 220, kind: "hidden" },

        { id: "y", name: "Y", tag: "OUT", fn: "Sigmoid", layer: 6, y: 220, kind: "output" },

        { id: "d1", name: "D1", tag: "DECOY", fn: "Sigmoid", layer: 3, y: 560, kind: "hidden" },
      ],
      danger: ["d1"],
    },

    {
      name: "FINAL / CLAMP (|x|-1)",
      goalEq: "y = σ( |x| − 1 )",
      operator:
        "마지막.\n절대값 만든 다음 1을 빼서 문턱을 걸어.\n|x| - 1을 Sigmoid로 눌러서 클램프해.",
      hint:
        "먼저 ABS 회로를 재구성.\n그 다음 -B(=-1)를 더해서 |x|-1 만들고 Sigmoid.",
      xRange: [-3, 3],
      yRange: [-0.1, 1.1],
      tolerance: 0.07,
      sampleXs: [-3, -2.4, -2, -1.6, -1.2, -1, -0.6, 0, 0.6, 1, 1.2, 1.6, 2, 2.4, 3],
      target: (x) => {
        const sig = (v) => 1 / (1 + Math.exp(-v));
        return sig(Math.abs(x) - 1);
      },
      nodes: [
        { id: "x", name: "X", tag: "IN", fn: "Identity", layer: 0, y: 220, kind: "input" },
        { id: "b", name: "B", tag: "BIAS", fn: "Identity", layer: 0, y: 520, kind: "bias", bias: 1 },

        { id: "rp", name: "P", tag: "L1", fn: "ReLU", layer: 1, y: 140, kind: "hidden" },
        { id: "negx", name: "NEG", tag: "L1", fn: "Neg", layer: 1, y: 320, kind: "hidden" },
        { id: "nb", name: "-B", tag: "L1", fn: "Neg", layer: 1, y: 520, kind: "hidden" },

        { id: "pp", name: "P'", tag: "L2", fn: "Identity", layer: 2, y: 140, kind: "hidden" },
        { id: "rn", name: "N", tag: "L2", fn: "ReLU", layer: 2, y: 320, kind: "hidden" },
        { id: "nb2", name: "-1", tag: "L2", fn: "Identity", layer: 2, y: 520, kind: "hidden" },

        { id: "abs", name: "|x|", tag: "L3", fn: "Identity", layer: 3, y: 220, kind: "hidden" },
        { id: "absM1", name: "|x|-1", tag: "L4", fn: "Identity", layer: 4, y: 220, kind: "hidden" },
        { id: "y", name: "Y", tag: "OUT", fn: "Sigmoid", layer: 5, y: 220, kind: "output" },

        { id: "d1", name: "D1", tag: "DECOY", fn: "Sigmoid", layer: 2, y: 600, kind: "hidden" },
      ],
      danger: ["d1"],
    },
  ];

  // ===== state =====
  const layerX = {};
  let stageIndex = 0;
  const state = { nodes: [], edges: [] };

  function getStage() {
    return STAGES[stageIndex];
  }
  function getNode(id) {
    return state.nodes.find((n) => n.id === id);
  }

  // ===== layout =====
  function computeLayerX() {
    const w = board.clientWidth;

    // ✅ CSS 변수(--nodeW)에서 실제 노드 폭 읽기 (px 문자열 -> 숫자)
    const cssNodeW = parseFloat(
      getComputedStyle(document.documentElement).getPropertyValue("--nodeW")
    );
    const nodeW = Number.isFinite(cssNodeW) ? cssNodeW : 148;

    const margin = 20;

    const maxLayer = Math.max(...state.nodes.map((n) => n.layer));
    const left = margin;
    const right = w - nodeW - margin;

    // ✅ 0.10~0.90 같은 비율 꼼수 말고, left~right를 “끝까지” 씀
    for (let L = 0; L <= maxLayer; L++) {
      const t = maxLayer === 0 ? 0 : L / maxLayer;
      const x = Math.floor(left + (right - left) * t);
      layerX[L] = clamp(x, left, right);
    }
  }


  // ✅ 버그 수정: 연속된 은닉층(+1)로만 연결
  function canConnect(fromN, toN) {
    if (!fromN || !toN) return false;
    if (toN.id === fromN.id) return false;
    return toN.layer === fromN.layer + 1;
  }

  // ===== forward (weight=1) =====
  function forward(x) {
    const out = {};
    const maxLayer = Math.max(...state.nodes.map((n) => n.layer));

    // layer 0 init
    for (const n of state.nodes) {
      if (n.kind === "input") out[n.id] = x;
      if (n.kind === "bias") out[n.id] = typeof n.bias === "number" ? n.bias : 1;
    }

    for (let L = 1; L <= maxLayer; L++) {
      for (const n of state.nodes.filter((nn) => nn.layer === L)) {
        const ins = state.edges.filter((e) => e.to === n.id);
        let s = 0;
        for (const e of ins) s += (out[e.from] ?? 0); // weight=1
        const fn = Act[n.fn] || Act.Identity;
        out[n.id] = fn(s);
      }
    }
    return out;
  }

  function outputAt(x) {
    const out = forward(x);
    const yNode =
      state.nodes.find((n) => n.kind === "output") ||
      state.nodes.slice().sort((a, b) => b.layer - a.layer)[0];
    return yNode ? (out[yNode.id] ?? 0) : 0;
  }

  // ===== expression builder =====
  function buildExpr() {
    const yNode =
      state.nodes.find((n) => n.kind === "output") ||
      state.nodes.slice().sort((a, b) => b.layer - a.layer)[0];
    if (!yNode) return "y = ?";

    const seen = new Set();
    function rec(id) {
      const key = String(id);
      if (seen.has(key)) return "…"; // cycle safety
      const n = getNode(id);
      if (!n) return "0";
      if (n.kind === "input") return "x";
      if (n.kind === "bias") return "1";

      const ins = state.edges.filter((e) => e.to === id);
      if (ins.length === 0) return n.fn === "Identity" ? "0" : `${n.fn}(0)`;

      seen.add(key);
      const terms = ins.map((e) => rec(e.from));
      seen.delete(key);

      const sum = terms.length > 1 ? `(${terms.join(" + ")})` : terms[0];
      if (n.fn === "Identity") return sum;
      return `${n.fn}(${sum})`;
    }

    return `y = ${rec(yNode.id)}`;
  }

  function flowText(x) {
    const out = forward(x);
    const lines = [];
    const maxLayer = Math.max(...state.nodes.map((n) => n.layer));
    for (let L = 0; L <= maxLayer; L++) {
      for (const n of state.nodes.filter((nn) => nn.layer === L)) {
        const v = out[n.id];
        if (v === undefined) continue;
        if (n.kind === "input") lines.push(`[L${L}] ${n.name} <= x = ${fmt(v, 3)}`);
        else if (n.kind === "bias") lines.push(`[L${L}] ${n.name} <= 1`);
        else lines.push(`[L${L}] ${n.name} <= ${fmt(v, 3)}`);
      }
    }
    return lines.join("\n");
  }

  // ===== plotting =====
  function resizeCanvas() {
    if (!plot || !ctx) return;
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const w = plot.clientWidth;
    const h = plot.clientHeight;
    plot.width = Math.floor(w * dpr);
    plot.height = Math.floor(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function plotFunctions() {
    if (!plot || !ctx) return;
    const st = getStage();
    const [xmin, xmax] = st.xRange;
    const [ymin, ymax] = st.yRange;
    const w = plot.clientWidth,
      h = plot.clientHeight;
    const N = 240;

    const mapX = (x) => ((x - xmin) / (xmax - xmin)) * w;
    const mapY = (y) => h - ((y - ymin) / (ymax - ymin)) * h;

    ctx.clearRect(0, 0, w, h);

    // grid
    ctx.strokeStyle = "rgba(183,255,205,.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 1; i < 6; i++) {
      ctx.moveTo((w * i) / 6, 0);
      ctx.lineTo((w * i) / 6, h);
      ctx.moveTo(0, (h * i) / 6);
      ctx.lineTo(w, (h * i) / 6);
    }
    ctx.stroke();

    // axes
    ctx.strokeStyle = "rgba(183,255,205,.14)";
    ctx.beginPath();
    ctx.moveTo(0, mapY(0));
    ctx.lineTo(w, mapY(0));
    ctx.moveTo(mapX(0), 0);
    ctx.lineTo(mapX(0), h);
    ctx.stroke();

    // target dashed
    ctx.save();
    ctx.setLineDash([6, 6]);
    ctx.strokeStyle = "rgba(183,255,205,.45)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= N; i++) {
      const x = xmin + ((xmax - xmin) * i) / N;
      const t = st.target(x);
      const px = mapX(x),
        py = mapY(t);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.restore();

    // current
    ctx.strokeStyle = "rgba(80,255,180,.95)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i <= N; i++) {
      const x = xmin + ((xmax - xmin) * i) / N;
      const y = outputAt(x);
      const px = mapX(x),
        py = mapY(y);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // cursor
    const x = parseFloat(xSlider.value);
    const y = outputAt(x);
    ctx.fillStyle = "rgba(255,211,106,.95)";
    ctx.beginPath();
    ctx.arc(mapX(x), mapY(y), 3.6, 0, Math.PI * 2);
    ctx.fill();
  }

  // ===== scoring =====
  function checkClear() {
    const st = getStage();
    let mse = 0;
    for (const x of st.sampleXs) {
      const t = st.target(x);
      const y = outputAt(x);
      mse += (y - t) * (y - t);
    }
    mse /= st.sampleXs.length;
    return { ok: mse <= st.tolerance, err: mse };
  }

  // ===== render nodes/edges =====
  function clampAllNodes() {
    const brH = board.clientHeight;
    const brW = board.clientWidth;
    const pad = 12;

    for (const n of state.nodes) {
      const el = nodesLayer.querySelector(`.node[data-id="${n.id}"]`);
      if (!el) continue;
      const h = el.offsetHeight;
      const w = el.offsetWidth;

      n.y = clamp(n.y, pad, brH - h - pad);
      el.style.top = n.y + "px";

      const x = clamp(layerX[n.layer], pad, brW - w - pad);
      el.style.left = x + "px";
    }
  }

  function getPortCenter(nodeId, portType) {
    const nodeEl = nodesLayer.querySelector(`.node[data-id="${nodeId}"]`);
    const portEl = nodeEl?.querySelector(`.port.${portType}`);
    if (!nodeEl || !portEl) return { x: 0, y: 0 };
    const pr = portEl.getBoundingClientRect();
    const br = board.getBoundingClientRect();
    return {
      x: (pr.left + pr.right) / 2 - br.left,
      y: (pr.top + pr.bottom) / 2 - br.top,
    };
  }

  function pathD(a, b) {
    const dx = Math.max(70, (b.x - a.x) * 0.5);
    const c1x = a.x + dx,
      c1y = a.y;
    const c2x = b.x - dx,
      c2y = b.y;
    return `M ${a.x} ${a.y} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${b.x} ${b.y}`;
  }

  function renderEdges() {
    const bw = board.clientWidth,
      bh = board.clientHeight;
    wires.setAttribute("viewBox", `0 0 ${bw} ${bh}`);
    wires.innerHTML = "";

    for (const e of state.edges) {
      const a = getPortCenter(e.from, "out");
      const b = getPortCenter(e.to, "in");
      const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
      p.setAttribute("d", pathD(a, b));
      p.setAttribute("class", "edge");
      p.addEventListener("click", (ev) => {
        ev.stopPropagation();
        state.edges = state.edges.filter((x) => !(x.from === e.from && x.to === e.to));
        log(`CUT: ${e.from} → ${e.to}`);
        bumpTrace(1);
        renderEdges();
        updateUI();
      });
      wires.appendChild(p);
    }

    if (edgeCountPill) edgeCountPill.textContent = `WIRES: ${state.edges.length}`;
  }

  function enableVerticalDrag(el, node) {
    let down = false,
      dragging = false,
      startY = 0,
      oy = 0;

    el.addEventListener("mousedown", (e) => {
      if (e.target.closest(".port")) return;
      down = true;
      dragging = false;
      startY = e.clientY;
      const r = el.getBoundingClientRect();
      oy = e.clientY - r.top;
      e.preventDefault();
    });

    const onMove = (e) => {
      if (!down) return;
      const dy = e.clientY - startY;
      if (!dragging) {
        if (Math.abs(dy) < 3) return;
        dragging = true;
        el.style.cursor = "grabbing";
      }
      const br = board.getBoundingClientRect();
      const h = el.offsetHeight;
      let ny = e.clientY - br.top - oy;
      ny = clamp(ny, 12, board.clientHeight - h - 12);
      node.y = ny;
      el.style.top = ny + "px";
      renderEdges();
    };

    const onUp = () => {
      if (!down) return;
      down = false;
      el.style.cursor = "grab";
      clampAllNodes();
      renderEdges();
      updateUI();
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    el.style.cursor = "grab";
  }

  let linking = null;

  function enablePortConnect(nodeEl, node) {
    const outPort = nodeEl.querySelector(".port.out");
    outPort.addEventListener("mousedown", (e) => {
      e.preventDefault();
      e.stopPropagation();

      const temp = document.createElementNS("http://www.w3.org/2000/svg", "path");
      temp.setAttribute("class", "edge temp");
      wires.appendChild(temp);

      const bad = document.createElementNS("http://www.w3.org/2000/svg", "path");
      bad.setAttribute("class", "edge bad");
      wires.appendChild(bad);

      linking = { fromId: node.id, tempPath: temp, badPath: bad, hoverEl: null };

      const onMove = (ev) => {
        const br = board.getBoundingClientRect();
        const a = getPortCenter(linking.fromId, "out");
        const b = { x: ev.clientX - br.left, y: ev.clientY - br.top };
        const d = pathD(a, b);
        linking.tempPath.setAttribute("d", d);
        linking.badPath.setAttribute("d", d);

        const under = document.elementFromPoint(ev.clientX, ev.clientY);
        const inPort = under ? under.closest(".port.in") : null;

        if (linking.hoverEl && linking.hoverEl !== inPort) {
          linking.hoverEl.classList.remove("hover");
          linking.hoverEl = null;
        }
        if (inPort && inPort !== linking.hoverEl) {
          linking.hoverEl = inPort;
          linking.hoverEl.classList.add("hover");
        }

        let ok = false;
        if (inPort) {
          const toId = inPort.closest(".node")?.dataset?.id;
          const fromN = getNode(linking.fromId);
          const toN = toId ? getNode(toId) : null;
          ok = !!(fromN && toN && canConnect(fromN, toN));
        }
        linking.tempPath.style.display = ok ? "block" : "none";
        linking.badPath.style.display = ok ? "none" : "block";
      };

      const onUp = (ev) => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);

        if (linking?.hoverEl) linking.hoverEl.classList.remove("hover");
        linking?.tempPath?.remove();
        linking?.badPath?.remove();

        const under = document.elementFromPoint(ev.clientX, ev.clientY);
        const inPort = under ? under.closest(".port.in") : null;
        if (!inPort) {
          linking = null;
          bumpTrace(1);
          return;
        }

        const toId = inPort.closest(".node")?.dataset?.id;
        const fromN = getNode(linking.fromId);
        const toN = toId ? getNode(toId) : null;

        if (!fromN || !toN) {
          linking = null;
          bumpTrace(1);
          return;
        }

        if (!canConnect(fromN, toN)) {
          log(`BLOCKED: L${fromN.layer} → L${toN.layer}`);
          showToast("bad", "FIREWALL REFUSED", "다음 layer(+1)로만 연결 가능");
          bumpTrace(2);
          linking = null;
          return;
        }

        const idx = state.edges.findIndex((ed) => ed.from === fromN.id && ed.to === toN.id);
        if (idx !== -1) {
          state.edges.splice(idx, 1);
          log(`TOGGLE OFF: ${fromN.id} → ${toN.id}`);
          showToast("cyan", "LINK REVOKED", "같은 연결 재시도 = 토글 해제");
          bumpTrace(1);
        } else {
          state.edges.push({ from: fromN.id, to: toN.id });
          log(`LINK: ${fromN.id} → ${toN.id}`);
          showToast("cyan", "LINK ESTABLISHED", `${fromN.name} → ${toN.name}`);
          bumpTrace(2);
        }

        renderEdges();
        updateUI();
        linking = null;
      };

      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });
  }

  function render() {
    nodesLayer.innerHTML = "";
    wires.innerHTML = "";

    computeLayerX();

    const st = getStage();
    const dangerSet = new Set(st.danger || []);

    for (const n of state.nodes) {
      const el = document.createElement("div");
      el.className = "node" + (dangerSet.has(n.id) ? " danger" : "");
      el.dataset.id = n.id;
      el.style.top = n.y + "px";
      el.style.left = layerX[n.layer] + "px";

      const hint =
        n.kind === "bias"
          ? "B=1 (문턱/오프셋)"
          : n.fn === "Neg"
          ? "부호 반전"
          : dangerSet.has(n.id)
          ? ""
          : "";

      el.innerHTML = `
        <div class="port in"></div>
        <div class="port out"></div>
        <div class="head">
          <div class="name">${n.name}</div>
          <div class="tag">${n.tag}</div>
        </div>
        <div class="fn"><span>fn</span><b>${n.fn}</b></div>
        <div class="hint">${hint}</div>
      `;

      nodesLayer.appendChild(el);
      enableVerticalDrag(el, n);
      enablePortConnect(el, n);
    }

    clampAllNodes();
    renderEdges();
    updateUI(true);
  }

  // ===== UI =====
  function updateUI(skipToast = false) {
    const st = getStage();

    if (stageBadge) stageBadge.textContent = `${st.name} (${stageIndex + 1}/${STAGES.length})`;
    if (targetEq) targetEq.textContent = st.goalEq;

    if (nodeCountPill) nodeCountPill.textContent = `NODES: ${state.nodes.length}`;

    const x = parseFloat(xSlider.value);
    if (xVal) xVal.textContent = fmt(x, 2);

    if (netEq) netEq.textContent = buildExpr();

    const y = outputAt(x);
    if (yVal) yVal.textContent = fmt(y, 3);

    if (flowLog) flowLog.textContent = flowText(x);

    const { ok, err } = checkClear();
    if (clearBadge) {
      clearBadge.textContent = ok
        ? `✓ OVERRIDE READY (mse=${fmt(err, 4)})`
        : `mse=${fmt(err, 4)} (<= ${st.tolerance})`;
    }
    btnCommit.disabled = !ok;

    resizeCanvas();
    plotFunctions();

    if (!skipToast) {
      // no-op
    }
  }

  function setOperator(text, sub = "") {
    if (agentMsg) agentMsg.textContent = text || "";
    if (agentSub) agentSub.textContent = sub || "";
  }

  function loadStage(i) {
    stageIndex = clamp(i, 0, STAGES.length - 1);
    const st = getStage();

    state.nodes = st.nodes.map((n) => ({ ...n }));
    state.edges = [];

    xSlider.min = st.xRange[0];
    xSlider.max = st.xRange[1];
    xSlider.value = 0;

    log(`\n=== ${st.name} ===`);
    showToast("cyan", "SESSION SYNCED", "그래프 스냅샷 고정");
    setOperator(st.operator, "HINT 버튼으로 개념 확인");
    render();
  }

  // ===== events =====
  xSlider.addEventListener("input", () => updateUI(true));

  btnHint.addEventListener("click", () => {
    const st = getStage();
    showToast("cyan", "CONTROL / HINT", st.hint || st.goalEq);
    bumpTrace(1);
  });

  btnReset.addEventListener("click", () => {
    state.edges = [];
    bumpTrace(2);
    showToast("cyan", "PURGE COMPLETE", "연결 흔적 삭제");
    log("PURGE: edges cleared");
    renderEdges();
    updateUI(true);
  });

  btnCommit.addEventListener("click", () => {
    const { ok, err } = checkClear();
    if (!ok) {
      showToast("bad", "COMMIT REJECTED", "목표 오버라이드 실패");
      bumpTrace(3);
      return;
    }

    const st = getStage();
    showToast("ok", "OVERRIDE SUCCESS", `mse=${fmt(err, 4)}`);
    bumpTrace(-6);

    // OPERATOR 칭찬
    const praise = [
      "좋아. 흔적 깔끔하다.",
      "정확해. 다음 구역으로 이동.",
      "오차 허용치 안쪽. 완벽.",
      "이 정도면 너 손에 익었네.",
      "끝까지 간다. 집중.",
      "딱 맞췄다. 계속.",
      "임무 완료. 돌아와.",
    ];
    const p = praise[Math.min(stageIndex, praise.length - 1)];

    setOperator(
      `${p}\n[${st.name}] 클리어.`,
      stageIndex < STAGES.length - 1 ? "다음 스테이지로 전환 중…" : "전체 시퀀스 종료"
    );

    log(`COMMIT OK (mse=${fmt(err, 4)})`);

    if (stageIndex < STAGES.length - 1) {
      setTimeout(() => loadStage(stageIndex + 1), 650);
    } else {
      showToast("ok", "MISSION COMPLETE", "타깃 출력 덮어쓰기 완료.");
      setOperator("임무 완료.\n출력 경로 완전 장악.", "이제 로그를 정리하고 빠져.");
      btnCommit.disabled = true;
    }
  });

  window.addEventListener("resize", () => render());

  // ===== start =====
  loadStage(0);
})();