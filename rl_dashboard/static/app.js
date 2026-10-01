// RL Visual Training Lab — frontend. No external libraries: charts, heatmaps,
// and the loss-surface contour are all drawn on <canvas> by hand so the page
// works fully offline.

// ---------------------------------------------------------------------------
// Tiny canvas chart helpers
// ---------------------------------------------------------------------------

function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(rect.width * dpr));
  const h = Math.max(1, Math.round((rect.height || 220) * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: rect.width, height: rect.height || 220 };
}

const COLORS = {
  grid: "#262f45",
  text: "#8b94ac",
  accent: "#6ee7c2",
  accent2: "#7c9bff",
  warn: "#ffb454",
  bad: "#ff6b6b",
};

function niceRange(min, max) {
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const pad = (max - min) * 0.08;
  return [min - pad, max + pad];
}

function drawLineChart(canvas, series, opts = {}) {
  const { ctx, width, height } = fitCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  const padL = 46, padR = 12, padT = 14, padB = 24;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const allPoints = series.flatMap((s) => s.points);
  if (allPoints.length === 0) {
    ctx.fillStyle = COLORS.text;
    ctx.font = "12px sans-serif";
    ctx.fillText(opts.emptyText || "waiting for data…", padL, height / 2);
    return;
  }

  let xs = allPoints.map((p) => p.x);
  let ys = allPoints.map((p) => p.y);
  let [xmin, xmax] = [Math.min(...xs), Math.max(...xs)];
  let [ymin, ymax] = niceRange(Math.min(...ys), Math.max(...ys));
  if (opts.yZero) ymin = Math.min(ymin, 0);

  const xTo = (x) => padL + ((x - xmin) / (xmax - xmin || 1)) * plotW;
  const yTo = (y) => padT + plotH - ((y - ymin) / (ymax - ymin || 1)) * plotH;

  // gridlines + y labels
  ctx.strokeStyle = COLORS.grid;
  ctx.fillStyle = COLORS.text;
  ctx.font = "10.5px monospace";
  ctx.lineWidth = 1;
  const rows = 4;
  for (let i = 0; i <= rows; i++) {
    const y = padT + (plotH * i) / rows;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(width - padR, y);
    ctx.stroke();
    const val = ymax - ((ymax - ymin) * i) / rows;
    ctx.fillText(val.toFixed(2), 4, y + 3);
  }

  series.forEach((s) => {
    if (s.points.length === 0) return;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    s.points.forEach((p, i) => {
      const x = xTo(p.x), y = yTo(p.y);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    if (s.showDot) {
      const last = s.points[s.points.length - 1];
      ctx.fillStyle = s.color;
      ctx.beginPath();
      ctx.arc(xTo(last.x), yTo(last.y), 3, 0, Math.PI * 2);
      ctx.fill();
    }
  });

  if (opts.legend) {
    let lx = padL;
    ctx.font = "11px sans-serif";
    series.forEach((s) => {
      ctx.fillStyle = s.color;
      ctx.fillRect(lx, 2, 8, 8);
      ctx.fillStyle = COLORS.text;
      ctx.fillText(s.label, lx + 12, 10);
      lx += ctx.measureText(s.label).width + 30;
    });
  }
}

function colorForValue(v, vmin, vmax) {
  // diverging blue -> slate -> warm, centered at 0 for weights
  const m = Math.max(Math.abs(vmin), Math.abs(vmax)) || 1;
  const t = Math.max(-1, Math.min(1, v / m));
  if (t >= 0) {
    const r = Math.round(30 + t * (255 - 30));
    const g = Math.round(40 + t * (180 - 40));
    const b = Math.round(60 + t * (120 - 60));
    return `rgb(${r},${g},${b})`;
  } else {
    const k = -t;
    const r = Math.round(30 + k * (60 - 30));
    const g = Math.round(40 + k * (140 - 40));
    const b = Math.round(60 + k * (255 - 60));
    return `rgb(${r},${g},${b})`;
  }
}

function drawHeatmap(canvas, matrix) {
  const { ctx, width, height } = fitCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  if (!matrix || matrix.length === 0) {
    ctx.fillStyle = COLORS.text;
    ctx.font = "12px sans-serif";
    ctx.fillText("waiting for a weight snapshot…", 10, height / 2);
    return;
  }
  const rows = matrix.length, cols = matrix[0].length;
  const flat = matrix.flat();
  const vmin = Math.min(...flat), vmax = Math.max(...flat);
  const cw = width / cols, ch = height / rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      ctx.fillStyle = colorForValue(matrix[r][c], vmin, vmax);
      ctx.fillRect(c * cw, r * ch, cw + 0.5, ch + 0.5);
    }
  }
}

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("view-" + btn.dataset.view).classList.add("active");
    if (btn.dataset.view === "report") fetchAndRenderReport();
  });
});

// ---------------------------------------------------------------------------
// Env / algo pickers
// ---------------------------------------------------------------------------
const envSelect = document.getElementById("env-select");
const algoSelect = document.getElementById("algo-select");
const envInfo = document.getElementById("env-info");
const envWarn = document.getElementById("env-warn");

async function loadEnvs() {
  const envs = await fetch("/api/envs").then((r) => r.json());
  const groups = {};
  envs.forEach((e) => {
    const key = e.featured ? "★ Featured (fast, renders well)" : e.family;
    (groups[key] = groups[key] || []).push(e);
  });
  envSelect.innerHTML = "";
  Object.entries(groups).forEach(([groupName, items]) => {
    const og = document.createElement("optgroup");
    og.label = groupName;
    items.forEach((e) => {
      const opt = document.createElement("option");
      opt.value = e.id;
      opt.textContent = e.requires ? `${e.id}  (needs ${e.requires})` : e.id;
      og.appendChild(opt);
    });
    envSelect.appendChild(og);
  });
  await loadAlgos();
}

async function loadAlgos() {
  const envId = envSelect.value;
  envWarn.innerHTML = "";
  envInfo.textContent = "";
  try {
    const info = await fetch(`/api/algos?env_id=${encodeURIComponent(envId)}`).then((r) => {
      if (!r.ok) return r.json().then((e) => Promise.reject(e));
      return r.json();
    });
    algoSelect.innerHTML = "";
    info.algos.forEach((a) => {
      const opt = document.createElement("option");
      opt.value = a;
      opt.textContent = a;
      algoSelect.appendChild(opt);
    });
    envInfo.textContent = `observation space: ${info.observation_space}   action space: ${info.action_space}`;
    if (info.algos.length === 0) {
      envWarn.innerHTML = `<div class="warn-box">No supported algorithm for this action space yet.</div>`;
    }
  } catch (err) {
    algoSelect.innerHTML = "";
    envWarn.innerHTML = `<div class="warn-box">Couldn't load this env: ${err.detail || err}</div>`;
  }
}

envSelect.addEventListener("change", loadAlgos);

// ---------------------------------------------------------------------------
// Training state + websocket
// ---------------------------------------------------------------------------
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const statusPill = document.getElementById("status-pill");

const trainState = {
  reward: [],
  grad: [],
  loss: { ploss: [], vloss: [], entropy: [], kl: [] },
  episodeCount: 0,
};

function resetTrainState() {
  trainState.reward = [];
  trainState.grad = [];
  trainState.loss = { ploss: [], vloss: [], entropy: [], kl: [] };
  trainState.episodeCount = 0;
  redrawAll();
}

function setStatus(state, text) {
  statusPill.className = "pill state-" + state;
  statusPill.innerHTML = `<span class="dot"></span> ${text}`;
}

function redrawAll() {
  drawLineChart(
    document.getElementById("chart-reward"),
    [
      { label: "reward", color: COLORS.accent, points: trainState.reward.map((v, i) => ({ x: i, y: v })), showDot: true },
    ],
    { yZero: false }
  );
  drawLineChart(
    document.getElementById("chart-grad"),
    [{ label: "‖grad‖", color: COLORS.warn, points: trainState.grad, showDot: true }],
    { yZero: true }
  );
  drawLineChart(
    document.getElementById("chart-loss"),
    [
      { label: "policy loss", color: COLORS.accent2, points: trainState.loss.ploss },
      { label: "value loss", color: COLORS.warn, points: trainState.loss.vloss },
      { label: "entropy", color: COLORS.accent, points: trainState.loss.entropy },
      { label: "approx KL", color: COLORS.bad, points: trainState.loss.kl },
    ],
    { legend: true }
  );
}

let ws;
function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/train`);
  ws.onmessage = (evt) => onEvent(JSON.parse(evt.data));
  ws.onclose = () => setTimeout(connectWs, 1000);
}

function onEvent(ev) {
  if (ev.type === "status") {
    if (ev.state === "starting") {
      resetTrainState();
      document.getElementById("stat-params").textContent = ev.n_parameters?.toLocaleString() ?? "–";
      setStatus("training", `starting ${ev.algo} on ${ev.env_id}`);
    } else if (ev.state === "training") {
      setStatus("training", "training…");
      startBtn.disabled = true;
      stopBtn.disabled = false;
    } else if (ev.state === "finished") {
      setStatus("finished", "finished");
      startBtn.disabled = false;
      stopBtn.disabled = true;
      if (document.getElementById("view-report").classList.contains("active")) fetchAndRenderReport();
    } else if (ev.state === "stopped") {
      setStatus("finished", "stopped");
      startBtn.disabled = false;
      stopBtn.disabled = true;
      if (document.getElementById("view-report").classList.contains("active")) fetchAndRenderReport();
    } else if (ev.state === "error") {
      setStatus("error", "error: " + ev.message);
      startBtn.disabled = false;
      stopBtn.disabled = true;
    }
    document.getElementById("stat-step").textContent = ev.step ?? 0;
    return;
  }

  document.getElementById("stat-step").textContent = ev.step;

  if (ev.type === "episode") {
    trainState.reward.push(ev.reward);
    if (trainState.reward.length > 300) trainState.reward.shift();
    document.getElementById("stat-reward").textContent = ev.reward.toFixed(1);
    document.getElementById("stat-length").textContent = ev.length;
    redrawAll();
  } else if (ev.type === "train_metrics") {
    const m = ev.metrics;
    const push = (arr, key) => {
      if (m[key] !== undefined) arr.push({ x: ev.step, y: m[key] });
    };
    push(trainState.loss.ploss, "train/policy_gradient_loss");
    push(trainState.loss.vloss, "train/value_loss");
    push(trainState.loss.entropy, "train/entropy_loss");
    push(trainState.loss.kl, "train/approx_kl");
    if (m["train/policy_gradient_loss"] !== undefined)
      document.getElementById("stat-ploss").textContent = m["train/policy_gradient_loss"].toFixed(4);
    if (m["train/value_loss"] !== undefined)
      document.getElementById("stat-vloss").textContent = m["train/value_loss"].toFixed(4);
    if (m["train/entropy_loss"] !== undefined)
      document.getElementById("stat-entropy").textContent = m["train/entropy_loss"].toFixed(4);
    if (m["train/approx_kl"] !== undefined)
      document.getElementById("stat-kl").textContent = m["train/approx_kl"].toFixed(4);
    redrawAll();
  } else if (ev.type === "grad_norm") {
    trainState.grad.push({ x: ev.step, y: ev.value });
    if (trainState.grad.length > 400) trainState.grad.shift();
    redrawAll();
  } else if (ev.type === "frame") {
    const wrap = document.getElementById("frame-wrap");
    wrap.innerHTML = `<img src="${ev.image}" />`;
  } else if (ev.type === "weights") {
    const names = Object.keys(ev.layers);
    if (names[0]) {
      document.getElementById("w-name-1").textContent = names[0];
      drawHeatmap(document.getElementById("heatmap-input"), ev.layers[names[0]]);
    }
    if (names[1]) {
      document.getElementById("w-name-2").textContent = names[1];
      drawHeatmap(document.getElementById("heatmap-action"), ev.layers[names[1]]);
    }
  }
}

startBtn.addEventListener("click", async () => {
  const body = {
    env_id: envSelect.value,
    algo: algoSelect.value,
    total_timesteps: parseInt(document.getElementById("timesteps-input").value, 10),
    learning_rate: parseFloat(document.getElementById("lr-input").value),
  };
  const seed = document.getElementById("seed-input").value;
  if (seed) body.seed = parseInt(seed, 10);

  const res = await fetch("/api/train/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json();
    setStatus("error", err.detail || "failed to start");
    return;
  }
  startBtn.disabled = true;
  stopBtn.disabled = false;
});

stopBtn.addEventListener("click", async () => {
  await fetch("/api/train/stop", { method: "POST" });
  stopBtn.disabled = true;
});

// ---------------------------------------------------------------------------
// Math Lab: gradient descent playground
// ---------------------------------------------------------------------------

function syncSlider(rangeId, numId) {
  const range = document.getElementById(rangeId);
  const num = document.getElementById(numId);
  range.addEventListener("input", () => (num.value = range.value));
  num.addEventListener("input", () => (range.value = num.value));
}
syncSlider("gd1d-lr", "gd1d-lr-val");
syncSlider("gd2d-lr", "gd2d-lr-val");

function animatePath(path, onFrame, onDone, stepMs = 60) {
  let i = 0;
  const timer = setInterval(() => {
    onFrame(path[i], i);
    i++;
    if (i >= path.length) {
      clearInterval(timer);
      if (onDone) onDone();
    }
  }, stepMs);
}

// --- 1D: rolling downhill ---
document.getElementById("gd1d-run").addEventListener("click", async () => {
  const lr = parseFloat(document.getElementById("gd1d-lr-val").value);
  const res = await fetch("/api/gradient-lab/1d", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ learning_rate: lr, steps: 40, start_x: 8.0 }),
  });
  const data = await res.json();
  const canvas = document.getElementById("chart-gd1d");
  const readout = document.getElementById("gd1d-readout");

  const drawStatic = (highlightIdx) => {
    const { ctx, width, height } = fitCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    const padL = 46, padR = 12, padT = 14, padB = 24;
    const plotW = width - padL - padR, plotH = height - padT - padB;
    const xs = data.curve.map((p) => p.x);
    const ys = data.curve.map((p) => p.loss);
    const [xmin, xmax] = [Math.min(...xs), Math.max(...xs)];
    const [ymin, ymax] = niceRange(0, Math.max(...ys));
    const xTo = (x) => padL + ((x - xmin) / (xmax - xmin)) * plotW;
    const yTo = (y) => padT + plotH - ((y - ymin) / (ymax - ymin)) * plotH;

    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, yTo(0));
    ctx.lineTo(width - padR, yTo(0));
    ctx.stroke();

    ctx.strokeStyle = COLORS.accent2;
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.curve.forEach((p, i) => {
      const x = xTo(p.x), y = yTo(p.loss);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // path taken so far
    ctx.strokeStyle = COLORS.warn;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i <= highlightIdx; i++) {
      const p = data.path[i];
      const x = xTo(p.x), y = yTo(p.loss);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    const cur = data.path[highlightIdx];
    ctx.fillStyle = COLORS.warn;
    ctx.beginPath();
    ctx.arc(xTo(cur.x), yTo(cur.loss), 5, 0, Math.PI * 2);
    ctx.fill();

    readout.textContent = `step ${cur.step} — x = ${cur.x.toFixed(3)}, f(x) = ${cur.loss.toFixed(4)}`;
  };

  animatePath(data.path.map((_, i) => i), (idx) => drawStatic(idx), null, 50);
});

// --- 2D: fitting a line ---
let lastGdData = null;

function drawFitPanel(canvas, gdData, wCur, bCur) {
  const { ctx, width, height } = fitCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const padL = 40, padR = 12, padT = 14, padB = 24;
  const plotW = width - padL - padR, plotH = height - padT - padB;
  const xs = gdData.data.x, ys = gdData.data.y;
  const [xmin, xmax] = niceRange(Math.min(...xs), Math.max(...xs));
  const [ymin, ymax] = niceRange(Math.min(...ys), Math.max(...ys));
  const xTo = (x) => padL + ((x - xmin) / (xmax - xmin)) * plotW;
  const yTo = (y) => padT + plotH - ((y - ymin) / (ymax - ymin)) * plotH;

  ctx.fillStyle = COLORS.accent2;
  xs.forEach((x, i) => {
    ctx.beginPath();
    ctx.arc(xTo(x), yTo(ys[i]), 3, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.strokeStyle = COLORS.warn;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(xTo(xmin), yTo(wCur * xmin + bCur));
  ctx.lineTo(xTo(xmax), yTo(wCur * xmax + bCur));
  ctx.stroke();
}

function drawSurfacePanel(canvas, gdData, pathIdx) {
  const { ctx, width, height } = fitCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const padL = 40, padR = 12, padT = 14, padB = 24;
  const plotW = width - padL - padR, plotH = height - padT - padB;
  const { w: ws, b: bs, loss: grid } = gdData.surface;
  const flat = grid.flat();
  const lmin = Math.min(...flat), lmax = Math.max(...flat);

  const cw = plotW / ws.length, ch = plotH / bs.length;
  for (let i = 0; i < bs.length; i++) {
    for (let j = 0; j < ws.length; j++) {
      const t = (grid[i][j] - lmin) / (lmax - lmin || 1);
      const c = Math.round(255 * (1 - t));
      ctx.fillStyle = `rgb(${20 + (1 - t) * 20},${40 + t * 40},${60 + (1 - t) * 150})`;
      ctx.fillRect(padL + j * cw, padT + i * ch, cw + 0.6, ch + 0.6);
    }
  }

  const wTo = (w) => padL + ((w - ws[0]) / (ws[ws.length - 1] - ws[0])) * plotW;
  const bTo = (b) => padT + ((b - bs[0]) / (bs[bs.length - 1] - bs[0])) * plotH;

  ctx.strokeStyle = COLORS.accent;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= pathIdx; i++) {
    const p = gdData.path[i];
    const x = wTo(p.w), y = bTo(p.b);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  const cur = gdData.path[pathIdx];
  ctx.fillStyle = COLORS.accent;
  ctx.beginPath();
  ctx.arc(wTo(cur.w), bTo(cur.b), 5, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#fff";
  const tw = gdData.true_params.w, tb = gdData.true_params.b;
  ctx.beginPath();
  ctx.arc(wTo(tw), bTo(tb), 4, 0, Math.PI * 2);
  ctx.fill();
}

document.getElementById("gd2d-run").addEventListener("click", async () => {
  const lr = parseFloat(document.getElementById("gd2d-lr-val").value);
  const noise = parseFloat(document.getElementById("gd2d-noise").value);
  const seed = lastGdData?.seedUsed ?? Math.floor(Math.random() * 10000);
  const res = await fetch("/api/gradient-lab/2d", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ learning_rate: lr, steps: 50, noise, seed }),
  });
  const gdData = await res.json();
  gdData.seedUsed = seed;
  lastGdData = gdData;
  const readout = document.getElementById("gd2d-readout");

  animatePath(
    gdData.path.map((_, i) => i),
    (idx) => {
      const p = gdData.path[idx];
      drawFitPanel(document.getElementById("chart-gd2d-fit"), gdData, p.w, p.b);
      drawSurfacePanel(document.getElementById("chart-gd2d-surface"), gdData, idx);
      readout.textContent = `step ${p.step} — w = ${p.w.toFixed(3)}, b = ${p.b.toFixed(3)}, loss = ${p.loss.toFixed(3)}  (true w=${gdData.true_params.w}, b=${gdData.true_params.b})`;
    },
    null,
    45
  );
});

document.getElementById("gd2d-regen").addEventListener("click", async () => {
  const noise = parseFloat(document.getElementById("gd2d-noise").value);
  const seed = Math.floor(Math.random() * 10000);
  const res = await fetch("/api/gradient-lab/2d", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ learning_rate: 0.0, steps: 0, noise, seed }),
  });
  const gdData = await res.json();
  gdData.seedUsed = seed;
  lastGdData = gdData;
  drawFitPanel(document.getElementById("chart-gd2d-fit"), gdData, gdData.path[0].w, gdData.path[0].b);
  drawSurfacePanel(document.getElementById("chart-gd2d-surface"), gdData, 0);
});

// ---------------------------------------------------------------------------
// Report tab
// ---------------------------------------------------------------------------

const ALGO_EXPLAIN = {
  PPO: "PPO (Proximal Policy Optimization) is a policy-gradient method: it directly adjusts a network that outputs " +
    "action probabilities. After each batch of experience it estimates the <strong>advantage</strong> of each " +
    "action taken (how much better it did than expected, via GAE), then takes several gradient steps on a " +
    "<em>clipped</em> objective that stops the policy changing too drastically in one update — that's exactly " +
    "what <code>clip_range</code> below controls.",
  A2C: "A2C (Advantage Actor-Critic) is PPO's simpler ancestor: also an advantage-weighted policy gradient, but it " +
    "takes a single gradient step per batch instead of PPO's multiple clipped epochs — less sample-efficient, " +
    "but simpler and faster per update.",
  DQN: "DQN (Deep Q-Network) is a value-based method: instead of a policy, it learns a network estimating the " +
    "value (Q) of every action, and acts by picking the highest one (with epsilon-greedy exploration during " +
    "training, controlled by <code>exploration_fraction</code>/<code>exploration_final_eps</code>). Its loss is " +
    "the squared TD-error against a target computed from a slower-updating copy of the network — that's what " +
    "<code>target_update_interval</code> controls.",
};

let lastReport = null;

function renderFlowDiagram(report) {
  const wrap = document.getElementById("flow-diagram");
  wrap.innerHTML = "";
  const trace = report.forward_trace;

  const addNode = (title, sub, values) => {
    const node = document.createElement("div");
    node.className = "flow-node";
    node.innerHTML = `<div class="flow-title">${title}</div><div class="flow-sub">${sub}</div>` +
      (values ? `<div class="flow-values">[${values.join(", ")}${values.length >= 12 ? ", …" : ""}]</div>` : "");
    wrap.appendChild(node);
  };
  const addArrow = () => {
    const a = document.createElement("div");
    a.className = "flow-arrow";
    a.textContent = "→";
    wrap.appendChild(a);
  };

  addNode("Input observation", `${trace.input.length} numbers from the environment`, trace.input);
  trace.activations.forEach((act) => {
    addArrow();
    const sub = act.type === "Linear"
      ? `z = W·x + b &nbsp; (${act.in_features}×${act.out_features})`
      : `${act.type} activation`;
    addNode(act.name, sub, act.sample);
  });
  addArrow();

  const out = trace.output;
  if (out.kind === "action_probs") {
    addNode("Action probabilities", `softmax(logits) → chosen action ${out.chosen_action}`, out.probs);
  } else if (out.kind === "q_values") {
    addNode("Q-values", `argmax → chosen action ${out.chosen_action}`, out.values);
  } else if (out.kind === "continuous_action") {
    addNode("Action (mean)", "continuous action space, deterministic mean", out.chosen_action);
  }
}

function renderHyperparams(hp) {
  const wrap = document.getElementById("report-hyperparams");
  wrap.innerHTML = "";
  const grid = document.createElement("div");
  grid.className = "hyperparam-grid";
  Object.entries(hp).forEach(([k, v]) => {
    const kv = document.createElement("div");
    kv.className = "kv";
    kv.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
    grid.appendChild(kv);
  });
  wrap.appendChild(grid);
}

async function fetchAndRenderReport() {
  const report = await fetch("/api/report").then((r) => r.json());
  const empty = document.getElementById("report-empty");
  const content = document.getElementById("report-content");

  if (!report.available) {
    empty.style.display = "";
    content.style.display = "none";
    return;
  }
  lastReport = report;
  empty.style.display = "none";
  content.style.display = "";

  document.getElementById("report-summary-line").textContent =
    `${report.algo} on ${report.env_id} — ${report.trained_at_step.toLocaleString()} / ${report.total_timesteps_requested.toLocaleString()} steps${report.still_training ? " (still training)" : ""}`;

  const caveat = document.getElementById("report-caveat");
  const trainedFraction = report.trained_at_step / Math.max(1, report.total_timesteps_requested);
  if (report.still_training) {
    caveat.innerHTML = `<div class="warn-box">Training is still running — these numbers are a live snapshot, not the final model.</div>`;
  } else if (trainedFraction < 0.6) {
    caveat.innerHTML = `<div class="warn-box">This run stopped at ${Math.round(trainedFraction * 100)}% of the requested timesteps — expect an under-trained policy in the test results below.</div>`;
  } else {
    caveat.innerHTML = "";
  }

  renderFlowDiagram(report);

  const doc = report.env_doc;
  document.getElementById("report-env-name").textContent = report.env_id;
  document.getElementById("report-env-doc").innerHTML = doc
    ? `<p class="explain"><strong>Reward function:</strong> ${doc.reward_function}</p>
       <p class="explain"><strong>Termination:</strong> ${doc.termination}</p>`
    : `<p class="explain">No hand-written notes for this env yet — check its Gymnasium docs page for the exact reward/termination rules.</p>`;

  document.getElementById("report-algo-name").textContent = report.algo;
  document.getElementById("report-algo-explain").innerHTML = `<p class="explain">${ALGO_EXPLAIN[report.algo] || ""}</p>`;
  renderHyperparams(report.hyperparams);
}

document.getElementById("report-retrace").addEventListener("click", fetchAndRenderReport);

document.getElementById("test-run-btn").addEventListener("click", async () => {
  const n = parseInt(document.getElementById("test-n-episodes").value, 10) || 10;
  const statusPill2 = document.getElementById("test-status");
  const btn = document.getElementById("test-run-btn");
  statusPill2.className = "pill state-training";
  statusPill2.innerHTML = `<span class="dot"></span> running ${n} episodes…`;
  btn.disabled = true;

  try {
    const res = await fetch("/api/test/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n_episodes: n, deterministic: true }),
    });
    if (!res.ok) {
      const err = await res.json();
      statusPill2.className = "pill state-error";
      statusPill2.innerHTML = `<span class="dot"></span> ${err.detail}`;
      return;
    }
    const data = await res.json();
    statusPill2.className = "pill state-finished";
    statusPill2.innerHTML = `<span class="dot"></span> done`;

    const s = data.summary;
    document.getElementById("test-summary").innerHTML = `
      <p class="explain" style="font-size:14px;">
        <strong>${s.n_success}/${s.n_episodes} episodes succeeded</strong> (${Math.round(s.success_rate * 100)}%) —
        average reward <strong>${s.avg_reward.toFixed(1)}</strong>, average length <strong>${s.avg_length.toFixed(0)}</strong> steps.
        ${s.success_rate === 1 ? " The agent solved this consistently." :
          s.success_rate === 0 ? " The agent hasn't learned a reliable strategy yet — more training timesteps should help." :
          " Mixed results — the policy has partly learned the task but isn't fully converged yet."}
      </p>`;

    const table = document.createElement("table");
    table.className = "report-table";
    table.innerHTML = `<thead><tr><th>#</th><th>Reward</th><th>Length</th><th>Ended by</th><th>Result</th><th>Why</th></tr></thead>`;
    const tbody = document.createElement("tbody");
    data.results.forEach((r) => {
      const tr = document.createElement("tr");
      const badge = r.success === true ? `<span class="badge success">success</span>` :
        r.success === false ? `<span class="badge fail">fail</span>` : "–";
      const endedBy = r.terminated ? "terminated" : r.truncated ? "truncated (time limit)" : "–";
      tr.innerHTML = `<td>${r.episode}</td><td>${r.reward}</td><td>${r.length}</td><td>${endedBy}</td><td>${badge}</td><td class="reason">${r.reason}</td>`;
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    const tableWrap = document.getElementById("test-table");
    tableWrap.innerHTML = "";
    tableWrap.appendChild(table);
  } finally {
    btn.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
loadEnvs();
connectWs();
redrawAll();
