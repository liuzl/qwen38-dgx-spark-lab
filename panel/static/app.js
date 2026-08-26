const $ = (id) => document.getElementById(id);
const chart = $("throughput-chart");
const chartEmpty = $("chart-empty");
const healthChip = $("health-chip");
const historyButtons = [...document.querySelectorAll("[data-hours]")];

let historyHours = 24;
let historySamples = [];
let lastSnapshot = null;

function text(id, value, fallback = "—") {
  $(id).textContent = value === null || value === undefined ? fallback : value;
}

function percentWidth(id, value) {
  $(id).style.width = `${Math.max(0, Math.min(100, value || 0))}%`;
}

function formatClock(timestamp) {
  if (!timestamp) return "—";
  return new Date(timestamp * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function setHealth(snapshot) {
  healthChip.className = `health-chip ${snapshot.online ? "online" : "offline"}`;
  text("health-label", snapshot.online ? "Serving" : "Offline");
  healthChip.title = snapshot.error || "vLLM endpoint is healthy";
}

function renderPositions(values) {
  const root = $("position-bars");
  if (!values || !values.some((value) => value !== null)) {
    root.innerHTML = '<div class="empty-copy">Acceptance appears while requests are decoding.</div>';
    return;
  }
  root.innerHTML = values.map((value, index) => {
    const safe = value === null ? 0 : Math.max(0, Math.min(100, value));
    const label = value === null ? "—" : `${value.toFixed(0)}%`;
    return `<div class="position-bar">
      <strong>${label}</strong>
      <div class="bar-track"><div class="bar-fill" style="height:${safe}%"></div></div>
      <span>P${index + 1}</span>
    </div>`;
  }).join("");
}

function renderSnapshot(snapshot) {
  lastSnapshot = snapshot;
  setHealth(snapshot);
  text("generation-rate", snapshot.generation_tok_s?.toFixed(1));
  text("prompt-rate", snapshot.prompt_tok_s?.toFixed(1));
  text("running-requests", snapshot.running_requests);
  text("waiting-requests", snapshot.waiting_requests);
  text("kv-percent", snapshot.kv_cache_percent?.toFixed(1));
  text("prefix-rate", snapshot.prefix_cache_hit_rate?.toFixed(1), "idle");
  text("accept-length", snapshot.acceptance_length?.toFixed(2), "idle");
  text("accept-rate", snapshot.acceptance_rate?.toFixed(1), "idle");
  text("accepted-rate", snapshot.accepted_tok_s?.toFixed(1));
  text("drafted-rate", snapshot.drafted_tok_s?.toFixed(1));
  text("preemptions", snapshot.preemptions_total);
  text("ttft", snapshot.latency?.ttft_p95_ms?.toFixed(1));
  text("tpot", snapshot.latency?.tpot_p95_ms?.toFixed(1));
  text("e2e", snapshot.latency?.e2e_p95_ms?.toFixed(1));
  text("model-name", snapshot.models?.join(" · ") || "No served model");
  text("last-update", `updated ${formatClock(snapshot.updated_at)}`);
  $("beszel-link").href = snapshot.beszel_url || "#";
  percentWidth("kv-meter", snapshot.kv_cache_percent);
  percentWidth("prefix-meter", snapshot.prefix_cache_hit_rate);
  renderPositions(snapshot.per_position_acceptance);
}

function drawChart() {
  const context = chart.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const rect = chart.getBoundingClientRect();
  chart.width = Math.max(1, Math.floor(rect.width * ratio));
  chart.height = Math.max(1, Math.floor(rect.height * ratio));
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);

  const values = historySamples.map((sample) => Number(sample.generation || 0));
  if (lastSnapshot?.generation_tok_s > 0) values.push(lastSnapshot.generation_tok_s);
  chartEmpty.hidden = values.length > 1;
  if (values.length < 2) return;

  const padding = { top: 14, right: 8, bottom: 22, left: 8 };
  const width = rect.width - padding.left - padding.right;
  const height = rect.height - padding.top - padding.bottom;
  const maximum = Math.max(10, ...values) * 1.12;

  context.strokeStyle = "#252a26";
  context.lineWidth = 1;
  for (let line = 0; line < 4; line += 1) {
    const y = padding.top + (height / 3) * line;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(rect.width - padding.right, y);
    context.stroke();
  }

  const points = values.map((value, index) => ({
    x: padding.left + (index / (values.length - 1)) * width,
    y: padding.top + height - (value / maximum) * height,
  }));

  const gradient = context.createLinearGradient(0, padding.top, 0, padding.top + height);
  gradient.addColorStop(0, "rgba(199, 255, 67, .20)");
  gradient.addColorStop(1, "rgba(199, 255, 67, 0)");
  context.beginPath();
  context.moveTo(points[0].x, padding.top + height);
  points.forEach((point) => context.lineTo(point.x, point.y));
  context.lineTo(points.at(-1).x, padding.top + height);
  context.closePath();
  context.fillStyle = gradient;
  context.fill();

  context.beginPath();
  points.forEach((point, index) => {
    if (index === 0) context.moveTo(point.x, point.y);
    else context.lineTo(point.x, point.y);
  });
  context.strokeStyle = "#c7ff43";
  context.lineWidth = 2;
  context.lineJoin = "round";
  context.stroke();

  const last = points.at(-1);
  context.beginPath();
  context.arc(last.x, last.y, 3.5, 0, Math.PI * 2);
  context.fillStyle = "#c7ff43";
  context.fill();
}

async function getJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function refreshSnapshot() {
  try {
    renderSnapshot(await getJson("/api/snapshot"));
    drawChart();
  } catch (error) {
    setHealth({ online: false, error: String(error) });
  }
}

async function refreshHistory() {
  try {
    const payload = await getJson(`/api/history?hours=${historyHours}`);
    historySamples = payload.samples || [];
    drawChart();
  } catch {
    historySamples = [];
    drawChart();
  }
}

historyButtons.forEach((button) => {
  button.addEventListener("click", () => {
    historyHours = Number(button.dataset.hours);
    historyButtons.forEach((item) => item.classList.toggle("active", item === button));
    refreshHistory();
  });
});

window.addEventListener("resize", drawChart);
refreshSnapshot();
refreshHistory();
setInterval(refreshSnapshot, 2000);
setInterval(refreshHistory, 60000);
