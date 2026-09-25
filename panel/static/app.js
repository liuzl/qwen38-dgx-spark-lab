const $ = (id) => document.getElementById(id);
const number = (value) =>
  value == null
    ? "—"
    : Number(value).toLocaleString("zh-CN", { maximumFractionDigits: 0 });
const duration = (ms) =>
  ms == null
    ? "—"
    : ms < 1000
      ? `${Math.round(ms)} 毫秒`
      : `${(ms / 1000).toFixed(2)} 秒`;
const clock = (ts) =>
  ts
    ? new Date(ts * 1000).toLocaleString("zh-CN", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      })
    : "暂无";
function relative(ts) {
  if (!ts) return "暂无记录";
  const s = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  return s < 60
    ? `${s} 秒前`
    : s < 3600
      ? `${Math.floor(s / 60)} 分钟前`
      : s < 86400
        ? `${Math.floor(s / 3600)} 小时前`
        : `${Math.floor(s / 86400)} 天前`;
}
function text(id, value) {
  $(id).textContent = value ?? "—";
}
function node(tag, content, className) {
  const el = document.createElement(tag);
  if (content != null) el.textContent = content;
  if (className) el.className = className;
  return el;
}
let snapshot = null,
  summary = null,
  history = [],
  auditKey = "",
  cursor = null,
  page = "overview";
let epoch = 0,
  detailEpoch = 0,
  summaryEpoch = 0,
  historyEpoch = 0,
  loadedOlder = false,
  answerText = "",
  selectedId = null;
let loadedRequests = [],
  lastListSignature = "";
let requestsLoading = false,
  summaryLoading = false;
const outcomes = {
  ok: "成功",
  error: "失败",
  incomplete: "未完成",
  disconnected: "连接中断",
  cancelled: "已取消",
};
function query() {
  return new URLSearchParams({
    hours: $("audit-hours").value,
    exclude_tests: $("exclude-tests").checked ? "1" : "0",
  });
}
async function fetchJson(path, privateData = false) {
  const response = await fetch(path, {
    cache: "no-store",
    headers: privateData ? { Authorization: `Bearer ${auditKey}` } : {},
  });
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(
      response.status === 401
        ? "查看密钥不正确或已失效，请重新解锁。"
        : response.status === 503
          ? "请求记录暂时不可用，请稍后重试。"
          : `加载失败（${response.status}），请重试。`,
    );
    error.status = response.status;
    throw error;
  }
  return data;
}
function showPage(next) {
  page = next;
  for (const view of ["overview", "requests"]) {
    $(`${view}-view`).hidden = view !== page;
    $(`tab-${view}`).setAttribute(
      "aria-current",
      view === page ? "page" : "false",
    );
  }
  location.hash = page;
  if (page === "overview") drawChart();
  else if (auditKey) loadRequests();
}
function activity() {
  if (!snapshot) return;
  const online = snapshot.online;
  $("health-chip").className = `health-chip ${online ? "online" : "offline"}`;
  text("health-label", online ? "服务在线" : "暂时无法连接");
  text(
    "activity-title",
    !online
      ? "服务暂时不可用"
      : snapshot.running_requests > 0
        ? "正在处理请求"
        : snapshot.waiting_requests > 0
          ? "请求等待调度"
          : "服务在线，当前空闲",
  );
  const last = summary?.latest_request ?? summary?.last_request;
  text(
    "activity-description",
    !online
      ? "暂时未获取到实时状态，请稍后刷新。"
      : last
        ? `最近一条${$("exclude-tests").checked ? "非验证" : ""}请求开始于 ${relative(last)}（${clock(last)}）`
        : "当前没有正在处理的请求；正在读取最近请求时间。",
  );
  if (online && summary && !last)
    text(
      "activity-description",
      "当前没有正在处理的请求，保留范围内暂无符合筛选的记录。",
    );
}
async function refreshSnapshot() {
  try {
    snapshot = await fetchJson("/api/snapshot");
    for (const [id, key] of [
      ["running-requests", "running_requests"],
      ["waiting-requests", "waiting_requests"],
      ["preemptions", "preemptions_total"],
    ])
      text(id, number(snapshot[key]));
    for (const [id, key] of [
      ["generation-rate", "generation_tok_s"],
      ["prompt-rate", "prompt_tok_s"],
      ["kv-percent", "kv_cache_percent"],
      ["accepted-rate", "accepted_tok_s"],
      ["drafted-rate", "drafted_tok_s"],
      ["accept-length", "acceptance_length"],
    ])
      text(id, snapshot[key]?.toFixed(1));
    text(
      "prefix-rate",
      snapshot.prefix_cache_hit_rate == null
        ? "暂无新请求"
        : `${snapshot.prefix_cache_hit_rate.toFixed(1)}%`,
    );
    text(
      "accept-rate",
      snapshot.acceptance_rate == null
        ? "—"
        : `${snapshot.acceptance_rate.toFixed(1)}%`,
    );
    for (const [id, key] of [
      ["ttft", "ttft_p95_ms"],
      ["tpot", "tpot_p95_ms"],
      ["e2e", "e2e_p95_ms"],
    ])
      text(id, snapshot.latency?.[key]?.toFixed(1));
    text("model-name", snapshot.models?.join(" · ") || "");
    text(
      "last-update",
      `更新于 ${new Date(snapshot.updated_at * 1000).toLocaleTimeString("zh-CN", { hour12: false })}`,
    );
    $("beszel-link").hidden = !snapshot.beszel_url;
    $("beszel-link").href = snapshot.beszel_url || "#";
    const model = $("audit-model"),
      existing = new Set([...model.options].map((o) => o.value));
    for (const value of snapshot.models || [])
      if (!existing.has(value)) {
        const option = node("option", value);
        option.value = value;
        model.append(option);
      }
    const bars = snapshot.per_position_acceptance || [];
    $("position-bars").replaceChildren(
      ...(bars.some((v) => v != null)
        ? bars.map((v, i) => {
            const wrap = node("div", null, "position-bar");
            const track = node("div", null, "bar-track");
            const fill = node("div", null, "bar-fill");
            fill.style.height = `${Math.max(0, Math.min(100, v || 0))}%`;
            track.append(fill);
            wrap.append(
              node("strong", v == null ? "—" : `${v.toFixed(0)}%`),
              track,
              node("span", `P${i + 1}`),
            );
            return wrap;
          })
        : [node("p", "暂无解码数据", "empty-copy")]),
    );
    activity();
  } catch {
    snapshot = { ...snapshot, online: false };
    activity();
    text("last-update", "更新失败");
  }
}
async function refreshSummary() {
  const ticket = ++summaryEpoch;
  summaryLoading = true;
  try {
    const data = await fetchJson(`/api/audit/summary?${query()}`);
    if (ticket !== summaryEpoch) return;
    summary = data;
    $("audit-status").className = "notice";
    text(
      "audit-status",
      `${$("audit-hours").selectedOptions[0].text} · ${number(data.requests)} 次已结束请求${$("exclude-tests").checked ? " · 已隐藏部署验证请求" : ""}`,
    );
    const hit =
      data.input_tokens && data.cached_tokens != null
        ? `${((data.cached_tokens / data.input_tokens) * 100).toFixed(1)}%`
        : "—";
    const cards = [
      [
        "请求数",
        number(data.requests),
        data.requests
          ? `${number(data.errors ?? 0)} 次异常 / 未完成`
          : "此范围内暂无请求",
      ],
      [
        "输入 token",
        number(data.input_tokens),
        `其中缓存命中 ${number(data.cached_tokens)}`,
      ],
      [
        "输出 token",
        number(data.output_tokens),
        `推理 token ${number(data.reasoning_tokens)}`,
      ],
      ["缓存命中率", hit, "命中缓存的 token ÷ 输入 token"],
    ];
    $("audit-cards").replaceChildren(
      ...cards.map(([label, value, note]) => {
        const card = node("article", null, "summary-card");
        card.append(
          node("span", label),
          node("strong", value),
          node("small", note),
        );
        return card;
      }),
    );
    text("recent-ttft", duration(data.ttft_p50_ms));
    text("recent-duration", duration(data.duration_p50_ms));
    text("recent-p95", duration(data.duration_p95_ms));
    text("recent-errors", number(data.errors ?? 0));
    text(
      "audit-quality",
      `缺少 token 统计：${number(data.missing_usage ?? 0)} 条；正文超限截断：${number(data.truncated ?? 0)} 条。`,
    );
    text(
      "retention-note",
      data.retained_since
        ? `最早保留记录：${clock(data.retained_since)}`
        : "记录将在请求完成后出现",
    );
    activity();
  } catch (error) {
    if (ticket !== summaryEpoch) return;
    summary = null;
    $("audit-status").className = "notice error";
    text("audit-status", error.message);
    $("audit-cards").replaceChildren();
    for (const id of [
      "recent-ttft",
      "recent-duration",
      "recent-p95",
      "recent-errors",
    ])
      text(id, "—");
  } finally {
    if (ticket === summaryEpoch) summaryLoading = false;
  }
}
async function refreshHistory() {
  const ticket = ++historyEpoch;
  try {
    const data = await fetchJson(
      `/api/history?hours=${$("audit-hours").value}`,
    );
    if (ticket !== historyEpoch) return;
    history = data.samples || [];
    drawChart();
  } catch {
    if (ticket === historyEpoch) {
      history = [];
      drawChart();
    }
  }
}
function drawChart() {
  const canvas = $("throughput-chart"),
    rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  const scale = devicePixelRatio || 1;
  canvas.width = rect.width * scale;
  canvas.height = rect.height * scale;
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  $("chart-empty").hidden = history.length > 1;
  if (history.length < 2) return;
  text("chart-start", clock(history[0].ts));
  text("chart-end", clock(history.at(-1).ts));
  const max = Math.max(1, ...history.map((s) => s.generation)) * 1.1;
  const left = 38,
    w = rect.width - left - 6,
    h = rect.height - 20;
  ctx.font = "10px system-ui";
  ctx.fillStyle = "#a1afa6";
  ctx.strokeStyle = "#303a33";
  ctx.lineWidth = 1;
  for (let i = 0; i < 4; i++) {
    const y = 10 + (h * i) / 3;
    ctx.fillText((max * (1 - i / 3)).toFixed(0), 1, y + 3);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(rect.width, y);
    ctx.stroke();
  }
  const start = history[0].ts,
    span = Math.max(1, history.at(-1).ts - start);
  ctx.strokeStyle = "#c3ee99";
  ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach((s, i) => {
    const x = left + ((s.ts - start) / span) * w,
      y = 10 + h - (s.generation / max) * h;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
}
function clearDetails() {
  detailEpoch++;
  selectedId = null;
  answerText = "";
  if ($("audit-detail").open) $("audit-detail").close();
  for (const id of ["detail-conversation", "detail-stats", "usage-list"])
    $(id).replaceChildren();
  for (const id of [
    "audit-input",
    "audit-output",
    "audit-usage",
    "audit-detail-note",
    "detail-meta",
    "copy-status",
  ])
    text(id, "");
}
function lock() {
  epoch++;
  auditKey = "";
  requestsLoading = false;
  $("audit-key").value = "";
  $("audit-login-card").hidden = false;
  $("audit-workspace").hidden = true;
  text("lock-indicator", "需解锁");
  $("audit-rows").replaceChildren();
  loadedRequests = [];
  clearDetails();
  text("audit-private-status", "已锁定，请求内容已从页面清除。");
}
function setPrivateStatus(message, isError = false) {
  text("audit-private-status", message);
  $("audit-private-status").className = `notice${isError ? " error" : ""}`;
}
async function loadRequests(append = false, automatic = false) {
  if (!auditKey || requestsLoading) return;
  const ticket = epoch;
  requestsLoading = true;
  const params = query();
  params.set("task", $("audit-task").value.trim());
  params.set("model", $("audit-model").value);
  if (append && cursor) params.set("before", cursor);
  if (!automatic) setPrivateStatus("正在加载请求…");
  $("audit-filter").disabled = true;
  try {
    const data = await fetchJson(`/api/audit/requests?${params}`, true);
    if (ticket !== epoch) return;
    $("audit-login-card").hidden = true;
    $("audit-workspace").hidden = false;
    text("lock-indicator", "已解锁");
    const signature = JSON.stringify(data.requests);
    if (automatic && signature === lastListSignature) return;
    if (!append) lastListSignature = signature;
    if (!append) {
      $("audit-rows").replaceChildren();
      loadedOlder = false;
      loadedRequests = [];
    }
    loadedRequests.push(...data.requests);
    for (const row of data.requests) {
      const tr = node("tr");
      const when = node("td");
      when.dataset.label = "时间 / 模型";
      when.append(
        node("span", clock(row.started)),
        node("small", row.model || "未知模型"),
      );
      const status = node("td");
      status.dataset.label = "结果";
      status.append(
        node(
          "span",
          outcomes[row.outcome] || row.outcome,
          `badge${row.outcome === "ok" ? "" : " error"}`,
        ),
      );
      if (row.truncated) status.append(node("small", "正文已截断"));
      const input = node("td");
      input.dataset.label = "输入 / 缓存";
      input.append(
        node("span", number(row.input_tokens)),
        node("small", `缓存 ${number(row.cached_tokens)}`),
      );
      const output = node("td", number(row.output_tokens));
      output.dataset.label = "输出 token";
      const elapsed = node("td");
      elapsed.dataset.label = "总耗时 / 首 token";
      elapsed.append(
        node("span", duration(row.duration_ms)),
        node("small", `首 token ${duration(row.ttft_ms)}`),
      );
      const action = node("td");
      const button = node("button", "查看对话");
      button.type = "button";
      button.setAttribute(
        "aria-label",
        `查看 ${clock(row.started)} 的请求对话`,
      );
      button.addEventListener("click", () => requestDetail(row.id));
      action.append(button);
      tr.append(when, status, input, output, elapsed, action);
      $("audit-rows").append(tr);
    }
    cursor = data.next_before;
    if (append) loadedOlder = true;
    $("audit-more").hidden = !cursor;
    $("request-empty").hidden = $("audit-rows").children.length > 0;
    setPrivateStatus("");
    text(
      "list-note",
      `${number($("audit-rows").children.length)} 条记录 · ${automatic ? "已自动更新" : "更新于 " + new Date().toLocaleTimeString("zh-CN", { hour12: false })}${loadedOlder ? " · 浏览历史时暂停自动更新" : ""}`,
    );
  } catch (error) {
    if (ticket !== epoch) return;
    if (error.status === 401) lock();
    setPrivateStatus(error.message, true);
  } finally {
    if (ticket === epoch) {
      requestsLoading = false;
      $("audit-filter").disabled = false;
    }
  }
}
function pretty(raw) {
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw || "无";
  }
}
function contentText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content))
    return content == null ? "" : JSON.stringify(content, null, 2);
  return content
    .map((part) => {
      if (typeof part === "string") return part;
      const type = part.type || "";
      if (type.includes("image")) return "[图片输入]";
      if (type.includes("audio")) return "[音频输入]";
      if (type.includes("video")) return "[视频输入]";
      return typeof part.text === "string"
        ? part.text
        : JSON.stringify(part, null, 2);
    })
    .join("\n");
}
function parseResponse(raw) {
  const result = { answer: "", reasoning: "", tools: [], notes: [] };
  let objects = [];
  try {
    objects = [JSON.parse(raw)];
  } catch {
    for (const line of (raw || "").split("\n"))
      if (line.startsWith("data:") && line.slice(5).trim() !== "[DONE]") {
        try {
          objects.push(JSON.parse(line.slice(5)));
        } catch {}
      }
  }
  const toolDeltas = new Map();
  let finalResponse = null;
  for (const obj of objects) {
    if (obj.error)
      result.notes.push(
        typeof obj.error === "string"
          ? obj.error
          : JSON.stringify(obj.error, null, 2),
      );
    if (obj.type === "response.output_text.delta")
      result.answer += obj.delta || "";
    if (obj.type?.includes("reasoning") && obj.type.endsWith(".delta"))
      result.reasoning += obj.delta || "";
    if (obj.response?.output) finalResponse = obj.response;
    if (obj.output) finalResponse = obj;
    for (const choice of obj.choices || []) {
      const message = choice.message || choice.delta;
      if (choice.text) result.answer += choice.text;
      if (!message) continue;
      result.answer += contentText(message.content);
      result.reasoning += contentText(
        message.reasoning_content || message.reasoning,
      );
      for (const tool of message.tool_calls || []) {
        if (choice.message) {
          result.tools.push(tool);
          continue;
        }
        const key = `${choice.index ?? 0}:${tool.index ?? 0}`;
        const existing = toolDeltas.get(key) || { name: "", arguments: "" };
        existing.name += tool.function?.name || "";
        existing.arguments += tool.function?.arguments || "";
        toolDeltas.set(key, existing);
      }
    }
  }
  if (finalResponse) {
    let answer = "",
      reasoning = "";
    for (const item of finalResponse.output || []) {
      if (item.type === "message") answer += contentText(item.content);
      else if (item.type === "reasoning")
        reasoning += contentText(item.summary || item.content);
      else if (item.type === "function_call")
        result.tools.push({ name: item.name, arguments: item.arguments });
    }
    if (answer) result.answer = answer;
    if (reasoning) result.reasoning = reasoning;
  }
  result.tools.push(...toolDeltas.values());
  return result;
}
function messageCard(role, body, className = "") {
  const article = node("article", null, `message ${className}`);
  article.append(node("div", role, "message-label"), node("pre", body));
  return article;
}
function renderConversation(data) {
  const container = $("detail-conversation");
  container.replaceChildren();
  let payload;
  try {
    payload = JSON.parse(data.request_body);
  } catch {
    container.append(
      node("p", "输入记录不完整，请切换到“原始数据”查看。", "raw-note"),
    );
  }
  const labels = {
    system: "系统指令",
    developer: "开发者指令",
    user: "用户",
    assistant: "历史回答",
    tool: "工具结果",
  };
  if (payload) {
    if (payload.instructions) {
      const d = node("details");
      d.append(
        node("summary", "系统指令"),
        node("pre", contentText(payload.instructions)),
      );
      container.append(d);
    }
    let messages = payload.messages ?? payload.input ?? payload.prompt;
    if (typeof messages === "string")
      messages = [{ role: "user", content: messages }];
    const earlier = node("details");
    earlier.append(node("summary", "展开历史消息与上下文"));
    let lastUser = -1;
    if (Array.isArray(messages))
      messages.forEach((msg, i) => {
        if (typeof msg === "string" || msg.role === "user") lastUser = i;
      });
    if (Array.isArray(messages))
      for (const [index, msg] of messages.entries()) {
        if (typeof msg === "number") continue;
        const role = msg.role || msg.type || "输入";
        const body =
          typeof msg === "string"
            ? msg
            : contentText(msg.content ?? msg.text ?? msg.output) +
              (msg.tool_calls
                ? "\n" + JSON.stringify(msg.tool_calls, null, 2)
                : "");
        if (!body) continue;
        if (role === "system" || role === "developer") {
          const item = node("details", null, "message");
          item.append(
            node("summary", labels[role], "message-label"),
            node("pre", body),
          );
          earlier.append(item);
        } else
          (index === lastUser ? container : earlier).append(
            messageCard(labels[role] || role, body),
          );
      }
    if (earlier.children.length > 1) container.append(earlier);
  }
  const parsed = parseResponse(data.response_body);
  answerText = parsed.answer;
  $("copy-answer").disabled = !answerText;
  const answer = messageCard(
    "本次回答",
    parsed.answer ||
      (parsed.tools.length
        ? "模型返回了工具调用。"
        : "没有可显示的文本回答，请查看原始数据。"),
    "answer",
  );
  if (parsed.reasoning) {
    const d = node("details");
    d.append(node("summary", "查看推理内容"), node("pre", parsed.reasoning));
    answer.append(d);
  }
  if (parsed.tools.length) {
    const d = node("details");
    d.append(
      node("summary", `查看工具调用（${parsed.tools.length}）`),
      node("pre", JSON.stringify(parsed.tools, null, 2)),
    );
    answer.append(d);
  }
  if (parsed.notes.length) answer.append(node("pre", parsed.notes.join("\n")));
  container.append(answer);
}
function detailView(view) {
  for (const name of ["conversation", "usage", "raw"])
    $(`detail-${name}`).hidden = name !== view;
  for (const button of document.querySelectorAll("[data-detail]"))
    button.setAttribute(
      "aria-current",
      button.dataset.detail === view ? "page" : "false",
    );
}
async function requestDetail(id) {
  clearDetails();
  selectedId = id;
  const index = loadedRequests.findIndex((row) => row.id === id);
  $("detail-newer").disabled = index <= 0;
  $("detail-older").disabled = index < 0 || index >= loadedRequests.length - 1;
  text(
    "detail-position",
    index < 0 ? "" : `${index + 1} / ${loadedRequests.length}`,
  );
  const ticket = ++detailEpoch;
  detailView("conversation");
  text("detail-title", "正在加载请求…");
  $("audit-detail").showModal();
  $("detail-conversation").append(node("p", "正在加载输入与回答…", "raw-note"));
  $("copy-answer").disabled = true;
  try {
    const data = await fetchJson(
      `/api/audit/requests/${encodeURIComponent(id)}`,
      true,
    );
    if (ticket !== detailEpoch) return;
    text(
      "detail-title",
      `${clock(data.started)} · ${outcomes[data.outcome] || data.outcome}`,
    );
    text("detail-meta", data.model || "未知模型");
    text(
      "audit-detail-note",
      data.truncated
        ? "正文达到记录上限，内容已截断。Token 统计独立采集。"
        : "",
    );
    const stats = [
      `输入 ${number(data.input_tokens)}`,
      `缓存 ${number(data.cached_tokens)}`,
      `输出 ${number(data.output_tokens)}`,
      `总耗时 ${duration(data.duration_ms)}`,
      `首 token ${duration(data.ttft_ms)}`,
    ];
    $("detail-stats").replaceChildren(
      ...stats.map((value) => node("span", value)),
    );
    text(
      "audit-usage",
      JSON.stringify(
        {
          请求ID: data.id,
          上游请求ID: data.metrics_json?.upstream_request_id,
          接口: data.endpoint,
          模型: data.model,
          任务或聊天ID: data.task || "未关联",
          调用方ID: data.metrics_json?.caller_id || "未知",
          HTTP状态: data.status,
          token: data.usage_json,
          服务端耗时: data.metrics_json,
        },
        null,
        2,
      ),
    );
    const usageRows = [
      ["请求 ID", data.id],
      ["任务 / 聊天 ID", data.task || "未关联"],
      ["调用方 ID", data.metrics_json?.caller_id || "未知"],
      ["输入 token（包含缓存）", number(data.input_tokens)],
      ["命中缓存 token", number(data.cached_tokens)],
      ["输出 token", number(data.output_tokens)],
      ["推理 token", number(data.reasoning_tokens)],
      ["首 token 延迟", duration(data.ttft_ms)],
      ["完整请求耗时", duration(data.duration_ms)],
      ["排队耗时", duration(data.metrics_json?.queue_time_ms)],
      [
        "生成速度",
        data.metrics_json?.tokens_per_second == null
          ? "—"
          : data.metrics_json.tokens_per_second.toFixed(1) + " token/s",
      ],
    ];
    $("usage-list").replaceChildren(
      ...usageRows.map(([label, value]) => {
        const row = node("div");
        row.append(node("dt", label), node("dd", value));
        return row;
      }),
    );
    text("audit-input", pretty(data.request_body));
    text("audit-output", pretty(data.response_body));
    renderConversation(data);
  } catch (error) {
    if (ticket !== detailEpoch) return;
    if (error.status === 401) {
      lock();
      setPrivateStatus(error.message, true);
    } else {
      text("detail-title", "请求加载失败");
      $("detail-conversation").replaceChildren(
        node("p", error.message, "notice error"),
      );
    }
  }
}
function reloadFilters() {
  epoch++;
  requestsLoading = false;
  loadedOlder = false;
  cursor = null;
  refreshSummary();
  refreshHistory();
  if (auditKey) loadRequests();
}
$("tab-overview").addEventListener("click", () => showPage("overview"));
$("tab-requests").addEventListener("click", () => showPage("requests"));
$("go-requests").addEventListener("click", () => showPage("requests"));
$("audit-hours").addEventListener("change", reloadFilters);
$("exclude-tests").addEventListener("change", reloadFilters);
$("refresh-all").addEventListener("click", () => {
  refreshSnapshot();
  reloadFilters();
});
$("audit-login").addEventListener("submit", (event) => {
  event.preventDefault();
  auditKey = $("audit-key").value.trim();
  $("audit-key").value = "";
  epoch++;
  requestsLoading = false;
  clearDetails();
  loadRequests();
});
$("audit-lock").addEventListener("click", lock);
$("audit-filter-form").addEventListener("submit", (event) => {
  event.preventDefault();
  epoch++;
  requestsLoading = false;
  loadRequests();
});
$("audit-filter").addEventListener("click", () => loadRequests());
$("audit-more").addEventListener("click", () => loadRequests(true));
$("audit-reset").addEventListener("click", () => {
  $("audit-model").value = "";
  $("audit-task").value = "";
  epoch++;
  requestsLoading = false;
  loadRequests();
});
$("empty-reset").addEventListener("click", () => {
  $("audit-hours").value = "168";
  $("exclude-tests").checked = false;
  $("audit-model").value = "";
  $("audit-task").value = "";
  reloadFilters();
});
$("detail-newer").addEventListener("click", () => {
  const i = loadedRequests.findIndex((row) => row.id === selectedId);
  if (i > 0) requestDetail(loadedRequests[i - 1].id);
});
$("detail-older").addEventListener("click", () => {
  const i = loadedRequests.findIndex((row) => row.id === selectedId);
  if (i >= 0 && i < loadedRequests.length - 1)
    requestDetail(loadedRequests[i + 1].id);
});
$("detail-close").addEventListener("click", clearDetails);
$("audit-detail").addEventListener("cancel", (event) => {
  event.preventDefault();
  clearDetails();
});
for (const button of document.querySelectorAll("[data-detail]"))
  button.addEventListener("click", () => detailView(button.dataset.detail));
$("copy-answer").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(answerText);
    text("copy-status", "回答已复制。");
  } catch {
    text("copy-status", "复制失败，请选中回答文字手动复制。");
  }
});
window.addEventListener("resize", drawChart);
$("advanced").addEventListener("toggle", drawChart);
window.addEventListener("hashchange", () => {
  const next = location.hash === "#requests" ? "requests" : "overview";
  if (next !== page) showPage(next);
});
showPage(location.hash === "#overview" ? "overview" : "requests");
refreshSnapshot();
refreshSummary();
refreshHistory();
setInterval(() => {
  if (!document.hidden) refreshSnapshot();
}, 3000);
setInterval(() => {
  if (document.hidden) return;
  if (!summaryLoading) refreshSummary();
  if (
    page === "requests" &&
    auditKey &&
    $("audit-auto").checked &&
    !loadedOlder &&
    !$("audit-detail").open &&
    !$("audit-rows").matches(":hover") &&
    !$("audit-workspace").contains(document.activeElement)
  )
    loadRequests(false, true);
}, 10000);
setInterval(() => {
  if (!document.hidden) refreshHistory();
}, 60000);
