"use strict";

const $ = (id) => document.getElementById(id);
const HISTORY_KEY = "cashguard.jobs";
const POLL_MS = 1500;
const HEALTH_MS = 15000;
const STAGES = ["upload", "queue", "analyze", "done"];

let selectedFile = null;
let localVideoUrl = null;   // видео из выбранного файла: играет сразу, без загрузки с сервера
let pollTimer = null;
let currentJob = null;
let marks = [];             // [{t, card, n}]

// ---------- утилиты ----------

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, "");
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const c of children) if (c != null) e.append(c);
  return e;
}

function icon(name, cls = "") {
  const ns = "http://www.w3.org/2000/svg";
  const s = document.createElementNS(ns, "svg");
  s.setAttribute("class", `ico ${cls}`.trim());
  s.setAttribute("aria-hidden", "true");
  const u = document.createElementNS(ns, "use");
  u.setAttribute("href", `#i-${name}`);
  s.append(u);
  return s;
}

function fmtTime(t) {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
}

function fmtSec(t) {
  return `${Number(t).toFixed(1).replace(".", ",")} с`;
}

function fmtSize(b) {
  return b > 1024 * 1024 ? `${(b / 1024 / 1024).toFixed(1).replace(".", ",")} МБ` : `${Math.ceil(b / 1024)} КБ`;
}

function fmtDate(ts) {
  return new Date(ts * 1000).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function plural(n, one, few, many) {
  const a = n % 10, b = n % 100;
  if (a === 1 && b !== 11) return one;
  if (a >= 2 && a <= 4 && (b < 12 || b > 14)) return few;
  return many;
}

function countViol(n) {
  return `${n} ${plural(n, "нарушение", "нарушения", "нарушений")}`;
}

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch { return []; }
}

function saveHistory(items) {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, 30))); } catch { /* нет доступа */ }
}

function upsertHistory(entry) {
  const items = loadHistory().filter((x) => x.job_id !== entry.job_id);
  const old = loadHistory().find((x) => x.job_id === entry.job_id) || {};
  items.unshift({ ...old, ...entry });
  saveHistory(items);
  renderHistory();
}

// ---------- состояние сервиса ----------

async function checkHealth() {
  const box = $("health");
  const text = $("healthText");
  try {
    const r = await fetch("/api/v1/health", { cache: "no-store" });
    const d = await r.json();
    const dev = { cuda: "GPU", mps: "GPU Apple", cpu: "CPU" }[d.device] || "";
    if (d.worker !== "ready") {
      box.dataset.state = "busy";
      text.textContent = "Загрузка моделей…";
    } else if (d.busy || d.queue > 0) {
      box.dataset.state = "busy";
      text.textContent = d.queue > 0 ? `Занят, в очереди: ${d.queue}` : "Занят: идёт анализ";
    } else {
      box.dataset.state = "ok";
      text.textContent = dev ? `Готов к проверке · ${dev}` : "Готов к проверке";
    }
  } catch {
    box.dataset.state = "down";
    text.textContent = "Сервис недоступен";
  }
  setTimeout(checkHealth, HEALTH_MS);
}

// ---------- выбор файла ----------

function pickFile(f) {
  if (!f) return;
  selectedFile = f;
  $("dropText").textContent = f.name;
  $("drop").classList.add("picked");
  $("fileInfo").textContent = fmtSize(f.size);
  $("start").disabled = false;
}

$("file").addEventListener("change", (e) => pickFile(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));

// ---------- статус ----------

function setStage(stage, failed) {
  const k = STAGES.indexOf(stage);
  document.querySelectorAll("#steps li").forEach((li, i) => {
    li.dataset.state = i < k ? "done" : i === k ? (failed ? "failed" : "current") : "";
    if (i === k) li.setAttribute("aria-current", "step"); else li.removeAttribute("aria-current");
  });
}

// progress: число 0..1, "indet" — бегущая полоса, null — без полосы
function showStatus(text, { time = "", progress = null, error = "", stage = null, failed = false } = {}) {
  $("status").classList.remove("hidden");
  $("statusText").textContent = text;
  $("statusTime").textContent = time;
  const p = $("progress");
  p.classList.toggle("hidden", progress === null);
  p.classList.toggle("indet", progress === "indet");
  $("bar").style.width = typeof progress === "number" ? `${Math.round(progress * 100)}%` : "";
  $("errorText").classList.toggle("hidden", !error);
  $("errorText").textContent = error;
  $("steps").classList.toggle("hidden", !stage);
  if (stage) setStage(stage, failed);
}

// ---------- загрузка ----------

$("start").addEventListener("click", () => {
  if (!selectedFile) return;
  const f = selectedFile;
  stopPolling();
  $("result").classList.add("hidden");
  $("start").disabled = true;
  if (localVideoUrl) URL.revokeObjectURL(localVideoUrl);
  // тип video/mp4: так браузеры воспроизводят H.264 в контейнере .mov
  localVideoUrl = URL.createObjectURL(new Blob([f], { type: "video/mp4" }));

  const form = new FormData();
  form.append("file", f);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/ui/api/analyze");
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      showStatus(`Загрузка видео: ${Math.round((e.loaded / e.total) * 100)} %`,
        { time: fmtSize(e.total), progress: e.loaded / e.total, stage: "upload" });
    }
  };
  xhr.onload = () => {
    $("start").disabled = false;
    let data = {};
    try { data = JSON.parse(xhr.responseText); } catch { /* не JSON */ }
    if (xhr.status !== 200 || !data.job_id) {
      showStatus("Видео не загружено", { error: data.detail || `Ошибка сервера (${xhr.status})`, stage: "upload", failed: true });
      return;
    }
    upsertHistory({ job_id: data.job_id, filename: f.name, created: Date.now() / 1000, local: true });
    openJob(data.job_id, true);
  };
  xhr.onerror = () => {
    $("start").disabled = false;
    showStatus("Видео не загружено", { error: "Нет связи с сервером. Проверьте подключение и попробуйте ещё раз.", stage: "upload", failed: true });
  };
  showStatus("Загрузка видео: 0 %", { time: fmtSize(f.size), progress: 0, stage: "upload" });
  xhr.send(form);
});

// ---------- опрос задачи ----------

function stopPolling() {
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = null;
}

function openJob(jobId, fresh = false) {
  stopPolling();
  currentJob = jobId;
  if (!fresh && localVideoUrl) { URL.revokeObjectURL(localVideoUrl); localVideoUrl = null; }
  history.replaceState(null, "", `#job=${jobId}`);
  $("result").classList.add("hidden");
  showStatus("Получение статуса…", { progress: "indet" });
  renderHistory();
  poll(jobId);
}

async function poll(jobId) {
  if (jobId !== currentJob) return;
  let job;
  try {
    const r = await fetch(`/ui/api/jobs/${encodeURIComponent(jobId)}`);
    if (r.status === 404) {
      showStatus("Проверка не найдена", { error: "Задачи нет на сервере: её результат мог быть удалён при перезапуске сервиса." });
      return;
    }
    job = await r.json();
  } catch {
    showStatus("Нет связи с сервером, повтор…", { progress: "indet" });
    pollTimer = setTimeout(() => poll(jobId), POLL_MS * 2);
    return;
  }
  if (jobId !== currentJob) return;
  if (job.state === "queued") {
    const n = job.queue_position || 0;
    showStatus(n > 0 ? `В очереди. Перед этим видео ${n} ${plural(n, "задача", "задачи", "задач")}` : "В очереди, обработка скоро начнётся",
      { time: `ожидание ${fmtSec(job.total_sec)}`, progress: "indet", stage: "queue" });
  } else if (job.state === "processing") {
    showStatus("Анализ видео: люди, сейф, деньги на столе…",
      { time: `прошло ${fmtSec(job.elapsed_sec)}`, progress: "indet", stage: "analyze" });
  } else if (job.state === "failed") {
    showStatus("Обработка не выполнена", { error: job.error || "Неизвестная ошибка", stage: "analyze", failed: true });
    upsertHistory({ job_id: jobId, filename: job.filename, created: job.created_at, outcome: "ошибка" });
    return;
  } else if (job.state === "done") {
    $("status").classList.add("hidden");
    renderResult(job);
    const n = job.violations.length;
    upsertHistory({
      job_id: jobId, filename: job.filename, created: job.created_at, n: job.verdict === "violation" ? n : 0,
      outcome: job.verdict === "violation" ? `нарушений: ${n}` : "не обнаружено",
    });
    return;
  }
  pollTimer = setTimeout(() => poll(jobId), POLL_MS);
}

// ---------- результат ----------

function seek(t, play = true) {
  const v = $("video");
  if (!v.src) return;
  const go = () => {
    v.currentTime = Math.min(t, (v.duration || t + 1) - 0.05);
    if (play) v.play().catch(() => {});
  };
  if (v.readyState >= 1) go(); else v.addEventListener("loadedmetadata", go, { once: true });
  if (window.matchMedia("(max-width: 760px)").matches) v.scrollIntoView({ behavior: "auto", block: "start" });
}

function renderResult(job) {
  document.body.classList.add("has-result");
  $("result").classList.remove("hidden");
  const bad = job.verdict === "violation";
  const n = job.violations.length;
  $("verdict").className = `case ${bad ? "bad" : "ok"}`;
  $("caseFile").textContent = job.filename || "";
  $("verdictTitle").textContent = bad ? "Обнаружены признаки возможных нарушений" : "Нарушение не обнаружено";
  const meta = job.meta || {};
  const parts = [];
  if (meta.duration_sec != null) parts.push(`длительность ${fmtSec(meta.duration_sec)}`);
  parts.push(`обработано за ${fmtSec(job.elapsed_sec)}`);
  if (job.created_at) parts.push(`загружено ${fmtDate(job.created_at)}`);
  $("verdictMeta").textContent = parts.join(" · ");
  $("verdictChip").replaceChildren(icon(bad ? "alert" : "check"), bad ? countViol(n) : "Нарушений нет");

  // видео
  const v = $("video");
  const src = localVideoUrl || job.video_url;
  $("noVideo").textContent = src ? "" : "Видео недоступно для просмотра: файл удалён с сервера.";
  if (src && v.getAttribute("src") !== src) v.src = src;
  if (!src) v.removeAttribute("src");

  // список нарушений
  const list = $("violations");
  list.replaceChildren();
  marks = [];
  if (!bad) {
    list.append(el("div", { class: "empty" },
      el("span", { class: "empty-ico" }, icon("check")),
      el("div", {},
        el("h3", {}, "Признаков нарушений не найдено"),
        el("p", {}, "На видео нет нарушений тех типов, которые распознаёт сервис. Под видео показано, "
          + "что он видел в кадре по времени: людей в помещении, сейф и предметы на столе."))));
  }
  job.violations.forEach((x, i) => {
    const ex = (job.explain || [])[i] || {};
    const frames = el("div", { class: "frames" });
    for (const f of x.evidence_frames) {
      const url = `data:image/jpeg;base64,${f.image_base64}`;
      const tc = fmtTime(f.timestamp_sec);
      frames.append(el("figure", {},
        el("button", { type: "button", class: "frame-btn", "aria-label": `Открыть кадр ${tc} крупно`,
          onclick: () => openViewer(url, `${x.type} · ${tc}`, f.timestamp_sec) },
          el("img", { src: url, alt: `Кадр ${tc}`, loading: "lazy" })),
        el("figcaption", {}, el("button", { type: "button", class: "textlink", onclick: () => seek(f.timestamp_sec) },
          icon("play"), tc))));
    }
    const card = el("article", { class: "viol", "aria-label": `Нарушение ${i + 1}` },
      el("header", { class: "viol-head" },
        el("span", { class: "num", "aria-hidden": "true" }, String(i + 1)),
        el("h3", {}, x.type),
        el("button", { type: "button", class: "btn soft small tc", title: "Перейти к моменту в видео",
          onclick: () => { seek(x.timecode_sec); highlight(i); } }, icon("play"), fmtTime(x.timecode_sec))),
      el("div", { class: "field" }, el("span", { class: "lbl" }, "Что видно на видео"),
        el("p", { class: "field-text" }, ex.observed || x.description)),
      normBlock(ex),
      x.evidence_frames.length ? el("div", { class: "field" }, el("span", { class: "lbl" }, "Кадры-доказательства"), frames) : null);
    list.append(card);
    marks.push({ t: x.timecode_sec, card, n: i + 1 });
  });

  $("json").textContent = JSON.stringify(apiJson(job), null, 2);
  layoutTimeline();
  renderLanes(job.timeline);
}

function normBlock(ex) {
  const lbl = el("span", { class: "lbl" }, "Норма",
    ex.direct === false
      ? el("span", { class: "pill", title: "Пункт прямо не называет это действие; несоответствие следует из его требования" }, "косвенно")
      : null);
  if (!ex.refs || !ex.refs.length) {
    return ex.norm ? el("div", { class: "field norm" }, lbl, el("p", { class: "norm-plain" }, ex.norm)) : null;
  }
  return el("div", { class: "field norm" }, lbl,
    ...ex.refs.map((r, k) => el("details", { class: "ref", open: k === 0 },
      el("summary", {}, icon("doc"),
        el("span", { class: "ref-text" }, el("span", { class: "ref-label" }, r.label),
          r.note ? el("span", { class: "ref-note" }, ` — ${r.note}`) : null),
        icon("chev", "chev")),
      el("div", { class: "ref-body" },
        el("div", { class: "ref-doc" }, r.doc),
        el("div", { class: "ref-quote" }, ...r.quote.map((q) => el("p", {}, q))),
        el("a", { class: "ref-link", href: r.url, target: "_blank", rel: "noopener" },
          "Открыть пункт на adilet.zan.kz", icon("ext"))))));
}

// ---------- дорожки событий ----------

function renderLanes(tl) {
  const box = $("lanes");
  box.replaceChildren();
  if (!tl || !tl.dur) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  const pos = (t) => `${Math.max(0, Math.min(100, (t / tl.dur) * 100))}%`;
  const width = (a, b) => `${Math.max(0.8, ((b - a) / tl.dur) * 100)}%`;
  const seg = (a, b, cls, label, tip) => {
    const s = el("button", { type: "button", class: `seg ${cls}`, title: tip, "aria-label": tip, onclick: () => seek(a) }, label || "");
    s.style.left = pos(a);
    s.style.width = width(a, b);
    return s;
  };
  const lane = (title, items, empty) => el("div", { class: "lane" },
    el("span", { class: "lane-lbl" }, title),
    el("div", { class: "lane-bar" }, ...items, items.length ? null : el("span", { class: "lane-empty" }, empty)));

  const grid = el("div", { class: "lanes-grid" });
  const people = (tl.people || []).filter((p) => p[2] > 0).map(([a, b, n]) =>
    seg(a, b, `p${Math.min(n, 3)}`, (b - a) / tl.dur > 0.06 ? String(n) : "",
      `${n} чел. в помещении · ${fmtTime(a)}–${fmtTime(b)}`));
  grid.append(lane("Люди в помещении", people, "никого"));
  if (tl.safe_open) {
    grid.append(lane("Сейф открыт", tl.safe_open.map(([a, b]) =>
      seg(a, b, "safe", "", `сейф открыт · ${fmtTime(a)}–${fmtTime(b)}`)), "закрыт всё время"));
  }
  grid.append(lane("Деньги или сумка на столе", (tl.desk_items || []).map(([a, b]) =>
    seg(a, b, "desk", "", `на столе деньги или сумка · ${fmtTime(a)}–${fmtTime(b)}`)), "не видно"));
  const ticks = (tl.events || []).map((e) => {
    const b = el("button", { type: "button", class: `tick ${e.kind}`, title: `${fmtTime(e.t)} — ${e.label}`,
      "aria-label": `${fmtTime(e.t)} — ${e.label}`, onclick: () => seek(e.t) });
    b.style.left = pos(e.t);
    return b;
  });
  grid.append(lane("События", ticks, "нет"));
  grid.append(el("div", { class: "lanes-head", id: "lanesHead" }));
  box.append(grid, el("p", { class: "lanes-note" },
    "Что сервис видел в кадре. Нажмите на отрезок или отметку, чтобы перейти к моменту."));
}

function apiJson(job) {
  // ответ API; base64 сокращён для читаемости
  return {
    job_id: job.job_id, status: job.status, verdict: job.verdict,
    violations: job.violations.map((x) => ({
      ...x,
      evidence_frames: x.evidence_frames.map((f) => ({
        timestamp_sec: f.timestamp_sec,
        image_base64: `${f.image_base64.slice(0, 24)}… (${Math.round(f.image_base64.length / 1024)} КБ)`,
      })),
    })),
  };
}

function highlight(i) {
  marks.forEach((m, k) => m.card.classList.toggle("active", k === i));
}

// ---------- таймлайн ----------

function layoutTimeline() {
  const tl = $("timeline");
  tl.querySelectorAll(".mark").forEach((m) => m.remove());
  const dur = $("video").duration;
  if (!dur || !isFinite(dur)) return;
  $("durLabel").textContent = fmtTime(dur).replace(/\.\d$/, "");
  marks.forEach((m, i) => {
    const b = el("button", { type: "button", class: "mark", title: `Нарушение ${m.n} · ${fmtTime(m.t)}`,
      "aria-label": `Нарушение ${m.n}, ${fmtTime(m.t)}`,
      onclick: (e) => { e.stopPropagation(); seek(m.t); highlight(i); m.card.scrollIntoView({ behavior: "smooth", block: "nearest" }); } },
      String(m.n));
    b.style.left = `${(m.t / dur) * 100}%`;
    tl.append(b);
  });
}

$("video").addEventListener("loadedmetadata", layoutTimeline);
$("video").addEventListener("timeupdate", () => {
  const v = $("video");
  if (!v.duration) return;
  const x = `${(v.currentTime / v.duration) * 100}%`;
  $("playhead").style.left = x;
  const lh = document.getElementById("lanesHead");
  if (lh) lh.style.left = x;
  if (!v.paused) {  // при воспроизведении подсвечиваем карточку нарушения, мимо которого идёт видео
    const k = marks.findIndex((m) => Math.abs(v.currentTime - m.t) < 1.5);
    if (k >= 0) highlight(k);
  }
});
$("timeline").addEventListener("click", (e) => {
  const v = $("video");
  if (!v.duration) return;
  const r = e.currentTarget.getBoundingClientRect();
  seek(((e.clientX - r.left) / r.width) * v.duration, false);
});

// ---------- просмотр кадра ----------

let viewerT = 0;
let viewerReturn = null;

function openViewer(url, caption, t) {
  viewerT = t;
  viewerReturn = document.activeElement;
  $("viewerImg").src = url;
  $("viewerCap").textContent = caption;
  $("viewer").classList.remove("hidden");
  $("viewerClose").focus();
}

function closeViewer() {
  if ($("viewer").classList.contains("hidden")) return;
  $("viewer").classList.add("hidden");
  if (viewerReturn && viewerReturn.focus) viewerReturn.focus();
}

$("viewerClose").addEventListener("click", closeViewer);
$("viewerSeek").addEventListener("click", () => { closeViewer(); seek(viewerT); });
$("viewer").addEventListener("click", (e) => { if (e.target.id === "viewer") closeViewer(); });
document.addEventListener("keydown", (e) => {
  if ($("viewer").classList.contains("hidden")) return;
  if (e.key === "Escape") closeViewer();
  if (e.key === "Tab") {  // фокус остаётся внутри окна просмотра
    const a = $("viewerSeek"), b = $("viewerClose");
    if (document.activeElement === a && e.shiftKey) { e.preventDefault(); b.focus(); }
    else if (document.activeElement === b && !e.shiftKey) { e.preventDefault(); a.focus(); }
  }
});

// ---------- история ----------

function outcomeChip(it) {
  if (it.outcome === "ошибка") return el("span", { class: "chip err" }, "ошибка обработки");
  if (!it.outcome) return el("span", { class: "chip wait" }, "в обработке");
  if (it.outcome.startsWith("нарушен")) {
    const n = it.n ?? Number(it.outcome.replace(/\D+/g, ""));
    return el("span", { class: "chip bad" }, n ? countViol(n) : it.outcome);
  }
  return el("span", { class: "chip ok" }, "нарушений нет");
}

function renderHistory() {
  const items = loadHistory();
  $("historyPanel").classList.toggle("hidden", items.length === 0);
  const body = $("historyBody");
  body.replaceChildren();
  for (const it of items) {
    const cur = it.job_id === currentJob;
    body.append(el("tr", { class: cur ? "current" : "", "aria-current": cur ? "true" : null },
      el("td", { class: "fname" }, it.filename || it.job_id),
      el("td", { class: "when" }, it.created ? fmtDate(it.created) : ""),
      el("td", {}, outcomeChip(it)),
      el("td", {}, cur ? el("span", { class: "cur-label" }, "Открыта")
        : el("button", { type: "button", class: "btn soft small", onclick: () => openJob(it.job_id) }, "Открыть"))));
  }
}

function openFromHash() {
  const h = location.hash.match(/job=([0-9a-f]+)/);
  if (h && h[1] !== currentJob) openJob(h[1]);
}

checkHealth();
renderHistory();
window.addEventListener("hashchange", openFromHash);
openFromHash();
