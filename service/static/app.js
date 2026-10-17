"use strict";

const $ = (id) => document.getElementById(id);
const HISTORY_KEY = "cashguard.jobs";
const POLL_MS = 1500;
const HEALTH_MS = 15000;
const MAX_MB = 200;
const LEAD_SEC = 2;          // кнопка таймкода включает видео чуть раньше события, чтобы был виден подход
const STAGES = ["upload", "queue", "analyze"];
const VIDEO_EXT = /\.(mov|mp4|m4v)$/i;

let lastFile = null;        // последний выбранный файл: для «Загрузить ещё раз»
let localVideoUrl = null;   // видео из выбранного файла: играет сразу, без загрузки с сервера
let pollTimer = null;
let currentJob = null;
let currentDur = 0;
let marks = [];             // [{t, card, n}]
let rules = [];             // типы, которые распознаёт сервис
let uploadPct = -1;
let lastJob = null;
let xhr = null;              // текущая загрузка: новая отменяет прежнюю
const isMobile = () => window.matchMedia("(max-width: 760px)").matches;          // показанный результат: сводку перерисовываем, когда придёт список правил

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

function fmtClock(t) {
  const s = Math.round(t);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function fmtSec(t) {
  return `${Math.round(Number(t))} с`;
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

// ---------- история в браузере ----------

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch { return []; }
}

function saveHistory(items) {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, 30))); } catch { /* нет доступа */ }
}

function upsertHistory(entry) {
  const items = loadHistory();
  const old = items.find((x) => x.job_id === entry.job_id) || {};
  saveHistory([{ ...old, ...entry }, ...items.filter((x) => x.job_id !== entry.job_id)]);
  renderHistory();
}

// отношение «время анализа / длительность видео» по прошлым проверкам: для оценки ожидания
function speedRatio() {
  const r = loadHistory().map((x) => x.ratio).filter((x) => x > 0);
  return r.length ? r.reduce((a, b) => a + b, 0) / r.length : 0.45;
}

// ---------- экраны ----------

function setView(view) {
  $("upload").classList.toggle("hidden", view !== "start");
  $("status").classList.toggle("hidden", view !== "busy");
  $("result").classList.toggle("hidden", view !== "result");
  document.body.classList.toggle("has-result", view === "result");
}

// ---------- состояние сервиса ----------

async function checkHealth() {
  const box = $("health");
  const text = $("healthText");
  try {
    const r = await fetch("/api/v1/health", { cache: "no-store" });
    const d = await r.json();
    box.title = { cuda: "Модели работают на GPU сервера", mps: "Модели работают на GPU", cpu: "Модели работают на CPU" }[d.device] || "";
    if (d.worker !== "ready") {
      box.dataset.state = "busy";
      text.textContent = "Загрузка моделей…";
    } else if (d.busy || d.queue > 0) {
      box.dataset.state = "busy";
      text.textContent = d.queue > 0 ? `Сервис занят, в очереди ${d.queue}` : "Сервис занят";
    } else {
      box.dataset.state = "ok";
      text.textContent = "Сервис готов";
    }
  } catch {
    box.dataset.state = "down";
    text.textContent = "Сервис недоступен";
  }
  setTimeout(checkHealth, HEALTH_MS);
}

// ---------- старт: примеры и список правил ----------

async function loadStartInfo() {
  try {
    rules = await (await fetch("/ui/api/rules")).json();
    $("rulesList").replaceChildren(...rules.map((r) => el("li", {}, r.name)));
    if (window.matchMedia("(min-width: 1101px)").matches) $("rulesBox").open = true;
    if (lastJob) renderCoverage(lastJob, lastJob.verdict === "violation");
  } catch { $("rulesBox").classList.add("hidden"); }
  try {
    const ex = await (await fetch("/ui/api/examples")).json();
    $("examples").classList.toggle("hidden", !ex.length);
    $("exampleList").replaceChildren(...ex.map((x) => el("li", {},
      el("button", { type: "button", class: "row-link", onclick: () => openJob(x.job_id, { push: true }) },
        el("span", { class: "row-main" }, x.filename,
          x.duration_sec ? el("span", { class: "row-sub" }, ` · ${fmtSec(x.duration_sec)}`) : null),
        x.verdict === "violation"
          ? el("span", { class: "outcome bad" }, x.types && x.types.length === 1 ? x.types[0] : countViol(x.n))
          : el("span", { class: "outcome ok" }, "нарушений нет"),
        icon("chev", "row-chev")))));
  } catch { $("examples").classList.add("hidden"); }
}

// ---------- выбор файла ----------

function pickFile(f) {
  if (!f) return;
  lastFile = f;
  if (!VIDEO_EXT.test(f.name) && !(f.type || "").startsWith("video/")) {
    failEarly(f, "Это не видеофайл. Нужна запись в формате .mov или .mp4.");
    return;
  }
  if (f.size > MAX_MB * 1024 * 1024) {
    failEarly(f, `Файл ${fmtSize(f.size)} больше допустимых ${MAX_MB} МБ.`);
    return;
  }
  startUpload(f);
}

function failEarly(f, msg) {
  if (xhr) xhr.abort();
  stopPolling();
  setView("busy");
  $("statusFile").textContent = f.name;
  $("preview").classList.add("hidden");
  showStatus("Файл не принят", { error: msg, stage: "upload", failed: true, retry: false });
}

$("file").addEventListener("change", (e) => { pickFile(e.target.files[0]); e.target.value = ""; });
$("newCheck").addEventListener("click", () => $("file").click());
$("pickOther").addEventListener("click", () => $("file").click());
$("retry").addEventListener("click", () => { if (lastFile) startUpload(lastFile); });

// файл можно бросить в любое место страницы
let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
window.addEventListener("dragenter", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth += 1;
  $("dropVeil").classList.remove("hidden");
});
window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
window.addEventListener("dragleave", (e) => {
  if (!hasFiles(e)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) $("dropVeil").classList.add("hidden");
});
window.addEventListener("drop", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  $("dropVeil").classList.add("hidden");
  pickFile(e.dataTransfer.files[0]);
});

// ---------- статус ----------

function setStage(stage, failed) {
  const k = STAGES.indexOf(stage);
  document.querySelectorAll("#steps li").forEach((li, i) => {
    li.dataset.state = i < k ? "done" : i === k ? (failed ? "failed" : "current") : "";
    li.querySelector("use").setAttribute("href", i === k && failed ? "#i-close" : "#i-tick");
    if (i === k) li.setAttribute("aria-current", "step"); else li.removeAttribute("aria-current");
  });
}

// progress: число 0..1, "indet" — бегущая полоса, null — без полосы
function showStatus(text, { time = "", progress = null, error = "", stage = null, failed = false, retry = true } = {}) {
  if ($("statusText").textContent !== text) $("statusText").textContent = text;
  $("statusTime").textContent = time;
  const p = $("progress");
  p.classList.toggle("hidden", progress === null);
  p.classList.toggle("indet", progress === "indet");
  $("bar").style.width = typeof progress === "number" ? `${Math.round(progress * 100)}%` : "";
  if (typeof progress === "number") p.setAttribute("aria-valuenow", String(Math.round(progress * 100)));
  else p.removeAttribute("aria-valuenow");
  $("errorText").classList.toggle("hidden", !error);
  $("errorText").textContent = error;
  $("statusNote").classList.toggle("hidden", !!error);
  $("statusActions").classList.toggle("hidden", !error);
  $("retry").classList.toggle("hidden", !(retry && lastFile));
  $("steps").classList.toggle("hidden", !stage);
  if (stage) setStage(stage, failed);
}

// ---------- загрузка ----------

function startUpload(f) {
  stopPolling();
  currentJob = null;
  setView("busy");
  $("statusFile").textContent = f.name;
  if (localVideoUrl) URL.revokeObjectURL(localVideoUrl);
  // тип video/mp4: так браузеры воспроизводят H.264 в контейнере .mov
  localVideoUrl = URL.createObjectURL(new Blob([f], { type: "video/mp4" }));
  const pv = $("preview");
  pv.src = localVideoUrl;
  pv.classList.remove("hidden");

  const form = new FormData();
  form.append("file", f);
  if (xhr) xhr.abort();
  const req = new XMLHttpRequest();
  xhr = req;
  req.open("POST", "/ui/api/analyze");
  uploadPct = -1;
  req.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const pct = Math.floor((e.loaded / e.total) * 10) * 10;  // текст меняется шагом 10 %, чтобы не частить для экранного диктора
    if (pct !== uploadPct) {
      uploadPct = pct;
      showStatus(`Загрузка видео: ${pct} %`, { time: fmtSize(e.total), progress: e.loaded / e.total, stage: "upload" });
    } else {
      $("bar").style.width = `${Math.round((e.loaded / e.total) * 100)}%`;
    }
  };
  req.onload = () => {
    if (xhr !== req) return;
    xhr = null;
    let data = {};
    try { data = JSON.parse(req.responseText); } catch { /* не JSON */ }
    if (req.status !== 200 || !data.job_id) {
      showStatus("Видео не загружено", { error: data.detail || `Сервер ответил ошибкой ${req.status}. Попробуйте ещё раз.`, stage: "upload", failed: true });
      return;
    }
    upsertHistory({ job_id: data.job_id, filename: f.name, created: Date.now() / 1000 });
    openJob(data.job_id, { fresh: true, push: true });
  };
  req.onerror = () => xhr === req && showStatus("Видео не загружено",
    { error: "Нет связи с сервером. Проверьте подключение и загрузите файл ещё раз.", stage: "upload", failed: true });
  showStatus("Загрузка видео: 0 %", { time: fmtSize(f.size), progress: 0, stage: "upload" });
  req.send(form);
}

// ---------- опрос задачи ----------

function stopPolling() {
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = null;
}

function openJob(jobId, { fresh = false, push = false } = {}) {
  stopPolling();
  currentJob = jobId;
  if (!fresh) {
    if (xhr) { xhr.abort(); xhr = null; }
    if (localVideoUrl) { URL.revokeObjectURL(localVideoUrl); localVideoUrl = null; }
    $("preview").classList.add("hidden");
    $("preview").removeAttribute("src");
    $("statusFile").textContent = "Проверка";
  }
  const url = `#job=${jobId}`;
  if (location.hash !== url) history[push ? "pushState" : "replaceState"](null, "", url);
  setView("busy");
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
      showStatus("Проверка не найдена", { error: "Этой проверки нет на сервере: её могли удалить при перезапуске сервиса. Загрузите видео заново.", retry: false });
      upsertHistory({ job_id: jobId, outcome: "ошибка" });
      return;
    }
    job = await r.json();
  } catch {
    showStatus("Нет связи с сервером, повтор…", { progress: "indet" });
    pollTimer = setTimeout(() => poll(jobId), POLL_MS * 2);
    return;
  }
  if (jobId !== currentJob) return;
  if (job.filename) $("statusFile").textContent = job.filename;
  if (job.state === "queued") {
    const n = job.queue_position || 0;
    showStatus(n > 0 ? `В очереди: перед этим видео ${n} ${plural(n, "проверка", "проверки", "проверок")}` : "В очереди, анализ скоро начнётся",
      { time: `ожидание ${fmtSec(job.total_sec)}`, progress: "indet", stage: "queue" });
  } else if (job.state === "processing") {
    const d = $("preview").duration;
    const est = d && isFinite(d) ? Math.max(8, d * speedRatio() * 1.3) : 0;
    const tail = !est ? "" : job.elapsed_sec <= est ? ` из ≈ ${fmtSec(est)}` : ", дольше обычного";
    showStatus("Анализ: люди, сейф, деньги на столе, дверь, экран",
      { time: `прошло ${fmtSec(job.elapsed_sec)}${tail}`, progress: "indet", stage: "analyze" });
  } else if (job.state === "failed") {
    showStatus("Анализ не выполнен", { error: `Сервис не смог обработать видео: ${job.error || "причина неизвестна"}.`, stage: "analyze", failed: true });
    upsertHistory({ job_id: jobId, filename: job.filename, created: job.created_at, outcome: "ошибка" });
    return;
  } else if (job.state === "done") {
    const n = job.violations.length;
    const dur = (job.meta || {}).duration_sec;
    upsertHistory({
      job_id: jobId, filename: job.filename, created: job.created_at, n: job.verdict === "violation" ? n : 0,
      outcome: job.verdict === "violation" ? `нарушений: ${n}` : "не обнаружено",
      ratio: dur ? job.elapsed_sec / dur : undefined,
    });
    renderResult(job);
    return;
  }
  pollTimer = setTimeout(() => poll(jobId), POLL_MS);
}

// ---------- результат ----------

function seek(t, play = true) {
  const v = $("video");
  if (!v.getAttribute("src")) return;
  const go = () => {
    v.currentTime = Math.max(0, Math.min(t, (v.duration || t + 1) - 0.05));
    if (play) v.play().catch(() => {}); else v.pause();
  };
  if (v.readyState >= 1) go(); else v.addEventListener("loadedmetadata", go, { once: true });
  if (isMobile()) v.scrollIntoView({ behavior: "auto", block: "start" });
}

const smooth = () => (window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth");

function goViolation(i, scroll = true) {
  const m = marks[i];
  if (!m) return;
  seek(m.t - LEAD_SEC);
  highlight(i);
  if (scroll && !isMobile()) m.card.scrollIntoView({ behavior: smooth(), block: "nearest" });
}

function renderResult(job) {
  lastJob = job;
  setView("result");
  const bad = job.verdict === "violation";
  const n = job.violations.length;
  const meta = job.meta || {};
  $("verdict").className = `verdict ${bad ? "bad" : "ok"}`;
  $("verdictIco").setAttribute("href", bad ? "#i-alert" : "#i-check");
  $("verdictTitle").textContent = bad ? "Обнаружены признаки возможных нарушений" : "Нарушение не обнаружено";
  const parts = [job.filename];
  if (meta.duration_sec != null) parts.push(`видео ${fmtSec(meta.duration_sec)}`);
  parts.push(`анализ ${fmtSec(job.elapsed_sec)}`);
  if (job.created_at) parts.push(fmtDate(job.created_at));
  const metaParts = [...(bad ? [el("b", {}, countViol(n))] : []), ...parts.map((x) => el("span", {}, x))];
  $("verdictMeta").replaceChildren(...metaParts.flatMap((x, k) => (k ? [" · ", x] : [x])));
  document.title = `${job.filename} — ${bad ? countViol(n) : "нарушений нет"} · CashGuard`;

  // видео
  const v = $("video");
  const src = localVideoUrl || job.video_url;
  $("noVideo").textContent = src ? "" : "Видео недоступно для просмотра: файл удалён с сервера. Отметки и кадры ниже сохранены.";
  if (src && v.getAttribute("src") !== src) v.src = src;
  if (!src) v.removeAttribute("src");

  // карточки нарушений
  const list = $("violations");
  list.replaceChildren();
  marks = [];
  job.violations.forEach((x, i) => {
    const ex = (job.explain || [])[i] || {};
    const card = el("article", { class: "viol", id: `v${i + 1}`, "aria-labelledby": `v${i + 1}t` },
      el("header", { class: "viol-head" },
        el("span", { class: "num", "aria-hidden": "true" }, String(i + 1)),
        el("h3", { id: `v${i + 1}t` }, x.type),
        el("button", { type: "button", class: "btn secondary small tc", title: `Воспроизвести с ${fmtTime(Math.max(0, x.timecode_sec - LEAD_SEC))}`,
          onclick: () => goViolation(i, false) }, icon("play"), fmtTime(x.timecode_sec))),
      el("div", { class: "viol-body" },
        el("div", { class: "viol-text" },
          el("div", { class: "field" }, el("span", { class: "lbl" }, "Что видно на видео"),
            el("p", { class: "field-text" }, ex.observed || x.description)),
          normBlock(ex, i === 0)),
        framesBlock(x)));
    list.append(card);
    marks.push({ t: x.timecode_sec, card, n: i + 1 });
  });

  // указатель, когда нарушений несколько
  const idx = $("vindex");
  idx.classList.toggle("hidden", n < 2);
  idx.replaceChildren(...(n < 2 ? [] : job.violations.map((x, i) => el("li", {},
    el("button", { type: "button", class: "textlink", onclick: () => goViolation(i) },
      el("span", { class: "vindex-t" }, fmtTime(x.timecode_sec)), x.type)))));

  renderCoverage(job, bad);
  $("json").textContent = JSON.stringify(apiJson(job), null, 2);
  currentDur = (job.timeline && job.timeline.dur) || meta.duration_sec || 0;
  renderChrono(job.timeline);
  $("verdictTitle").focus({ preventScroll: true });
}

function framesBlock(x) {
  if (!x.evidence_frames.length) return null;
  const urls = x.evidence_frames.map((f) => ({ url: `data:image/jpeg;base64,${f.image_base64}`, t: f.timestamp_sec }));
  return el("div", { class: "field frames-field" }, el("span", { class: "lbl" }, "Кадры-доказательства"),
    el("div", { class: "frames" }, ...urls.map((f, k) => el("figure", {},
      el("button", { type: "button", class: "frame-btn", "aria-label": `Открыть кадр ${fmtTime(f.t)} крупно`,
        onclick: () => openViewer(urls, k, x.type) },
        el("img", { src: f.url, alt: `${x.type}, кадр ${fmtTime(f.t)}`, loading: "lazy" })),
      el("figcaption", {}, el("button", { type: "button", class: "textlink", title: "Перейти к кадру и остановить видео",
        onclick: () => seek(f.t, false) }, icon("play"), fmtTime(f.t)))))));
}

function normBlock(ex, openFirst) {
  const lbl = el("span", { class: "lbl" }, "Норма",
    ex.direct === false ? el("span", { class: "lbl-note" }, " · косвенная: пункт задаёт требование, а не называет само действие") : null);
  if (!ex.refs || !ex.refs.length) {
    return ex.norm ? el("div", { class: "field norm" }, lbl, el("p", { class: "norm-plain" }, ex.norm)) : null;
  }
  return el("div", { class: "field norm" }, lbl,
    ...ex.refs.map((r, k) => el("details", { class: "ref", open: openFirst && k === 0 },
      el("summary", {}, el("span", { class: "ref-label" }, r.label),
        r.note ? el("span", { class: "ref-note" }, ` — ${r.note}`) : null, icon("chev", "chev")),
      el("blockquote", { class: "ref-quote" }, ...r.quote.map((q) => el("p", {}, q))),
      el("p", { class: "ref-src" }, `${r.doc} `,
        el("a", { href: r.url, target: "_blank", rel: "noopener" }, "Пункт на adilet.zan.kz", icon("ext"))))));
}

// что сервис нашёл в кадре и какие типы проверял: на чистом результате это главный ответ на «почему нет»
function renderCoverage(job, bad) {
  const sc = (job.meta || {}).scene || {};
  const found = [["стол", sc.desk], ["дверь", sc.door], ["сейф", sc.safe]];
  const off = [];
  if (!sc.safe) off.push("сейф не найден в кадре — открытый сейф не проверялся");
  if (sc.vlm && sc.vlm !== "ok") off.push("локальная VLM не работала — дверь и экран компьютера не проверялись");
  if (sc.safe && sc.safe_classifier && sc.safe_classifier !== "ok") off.push("классификатор сейфа не работал");
  const items = [
    el("p", {}, "В кадре найдены: ", found.filter((x) => x[1]).map((x) => x[0]).join(", ") || "зоны не найдены",
      sc.people_tracks != null ? `. Людей за ролик: ${sc.people_tracks}.` : "."),
  ];
  if (off.length) items.push(el("p", { class: "coverage-warn" }, `Ограничения этой проверки: ${off.join("; ")}.`));
  if (rules.length) {
    items.push(el("p", { class: "lbl" }, "Типы нарушений, которые распознаёт сервис"));
    items.push(el("ul", { class: "rules-list" }, ...rules.map((r) => el("li", {}, r.name))));
  }
  $("coverageBody").replaceChildren(...items);
  $("coverage").open = !bad;
  $("violations").classList.toggle("hidden", !bad);
}

// ---------- хронология ----------

const pct = (t) => `${Math.max(0, Math.min(100, (t / currentDur) * 100))}%`;

function renderChrono(tl) {
  const row = $("timeline");
  const lanes = $("lanes");
  const guides = $("guides");
  row.replaceChildren();
  lanes.replaceChildren();
  guides.querySelectorAll(".guide").forEach((g) => g.remove());
  $("chrono").closest(".chrono").classList.toggle("hidden", !currentDur);
  if (!currentDur) return;

  // нарушения; близкие отметки (ближе 4 % ширины) объединяются в одну
  const groups = [];
  const rowW = row.getBoundingClientRect().width || 600;
  marks.forEach((m, i) => {
    const g = groups[groups.length - 1];
    if (g && ((m.t - g.t) / currentDur) * rowW < 34) g.idx.push(i); else groups.push({ t: m.t, idx: [i] });
  });
  for (const g of groups) {
    const label = g.idx.length > 1 ? `${g.idx[0] + 1}–${g.idx[g.idx.length - 1] + 1}` : String(g.idx[0] + 1);
    const b = el("button", { type: "button", class: "mark", "data-i": g.idx[0],
      title: g.idx.map((i) => `${i + 1}: ${fmtTime(marks[i].t)}`).join(", "),
      "aria-label": `Нарушение ${label}, ${fmtTime(g.t)}`,
      onclick: (e) => { e.stopPropagation(); goViolation(g.idx[0]); } }, label);
    b.style.left = pct(g.t);
    row.append(b);
  }
  if (!marks.length) row.append(el("span", { class: "row-empty" }, "нет"));
  for (const m of marks) {
    const gl = el("div", { class: "guide" });
    gl.style.left = pct(m.t);
    guides.append(gl);
  }

  // факты сцены
  const seg = (a, b, cls, label, tip) => {
    const s = el("button", { type: "button", class: `seg ${cls}`, title: tip, tabindex: "-1", "aria-hidden": "true",
      onclick: (e) => { e.stopPropagation(); seek(a); } }, label || "");
    s.style.left = pct(a);
    s.style.width = `${Math.max(0.8, ((b - a) / currentDur) * 100)}%`;
    return s;
  };
  const lane = (title, items, empty) => {
    lanes.append(el("span", { class: "row-lbl" }, title),
      el("div", { class: "row-bar" }, ...items, items.length ? null : el("span", { class: "row-empty" }, empty)));
  };
  if (tl) {
    lane("Людей в помещении", (tl.people || []).filter((p) => p[2] > 0).map(([a, b, n]) =>
      seg(a, b, `p${Math.min(n, 3)}`, (b - a) / currentDur > 0.05 ? String(n) : "", `${n} чел. · ${fmtTime(a)}–${fmtTime(b)}`)), "никого");
    if (tl.safe_open) {
      lane("Сейф открыт", tl.safe_open.map(([a, b]) => seg(a, b, "safe", "", `сейф открыт · ${fmtTime(a)}–${fmtTime(b)}`)), "закрыт весь ролик");
    }
    lane("Деньги или сумка на столе", (tl.desk_items || []).map(([a, b]) =>
      seg(a, b, "desk", "", `на столе деньги или сумка · ${fmtTime(a)}–${fmtTime(b)}`)), "не видно");
    lane("Входы и уходы", (tl.events || []).map((e) => {
      const b = el("button", { type: "button", class: `tick ${e.kind}`, title: `${fmtTime(e.t)} — ${e.label}`,
        tabindex: "-1", "aria-hidden": "true", onclick: (ev) => { ev.stopPropagation(); seek(e.t); } });
      b.style.left = pct(e.t);
      return b;
    }), "нет");
  }
  $("chronoNote").classList.toggle("hidden", !tl);
  $("chronoHint").classList.toggle("hidden", !marks.length);
  $("chronoLegend").classList.toggle("hidden", !(tl && tl.events && tl.events.length));

  // ось: шаг делений подбирается так, чтобы подписи не налезали друг на друга
  const pxPerSec = rowW / currentDur;
  const step = [5, 10, 15, 20, 30, 60, 120].find((s) => s * pxPerSec >= 64) || 300;
  const ticks = [el("span", { class: "first" }, "00:00")];
  for (let t = step; (currentDur - t) * pxPerSec >= 48; t += step) {
    const s = el("span", {}, fmtClock(t));
    s.style.left = pct(t);
    ticks.push(s);
  }
  const last = el("span", { class: "last", id: "durLabel" }, fmtClock(currentDur));
  last.style.left = "100%";
  $("axis").replaceChildren(...ticks, last);
}

// клик по свободному месту дорожки — переход к этому моменту
$("chrono").addEventListener("click", (e) => {
  const bar = e.target.closest(".row-bar");
  if (!bar || !currentDur) return;
  const r = bar.getBoundingClientRect();
  seek(((e.clientX - r.left) / r.width) * currentDur, false);
});

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

let activeIdx = -1;
function highlight(i) {
  activeIdx = i;
  marks.forEach((m, k) => m.card.classList.toggle("active", k === i));
  document.querySelectorAll("#timeline .mark").forEach((b) => b.classList.toggle("active", Number(b.dataset.i) === i));
}

$("video").addEventListener("timeupdate", () => {
  const v = $("video");
  const d = currentDur || v.duration;
  if (!d) return;
  $("playhead").style.left = `${Math.min(100, (v.currentTime / d) * 100)}%`;
  if (!v.paused) {  // при воспроизведении подсвечивается нарушение, мимо которого идёт видео
    const k = marks.findIndex((m) => v.currentTime >= m.t - LEAD_SEC && v.currentTime <= m.t + 3);
    if (k >= 0 && k !== activeIdx) highlight(k);
  }
});

// ---------- просмотр кадра ----------

let viewer = { frames: [], k: 0, type: "", back: null };

function showFrame() {
  const f = viewer.frames[viewer.k];
  $("viewerImg").src = f.url;
  $("viewerImg").alt = `${viewer.type}, кадр ${fmtTime(f.t)}`;
  const many = viewer.frames.length > 1;
  $("viewerPrev").classList.toggle("hidden", !many);
  $("viewerNext").classList.toggle("hidden", !many);
  const count = many ? ` · ${viewer.k + 1} из ${viewer.frames.length}` : "";
  $("viewerCap").textContent = `${viewer.type} · ${fmtTime(f.t)}${count}`;
}

function openViewer(frames, k, type) {
  viewer = { frames, k, type, back: document.activeElement };
  showFrame();
  $("viewer").classList.remove("hidden");
  setInert(true);
  $("viewerClose").focus();
}

function setInert(on) {
  for (const sel of [".appbar", "main", ".foot"]) document.querySelector(sel).inert = on;
  document.body.classList.toggle("locked", on);
}

function closeViewer() {
  if ($("viewer").classList.contains("hidden")) return;
  $("viewer").classList.add("hidden");
  setInert(false);
  if (viewer.back && viewer.back.focus) viewer.back.focus();
}

function pageViewer(dir) {
  const n = viewer.frames.length;
  viewer.k = (viewer.k + dir + n) % n;
  showFrame();
}

$("viewerClose").addEventListener("click", closeViewer);
$("viewerPrev").addEventListener("click", () => pageViewer(-1));
$("viewerNext").addEventListener("click", () => pageViewer(1));
$("viewerSeek").addEventListener("click", () => { closeViewer(); seek(viewer.frames[viewer.k].t, false); });
$("viewer").addEventListener("click", (e) => { if (!e.target.closest("button, img")) closeViewer(); });

// ---------- клавиатура ----------

document.addEventListener("keydown", (e) => {
  if (!$("viewer").classList.contains("hidden")) {
    if (e.key === "Escape") closeViewer();
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") pageViewer(e.key === "ArrowRight" ? 1 : -1);
    if (e.key === "Tab") {  // фокус остаётся внутри окна просмотра
      const btns = [...$("viewer").querySelectorAll("button:not(.hidden)")];
      const first = btns[0], last = btns[btns.length - 1];
      if (document.activeElement === first && e.shiftKey) { e.preventDefault(); last.focus(); }
      else if (document.activeElement === last && !e.shiftKey) { e.preventDefault(); first.focus(); }
    }
    return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey || $("result").classList.contains("hidden")) return;
  if (e.code === "BracketRight" || e.code === "BracketLeft") {
    if (!marks.length) return;
    e.preventDefault();
    const dir = e.code === "BracketRight" ? 1 : -1;
    const i = activeIdx < 0 ? (dir > 0 ? 0 : marks.length - 1) : (activeIdx + dir + marks.length) % marks.length;
    goViolation(i);
  }
});

// ---------- история ----------

function outcomeEl(it) {
  if (it.outcome === "ошибка") return el("span", { class: "outcome err" }, "ошибка");
  if (!it.outcome) return el("span", { class: "outcome wait" }, "в обработке");
  if (it.outcome.startsWith("нарушен")) {
    const n = it.n ?? Number(it.outcome.replace(/\D+/g, ""));
    return el("span", { class: "outcome bad" }, n ? countViol(n) : it.outcome);
  }
  return el("span", { class: "outcome ok" }, "нарушений нет");
}

function renderHistory() {
  const items = loadHistory();
  $("historyPanel").classList.toggle("hidden", items.length === 0);
  $("historyBody").replaceChildren(...items.map((it) => {
    const name = it.filename || it.job_id;
    const body = [
      el("span", { class: "row-main", title: name }, name,
        it.created ? el("span", { class: "row-sub" }, ` · ${fmtDate(it.created)}`) : null),
      outcomeEl(it),
    ];
    if (it.job_id === currentJob) {
      return el("li", { "aria-current": "true" }, el("div", { class: "row-link current" }, ...body, el("span", { class: "row-chev cur-label" }, "открыта")));
    }
    return el("li", {}, el("button", { type: "button", class: "row-link", onclick: () => openJob(it.job_id, { push: true }) },
      ...body, icon("chev", "row-chev")));
  }));
}

// проверки, оставшиеся «в обработке» с прошлого визита, уточняются один раз
async function refreshPending() {
  for (const it of loadHistory().filter((x) => !x.outcome).slice(0, 5)) {
    try {
      const r = await fetch(`/ui/api/jobs/${encodeURIComponent(it.job_id)}`);
      if (r.status === 404) { upsertHistory({ job_id: it.job_id, outcome: "ошибка" }); continue; }
      const j = await r.json();
      if (j.state === "done") {
        upsertHistory({ job_id: it.job_id, n: j.violations.length,
          outcome: j.verdict === "violation" ? `нарушений: ${j.violations.length}` : "не обнаружено" });
      } else if (j.state === "failed") upsertHistory({ job_id: it.job_id, outcome: "ошибка" });
    } catch { /* сеть недоступна: оставляем как есть */ }
  }
}

function route() {
  const h = location.hash.match(/job=([0-9a-f]+)/);
  if (h) { if (h[1] !== currentJob) openJob(h[1]); return; }
  stopPolling();
  currentJob = null;
  setView("start");
  renderHistory();
}

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (lastJob && !$("result").classList.contains("hidden")) renderChrono(lastJob.timeline); }, 150);
});

checkHealth();
loadStartInfo();
renderHistory();
refreshPending();
window.addEventListener("popstate", route);
window.addEventListener("hashchange", route);
route();
