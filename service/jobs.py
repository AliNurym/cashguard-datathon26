"""Задачи: хранение (память + диск) и диспетчер одного воркер-процесса.

Статусы внутри: queued → processing → done | failed. В API queued отдаётся как processing.
Каждая задача — data/service/jobs/<job_id>.json; после перезапуска незавершённые задачи
снова встают в очередь в порядке загрузки (не больше MAX_ATTEMPTS попыток на задачу).
"""
import json
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path

from .worker import worker_main

log = logging.getLogger("cashguard.service")

MAX_ATTEMPTS = 2
LIMIT_SEC = 300  # ТЗ: результат не позже 5 минут после загрузки


class JobStore:
    def __init__(self, root: Path):
        self.dir = root / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        for p in self.dir.glob("*.json"):
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
                self.jobs[j["job_id"]] = j
            except (OSError, ValueError, KeyError):
                log.warning("skip unreadable job file %s", p.name)

    def _save(self, j):
        p = self.dir / f"{j['job_id']}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)

    def add(self, job_id, source, filename, video, size) -> dict:
        j = {"job_id": job_id, "source": source, "filename": filename, "video": str(video), "size": size,
             "status": "queued", "created_at": time.time(), "started_at": None, "finished_at": None,
             "attempts": 0, "result": None, "error": None}
        with self.lock:
            self.jobs[job_id] = j
            self._save(j)
        return j

    def get(self, job_id) -> dict | None:
        with self.lock:
            j = self.jobs.get(job_id)
            return dict(j) if j else None

    def update(self, job_id, **kw) -> dict | None:
        with self.lock:
            j = self.jobs.get(job_id)
            if j is None:
                return None
            j.update(kw)
            self._save(j)
            return dict(j)

    def unfinished(self) -> list[dict]:
        with self.lock:
            js = [dict(j) for j in self.jobs.values() if j["status"] in ("queued", "processing")]
        return sorted(js, key=lambda j: j["created_at"])


class Dispatcher:
    """Один воркер-процесс, задачи строго по очереди. Следит за падением и зависанием воркера."""

    def __init__(self, store: JobStore, log_file: str, timeout_sec: float, keep_api_videos: bool):
        self.store = store
        self.log_file = log_file
        self.timeout = timeout_sec
        self.keep_api_videos = keep_api_videos
        self.ctx = mp.get_context("spawn")
        self.pending: deque[str] = deque()
        self.lock = threading.Lock()
        self.stop_ev = threading.Event()
        self.proc = None
        self.tasks = self.events = None
        self.state = "down"           # down | loading | ready
        self.info: dict = {}
        self.current: str | None = None
        self.current_started = 0.0
        self.fails = 0                # падений подряд до готовности
        self.next_spawn = 0.0
        self.thread = threading.Thread(target=self._loop, name="dispatcher", daemon=True)

    # --- публичное ---
    def start(self):
        for j in self.store.unfinished():
            if j["attempts"] >= MAX_ATTEMPTS:
                self._fail(j["job_id"], "обработка прервалась несколько раз (перезапуск сервиса)")
                continue
            self.store.update(j["job_id"], status="queued")
            self.pending.append(j["job_id"])
        if self.pending:
            log.info("requeued %d unfinished jobs after restart", len(self.pending))
        self.thread.start()

    def stop(self):
        self.stop_ev.set()
        self.thread.join(timeout=5)
        if self.proc is not None and self.proc.is_alive():
            try:
                self.tasks.put(None)
            except (OSError, ValueError):
                pass
            self.proc.join(timeout=3)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=3)

    def submit(self, job_id: str):
        with self.lock:
            self.pending.append(job_id)
        log.info("job %s queued (queue length %d)", job_id, len(self.pending))

    def position(self, job_id: str) -> int | None:
        """Сколько задач будет обработано раньше этой (0 — следующая)."""
        with self.lock:
            if job_id not in self.pending:
                return None
            return list(self.pending).index(job_id) + (1 if self.current else 0)

    def health(self) -> dict:
        with self.lock:
            n = len(self.pending)
        return {"worker": self.state, "device": self.info.get("device"), "queue": n,
                "busy": self.current is not None}

    # --- цикл диспетчера ---
    def _spawn(self):
        self.tasks, self.events = self.ctx.Queue(), self.ctx.Queue()
        self.proc = self.ctx.Process(target=worker_main, args=(self.tasks, self.events, self.log_file),
                                     name="cashguard-worker", daemon=True)
        self.proc.start()
        self.state = "loading"
        log.info("worker started, pid %s, loading models", self.proc.pid)

    def _loop(self):
        while not self.stop_ev.is_set():
            try:
                self._tick()
            except Exception:  # noqa: BLE001 — диспетчер не должен умирать
                log.exception("dispatcher error")
                time.sleep(1)

    def _tick(self):
        if self.proc is None or not self.proc.is_alive():
            if self.proc is not None:
                self._on_worker_exit()
            if time.time() < self.next_spawn:
                time.sleep(0.2)
                return
            self._spawn()
        if self.state == "ready" and self.current is None:
            with self.lock:
                jid = self.pending.popleft() if self.pending else None
            if jid:
                self._dispatch(jid)
        try:
            ev = self.events.get(timeout=0.5)
        except queue.Empty:
            ev = None
        if ev is not None:
            self._handle(ev)
        if self.current and time.time() - self.current_started > self.timeout:
            jid = self.current
            log.error("job %s exceeded %.0fs, killing worker", jid, self.timeout)
            self.current = None
            self.proc.kill()
            self.proc.join(timeout=5)
            self._fail(jid, f"превышен лимит обработки {self.timeout:.0f} с")
            self.state = "down"
            self.proc = None

    def _dispatch(self, jid):
        j = self.store.get(jid)
        if j is None or j["status"] not in ("queued", "processing"):
            return
        now = time.time()
        self.store.update(jid, status="processing", started_at=now, attempts=j["attempts"] + 1)
        self.current, self.current_started = jid, now
        self.tasks.put((jid, j["video"]))
        log.info("job %s started (waited in queue %.1fs)", jid, now - j["created_at"])

    def _handle(self, ev):
        kind = ev[0]
        if kind == "ready":
            self.state, self.info, self.fails = "ready", ev[1], 0
            log.info("worker ready on %s (models %.1fs)", ev[1]["device"], ev[1]["load_sec"])
        elif kind == "done":
            jid, res = ev[1], ev[2]
            if jid == self.current:
                self.current = None
            self._finish(jid, "done", result=res)
        elif kind == "failed":
            jid = ev[1]
            if jid == self.current:
                self.current = None
            self._fail(jid, ev[2])

    def _on_worker_exit(self):
        # события, успевшие прийти до смерти процесса
        while True:
            try:
                self._handle(self.events.get_nowait())
            except (queue.Empty, OSError, ValueError, EOFError):
                break
        code = self.proc.exitcode
        log.error("worker exited with code %s (state %s)", code, self.state)
        if self.current:
            jid, self.current = self.current, None
            j = self.store.get(jid)
            if j and j["attempts"] < MAX_ATTEMPTS:
                self.store.update(jid, status="queued")
                with self.lock:
                    self.pending.appendleft(jid)
                log.warning("job %s requeued after worker crash", jid)
            else:
                self._fail(jid, "воркер аварийно завершился при обработке видео")
        if self.state != "ready":
            self.fails += 1
        delay = min(60, 5 * 2 ** max(0, self.fails - 1)) if self.fails else 1
        self.next_spawn = time.time() + delay
        self.state = "down"
        self.proc = None

    def _fail(self, jid, error):
        self._finish(jid, "failed", error=error)

    def _finish(self, jid, status, result=None, error=None):
        j = self.store.get(jid)
        if j is None:
            return
        now = time.time()
        self.store.update(jid, status=status, finished_at=now, result=result, error=error)
        total = now - j["created_at"]
        proc = now - j["started_at"] if j["started_at"] else 0.0
        verdict = result["verdict"] if result else "-"
        n = len(result["violations"]) if result else 0
        msg = "job %s %s: verdict %s, %d violations, processing %.1fs, total since upload %.1fs"
        args = (jid, status, verdict, n, proc, total)
        if error:
            msg += ", error: %s"
            args += (error,)
        (log.warning if total > LIMIT_SEC or status == "failed" else log.info)(msg, *args)
        if total > LIMIT_SEC:
            log.warning("job %s exceeded the 5-minute limit (%.1fs)", jid, total)
        if j["source"] == "api" and not self.keep_api_videos:
            try:
                Path(j["video"]).unlink(missing_ok=True)
            except OSError:
                log.warning("could not delete %s", j["video"])
