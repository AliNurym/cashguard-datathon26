"""Воркер-процесс: грузит модели один раз и обрабатывает задачи по одной.

Запускается диспетчером (service/jobs.py) через multiprocessing (spawn). Протокол:
    tasks:  (job_id, video_path) | None (остановка)
    events: ("ready", info) | ("started", job_id, t) | ("done", job_id, result) | ("failed", job_id, error)
"""
import logging
import os
import time


def worker_main(tasks, events, log_file: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # веса уже в кэше, сеть не нужна
    os.environ.pop("CASHGUARD_VARIANT", None)     # сервис работает с оригинальными видео
    from service.logs import setup_logging
    setup_logging(log_file, "worker")
    log = logging.getLogger("cashguard.worker")

    from cashguard.pipeline import Analyzer
    an = Analyzer()
    an.warmup()
    events.put(("ready", {"device": an.device, "load_sec": an.load_sec, "pid": os.getpid()}))
    while True:
        task = tasks.get()
        if task is None:
            break
        job_id, path = task
        events.put(("started", job_id, time.time()))
        try:
            events.put(("done", job_id, an.run(path, job_id=job_id)))
        except Exception as e:  # noqa: BLE001 — любая ошибка видео не должна ронять воркер
            log.exception("job %s failed", job_id)
            events.put(("failed", job_id, f"{type(e).__name__}: {e}"))
