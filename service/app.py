"""HTTP-сервис CashGuard: асинхронный API для жюри (X-API-Key) и веб-интерфейс.

    CASHGUARD_API_KEY=... uvicorn service.app:app --host 0.0.0.0 --port 8000

Запускать одним процессом uvicorn (без --workers и --reload): задачи обрабатывает
один воркер-процесс, которым управляет этот процесс.
"""
import os
import secrets
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles

from cashguard.registry import ACTIVE, REGISTRY, refs

from .jobs import Dispatcher, JobStore
from .logs import setup_logging

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
DATA = Path(os.environ.get("CASHGUARD_DATA_DIR", ROOT / "data" / "service"))
API_KEY = os.environ.get("CASHGUARD_API_KEY", "")
TIMEOUT_SEC = float(os.environ.get("CASHGUARD_JOB_TIMEOUT", "300"))
MAX_UPLOAD = int(float(os.environ.get("CASHGUARD_MAX_UPLOAD_MB", "200")) * 1024 * 1024)
KEEP_API_VIDEOS = os.environ.get("CASHGUARD_KEEP_API_VIDEOS", "0") == "1"
# готовые проверки, которые интерфейс показывает как примеры (job_id через запятую)
EXAMPLES = [x.strip() for x in os.environ.get("CASHGUARD_EXAMPLES", "").split(",") if x.strip()]

# названия типов из прежних версий: старые задачи в интерфейсе показываются с текущим названием
NAME2CODE = {r["name"]: code for code, r in REGISTRY.items()}
NAME2CODE["Наличные предыдущей операции открыто на столе"] = "C_cash_on_desk_next_client"

UPLOADS = DATA / "uploads"
LOG_FILE = DATA / "service.log"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY:
        raise RuntimeError("Задайте ключ API: CASHGUARD_API_KEY=<ключ> uvicorn service.app:app ...")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    setup_logging(str(LOG_FILE), "server")
    store = JobStore(DATA)
    disp = Dispatcher(store, str(LOG_FILE), TIMEOUT_SEC, KEEP_API_VIDEOS)
    disp.start()
    app.state.store, app.state.disp = store, disp
    yield
    disp.stop()


app = FastAPI(title="CashGuard API", version="1.0",
              description="Поиск признаков нарушений кассовой дисциплины на видео. Все модели локальные.",
              lifespan=lifespan)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False,
                              description="Ключ доступа к API (выдаётся жюри)")


def key_ok(key: str | None) -> bool:
    return bool(key) and secrets.compare_digest(key.encode(), API_KEY.encode())


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    """Ключ проверяется до чтения тела запроса: без ключа видео даже не загружается."""
    path = request.url.path
    if path.startswith("/api/") and path != "/api/v1/health" and not key_ok(request.headers.get("X-API-Key")):
        return JSONResponse({"detail": "Invalid or missing API key"}, status_code=401,
                            headers={"WWW-Authenticate": "X-API-Key"})
    return await call_next(request)


def require_key(key: str | None = Security(api_key_header)):
    # дублирует middleware; нужно, чтобы /docs показывал авторизацию по X-API-Key
    if not key_ok(key):
        raise HTTPException(401, "Invalid or missing API key")


def save_upload(request: Request, file: UploadFile, source: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    name = Path(file.filename or "video.mov").name
    ext = Path(name).suffix.lower() or ".mov"
    dst = UPLOADS / f"{job_id}{ext}"
    size = 0
    with open(dst, "wb") as out:
        while chunk := file.file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                out.close()
                dst.unlink(missing_ok=True)
                raise HTTPException(413, f"Файл больше {MAX_UPLOAD // (1024 * 1024)} МБ")
            out.write(chunk)
    if size == 0:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, "Пустой файл: передайте видео в поле file")
    request.app.state.store.add(job_id, source, name, dst, size)
    request.app.state.disp.submit(job_id)
    return job_id


def api_view(j: dict) -> dict:
    """Ответ по схеме ТЗ (раздел 10)."""
    status = "processing" if j["status"] in ("queued", "processing") else j["status"]
    out = {"job_id": j["job_id"], "status": status, "verdict": None, "violations": []}
    if status == "done":
        out["verdict"] = j["result"]["verdict"]
        out["violations"] = j["result"]["violations"]
    elif status == "failed":
        out["error"] = j["error"]
    return out


# ---------------- API для жюри ----------------

@app.post("/api/v1/analyze", dependencies=[Security(require_key)], tags=["api"],
          summary="Загрузить видео (.mov, поле file); в ответ job_id")
def analyze(request: Request, file: UploadFile = File(..., description="Видео .mov")):
    return {"job_id": save_upload(request, file, "api")}


@app.get("/api/v1/jobs/{job_id}", dependencies=[Security(require_key)], tags=["api"],
         summary="Статус (processing, done, failed) и результат")
def job_status(request: Request, job_id: str):
    j = request.app.state.store.get(job_id)
    if j is None:
        raise HTTPException(404, "Задача не найдена")
    return api_view(j)


@app.get("/api/v1/health", tags=["api"], summary="Состояние сервиса (без ключа)")
def health(request: Request):
    return {"status": "ok", **request.app.state.disp.health()}


# ---------------- веб-интерфейс (без авторизации) ----------------
# Видны только задачи, загруженные через интерфейс: результаты и видео жюри сюда не попадают.

def ui_job(request: Request, job_id: str) -> dict:
    j = request.app.state.store.get(job_id)
    if j is None or j["source"] != "ui":
        raise HTTPException(404, "Задача не найдена")
    return j


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.post("/ui/api/analyze", include_in_schema=False)
def ui_analyze(request: Request, file: UploadFile = File(...)):
    return {"job_id": save_upload(request, file, "ui")}


@app.get("/ui/api/jobs/{job_id}", include_in_schema=False)
def ui_status(request: Request, job_id: str):
    j = ui_job(request, job_id)
    out = api_view(j)
    now = time.time()
    res = j["result"] or {}
    out.update({
        "state": j["status"],
        "filename": j["filename"],
        "created_at": j["created_at"],
        "queue_position": request.app.state.disp.position(job_id),
        "elapsed_sec": round((j["finished_at"] or now) - j["started_at"], 1) if j["started_at"] else 0.0,
        "total_sec": round((j["finished_at"] or now) - j["created_at"], 1),
        "explain": res.get("explain", []),
        "meta": res.get("meta", {}),
        "timeline": res.get("timeline"),
        "video_url": f"/ui/video/{job_id}" if Path(j["video"]).exists() else None,
    })
    return ui_normalize(out)


def ui_normalize(out: dict) -> dict:
    """Задачи, обработанные прежней версией: текущее название типа и нормы с цитатами (копии, хранилище не меняется)."""
    out["violations"] = [dict(v) for v in out.get("violations") or []]
    out["explain"] = [dict(x) for x in out.get("explain") or []]
    for i, v in enumerate(out["violations"]):
        code = NAME2CODE.get(v.get("type"))
        if not code:
            continue
        v["type"] = REGISTRY[code]["name"]
        if i < len(out["explain"]) and not out["explain"][i].get("refs"):
            out["explain"][i]["refs"] = refs(code)
    return out


@app.get("/ui/api/rules", include_in_schema=False)
def ui_rules():
    return [{"code": code[0], "name": REGISTRY[code]["name"]} for code in ACTIVE]


@app.get("/ui/api/examples", include_in_schema=False)
def ui_examples(request: Request):
    out = []
    for job_id in EXAMPLES:
        j = request.app.state.store.get(job_id)
        if not j or j["source"] != "ui" or j["status"] != "done":
            continue
        res = j["result"] or {}
        out.append({"job_id": job_id, "filename": j["filename"], "verdict": res.get("verdict"),
                    "n": len(res.get("violations") or []),
                    "types": [REGISTRY[NAME2CODE[v["type"]]]["name"] if v["type"] in NAME2CODE else v["type"]
                              for v in res.get("violations") or []],
                    "duration_sec": (res.get("meta") or {}).get("duration_sec")})
    return out


@app.get("/ui/video/{job_id}", include_in_schema=False)
def ui_video(request: Request, job_id: str):
    j = ui_job(request, job_id)
    p = Path(j["video"])
    if not p.exists():
        raise HTTPException(404, "Видео удалено")
    # H.264 в .mov браузеры воспроизводят, если отдать его как video/mp4
    media = "video/mp4" if p.suffix in (".mov", ".mp4", ".m4v") else None
    return FileResponse(p, media_type=media)
