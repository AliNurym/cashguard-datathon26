"""Сквозная проверка сервиса через API жюри (ТЗ, раздел 10) на всех уникальных роликах.

    python scripts/api_e2e.py --mode seq   [--base http://127.0.0.1:8000] [--key-file data/service/dev_api_key.txt]
    python scripts/api_e2e.py --mode burst  # все ролики отправляются разом, как может сделать жюри

Проверяет: 401 без ключа и с неверным ключом, отказ на не-видео, схему JSON ответа (типы полей,
verdict, 1–3 JPEG-кадра в base64), время от отправки до результата (лимит ТЗ — 5 минут).
Пишет runs/api_<mode>.json (предсказания для scripts/evaluate.py) и runs/api_<mode>_report.json.
"""
import argparse
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cashguard.labels import load_videos  # noqa: E402

LIMIT = 300.0


def check_schema(res: dict) -> list:
    """Список нарушений схемы ТЗ (пусто — всё верно)."""
    err = []
    for k in ("job_id", "status", "verdict", "violations"):
        if k not in res:
            err.append(f"нет поля {k}")
    if res.get("status") != "done":
        err.append(f"status={res.get('status')}")
    if res.get("verdict") not in ("violation", "no_violation"):
        err.append(f"verdict={res.get('verdict')}")
    vs = res.get("violations", [])
    if not isinstance(vs, list):
        return err + ["violations не список"]
    if (res.get("verdict") == "violation") != bool(vs):
        err.append("verdict не согласован с violations")
    for i, v in enumerate(vs):
        for k, typ in (("type", str), ("timecode_sec", (int, float)), ("description", str), ("evidence_frames", list)):
            if not isinstance(v.get(k), typ):
                err.append(f"v{i}.{k}: {type(v.get(k)).__name__}")
        fr = v.get("evidence_frames", [])
        if not 1 <= len(fr) <= 3:
            err.append(f"v{i}: кадров {len(fr)}")
        for j, f in enumerate(fr):
            if not isinstance(f.get("timestamp_sec"), (int, float)):
                err.append(f"v{i}.f{j}.timestamp_sec")
            try:
                raw = base64.b64decode(f.get("image_base64", ""), validate=True)
                if raw[:3] != b"\xff\xd8\xff":
                    err.append(f"v{i}.f{j}: не JPEG")
            except Exception:
                err.append(f"v{i}.f{j}: base64 не декодируется")
    return err


def run_one(base, key, v, t_submit=None):
    t0 = t_submit or time.time()
    with open(v.path, "rb") as f:
        r = requests.post(f"{base}/api/v1/analyze", headers={"X-API-Key": key},
                          files={"file": (v.path.name, f, "video/quicktime")}, timeout=120)
    if r.status_code != 200:
        return {"vid": v.vid, "error": f"POST {r.status_code} {r.text[:200]}"}
    job = r.json()["job_id"]
    res = {}
    while time.time() - t0 < LIMIT + 60:
        g = requests.get(f"{base}/api/v1/jobs/{job}", headers={"X-API-Key": key}, timeout=30)
        res = g.json()
        if res.get("status") in ("done", "failed"):
            break
        time.sleep(2)
    lat = time.time() - t0
    return {"vid": v.vid, "job_id": job, "latency": round(lat, 1), "status": res.get("status"),
            "schema_errors": check_schema(res) if res.get("status") == "done" else ["не done"],
            "result": {k: res.get(k) for k in ("verdict", "violations")}}


def security_checks(base, key):
    out = {}
    v = load_videos()[0]
    with open(v.path, "rb") as f:
        out["post_no_key"] = requests.post(f"{base}/api/v1/analyze", files={"file": f}, timeout=60).status_code
    with open(v.path, "rb") as f:
        out["post_bad_key"] = requests.post(f"{base}/api/v1/analyze", headers={"X-API-Key": "wrong"},
                                            files={"file": f}, timeout=60).status_code
    out["get_no_key"] = requests.get(f"{base}/api/v1/jobs/whatever", timeout=30).status_code
    out["get_unknown_job"] = requests.get(f"{base}/api/v1/jobs/nope", headers={"X-API-Key": key}, timeout=30).status_code
    r = requests.post(f"{base}/api/v1/analyze", headers={"X-API-Key": key},
                      files={"file": ("note.txt", b"not a video", "text/plain")}, timeout=60)
    out["post_not_video"] = r.status_code
    if r.status_code == 200:  # принято в очередь — должно закончиться failed, а не повесить воркер
        job = r.json()["job_id"]
        for _ in range(60):
            s = requests.get(f"{base}/api/v1/jobs/{job}", headers={"X-API-Key": key}, timeout=30).json()
            if s.get("status") in ("done", "failed"):
                break
            time.sleep(1)
        out["not_video_job_status"] = s.get("status")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--key-file", default=str(ROOT / "data" / "service" / "dev_api_key.txt"))
    ap.add_argument("--mode", choices=["seq", "burst"], default="seq")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    key = Path(a.key_file).read_text().strip()
    vids = [v for v in load_videos() if not a.only or v.vid in a.only.split(",")]
    report = {"security": security_checks(a.base, key)}
    print("security:", report["security"], flush=True)
    t_all = time.time()
    if a.mode == "seq":
        rows = []
        for v in vids:
            rows.append(run_one(a.base, key, v))
            r = rows[-1]
            print(f"{v.vid:5} {r.get('status')} {r.get('latency')}s verdict={r.get('result', {}).get('verdict')} "
                  f"n={len((r.get('result') or {}).get('violations') or [])} schema={r.get('schema_errors') or r.get('error')}",
                  flush=True)
    else:
        t_submit = time.time()
        with ThreadPoolExecutor(max_workers=len(vids)) as ex:
            rows = list(ex.map(lambda v: run_one(a.base, key, v, t_submit), vids))
    report["rows"] = rows
    lat = [r["latency"] for r in rows if "latency" in r]
    report["summary"] = {
        "videos": len(rows), "done": sum(r.get("status") == "done" for r in rows),
        "failed": sum(r.get("status") == "failed" for r in rows),
        "errors": [r for r in rows if r.get("error")],
        "schema_bad": [r["vid"] for r in rows if r.get("schema_errors")],
        "over_limit": [r["vid"] for r in rows if r.get("latency", 0) > LIMIT],
        "latency_max": max(lat) if lat else None, "latency_median": sorted(lat)[len(lat) // 2] if lat else None,
        "wall_sec": round(time.time() - t_all, 1),
    }
    print("summary:", json.dumps(report["summary"], ensure_ascii=False), flush=True)
    preds = {r["vid"]: r["result"] for r in rows if r.get("status") == "done"}
    (ROOT / "runs" / f"api_{a.mode}.json").write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")
    (ROOT / "runs" / f"api_{a.mode}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
