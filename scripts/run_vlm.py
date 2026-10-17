"""Прогон VLM в моменты событий сцены; ответы кэшируются в data/vlm_cache/<vid>.json.

    python scripts/run_vlm.py [--only X-20,X-2] [--safe-every 3] [--count-every 2]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cashguard.labels import load_videos  # noqa: E402
from cashguard.machine import box_from_norm, roi_activity  # noqa: E402
from cashguard.scene import CACHE, build_scene  # noqa: E402
from cashguard.video import read_frame  # noqa: E402
from cashguard.vlm_plan import plan_queries  # noqa: E402
from cashguard.vlm import VLM, VLMCache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--safe-every", type=float, default=3.0)
    ap.add_argument("--count-every", type=float, default=2.0)
    ap.add_argument("--qids", default="", help="только эти вопросы, напр. laptop")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    vlm = VLM()
    for v in load_videos():
        if (only and v.vid not in only) or not (CACHE / f"{v.vid}.json").exists():
            continue
        sc = build_scene(v.vid)
        cache = VLMCache(v.vid)
        t0 = time.time()
        n = 0
        for qid, t in plan_queries(sc, a.safe_every, a.count_every):
            if a.qids and qid not in a.qids.split(","):
                continue
            if cache.get(qid, t) is not None:
                continue
            fr = read_frame(v.path, t)
            if fr is None:
                continue
            cache.put(qid, t, vlm.ask_q([fr], qid, desk=sc.desk, safe=sc.safe, laptop=sc.laptop, door=sc.door))
            n += 1
        # уточнение начала интервала «сейф открыт, у сейфа никого»
        for qid, t in plan_queries(sc, a.safe_every, a.count_every, cache=cache, refine=True):
            if cache.get(qid, t) is None:
                fr = read_frame(v.path, t)
                if fr is not None:
                    cache.put(qid, t, vlm.ask_q([fr], qid, desk=sc.desk, safe=sc.safe, laptop=sc.laptop, door=sc.door))
                    n += 1
        # активность в зоне счётной машины (рамка — от VLM, движение — по пикселям)
        m = cache.get("machine", 0.5)
        if m and m.get("boxes") and "_machine_activity" not in cache.data:
            box = box_from_norm(m["boxes"][0], sc.W, sc.H)
            cache.data["_machine_box"] = box
            cache.data["_machine_activity"] = roi_activity(v.path, box)
        cache.save()
        print(f"{v.vid}: {n} запросов за {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
