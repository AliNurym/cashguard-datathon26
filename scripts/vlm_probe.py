"""Проверка надёжности VLM на кадрах с известным ответом (из ручного просмотра).

    python scripts/vlm_probe.py [--model Qwen/Qwen3-VL-4B-Instruct]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cashguard.scene import build_scene  # noqa: E402
from cashguard.video import read_frame, video_path  # noqa: E402
from cashguard.vlm import crop_region, region_for  # noqa: E402
from cashguard.vlm import VLM, VLM_ID  # noqa: E402

# (видео, t, вопрос, ожидаемые значения полей)
CASES = [
    ("X-20", 7.0, "safe", {"safe_open": True}),
    ("X-20", 10.0, "safe", {"safe_open": True, "person_at_safe": False}),
    ("X-13", 49.5, "safe", {"safe_open": True}),
    ("X-3", 5.0, "safe", {"safe_open": False}),
    ("X-12", 20.0, "safe", {"safe_open": False}),
    ("X-11", 5.0, "safe", {"safe_open": False}),
    ("X-29", 14.0, "safe", {"safe_open": False}),
    ("X-26", 13.0, "desk", {"cash_on_desk": True}),
    ("X-38", 30.0, "desk", {"cash_on_desk": True}),
    ("X-4", 55.0, "desk", {"cash_on_desk": True}),
    ("X-15", 18.0, "desk", {"bag_on_desk": True}),
    ("X-23", 16.0, "desk", {"cash_on_desk": False, "bag_on_desk": False}),
    ("X-29", 14.0, "desk", {"cash_on_desk": False, "bag_on_desk": False}),
    ("X-28", 12.0, "desk", {"cash_on_desk": False}),
    ("X-2", 12.0, "count", {"cash_in_hands": True, "hand_at_counting_machine": False}),
    ("X-4", 68.0, "count", {"cash_in_hands": True, "hand_at_counting_machine": False}),
    ("X-3", 16.5, "count", {"hand_at_counting_machine": True}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=VLM_ID)
    ap.add_argument("--full", action="store_true", help="весь кадр без кропа")
    ap.add_argument("--save", action="store_true", help="сохранить кропы в data/debug/vlm_*")
    a = ap.parse_args()
    t0 = time.time()
    vlm = VLM(a.model)
    print(f"load {time.time() - t0:.1f}s on {vlm.dev}")
    ok = total = 0
    for vid, t, qid, exp in CASES:
        fr = read_frame(video_path(vid), t)
        sc = build_scene(vid)
        desk, safe = (None, None) if a.full else (sc.desk, sc.safe)
        if a.save and region_for(qid, desk, safe):
            import cv2
            cv2.imwrite(f"data/debug/vlm_{vid}_{t}_{qid}.jpg", crop_region(fr, **region_for(qid, desk, safe)))
        ans = vlm.ask_q([fr], qid, desk=desk, safe=safe)
        res = []
        for k, v in exp.items():
            got = ans.get(k)
            total += 1
            ok += got == v
            res.append(f"{k}={got}{'' if got == v else ' ✗(exp ' + str(v) + ')'}")
        print(f"{vid:5} {t:5.1f} {qid:5} {ans['_sec']:5.2f}s  " + ", ".join(res))
    print(f"\nверно {ok}/{total}")


if __name__ == "__main__":
    main()
