"""Скачать веса всех локальных моделей (один раз, дальше сервис работает офлайн).

    python scripts/download_models.py [--vlm Qwen/Qwen3-VL-4B-Instruct]
"""
import argparse
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cashguard.detectors import OWL_ID, RTDETR_ID  # noqa: E402
from cashguard.embed import DINO_ID  # noqa: E402
from cashguard.vlm import VLM_ID  # noqa: E402


# только то, что нужно transformers (без TF/Flax/ONNX-копий весов)
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "tokenizer*", "vocab*", "merges*"]


def fetch(repo, tries=30):
    for i in range(tries):
        try:
            p = snapshot_download(repo, max_workers=4, etag_timeout=60, allow_patterns=PATTERNS)
            print(f"OK {repo} -> {p}", flush=True)
            return p
        except Exception as e:  # сеть нестабильна: докачиваем с места обрыва
            print(f"retry {i + 1} {repo}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(5)
    raise SystemExit(f"FAILED {repo}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default=VLM_ID)
    a = ap.parse_args()
    for repo in (a.vlm, RTDETR_ID, OWL_ID, DINO_ID):
        fetch(repo)
