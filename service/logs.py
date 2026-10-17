import logging
import sys


def setup_logging(log_file: str, proc: str):
    """Логгер cashguard.*: stdout + общий файл (сервер и воркер пишут в один)."""
    root = logging.getLogger("cashguard")
    if root.handlers:
        return
    fmt = logging.Formatter(f"%(asctime)s {proc:6} %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    root.setLevel(logging.INFO)
    root.propagate = False
