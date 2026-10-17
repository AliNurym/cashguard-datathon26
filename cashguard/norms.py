"""Дословные тексты пунктов НПА и ссылки на абзацы ИПС «Әділет» для карточек нарушений.

Источник — проверенные реестры docs/npa_registry.md и docs/npa_workstation_lock.md (тексты сверены
с adilet.zan.kz). Здесь ничего не набирается вручную: пункт берётся из реестра или не показывается.
"""
import re
from functools import lru_cache
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"

DOC_INFO = {
    "231": {"short": "Правила № 231", "url": "https://adilet.zan.kz/rus/docs/V1900019680",
            "title": "Правила осуществления кассовых операций и операций по инкассации… (пост. НБРК от 29.11.2019 № 231)"},
    "14": {"short": "Правила № 14", "url": "https://adilet.zan.kz/rus/docs/V2000020075",
           "title": "Правила организации охраны и устройства помещений банков… (пост. НБРК от 24.02.2020 № 14)"},
    "53": {"short": "Требования № 53", "url": "https://adilet.zan.kz/rus/docs/V2600038336",
           "title": "Требования к обеспечению информационной безопасности банков… (пост. АРРФР от 03.04.2026 № 53)"},
}


def _quote_lines(block: str) -> list[str]:
    """Абзацы первой цитаты (строки с '>'): каждая непустая строка — абзац.
    Служебные пометки ИПС («Сноска.», «Примечание ИЗПИ!») в цитату не берутся."""
    paras, started = [], False
    for line in block.splitlines():
        if line.startswith(">"):
            started = True
            text = line[1:].strip()
            if text and not text.startswith(("Сноска.", "Примечание ИЗПИ")):
                paras.append(text)
        elif started and line.strip():
            break
    return paras


@lru_cache(maxsize=1)
def load_points() -> dict:
    """{(doc, point): {"url": str, "paras": [str]}}"""
    out = {}
    reg = DOCS / "npa_registry.md"
    if reg.exists():
        t = reg.read_text(encoding="utf-8")
        heads = list(re.finditer(r"^#### п\. ([\d\-]+) — (?:Правила|Требования) № (\d+)", t, re.M))
        for i, m in enumerate(heads):
            block = t[m.end(): heads[i + 1].start() if i + 1 < len(heads) else len(t)]
            src = re.search(r"\*\*Источник:\*\*.*?\((https://adilet\.zan\.kz/[^)\s]+)\)", block)
            body = block[block.find("**Текст дословно:**"):] if "**Текст дословно:**" in block else block
            paras = _quote_lines(body)
            if paras and (m.group(2), m.group(1)) not in out:
                out[(m.group(2), m.group(1))] = {"url": src.group(1) if src else DOC_INFO.get(m.group(2), {}).get("url"),
                                                 "paras": paras}
    ws = DOCS / "npa_workstation_lock.md"
    if ws.exists():  # п. 26 Требований № 53 (ответственность работника за ИБ)
        t = ws.read_text(encoding="utf-8")
        i = t.find("**п. 26**")
        if i >= 0:
            paras = _quote_lines(t[i:].split("\n", 1)[1])
            if paras:
                out[("53", "26")] = {"url": DOC_INFO["53"]["url"], "paras": paras}
    return out


def ref(doc: str, point: str, part: str | None = None, direct: bool = True, note: str = "") -> dict | None:
    """Ссылка с цитатой. part: 'пп. N' — подпункт, 'ч. N' — N-й абзац пункта, None — весь пункт."""
    p = load_points().get((doc, point))
    if not p:
        return None
    paras = p["paras"]
    quote = paras
    if part:
        kind, n = part.split()
        if kind == "пп.":
            sub = [x for x in paras if x.startswith(f"{n})")]
            lead = next((x for x in paras if x.startswith(f"{point}.")), paras[0] if paras else "")
            quote = ([lead] if lead and lead not in sub else []) + sub if sub else paras
        elif kind == "ч." and n.isdigit() and int(n) <= len(paras):
            quote = [paras[int(n) - 1]]
    label = f"{DOC_INFO[doc]['short']}, п. {point}" + (f" {part}" if part else "")
    return {"label": label, "doc": DOC_INFO[doc]["title"], "url": p["url"], "quote": quote,
            "direct": direct, "note": note}
