"""Глоссарий и пост-правки перевода для дубляжа (еда / мемные повторы)."""

from __future__ import annotations

import re

# EN → предпочтительный RU (подсказка LLM + пост-фикс)
FOOD_GLOSSARY_EN_RU: dict[str, str] = {
    "pickle": "маринованный огурец",
    "pickles": "маринованные огурцы",
    "burger": "бургер",
    "butter": "масло",
    "bun": "булочка",
    "buns": "булочки",
    "patty": "котлета",
    "patties": "котлеты",
    "beef": "говядина",
    "cheese": "сыр",
    "onion": "лук",
    "onions": "лук",
    "lettuce": "салат",
    "tomato": "помидор",
    "tomatoes": "помидоры",
    "sauce": "соус",
    "mayo": "майонез",
    "mayonnaise": "майонез",
    "ketchup": "кетчуп",
    "mustard": "горчица",
    "sesame": "кунжут",
    "grill": "гриль",
    "griddle": "сковорода",
    "handful": "горсть",
    "handfuls": "горсти",
}

# Типичные галлюцинации / путаница LLM при дубляже рецептов
_SOURCE_FIXES: list[tuple[re.Pattern[str], re.Pattern[str], str]] = [
    # source has pickle*, dest has щекоч* → огурцы
    (
        re.compile(r"pickle", re.I),
        re.compile(r"щекоч\w*", re.I),
        "маринованные огурцы",
    ),
    (
        re.compile(r"handful", re.I),
        re.compile(r"туфл\w*", re.I),
        "горсть",
    ),
    (
        re.compile(r"\bbeef\b", re.I),
        re.compile(r"ковяж\w*", re.I),
        "говяжий",
    ),
]

_HALLUCINATION_SWAPS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bковяж\w*\b", re.I), "говяжий"),
    (re.compile(r"\b[Тт]энсо\b"), ""),
    (re.compile(r"\b[Тт]енсо\b"), ""),
    (re.compile(r"\bщекочущ\w*\b", re.I), "маринованные огурцы"),
    (re.compile(r"\bщекоч\w*\b", re.I), "огурцы"),
]

_STEP_RE = re.compile(
    r"(?i)(?<![\w>])(шаг\s+(?:один|два|три|четыре|пять|\d+))",
)


def glossary_hint_for_prompt() -> str:
    pairs = ", ".join(f"{k}→{v}" for k, v in list(FOOD_GLOSSARY_EN_RU.items())[:14])
    return (
        "Кулинарный глоссарий (обязательно): "
        f"{pairs}. "
        "Особенно: pickles = маринованные огурцы (НЕ щекочущие/щекочи). "
        "handful = горсть (НЕ туфли)."
    )


def apply_glossary_to_source(text: str) -> str:
    """Подсказка в исходнике перед LLM: pickle → pickle (маринованные огурцы)."""
    out = text or ""
    for en, ru in sorted(FOOD_GLOSSARY_EN_RU.items(), key=lambda x: -len(x[0])):
        def repl(m: re.Match[str], e: str = en, r: str = ru) -> str:
            return f"{m.group(0)} ({r})"

        out = re.sub(rf"\b{re.escape(en)}\b", repl, out, flags=re.IGNORECASE)
    return out


def sanitize_dub_translation(source: str, translated: str) -> str:
    """Чинит типичные ошибки перевода и галлюцинации TTS-текста."""
    src = source or ""
    dst = (translated or "").strip()
    if not dst:
        return dst

    for src_pat, dst_pat, replacement in _SOURCE_FIXES:
        if src_pat.search(src) and dst_pat.search(dst):
            dst = dst_pat.sub(replacement, dst)

    # если в источнике pickles, а в переводе нет огурц* — форс-замена щекоч/tickl
    if re.search(r"pickle", src, re.I) and not re.search(r"огурц", dst, re.I):
        dst2 = re.sub(r"щекоч\w*", "маринованные огурцы", dst, flags=re.I)
        dst2 = re.sub(r"tickl\w*", "маринованные огурцы", dst2, flags=re.I)
        if dst2 != dst:
            dst = dst2
        elif re.fullmatch(r"[\s\W]*", dst) or len(dst.split()) <= 3:
            # короткая реплика из одних pickles
            if re.fullmatch(r"(?i)[\s,]*pickle[s]?([\s,]+pickle[s]?)*[\s,!.]*", src.strip()):
                n = len(re.findall(r"(?i)pickle", src))
                dst = ", ".join(["огурцы"] * max(1, n))

    for pat, repl in _HALLUCINATION_SWAPS:
        dst = pat.sub(repl, dst)

    dst = re.sub(r"\s{2,}", " ", dst).strip(" ,;.")
    return dst


def inject_step_pause_ssml(text: str) -> str:
    """Пауза перед «Шаг N» как в рецептах."""
    raw = (text or "").strip()
    if not raw:
        return raw
    if raw.lstrip().lower().startswith("<speak"):
        return _STEP_RE.sub(r'<break time="600ms"/>\1', raw)
    if _STEP_RE.search(raw):
        body = _STEP_RE.sub(r'<break time="600ms"/>\1', raw)
        return f"<speak>{body}</speak>"
    return raw
