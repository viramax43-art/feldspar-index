"""Non-lexical vocalizations (uh-huh, moans…) — keep as original bed, not TTS."""

from __future__ import annotations

import re
from typing import TypeVar

T = TypeVar("T")

# Exact short tokens that alone (or only with each other) are backchannel / moans.
_FILLER_TOKENS = frozenset(
    {
        # EN fillers / backchannels
        "uh",
        "uhh",
        "uhhh",
        "um",
        "umm",
        "ummh",
        "er",
        "erm",
        "ah",
        "ahh",
        "ahhh",
        "aah",
        "aahh",
        "oh",
        "ooh",
        "oooh",
        "ohh",
        "ohhh",
        "hm",
        "hmm",
        "hmmm",
        "mm",
        "mmm",
        "mmmm",
        "mmhm",
        "mhm",
        "huh",
        "huhh",
        "ugh",
        "uhhuh",
        "uhuh",
        "aha",
        "eh",
        "ehh",
        "ow",
        "oohh",
        "ha",
        "hah",
        "haha",
        "heh",
        "ngh",
        "nngh",
        "nnngh",
        "ungh",
        "umph",
        "mph",
        "moan",
        "moans",
        "moaning",
        "groan",
        "groans",
        "groaning",
        "sigh",
        "sighs",
        "sighing",
        "gasp",
        "gasps",
        # RU
        "эм",
        "ээ",
        "эээ",
        "мм",
        "ммм",
        "мммм",
        "ага",
        "угу",
        "ах",
        "ахх",
        "ох",
        "охх",
        "ой",
        "эй",
        "хе",
        "хех",
        "ха",
        "хах",
        "нь",
        "нн",
        "стон",
        "стоны",
        "стонет",
        "стонать",
        "вздох",
        "вздохи",
    }
)

# Elongated moan-like tokens: aaaa, ooooh, mmmm, nnngh, аааа, мммм…
_MOAN_TOKEN_RE = re.compile(
    r"^(?:"
    r"[aeiouy]{2,}h*"  # aa, aah, ooooh
    r"|[hmn]{2,}h*"  # mm, mmm, nnngh-ish without g
    r"|n+g+h*"  # ngh, nngh
    r"|u+n+g+h*"  # ungh
    r"|[аеёиоуыэюя]{2,}х*"  # ааа, оох
    r"|[хмнь]{2,}"  # ммм, ннь, ххх
    r")$",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE | re.UNICODE)

# Real short words that are all-vowel and must NOT be treated as moans.
_VOWEL_WORD_WHITELIST = frozenset({"её", "ее", "i", "a", "о", "у", "и", "я", "е", "а"})

_EN_VOWELS = set("aeiouy")
_RU_VOWELS = set("аеёиоуыэюя")


def _tokens(text: str) -> list[str]:
    # "Uh-huh" / "Uh -uh -uh" / "а-а-а" → separate tokens
    norm = re.sub(r"[-–—_/]+", " ", (text or "").lower())
    return _WORD_RE.findall(norm)


def _is_periodic_babble(t: str) -> bool:
    """оаоаооаоа / ohohoh / лалала — short pattern repeated through the token."""
    n = len(t)
    if n < 5:
        return False
    for period in (1, 2, 3):
        if n < period * 2:
            continue
        unit = t[:period]
        # Allow ragged tail and small imperfections (оаоаООаоа).
        matches = sum(
            1 for i in range(0, n - period + 1, period) if t[i : i + period] == unit
        )
        covered = matches * period
        if covered >= max(4, int(0.8 * n)):
            return True
    return False


def _vowel_noise(t: str) -> bool:
    """Token made only of vowels (+h/х/ь) in any order: aoaoa, оаоуа, ааахх."""
    if len(t) < 3:
        return False
    if t in _VOWEL_WORD_WHITELIST:
        return False
    chars = set(t)
    if chars <= (_EN_VOWELS | {"h"}):
        return True
    if chars <= (_RU_VOWELS | {"х", "ь", "й"}):
        return True
    return False


def _is_moan_like_token(token: str) -> bool:
    t = (token or "").lower().strip()
    if not t:
        return True
    if t in _FILLER_TOKENS:
        return True
    if _MOAN_TOKEN_RE.match(t):
        return True
    if _vowel_noise(t):
        return True
    if _is_periodic_babble(t):
        return True
    # Consonant hums: mmm, нннь, хмм in any mix of m/n/h (+ь)
    if len(t) >= 2 and (set(t) <= set("mnh") or set(t) <= set("мнхь")):
        return True
    return False


def is_background_vocalization(text: str) -> bool:
    """True for cues that are only fillers / moans / backchannels (not real speech)."""
    tokens = _tokens(text)
    if not tokens:
        return True
    if any(t.isdigit() for t in tokens):
        return False
    # "а а а а" / "o a o a o" — stream of bare vowels: shaking/moaning, not speech.
    if len(tokens) >= 3 and all(len(t) <= 2 for t in tokens):
        joined = "".join(tokens)
        if set(joined) <= _EN_VOWELS or set(joined) <= _RU_VOWELS:
            return True
    return all(_is_moan_like_token(t) for t in tokens)


def drop_background_vocalizations(segments: list[T]) -> list[T]:
    """Remove filler/moan-only cues so original audio stays in the bed."""
    out: list[T] = []
    for seg in segments:
        text = getattr(seg, "text", None)
        if text is None and isinstance(seg, str):
            text = seg
        if is_background_vocalization(str(text or "")):
            continue
        out.append(seg)
    return out


def remove_vocalization_tokens(text: str) -> str:
    """Strip moan/filler tokens from MIXED cues: «ах, ах, да, не останавливайся!»
    → «да, не останавливайся!». Real words (and digits) are kept, punctuation
    re-joined. Returns "" when nothing lexical remains."""
    raw = (text or "").strip()
    if not raw:
        return ""
    kept: list[str] = []
    for chunk in re.split(r"\s+", raw):
        if not chunk:
            continue
        # Word cores inside the chunk ("ах-х-х" → ["ах","х","х"]; "(ох!" → ["ох"]).
        cores = _WORD_RE.findall(re.sub(r"[-–—_/]+", " ", chunk.lower()))
        if not cores:
            continue  # stray punctuation — drop, re-join below
        if any(c.isdigit() for c in cores):
            kept.append(chunk)
            continue
        if all(_is_moan_like_token(c) for c in cores):
            continue  # pure moan/filler token
        kept.append(chunk)
    out = " ".join(kept)
    # Re-attach punctuation orphaned by removed tokens and tidy the edges.
    out = re.sub(r"\s+([,.!?…;:])", r"\1", out)
    out = re.sub(r"([,.!?…;:])\1+", r"\1", out)
    out = re.sub(r"^[\s,.!?…;:—–-]+", "", out)
    out = re.sub(r"[\s,;:—–-]+$", "", out).strip()
    # Capitalize the first letter if the original started uppercase.
    if out and raw[0].isupper() and out[0].islower():
        out = out[0].upper() + out[1:]
    return out
