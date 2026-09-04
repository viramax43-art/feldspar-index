"""Фонетический фронтенд для английского Tacotron MockingBird.

Чекпоинт RTVC знает только ASCII. Кириллица, хангыль, кана и диакритика
переводятся в английскую фонетическую запись, которую модель может прочитать.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from num2words import num2words

from app.text.language import resolve_language

logger = logging.getLogger(__name__)

# Алфавит English RTVC synthesizer.pt (66 символов вместе с _ ~)
_MB_OK = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'\"(),-.:;? ")
_NUM_RE = re.compile(r"(?<!\w)-?\d[\d\s]*(?:[.,]\d+)?(?!\w)")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

_NUM2WORDS_LANG = {
    "ru": "ru",
    "de": "de",
    "fr": "fr",
    "ja": "ja",
    "ko": "ko",
    "en": "en",
}

# --- Russian: практическая фонетика под английский Tacotron ---
_RU_CONS = set("бвгджзклмнпрстфхцчшщйБВГДЖЗКЛМНПРСТФХЦЧШЩЙ")
_RU_IOTED = {
    "е": ("ye", "e"),
    "ё": ("yo", "o"),
    "ю": ("yu", "u"),
    "я": ("ya", "a"),
}
_RU_BASE = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "ж": "zh",
    "з": "z",
    "и": "ee",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "oo",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ы": "ih",
    "э": "e",
    "ъ": "",
    "ь": "",
}


def _ru_word(word: str) -> str:
    out: list[str] = []
    lower = word.lower()
    for i, ch in enumerate(lower):
        prev = lower[i - 1] if i else ""
        if ch in _RU_IOTED:
            y_form, plain = _RU_IOTED[ch]
            if i == 0 or prev in "ъьаеёиоуыэюя":
                out.append(y_form)
            else:
                out.append(plain)
            continue
        out.append(_RU_BASE.get(ch, ch if ch.isascii() else ""))
    return "".join(out)


def _russian_phonetic(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if not re.search(r"[а-яё]", token, re.I):
            return token
        return _ru_word(token)

    return _WORD_RE.sub(repl, text)


# --- German ---
_DE_REPL = (
    (re.compile(r"tsch", re.I), "ch"),
    (re.compile(r"sch", re.I), "sh"),
    (re.compile(r"tion", re.I), "tseeon"),
    (re.compile(r"ck", re.I), "k"),
    (re.compile(r"ie", re.I), "ee"),
    (re.compile(r"ei", re.I), "eye"),
    (re.compile(r"eu|äu", re.I), "oy"),
    (re.compile(r"au", re.I), "ow"),
    (re.compile(r"äh|ä", re.I), "eh"),
    (re.compile(r"öh|ö", re.I), "er"),
    (re.compile(r"üh|ü", re.I), "ue"),
    (re.compile(r"ß", re.I), "ss"),
    (re.compile(r"(?<=[aouAOU])ch"), "kh"),
    (re.compile(r"ch", re.I), "kh"),
    (re.compile(r"^sp", re.I), "shp"),
    (re.compile(r"^st", re.I), "sht"),
    (re.compile(r"v", re.I), "f"),
    (re.compile(r"w", re.I), "v"),
    (re.compile(r"z", re.I), "ts"),
    (re.compile(r"j", re.I), "y"),
    (re.compile(r"ig\b", re.I), "ikh"),
)


def _german_phonetic(text: str) -> str:
    def word(match: re.Match[str]) -> str:
        w = match.group(0)
        for pat, repl in _DE_REPL:
            w = pat.sub(repl, w)
        return w

    return _WORD_RE.sub(word, text)


# --- French (грубая, но разборчивая запись для английского Tacotron) ---
_FR_REPL = (
    (re.compile(r"eau", re.I), "oh"),
    (re.compile(r"ault|aux\b", re.I), "oh"),
    (re.compile(r"ain|ein|aim|im\b|in\b", re.I), "an"),
    (re.compile(r"oin", re.I), "wan"),
    (re.compile(r"oi", re.I), "wah"),
    (re.compile(r"ou", re.I), "oo"),
    (re.compile(r"au", re.I), "oh"),
    (re.compile(r"eu|œu", re.I), "uh"),
    (re.compile(r"on|om", re.I), "on"),
    (re.compile(r"un|um", re.I), "uhn"),
    (re.compile(r"en|em", re.I), "ahn"),
    (re.compile(r"ch", re.I), "sh"),
    (re.compile(r"gn", re.I), "ny"),
    (re.compile(r"qu", re.I), "k"),
    (re.compile(r"ph", re.I), "f"),
    (re.compile(r"ç", re.I), "s"),
    (re.compile(r"[éèêë]", re.I), "ay"),
    (re.compile(r"[àâ]", re.I), "ah"),
    (re.compile(r"[ùû]", re.I), "ue"),
    (re.compile(r"[ô]", re.I), "oh"),
    (re.compile(r"[ïî]", re.I), "ee"),
    (re.compile(r"œ", re.I), "uh"),
    (re.compile(r"æ", re.I), "eh"),
    (re.compile(r"ÿ", re.I), "ee"),
    (re.compile(r"er\b|ez\b|et\b", re.I), "ay"),
    (re.compile(r"ent\b", re.I), ""),
    (re.compile(r"e\b", re.I), ""),
    (re.compile(r"(?<![csCS])h", re.I), ""),
)


def _french_phonetic(text: str) -> str:
    def word(match: re.Match[str]) -> str:
        w = match.group(0)
        for pat, repl in _FR_REPL:
            w = pat.sub(repl, w)
        return w

    return _WORD_RE.sub(word, text)


# --- Japanese ---
_HIRAGANA = {
    "あ": "a",
    "い": "i",
    "う": "u",
    "え": "e",
    "お": "o",
    "か": "ka",
    "き": "ki",
    "く": "ku",
    "け": "ke",
    "こ": "ko",
    "さ": "sa",
    "し": "shi",
    "す": "su",
    "せ": "se",
    "そ": "so",
    "た": "ta",
    "ち": "chi",
    "つ": "tsu",
    "て": "te",
    "と": "to",
    "な": "na",
    "に": "ni",
    "ぬ": "nu",
    "ね": "ne",
    "の": "no",
    "は": "ha",
    "ひ": "hi",
    "ふ": "fu",
    "へ": "he",
    "ほ": "ho",
    "ま": "ma",
    "み": "mi",
    "む": "mu",
    "め": "me",
    "も": "mo",
    "や": "ya",
    "ゆ": "yu",
    "よ": "yo",
    "ら": "ra",
    "り": "ri",
    "る": "ru",
    "れ": "re",
    "ろ": "ro",
    "わ": "wa",
    "を": "o",
    "ん": "n",
    "が": "ga",
    "ぎ": "gi",
    "ぐ": "gu",
    "げ": "ge",
    "ご": "go",
    "ざ": "za",
    "じ": "ji",
    "ず": "zu",
    "ぜ": "ze",
    "ぞ": "zo",
    "だ": "da",
    "ぢ": "ji",
    "づ": "zu",
    "で": "de",
    "ど": "do",
    "ば": "ba",
    "び": "bi",
    "ぶ": "bu",
    "べ": "be",
    "ぼ": "bo",
    "ぱ": "pa",
    "ぴ": "pi",
    "ぷ": "pu",
    "ぺ": "pe",
    "ぽ": "po",
    "きゃ": "kya",
    "きゅ": "kyu",
    "きょ": "kyo",
    "しゃ": "sha",
    "しゅ": "shu",
    "しょ": "sho",
    "ちゃ": "cha",
    "ちゅ": "chu",
    "ちょ": "cho",
    "にゃ": "nya",
    "にゅ": "nyu",
    "にょ": "nyo",
    "ひゃ": "hya",
    "ひゅ": "hyu",
    "ひょ": "hyo",
    "みゃ": "mya",
    "みゅ": "myu",
    "みょ": "myo",
    "りゃ": "rya",
    "りゅ": "ryu",
    "りょ": "ryo",
    "ぎゃ": "gya",
    "ぎゅ": "gyu",
    "ぎょ": "gyo",
    "じゃ": "ja",
    "じゅ": "ju",
    "じょ": "jo",
    "びゃ": "bya",
    "びゅ": "byu",
    "びょ": "byo",
    "ぴゃ": "pya",
    "ぴゅ": "pyu",
    "ぴょ": "pyo",
}
_KATAKANA_OFF = ord("ア") - ord("あ")


def _kata_to_hira(ch: str) -> str:
    code = ord(ch)
    if 0x30A1 <= code <= 0x30F6:
        return chr(code - _KATAKANA_OFF)
    return ch


def _kana_to_romaji(text: str) -> str:
    chars = [(_kata_to_hira(ch) if "ァ" <= ch <= "ヶ" else ch) for ch in text]
    out: list[str] = []
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch in {"っ", "ッ"}:
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            romaji = _HIRAGANA.get(nxt, "")
            out.append(romaji[:1] if romaji else "t")
            i += 1
            continue
        if ch == "ー" and out:
            prev = out[-1]
            out.append(prev[-1] if prev else "")
            i += 1
            continue
        pair = "".join(chars[i : i + 2])
        if pair in _HIRAGANA:
            out.append(_HIRAGANA[pair])
            i += 2
            continue
        if ch in _HIRAGANA:
            out.append(_HIRAGANA[ch])
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


@lru_cache(maxsize=1)
def _kakasi_convert():
    try:
        import pykakasi

        kks = pykakasi.kakasi()
        if hasattr(kks, "convert"):
            def convert(text: str) -> str:
                parts = []
                for item in kks.convert(text):
                    piece = (item.get("hepburn") or item.get("orig") or "").strip()
                    if piece:
                        parts.append(piece)
                return " ".join(parts)

            return convert
        kks.setMode("J", "a")
        kks.setMode("H", "a")
        kks.setMode("K", "a")
        conv = kks.getConverter()
        return conv.do
    except Exception as exc:
        logger.info("pykakasi недоступен (%s), кана — таблицей, кандзи останутся", exc)
        return None


def _japanese_phonetic(text: str) -> str:
    convert = _kakasi_convert()
    if convert is not None:
        try:
            return convert(text)
        except Exception as exc:
            logger.warning("pykakasi не смог прочитать текст: %s", exc)
    return _kana_to_romaji(text)


# --- Korean: Revised Romanization, озвучка ближе к английскому ---
_CHO = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
_JUNG = [
    "a",
    "eh",
    "ya",
    "yeh",
    "aw",
    "eh",
    "yaw",
    "yeh",
    "o",
    "wa",
    "weh",
    "weh",
    "yo",
    "oo",
    "waw",
    "weh",
    "wee",
    "yoo",
    "eu",
    "ee",
    "ee",
]
_JONG = [
    "",
    "k",
    "k",
    "ks",
    "n",
    "nj",
    "nh",
    "t",
    "l",
    "lg",
    "lm",
    "lb",
    "ls",
    "lt",
    "lp",
    "lh",
    "m",
    "p",
    "ps",
    "t",
    "t",
    "ng",
    "t",
    "t",
    "k",
    "t",
    "p",
    "t",
]


def _hangul_syllable(code: int) -> str:
    s = code - 0xAC00
    cho, jung, jong = s // 588, (s % 588) // 28, s % 28
    return f"{_CHO[cho]}{_JUNG[jung]}{_JONG[jong]}"


def _korean_phonetic(text: str) -> str:
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            out.append(_hangul_syllable(code))
        else:
            out.append(ch)
    return "".join(out)


def _expand_numbers(text: str, lang: str) -> str:
    nw = _NUM2WORDS_LANG.get(lang, "en")

    def repl(match: re.Match[str]) -> str:
        raw = match.group(0).replace(" ", "").replace("\u00a0", "")
        raw = raw.replace(",", ".")
        try:
            value: int | float
            if "." in raw:
                value = float(raw)
                if value.is_integer():
                    value = int(value)
            else:
                value = int(raw)
            try:
                return num2words(value, lang=nw)
            except NotImplementedError:
                return num2words(int(value), lang="en")
        except Exception:
            return match.group(0)

    return _NUM_RE.sub(repl, text)


def _unify_punct(text: str) -> str:
    repl = {
        "。": ".",
        "．": ".",
        "！": "!",
        "？": "?",
        "、": ",",
        "，": ",",
        "；": ",",
        "：": ",",
        "「": '"',
        "」": '"',
        "『": '"',
        "』": '"',
        "«": '"',
        "»": '"',
        "„": '"',
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": ",",
        "–": ",",
        "…": "...",
        "・": " ",
        "·": " ",
    }
    for src, dst in repl.items():
        text = text.replace(src, dst)
    text = re.sub(r"[%‰]", " percent ", text)
    return text


def _sanitize_ascii(text: str) -> str:
    cleaned: list[str] = []
    for ch in text:
        if ch in _MB_OK:
            cleaned.append(ch)
        elif ch in "\n\t":
            cleaned.append(" ")
        elif ch.isascii() and ch.isdigit():
            cleaned.append(" ")
        else:
            cleaned.append(" ")
    out = re.sub(r" {2,}", " ", "".join(cleaned)).strip()
    return out


def to_mockingbird_text(text: str, language: str | None = None, default: str = "ru") -> str:
    """Готовит фразу для English RTVC: фонетическая латиница + только допустимые символы."""
    if not text or not text.strip():
        return text
    lang = resolve_language(text, default if language is None else language)
    text = _unify_punct(text)
    text = _expand_numbers(text, lang)
    if lang == "ru":
        text = _russian_phonetic(text)
    elif lang == "de":
        text = _german_phonetic(text)
    elif lang == "fr":
        text = _french_phonetic(text)
    elif lang == "ja":
        text = _japanese_phonetic(text)
    elif lang == "ko":
        text = _korean_phonetic(text)
    text = _sanitize_ascii(text)
    logger.debug("MockingBird frontend lang=%s → %s", lang, text[:80])
    return text
