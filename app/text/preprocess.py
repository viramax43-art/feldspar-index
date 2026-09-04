"""Предобработка русского текста для TTS."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from num2words import num2words

from app.text.ssml import strip_ssml

INTONATION_PRESETS = {
    "neutral": {
        "temperature": 0.75,
        "speed": 1.0,
        "repetition_penalty": 2.0,
        "top_k": 50,
        "top_p": 0.85,
        "length_penalty": 1.0,
    },
    "calm": {
        "temperature": 0.70,
        "speed": 0.97,
        "repetition_penalty": 2.1,
        "top_k": 45,
        "top_p": 0.82,
        "length_penalty": 1.0,
    },
    "expressive": {
        "temperature": 0.78,
        "speed": 1.0,
        "repetition_penalty": 2.0,
        "top_k": 50,
        "top_p": 0.88,
        "length_penalty": 1.0,
    },
}

ABBREVIATIONS = {
    "т.д.": "так далее",
    "т.п.": "тому подобное",
    "т.е.": "то есть",
    "т.к.": "так как",
    "др.": "другие",
    "пр.": "прочее",
    "г.": "город",
    "ул.": "улица",
    "кв.": "квартира",
    "руб.": "рублей",
    "коп.": "копеек",
    "млн": "миллионов",
    "млрд": "миллиардов",
    "тыс.": "тысяч",
}

CURRENCY_PATTERN = re.compile(
    r"(\d[\d\s]*)(?:\s*)(₽|руб\.?|рублей|долл\.?|usd|\$|€|eur)",
    re.IGNORECASE,
)
PERCENT_PATTERN = re.compile(r"(\d[\d\s]*)\s*%")
# Время: не трогаем «т.д.» / «т.п.» — двоеточие только у цифр
TIME_PATTERN = re.compile(r"(?<![а-яА-ЯёЁa-zA-Z])(\d{1,2}):(\d{2})(?!\d)")
DATE_PATTERN = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{2,4})\b")
NUMBER_PATTERN = re.compile(
    # Дробь только без пробела после .|, (5.4 / 5,4). Списки «5, 4, 3» — не дробь.
    r"(?<!\w)-?(?:\d+(?:[.,]\d+)?|\d+(?:\s+\d+)*)(?!\w)"
)
# +7 999 123-45-67, 8(999)123-45-67, 999-123-45-67
PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+7|8)?[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\w)"
)
# Ударение перед гласной: зам+ок → замо́к
INLINE_STRESS_PATTERN = re.compile(
    r"([а-яА-ЯёЁ]*)\+([аеёиоуыэюяАЕЁИОУЫЭЮЯ])([а-яА-ЯёЁ]*)"
)
# Legacy: +слово → ударение на первую гласную
LEGACY_STRESS_PATTERN = re.compile(r"(?<![а-яА-ЯёЁ])\+([а-яА-ЯёЁ]+)")
COMBINING_ACUTE = "\u0301"
VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
# Границы аббревиатур с точками — не режем sentence splitter
ABBREV_BOUNDARY_PATTERN = re.compile(
    r"\b(т\.д|т\.п|т\.е|т\.к|др|пр|г|ул|кв|руб|коп|тыс)\.",
    re.IGNORECASE,
)

# Частые слова, которые XTTS часто читает с неверным ударением
BUILTIN_STRESS: dict[str, str] = {
    "договор": "догово́р",
    "договора": "догово́ра",
    "договоры": "догово́ры",
    "звонит": "звони́т",
    "звонят": "звоня́т",
    "помощник": "помо́щник",
    "помощника": "помо́щника",
    "помощнику": "помо́щнику",
    "помощником": "помо́щником",
    "помощники": "помо́щники",
    "включит": "включи́т",
    "красивее": "краси́вее",
    "обеспечение": "обеспе́чение",
    "каталог": "катало́г",
    "квартал": "кварта́л",
    "эксперт": "экспе́рт",
    "средства": "сре́дства",
    "торты": "то́рты",
    "банты": "ба́нты",
    "шарфы": "ша́рфы",
    "свекла": "свёкла",
    "одновременно": "одновреме́нно",
    "поняла": "поняла́",
    "заняла": "заняла́",
    "приняла": "приняла́",
    "создала": "создала́",
    "ждала": "ждала́",
    "брала": "брала́",
    "дала": "дала́",
    "жила": "жила́",
    "взяла": "взяла́",
    "началась": "начала́сь",
    "начался": "начался́",
    "документ": "докуме́нт",
    "документы": "докуме́нты",
    "алфавит": "алфави́т",
    "конечно": "коне́чно",
    "сегодня": "сево́дня",
    "пожалуйста": "пожа́луйста",
    "здравствуйте": "здра́ствуйте",
    "сейчас": "сейча́с",
    "вообще": "вообще́",
    "например": "наприме́р",
    "человек": "челове́к",
    "человека": "челове́ка",
    "результат": "результа́т",
    "информация": "информа́ция",
    "компьютер": "компью́тер",
    "интернет": "интерне́т",
    "телефон": "телефо́н",
    "магазин": "магази́н",
    "проект": "прое́кт",
    "синтез": "си́нтез",
}


@dataclass
class TextChunk:
    text: str
    pause_after: float = 0.25


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def expand_abbreviations(text: str) -> str:
    for abbr, full in ABBREVIATIONS.items():
        pattern = re.compile(
            rf"(?:^|[\s]){re.escape(abbr)}(?=[\s,.!?;:]|$)",
            re.IGNORECASE,
        )

        def repl(match: re.Match[str], replacement: str = full) -> str:
            matched = match.group(0)
            if matched and matched[0].isspace():
                return matched[0] + replacement
            return replacement

        text = pattern.sub(repl, text)
    return text


def _digits_to_words(value: str) -> str:
    raw = (value or "").strip()
    # «5 4 3» / «5, 4, 3» — отдельные цифры, НЕ «пять целых четыре»
    spaced = re.findall(r"\d+", raw)
    if len(spaced) >= 2 and re.search(r"[,\s]", raw):
        # если это настоящая дробь вида 5.4 / 5,4 без пробела — одна точка/запятая
        compact = raw.replace(" ", "")
        if re.fullmatch(r"-?\d+[.,]\d+", compact):
            cleaned = compact.replace(",", ".")
            left, right = cleaned.split(".", 1)
            left_words = num2words(int(left), lang="ru")
            right_words = " ".join(num2words(int(d), lang="ru") for d in right)
            return f"{left_words} целая {right_words}"
        return " ".join(num2words(int(d), lang="ru") for d in spaced)

    cleaned = raw.replace(" ", "").replace(",", ".")
    if "." in cleaned:
        left, right = cleaned.split(".", 1)
        if left.lstrip("-").isdigit() and right.isdigit():
            # одноразрядная «дробь» 5.4 / 3.2 — почти всегда склейка countdown
            if len(left.lstrip("-")) == 1 and len(right) == 1:
                return " ".join(
                    num2words(int(d), lang="ru")
                    for d in (left.lstrip("-") + right)
                    if d.isdigit()
                )
            left_words = num2words(int(left), lang="ru")
            right_words = " ".join(num2words(int(d), lang="ru") for d in right)
            return f"{left_words} целая {right_words}"
    if cleaned.lstrip("-").isdigit():
        return num2words(int(cleaned), lang="ru")
    return value


def expand_numbers(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return _digits_to_words(match.group(0))

    return NUMBER_PATTERN.sub(repl, text)


def expand_currency(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        amount = _digits_to_words(match.group(1))
        unit = match.group(2).lower()
        if unit in {"₽", "руб", "руб.", "рублей"}:
            return f"{amount} рублей"
        if unit in {"$", "usd", "долл", "долл."}:
            return f"{amount} долларов"
        return f"{amount} евро"

    return CURRENCY_PATTERN.sub(repl, text)


def expand_percent(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return f"{_digits_to_words(match.group(1))} процентов"

    return PERCENT_PATTERN.sub(repl, text)


def expand_time(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        return f"{num2words(hours, lang='ru')} {_minute_form(minutes)}"

    return TIME_PATTERN.sub(repl, text)


def _minute_form(minutes: int) -> str:
    if minutes == 0:
        return "ноль минут"
    word = num2words(minutes, lang="ru")
    if minutes % 10 == 1 and minutes % 100 != 11:
        return f"{word} минута"
    if 2 <= minutes % 10 <= 4 and not (12 <= minutes % 100 <= 14):
        return f"{word} минуты"
    return f"{word} минут"


def expand_dates(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))
        if year < 100:
            year += 2000
        months = [
            "января", "февраля", "марта", "апреля", "мая", "июня",
            "июля", "августа", "сентября", "октября", "ноября", "декабря",
        ]
        month_name = months[month - 1] if 1 <= month <= 12 else str(month)
        return f"{num2words(day, lang='ru')} {month_name} {num2words(year, lang='ru')} года"

    return DATE_PATTERN.sub(repl, text)


def expand_phones(text: str) -> str:
    """Телефоны → поцифровое чтение (без ложного num2words на весь номер)."""

    def repl(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) < 7:
            return match.group(0)
        return " ".join(num2words(int(d), lang="ru") for d in digits)

    return PHONE_PATTERN.sub(repl, text)


def acute_to_silero_plus(text: str) -> str:
    """
    Адаптер ударений для Silero: гласная+́ → +гласная (markup put_accent).
    Для XTTS оставляем комбинированный акут.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if (
            ch in VOWELS
            and i + 1 < len(text)
            and text[i + 1] == COMBINING_ACUTE
        ):
            out.append("+")
            out.append(ch)
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def strip_stress_marks(text: str) -> str:
    """Убирает все маркеры ударений.

    XTTS не обучен на U+0301 / '+гласная' — токенизатор режет слова
    на странные куски, и речь звучит как «чтение по слогам».
    Ударения оставляем только для Silero.
    """
    text = text.replace(COMBINING_ACUTE, "")
    # leftover inline markers: зам+ок → замок, +звонит → звонит
    vowels = "".join(sorted(VOWELS))
    text = re.sub(rf"\+(?=[{re.escape(vowels)}])", "", text)
    text = text.replace("+", "")
    return text


def accents_for_xtts(text: str) -> str:
    """Ударения для XTTS: снимаем ́ и +, буквы не меняем.

    Раньше ударная «е» принудительно становилась «ё» — из‑за этого
    «неделя/время/телефон» звучали как с «ё». Настоящая «ё»
    (всё, ещё, чёрный) сохраняется, если она уже в тексте.
    """
    return strip_stress_marks(text)


def protect_abbrev_boundaries(text: str) -> str:
    """Временно заменяем точки в т.д./т.п. чтобы не резать фразы."""

    def repl(match: re.Match[str]) -> str:
        return match.group(0).replace(".", "\uE000")

    return ABBREV_BOUNDARY_PATTERN.sub(repl, text)


def restore_abbrev_boundaries(text: str) -> str:
    return text.replace("\uE000", ".")


def stress_vowel(word: str, vowel_index: int) -> str:
    """Ставит комбинированный акут U+0301 после ударной гласной."""
    if vowel_index < 0 or vowel_index >= len(word):
        return word
    ch = word[vowel_index]
    if ch not in VOWELS:
        return word
    if ch in "ёЁ":
        return word
    if vowel_index + 1 < len(word) and word[vowel_index + 1] == COMBINING_ACUTE:
        return word
    return word[: vowel_index + 1] + COMBINING_ACUTE + word[vowel_index + 1 :]


def apply_inline_stress_marks(text: str) -> str:
    """
    Ручные ударения:
      зам+ок   → замо́к
      догово́р  → без изменений
      +звонит  → ударение на первую гласную (legacy)
    """

    def repl_inline(match: re.Match[str]) -> str:
        prefix, vowel, suffix = match.group(1), match.group(2), match.group(3)
        stressed = vowel if vowel in "ёЁ" else vowel + COMBINING_ACUTE
        return f"{prefix}{stressed}{suffix}"

    text = INLINE_STRESS_PATTERN.sub(repl_inline, text)

    def repl_legacy(match: re.Match[str]) -> str:
        word = match.group(1)
        for idx, ch in enumerate(word):
            if ch in VOWELS:
                return stress_vowel(word, idx)
        return word

    return LEGACY_STRESS_PATTERN.sub(repl_legacy, text)


def _match_case(source: str, replacement: str) -> str:
    if not source:
        return replacement
    if source.isupper():
        return replacement.upper()
    if source[0].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def apply_pronunciation_dict(text: str, dictionary: dict[str, str]) -> str:
    if not dictionary:
        return text
    for word, pronunciation in sorted(dictionary.items(), key=lambda x: -len(x[0])):
        if word.startswith("_"):
            continue
        pattern = re.compile(
            rf"(?<![а-яА-ЯёЁa-zA-Z]){re.escape(word)}(?![а-яА-ЯёЁa-zA-Z])",
            re.IGNORECASE,
        )

        def repl(match: re.Match[str], p: str = pronunciation) -> str:
            return _match_case(match.group(0), p)

        text = pattern.sub(repl, text)
    return text


def load_pronunciation_dict(path: Path) -> dict[str, str]:
    merged = dict(BUILTIN_STRESS)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if str(key).startswith("_"):
                continue
            merged[str(key).lower()] = str(value)
    return merged


def split_sentences(text: str) -> list[str]:
    protected = protect_abbrev_boundaries(text)
    parts = re.split(r"(?<=[.!?…])\s+", protected)
    return [restore_abbrev_boundaries(p.strip()) for p in parts if p.strip()]


def split_long_sentence(
    sentence: str,
    max_chars: int,
    min_chars: int = 24,
) -> list[str]:
    """Phrase chunker: старается держать фразы в [min_chars, max_chars]."""
    if len(sentence) <= max_chars:
        return [sentence]
    chunks: list[str] = []
    for part in re.split(r"(?<=[,;:—-])\s+", sentence):
        if len(part) <= max_chars:
            if (
                chunks
                and len(chunks[-1]) < min_chars
                and len(chunks[-1]) + 1 + len(part) <= max_chars
            ):
                chunks[-1] = f"{chunks[-1]} {part}"
            else:
                chunks.append(part)
            continue
        words = part.split()
        current: list[str] = []
        length = 0
        for word in words:
            add = len(word) + (1 if current else 0)
            if current and length + add > max_chars:
                chunks.append(" ".join(current))
                current = [word]
                length = len(word)
            else:
                current.append(word)
                length += add
        if current:
            piece = " ".join(current)
            if (
                chunks
                and len(chunks[-1]) < min_chars
                and len(chunks[-1]) + 1 + len(piece) <= max_chars
            ):
                chunks[-1] = f"{chunks[-1]} {piece}"
            else:
                chunks.append(piece)
    return chunks


class RussianPhraseChunker:
    """
    Режет текст на фразы и ОБЪЕДИНЯЕТ слишком короткие предложения:
    отдельная генерация коротких фраз даёт «дикторский» темп и underrun.
    Цель — держать фразы в диапазоне [min_chars, soft_max_chars],
    допуская до hard_max_chars, если иначе фраза окажется короче min_chars.
    """

    def __init__(
        self,
        min_chars: int = 40,
        soft_max_chars: int = 110,
        hard_max_chars: int = 170,
    ) -> None:
        self.min_chars = min_chars
        self.soft_max_chars = max(min_chars, soft_max_chars)
        self.hard_max_chars = max(self.soft_max_chars, hard_max_chars)

    def chunk(self, text: str) -> list[TextChunk]:
        pieces: list[str] = []
        for sentence in split_sentences(text):
            if len(sentence) <= self.hard_max_chars:
                pieces.append(sentence)
            else:
                pieces.extend(
                    split_long_sentence(
                        sentence,
                        self.hard_max_chars,
                        min_chars=self.min_chars,
                    )
                )

        merged: list[str] = []
        buf = ""
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if not buf:
                buf = piece
                continue
            candidate_len = len(buf) + 1 + len(piece)
            # Добираем до min_chars (до hard_max), затем объединяем до soft_max
            if len(buf) < self.min_chars and candidate_len <= self.hard_max_chars:
                buf = f"{buf} {piece}"
            elif candidate_len <= self.soft_max_chars:
                buf = f"{buf} {piece}"
            else:
                merged.append(buf)
                buf = piece
        if buf:
            merged.append(buf)

        return [
            TextChunk(text=m, pause_after=pause_for_punctuation(m)) for m in merged
        ]


def pause_for_punctuation(fragment: str) -> float:
    """Пауза после фразы (сек). Между предложениями не меньше 0.5с."""
    tail = fragment.rstrip()
    if tail.endswith("...") or tail.endswith("…"):
        return 0.55
    if tail.endswith("?"):
        return 0.55
    if tail.endswith("!"):
        return 0.50
    if tail.endswith("."):
        return 0.50
    if tail.endswith(";"):
        return 0.25
    if tail.endswith(":"):
        return 0.20
    if tail.endswith(","):
        return 0.08
    if tail.endswith("—") or tail.endswith("-"):
        return 0.12
    return 0.08


def enhance_prosody(text: str) -> str:
    """Нормализует многоточия для пауз TTS. Без насильственных запятых —
    они рвут фразу и усиливают ощущение «чтения по слогам».
    """
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"(?<!\.)\.\.(?!\.)", "...", text)
    return text


def prepare_text_for_tts(
    text: str,
    max_chunk_chars: int = 170,
    pronunciation_dict: dict[str, str] | None = None,
    min_chunk_chars: int = 40,
    soft_max_chunk_chars: int | None = None,
    engine: str = "xtts",
    language: str = "ru",
) -> list[TextChunk]:
    text = normalize_whitespace(text)
    text = strip_ssml(text)
    text = (
        text.replace("。", ".")
        .replace("！", "!")
        .replace("？", "?")
        .replace("、", ",")
    )
    if language == "ru":
        text = expand_abbreviations(text)
        text = expand_phones(text)
        text = expand_currency(text)
        text = expand_percent(text)
        text = expand_time(text)
        text = expand_dates(text)
        text = expand_numbers(text)
    text = enhance_prosody(text)
    dict_to_use = (
        pronunciation_dict if pronunciation_dict is not None else dict(BUILTIN_STRESS)
    )
    text = apply_pronunciation_dict(text, dict_to_use)
    text = apply_inline_stress_marks(text)
    if engine == "silero":
        text = acute_to_silero_plus(text)
    else:
        # XTTS: без U+0301/«+» (ломают токенизатор); «е» не трогаем
        text = accents_for_xtts(text)

    if soft_max_chunk_chars is None:
        soft_max_chunk_chars = max(min_chunk_chars, int(max_chunk_chars * 0.65))

    chunker = RussianPhraseChunker(
        min_chars=min_chunk_chars,
        soft_max_chars=soft_max_chunk_chars,
        hard_max_chars=max_chunk_chars,
    )
    return chunker.chunk(text)


def get_inference_params(
    intonation: str,
    speed: float | None = None,
    temperature: float | None = None,
) -> dict[str, float]:
    preset = INTONATION_PRESETS.get(intonation, INTONATION_PRESETS["neutral"]).copy()
    if speed is not None:
        preset["speed"] = max(0.7, min(1.3, speed))
    if temperature is not None:
        preset["temperature"] = max(0.3, min(1.0, temperature))
    return preset
