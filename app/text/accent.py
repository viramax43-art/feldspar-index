"""Локальная расстановка русских ударений: RUAccent + доменный словарь."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)

VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
STRESS_PLUS_RE = re.compile(rf"\+(?=[{VOWELS}])")
RUSSIAN_TOKEN_RE = re.compile(r"[+а-яА-ЯёЁ]+")

DEFAULT_CUSTOM_ACCENTS: dict[str, str] = {
    "войсер": "в+ойсер",
    "звонит": "звон+ит",
    "звонят": "звон+ят",
    "договор": "догов+ор",
    "договоры": "догов+оры",
    "каталог": "катал+ог",
    "маркетинг": "марк+етинг",
    "жалюзи": "жалюз+и",
    "помощник": "пом+ощник",
    "помощника": "пом+ощника",
    "помощнику": "пом+ощнику",
    "помощником": "пом+ощником",
    "помощники": "пом+ощники",
}


def strip_stress_plus(text: str) -> str:
    """Удаляет только TTS-маркеры перед русской гласной, не обычный знак плюс."""
    return STRESS_PLUS_RE.sub("", text)


def _canonical_for_validation(text: str) -> str:
    """RUAccent вправе восстанавливать ё; это не считается изменением текста."""
    return strip_stress_plus(text).replace("ё", "е").replace("Ё", "Е")


def constrain_yo_to_source(result: str, source: str) -> str:
    """
    Не даём RUAccent/модели массово менять «е»→«ё».
    «ё» оставляем только там, где в source уже была «ё»/«Ё».
    """
    src_letters = [ch for ch in strip_stress_plus(source) if ch.isalpha()]
    src_idx = 0
    out: list[str] = []
    for ch in result:
        if ch.isalpha():
            if src_idx < len(src_letters):
                src_ch = src_letters[src_idx]
                if ch == "ё" and src_ch in "еЕ":
                    ch = "е" if src_ch == "е" else "Е"
                elif ch == "Ё" and src_ch in "еЕ":
                    ch = "Е" if src_ch == "Е" else "е"
                src_idx += 1
            out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def _letters_fingerprint(text: str) -> str:
    """Сравнение без пунктуации/пробелов — RUAccent часто их чуть двигает."""
    return "".join(
        ch.lower()
        for ch in _canonical_for_validation(text)
        if ch.isalpha()
    )


def _match_case(source: str, replacement: str) -> str:
    plain_source = strip_stress_plus(source)
    if plain_source.isupper():
        return replacement.upper()
    if plain_source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def apply_custom_accents(text: str, accents: dict[str, str]) -> str:
    """Применяет словарь после RUAccent, заменяя его вариант ударения."""

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        key = strip_stress_plus(token).lower()
        accented = accents.get(key)
        return _match_case(token, accented) if accented else token

    return RUSSIAN_TOKEN_RE.sub(replace, text)


def _stress_vowel_ordinal(token: str) -> int | None:
    plus_index = token.find("+")
    if plus_index < 0:
        return None
    ordinal = sum(1 for char in token[:plus_index] if char in VOWELS)
    return ordinal


def _put_stress_by_ordinal(token: str, ordinal: int) -> str:
    current = 0
    for index, char in enumerate(token):
        if char in VOWELS:
            if current == ordinal:
                return token[:index] + "+" + token[index:]
            current += 1
    return token


def merge_gigachat_accents(result: str, gigachat_text: str) -> str:
    """
    Приоритет: уже расставленное RUAccent (+), затем метки GigaChat
    только для слов без ударения, затем авто для односложных.
    """
    source_tokens = RUSSIAN_TOKEN_RE.findall(gigachat_text)
    source_index = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal source_index
        token = match.group(0)
        source = (
            source_tokens[source_index] if source_index < len(source_tokens) else ""
        )
        source_index += 1

        if "+" in token:
            return token

        same_word = (
            _canonical_for_validation(token).lower()
            == _canonical_for_validation(source).lower()
        )
        if same_word:
            ordinal = _stress_vowel_ordinal(source)
            if ordinal is not None:
                return _put_stress_by_ordinal(token, ordinal)

        vowel_count = sum(1 for char in token if char in VOWELS)
        if vowel_count == 1:
            return _put_stress_by_ordinal(token, 0)
        return token

    return RUSSIAN_TOKEN_RE.sub(replace, result)


class AccentService:
    """RUAccent грузится лениво при первом ответе, чтобы не замедлять startup."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._accentizer: Any | None = None
        self._load_attempted = False
        self._load_lock = asyncio.Lock()
        self._process_lock = asyncio.Lock()
        self.custom_accents = self._load_custom_accents(
            settings.custom_accents_path
        )

    @staticmethod
    def _load_custom_accents(path: Path) -> dict[str, str]:
        result = dict(DEFAULT_CUSTOM_ACCENTS)
        if not path.exists():
            return result
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for word, accented in payload.items():
                    word = str(word).strip().lower()
                    accented = str(accented).strip()
                    if word and accented and strip_stress_plus(accented).lower() == word:
                        result[word] = accented
                    else:
                        logger.warning(
                            "Пропущено некорректное ударение %r: %r",
                            word,
                            accented,
                        )
        except Exception as exc:
            logger.warning("Не удалось прочитать %s: %s", path, exc)
        return result

    async def _ensure_loaded(self) -> Any | None:
        if not self.settings.enable_ruaccent:
            return None
        if self._accentizer is not None or self._load_attempted:
            return self._accentizer

        async with self._load_lock:
            if self._accentizer is not None or self._load_attempted:
                return self._accentizer
            self._load_attempted = True
            self.settings.ruaccent_workdir.mkdir(parents=True, exist_ok=True)

            def load() -> Any:
                from ruaccent import RUAccent

                accentizer = RUAccent()
                accentizer.load(
                    omograph_model_size=self.settings.ruaccent_model_size,
                    use_dictionary=self.settings.ruaccent_use_dictionary,
                    custom_dict=self.custom_accents,
                    device="CPU",
                    workdir=str(self.settings.ruaccent_workdir),
                )
                return accentizer

            try:
                self._accentizer = await asyncio.to_thread(load)
                logger.info(
                    "RUAccent загружен (model=%s, dictionary=%s)",
                    self.settings.ruaccent_model_size,
                    self.settings.ruaccent_use_dictionary,
                )
            except ImportError:
                logger.warning(
                    "ruaccent не установлен; используется GigaChat + custom accents"
                )
            except Exception as exc:
                logger.warning(
                    "RUAccent недоступен (%s); используется GigaChat + custom accents",
                    exc,
                )
            return self._accentizer

    async def add_accents(self, text: str) -> str:
        if not text.strip():
            return text

        from app.text.language import resolve_language

        if resolve_language(text, "ru") != "ru":
            return text

        accentizer = await self._ensure_loaded()
        result = text
        source_plain = strip_stress_plus(text)
        if accentizer is not None:
            # RUAccent — основная расстановка; GigaChat добивает только пропуски.
            plain = source_plain
            async with self._process_lock:
                try:
                    accented = await asyncio.to_thread(accentizer.process_all, plain)
                    if _canonical_for_validation(accented) == _canonical_for_validation(
                        plain
                    ):
                        result = accented
                    elif _letters_fingerprint(accented) == _letters_fingerprint(plain):
                        # Пунктуация/пробелы съехали — оставляем исходный текст,
                        # переносим только позиции ударений по словам.
                        result = merge_gigachat_accents(plain, accented)
                        logger.debug(
                            "RUAccent сдвинул пунктуацию; ударения перенесены "
                            "(|%s| → |%s|)",
                            plain[:60],
                            accented[:60],
                        )
                    else:
                        result = merge_gigachat_accents(plain, accented)
                        logger.warning(
                            "RUAccent изменил слова; ударения перенесены частично "
                            "(|%s| → |%s|)",
                            plain[:60],
                            accented[:60],
                        )
                except Exception as exc:
                    logger.warning("Ошибка RUAccent: %s", exc)

        # RUAccent любит е→ё; откатываем, если в исходнике была «е»
        result = constrain_yo_to_source(result, source_plain)
        result = merge_gigachat_accents(result, text)
        return apply_custom_accents(result, self.custom_accents)
