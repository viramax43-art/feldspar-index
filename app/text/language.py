"""Определение языка реплики по письму (без внешних моделей)."""

from __future__ import annotations

import re

SUPPORTED_LANGS = ("ru", "de", "fr", "ja", "ko", "en")

_HANGUL = re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")
_KANA = re.compile(r"[\u3040-\u30ff]")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_CYRILLIC = re.compile(r"[а-яА-ЯёЁ]")
_GERMAN = re.compile(r"[äöüßÄÖÜ]")
_FRENCH = re.compile(
    r"[àâæçéèêëïîôùûüÿœÀÂÆÇÉÈÊËÏÎÔÙÛÜŸŒ]"
)
_LATIN = re.compile(r"[a-zA-Z]")


def detect_script_language(text: str) -> str | None:
    """Язык по доминантному письму. None — только латиница без диакритики."""
    if not text or not text.strip():
        return None
    scores = {
        "ko": len(_HANGUL.findall(text)),
        "ja": len(_KANA.findall(text)) + len(_CJK.findall(text)),
        "ru": len(_CYRILLIC.findall(text)),
        "de": len(_GERMAN.findall(text)),
        "fr": len(_FRENCH.findall(text)),
    }
    # кандзи без каны чаще японский в этом боте (китайский не просили)
    if scores["ko"] and scores["ko"] >= scores["ja"] and scores["ko"] >= scores["ru"]:
        return "ko"
    if scores["ja"] and scores["ja"] >= scores["ru"]:
        return "ja"
    if scores["ru"]:
        return "ru"
    if scores["de"] > scores["fr"]:
        return "de"
    if scores["fr"]:
        return "fr"
    if _LATIN.search(text):
        return None
    return None


def detect_transcript_language(text: str, default: str = "en") -> str:
    """Язык транскрипта: латиница без диакритики → английский, не default."""
    detected = detect_script_language(text)
    if detected:
        return detected
    if text and _LATIN.search(text):
        return "en"
    default = (default or "en").lower()
    return default if default in SUPPORTED_LANGS else "en"


def resolve_dub_source_language(
    text: str,
    *,
    target_lang: str,
    stt_language: str | None = None,
    countdown: bool = False,
    default: str = "en",
) -> str:
    """Source language for dubbing (prefer Whisper STT over post-normalize script).

    Countdown cues are rewritten to Russian digit words early, so script
    detection falsely matches ``ru`` and re-enables English voice clone.
    """
    tgt = (target_lang or default or "en").lower().strip()
    stt = (stt_language or "").lower().strip()
    if stt in SUPPORTED_LANGS:
        return stt
    detected = detect_transcript_language(text, default=default)
    if countdown and tgt != "en" and detected == tgt:
        return "en"
    return detected


def resolve_language(text: str, default: str = "ru") -> str:
    detected = detect_script_language(text)
    if detected:
        return detected
    default = (default or "en").lower()
    if default in SUPPORTED_LANGS:
        return default
    return "en"


def _letter_count(text: str) -> int:
    return (
        len(_CYRILLIC.findall(text))
        + len(_LATIN.findall(text))
        + len(_HANGUL.findall(text))
        + len(_KANA.findall(text))
        + len(_CJK.findall(text))
    )


def leftover_source_language(text: str, *, source: str, target: str) -> bool:
    """True, если перевод почти целиком остался на языке исходника (не имя)."""
    src = (source or "").lower()
    tgt = (target or "").lower()
    if not tgt or src == tgt:
        return False
    blob = (text or "").strip()
    if not blob or _letter_count(blob) < 10:
        return False
    detected = detect_transcript_language(blob, default=tgt)
    return detected == src and detected != tgt
