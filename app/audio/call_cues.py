"""Генерация телефонного гудка и разбиение ответа на фразы."""

from __future__ import annotations

import logging
import math
import re
import struct
from pathlib import Path

from app.audio import convert_pcm16_to_ogg_opus

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def split_first_utterance(
    text: str,
    *,
    min_first_chars: int = 12,
    min_rest_chars: int = 24,
) -> tuple[str, str]:
    """
    Разделить ответ на первую фразу и хвост.
    Если разбивать бессмысленно — вернуть (весь текст, "").
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return "", ""

    parts = _SENTENCE_END.split(cleaned, maxsplit=1)
    if len(parts) < 2:
        return cleaned, ""

    first = parts[0].strip()
    rest = parts[1].strip()
    if (
        len(first) < min_first_chars
        or len(rest) < min_rest_chars
        or not first
        or not rest
    ):
        return cleaned, ""
    return first, rest


def generate_ringback_ogg(
    path: Path,
    *,
    sample_rate: int = 48000,
    tone_hz: float = 425.0,
    duration_sec: float = 1.25,
    gap_sec: float = 0.15,
) -> Path:
    """Синтетический гудок (425 Гц) без внешних TTS-моделей."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path

    n_tone = int(sample_rate * duration_sec)
    n_gap = int(sample_rate * gap_sec)
    amplitude = 0.28
    pcm = bytearray()
    for i in range(n_tone):
        env = 1.0
        fade = int(0.02 * sample_rate)
        if i < fade:
            env = i / fade
        elif i > n_tone - fade:
            env = max(0.0, (n_tone - i) / fade)
        sample = amplitude * env * math.sin(2.0 * math.pi * tone_hz * i / sample_rate)
        pcm.extend(struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767)))
    pcm.extend(b"\x00\x00" * n_gap)
    convert_pcm16_to_ogg_opus(bytes(pcm), sample_rate, path)
    logger.info("Сгенерирован ringback: %s", path)
    return path
