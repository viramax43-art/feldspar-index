"""Паравербалика: смех, дыхание, тихие non-speech vocals после речи."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def detect_soft_vocal_events(
    audio: np.ndarray,
    sample_rate: int,
    *,
    after_sec: float,
    until_sec: float | None = None,
    min_event_sec: float = 0.18,
    max_events: int = 4,
) -> list[tuple[float, float]]:
    """Ищет тихие vocal-события (смешок/выдох) после речи.

    Возвращает (start, end) в секундах относительно начала файла.
    """
    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    sr = int(sample_rate)
    if wav.size < sr // 2:
        return []
    a = max(0, int(float(after_sec) * sr))
    b = int(float(until_sec) * sr) if until_sec is not None else wav.size
    b = min(wav.size, max(a + 1, b))
    clip = wav[a:b]
    if clip.size < int(min_event_sec * sr):
        return []

    # относительно локального пика — ASMR тихий
    frame = max(1, int(0.02 * sr))
    hop = frame
    peak = float(np.max(np.abs(clip)) or 1e-6)
    # смешок обычно ниже пика речи, но выше тишины
    lo = max(0.008, 0.04 * peak)
    hi = max(lo * 1.2, 0.55 * peak)
    active = False
    start_i = 0
    events: list[tuple[float, float]] = []
    i = 0
    while i + frame <= clip.size:
        rms = float(np.sqrt(np.mean(np.square(clip[i : i + frame]))))
        if lo <= rms <= hi:
            if not active:
                active = True
                start_i = i
        else:
            if active:
                end_i = i
                dur = (end_i - start_i) / float(sr)
                if dur >= min_event_sec:
                    events.append((a / float(sr) + start_i / float(sr), a / float(sr) + end_i / float(sr)))
                    if len(events) >= max_events:
                        return events
                active = False
        i += hop
    if active:
        end_i = clip.size
        dur = (end_i - start_i) / float(sr)
        if dur >= min_event_sec:
            events.append((a / float(sr) + start_i / float(sr), a / float(sr) + end_i / float(sr)))
    return events


def extract_breath_bed(
    audio: np.ndarray,
    sample_rate: int,
    *,
    gaps: list[tuple[float, float]],
    level_db: float = -24.0,
) -> np.ndarray:
    """Тихий breath/noise bed в паузах (из исходных промежутков или розовый шум)."""
    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    sr = int(sample_rate)
    out = np.zeros_like(wav)
    gain = float(10 ** (level_db / 20.0))
    for g0, g1 in gaps:
        a = max(0, int(float(g0) * sr))
        b = min(wav.size, int(float(g1) * sr))
        if b - a < sr // 20:
            continue
        src = wav[a:b]
        src_rms = float(np.sqrt(np.mean(np.square(src))) or 0.0)
        if src_rms > 1e-4:
            bed = src * (gain / max(src_rms, 1e-6))
        else:
            # мягкий breath-like noise
            noise = np.random.randn(b - a).astype(np.float32) * 0.015
            # простой lowpass через moving average
            k = max(3, int(0.004 * sr) | 1)
            kernel = np.ones(k, dtype=np.float32) / float(k)
            bed = np.convolve(noise, kernel, mode="same") * gain * 8.0
        fade = min(int(0.04 * sr), (b - a) // 3)
        if fade > 1:
            bed = bed.copy()
            bed[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            bed[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        out[a:b] += bed
    peak = float(np.max(np.abs(out)) or 1.0)
    if peak > 0.25:
        out *= 0.25 / peak
    return out


def apply_tail_fade(wav: np.ndarray, sample_rate: int, fade_sec: float = 0.55) -> np.ndarray:
    out = np.asarray(wav, dtype=np.float32).reshape(-1).copy()
    n = max(1, int(float(fade_sec) * sample_rate))
    n = min(n, out.size // 2 if out.size > 4 else out.size)
    if n > 1:
        out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


def overlay_source_events(
    timeline: np.ndarray,
    source: np.ndarray,
    sample_rate: int,
    events: list[tuple[float, float]],
    *,
    gain: float = 0.85,
) -> np.ndarray:
    """Накладывает смех/выдох с оригинала на таймлайн дубляжа (в тех же таймкодах)."""
    out = np.asarray(timeline, dtype=np.float32).reshape(-1).copy()
    src = np.asarray(source, dtype=np.float32).reshape(-1)
    sr = int(sample_rate)
    n = min(out.size, src.size)
    for e0, e1 in events:
        a = max(0, int(float(e0) * sr))
        b = min(n, int(float(e1) * sr))
        if b <= a + 8:
            continue
        piece = src[a:b] * float(gain)
        fade = min(int(0.03 * sr), (b - a) // 4)
        if fade > 1:
            piece = piece.copy()
            piece[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            piece[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        out[a:b] += piece
    peak = float(np.max(np.abs(out)) or 1.0)
    if peak > 0.98:
        out *= 0.97 / peak
    return out


def synthesize_soft_giggle(
    sample_rate: int,
    *,
    duration_sec: float = 0.42,
    level_db: float = -20.0,
) -> np.ndarray:
    """Тихий синтетический смешок, если оригинал не детектировался."""
    sr = int(sample_rate)
    n = max(8, int(float(duration_sec) * sr))
    t = np.arange(n, dtype=np.float32) / float(sr)
    # 2–3 мягких «ха» как амплитудные вспышки + breathy noise
    env = np.zeros(n, dtype=np.float32)
    bursts = (0.05, 0.16, 0.28)
    for b in bursts:
        c = int(b * sr)
        w = int(0.07 * sr)
        a = max(0, c - w // 2)
        b2 = min(n, c + w // 2)
        if b2 > a:
            env[a:b2] += np.hanning(b2 - a).astype(np.float32)
    noise = np.random.randn(n).astype(np.float32)
    k = max(3, int(0.003 * sr) | 1)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    breath = np.convolve(noise, kernel, mode="same")
    # лёгкий тон ~320 Hz под envelope
    tone = np.sin(2.0 * np.pi * 320.0 * t) * 0.35
    tone += np.sin(2.0 * np.pi * 480.0 * t) * 0.12
    wav = (breath * 0.55 + tone * 0.45) * env
    gain = float(10 ** (level_db / 20.0))
    peak = float(np.max(np.abs(wav)) or 1e-6)
    wav = wav * (gain / peak)
    fade = min(int(0.05 * sr), n // 4)
    if fade > 1:
        wav[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        wav[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return wav.astype(np.float32)


def place_clip_at(
    timeline: np.ndarray,
    clip: np.ndarray,
    sample_rate: int,
    start_sec: float,
) -> np.ndarray:
    out = np.asarray(timeline, dtype=np.float32).reshape(-1).copy()
    piece = np.asarray(clip, dtype=np.float32).reshape(-1)
    a = max(0, int(float(start_sec) * sample_rate))
    b = min(out.size, a + piece.size)
    if b <= a:
        return out
    out[a:b] += piece[: b - a]
    peak = float(np.max(np.abs(out)) or 1.0)
    if peak > 0.98:
        out *= 0.97 / peak
    return out
