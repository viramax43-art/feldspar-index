"""Шумоподавление референсов для клонирования голоса из видео.

Обычный noisereduce на всех клипах даёт металлическую рябь в TTS.
Здесь: оцениваем SNR, шумодав только на грязных кусках, сохраняем RMS речи.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _frame_rms(audio: np.ndarray, sample_rate: int, frame_ms: float = 25.0) -> np.ndarray:
    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame = max(1, int(float(frame_ms) * 0.001 * sample_rate))
    hop = max(1, frame // 2)
    if wav.size < frame:
        return np.asarray([float(np.sqrt(np.mean(np.square(wav))))], dtype=np.float32)
    vals: list[float] = []
    for i in range(0, wav.size - frame + 1, hop):
        chunk = wav[i : i + frame]
        vals.append(float(np.sqrt(np.mean(np.square(chunk)))))
    return np.asarray(vals, dtype=np.float32)


def estimate_snr_db(audio: np.ndarray, sample_rate: int) -> float:
    """Грубая оценка SNR: громкие кадры vs тихие.

    Непрерывный чистый тон без пауз даёт p15≈p90 — это не шум, а отсутствие
    динамики; такие клипы считаем достаточно чистыми (высокий SNR).
    """
    rms = _frame_rms(audio, sample_rate)
    if rms.size < 4:
        peak = float(np.max(np.abs(audio)) or 0.0)
        return 40.0 if peak > 0.05 else 0.0
    speech = float(np.percentile(rms, 90))
    noise = float(np.percentile(rms, 15))
    if speech < 1e-6:
        return 0.0
    noise = max(noise, 1e-8)
    snr = float(20.0 * np.log10(speech / noise))
    # Low frame dynamics on a loud clip → no audible noise floor.
    if speech > 0.01 and (speech - noise) / speech < 0.28:
        return max(snr, 22.0)
    return snr


def extract_noise_sample(
    audio: np.ndarray,
    sample_rate: int,
    *,
    max_sec: float = 0.45,
) -> np.ndarray | None:
    """Тихие кадры клипа как профиль шума (если их достаточно)."""
    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame = max(1, int(0.025 * sample_rate))
    hop = max(1, frame // 2)
    if wav.size < frame * 4:
        return None
    scores: list[tuple[float, int]] = []
    for i in range(0, wav.size - frame + 1, hop):
        chunk = wav[i : i + frame]
        scores.append((float(np.sqrt(np.mean(np.square(chunk)))), i))
    scores.sort(key=lambda x: x[0])
    need = max(frame, int(float(max_sec) * sample_rate))
    pieces: list[np.ndarray] = []
    used = 0
    for _rms, start in scores:
        if used >= need:
            break
        piece = wav[start : start + frame]
        pieces.append(piece)
        used += piece.size
    if used < int(0.12 * sample_rate):
        return None
    noise = np.concatenate(pieces).astype(np.float32)
    # Noise must stay quieter than speech — otherwise we subtract voice.
    speech_p = float(np.percentile(_frame_rms(wav, sample_rate), 75))
    noise_rms = float(np.sqrt(np.mean(np.square(noise))))
    if speech_p > 1e-6 and noise_rms > speech_p * 0.55:
        return None
    return noise


def _match_speech_rms(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Восстановить громкость речи после спектрального гейта."""
    b = np.asarray(before, dtype=np.float32).reshape(-1)
    a = np.asarray(after, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0:
        return a
    # Use upper-percentile RMS so silence doesn't dominate.
    def _p90(x: np.ndarray) -> float:
        frame = max(1, x.size // 40)
        if x.size < frame:
            return float(np.sqrt(np.mean(np.square(x))))
        vals = [
            float(np.sqrt(np.mean(np.square(x[i : i + frame]))))
            for i in range(0, x.size - frame + 1, frame)
        ]
        return float(np.percentile(vals, 90)) if vals else 0.0

    rb, ra = _p90(b), _p90(a)
    if rb < 1e-6 or ra < 1e-6:
        return a
    gain = min(2.2, max(0.55, rb / ra))
    out = a * gain
    peak = float(np.max(np.abs(out)) or 0.0)
    if peak > 0.97:
        out *= 0.95 / peak
    return out.astype(np.float32)


def denoise_for_voice_clone(
    audio: np.ndarray,
    sample_rate: int,
    *,
    snr_threshold_db: float = 14.0,
    prop_decrease: float = 0.72,
    force: bool = False,
    soft: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Шумодав для клон-референса.

    Returns
    -------
    cleaned, info
        info: snr_db, applied, prop, reason
    """
    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    info: dict[str, Any] = {
        "snr_db": None,
        "applied": False,
        "prop": 0.0,
        "reason": "skip",
    }
    if wav.size < max(256, sample_rate // 5):
        info["reason"] = "too_short"
        return wav, info

    snr = estimate_snr_db(wav, sample_rate)
    info["snr_db"] = round(snr, 2)
    if soft:
        # ASMR/whisper: very light or skip — denoise kills breath timbre.
        if not force and snr >= max(8.0, float(snr_threshold_db) - 4.0):
            info["reason"] = "whisper_clean"
            return wav, info
        prop = min(0.45, float(prop_decrease) * 0.55)
    else:
        if not force and snr >= float(snr_threshold_db):
            info["reason"] = "clean_enough"
            return wav, info
        # Dirtier → stronger gate, capped to avoid metallic artifacts.
        if snr < 6.0:
            prop = min(0.88, float(prop_decrease) + 0.12)
        elif snr < 10.0:
            prop = float(prop_decrease)
        else:
            prop = max(0.45, float(prop_decrease) - 0.12)

    try:
        import noisereduce as nr
    except ImportError:
        info["reason"] = "noisereduce_missing"
        logger.warning("noisereduce not installed — clone denoise skipped")
        return wav, info

    noise = extract_noise_sample(wav, sample_rate)
    try:
        if noise is not None and noise.size >= int(0.12 * sample_rate):
            cleaned = nr.reduce_noise(
                y=wav,
                sr=sample_rate,
                y_noise=noise,
                stationary=False,
                prop_decrease=prop,
            )
        else:
            cleaned = nr.reduce_noise(
                y=wav,
                sr=sample_rate,
                stationary=True,
                prop_decrease=prop,
            )
    except Exception:
        logger.exception("clone denoise failed")
        info["reason"] = "error"
        return wav, info

    cleaned = np.asarray(cleaned, dtype=np.float32).reshape(-1)
    if cleaned.size != wav.size:
        n = min(cleaned.size, wav.size)
        tmp = wav.copy()
        tmp[:n] = cleaned[:n]
        cleaned = tmp
    # Reject if we crushed the voice (over-suppression).
    before_rms = float(np.sqrt(np.mean(np.square(wav))))
    after_rms = float(np.sqrt(np.mean(np.square(cleaned))))
    if before_rms > 1e-6 and after_rms < before_rms * 0.28:
        info["reason"] = "over_suppressed"
        return wav, info
    cleaned = _match_speech_rms(wav, cleaned)
    info["applied"] = True
    info["prop"] = round(float(prop), 3)
    info["reason"] = "denoised"
    return cleaned.astype(np.float32), info
