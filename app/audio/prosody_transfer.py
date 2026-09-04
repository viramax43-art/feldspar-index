"""Перенос интонации (F0) и энергии с оригинала на синтез — CPU, без VRAM."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _f0_median(wav: np.ndarray, sample_rate: int) -> float | None:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size < sample_rate // 4:
        return None
    try:
        import librosa

        f0, voiced, _ = librosa.pyin(
            wav,
            fmin=65.0,
            fmax=420.0,
            sr=sample_rate,
            frame_length=2048,
        )
        vals = f0[np.asarray(voiced, dtype=bool)]
        vals = vals[np.isfinite(vals) & (vals > 40)]
        if vals.size < 4:
            return None
        return float(np.median(vals))
    except Exception as exc:
        logger.debug("F0 extract failed: %s", exc)
        return None


def _f0_slope_semitones(wav: np.ndarray, sample_rate: int) -> float:
    """Грубый наклон мелодии: конец − начало (в полутонах)."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size < sample_rate // 3:
        return 0.0
    try:
        import librosa

        f0, voiced, _ = librosa.pyin(
            wav,
            fmin=65.0,
            fmax=420.0,
            sr=sample_rate,
            frame_length=2048,
        )
        mask = np.asarray(voiced, dtype=bool) & np.isfinite(f0) & (f0 > 40)
        idx = np.flatnonzero(mask)
        if idx.size < 6:
            return 0.0
        early = float(np.median(f0[idx[: max(2, idx.size // 4)]]))
        late = float(np.median(f0[idx[-max(2, idx.size // 4) :]]))
        if early < 40 or late < 40:
            return 0.0
        return float(12.0 * np.log2(late / early))
    except Exception:
        return 0.0


def pitch_hint_from_audio(wav: np.ndarray, sample_rate: int) -> str:
    """SSML pitch hint по наклону F0 оригинала."""
    slope = _f0_slope_semitones(wav, sample_rate)
    if slope >= 1.8:
        return "+10%"
    if slope >= 0.7:
        return "+5%"
    if slope <= -1.8:
        return "-8%"
    if slope <= -0.7:
        return "-4%"
    return "medium"


def transfer_prosody_from_source(
    tts: np.ndarray,
    source: np.ndarray,
    sample_rate: int,
    *,
    max_shift_semitones: float = 0.0,
    match_energy: bool = True,
) -> np.ndarray:
    """Выравнивает громкость TTS под оригинал. Pitch-shift отключён — ломает тембр клона."""
    out = np.asarray(tts, dtype=np.float32).reshape(-1)
    src = np.asarray(source, dtype=np.float32).reshape(-1)
    if out.size < 16 or src.size < 16:
        return out

    if max_shift_semitones > 0.2:
        src_f0 = _f0_median(src, sample_rate)
        tts_f0 = _f0_median(out, sample_rate)
        if src_f0 and tts_f0 and tts_f0 > 40 and src_f0 > 40:
            semitones = 12.0 * np.log2(src_f0 / tts_f0)
            semitones = float(np.clip(semitones, -max_shift_semitones, max_shift_semitones))
            if abs(semitones) >= 0.35:
                try:
                    import librosa

                    shifted = librosa.effects.pitch_shift(
                        out, sr=sample_rate, n_steps=semitones
                    )
                    out = np.asarray(shifted, dtype=np.float32).reshape(-1)
                except Exception as exc:
                    logger.debug("pitch_shift skipped: %s", exc)

    if match_energy:
        src_rms = float(np.sqrt(np.mean(np.square(src))) or 0.0)
        out_rms = float(np.sqrt(np.mean(np.square(out))) or 0.0)
        if src_rms > 1e-4 and out_rms > 1e-4:
            gain = float(np.clip(src_rms / out_rms, 0.85, 1.18))
            out = out * gain

    peak = float(np.max(np.abs(out)) or 1.0)
    if peak > 0.98:
        out = out * (0.97 / peak)
    return out
