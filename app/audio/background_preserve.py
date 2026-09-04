"""Language replacement on the **original** soundtrack.

Design (rewrite)
----------------
The bed is never synthesized and never replaced by a full-track Demucs stem.

* **Outside speech windows** → bit-copy of the original mix
  (music, heels, clicks, chewing, laughter, crying, room…).
* **Inside speech windows** → Demucs accompaniment (music/SFX without lead
  dialogue) when available; otherwise ``original - vocals``.
* **Always** → add the new TTS voice on top (no ducking of the bed).

Speech windows are taken from **original ASR timing**, not from expanded TTS
placements.  Long translations may spill into pauses; those pauses keep the
original soundtrack and simply gain the new voice.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _mono(audio: np.ndarray) -> np.ndarray:
    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    return wav.reshape(-1)


def _fit(audio: np.ndarray, n: int) -> np.ndarray:
    wav = _mono(audio)
    if n <= 0:
        return wav
    if wav.size == n:
        return wav.copy()
    if wav.size < n:
        return np.pad(wav, (0, n - wav.size))
    return wav[:n].copy()


def _align_length(audio: np.ndarray, target_n: int) -> np.ndarray:
    """Match duration by resampling — never truncate a ~2× buffer (that races the bed)."""
    wav = _mono(audio)
    if target_n <= 0:
        return wav
    if wav.size == 0:
        return np.zeros(target_n, dtype=np.float32)
    if abs(wav.size - target_n) <= max(8, target_n // 500):
        return _fit(wav, target_n)
    ratio = wav.size / float(target_n)
    import librosa

    aligned = librosa.resample(
        wav.astype(np.float64), orig_sr=int(wav.size), target_sr=int(target_n)
    ).astype(np.float32)
    if 1.7 <= ratio <= 2.3 or 0.43 <= ratio <= 0.58:
        logger.warning(
            "bg_rewrite: stem length ratio=%.2f → resampled %d→%d (tempo fix)",
            ratio,
            wav.size,
            target_n,
        )
    return _fit(aligned, target_n)


def _slow_bed(bed: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    """Slow the soundtrack (speed=0.5 → half rate). Output length stays ``len(bed)``."""
    wav = _mono(bed)
    n = wav.size
    speed = float(np.clip(speed, 0.25, 2.0))
    if n < 32 or abs(speed - 1.0) < 0.02:
        return wav
    # speed=0.5 → pretend the clip was captured at sr/2, then resample to sr.
    # That doubles duration (half tempo).  Keep the video slot via crop/pad.
    from_sr = max(1, int(round(float(sample_rate) * speed)))
    slowed = _resample_mono(wav, from_sr, int(sample_rate))
    logger.info(
        "bg_rewrite: background_speed=%.2f (%d → %d samples, keep %d)",
        speed,
        n,
        slowed.size,
        n,
    )
    return _fit(slowed, n)


def _resample_mono(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    wav = _mono(audio)
    if int(orig_sr) == int(target_sr) or wav.size == 0:
        return wav
    import librosa

    return librosa.resample(
        wav, orig_sr=int(orig_sr), target_sr=int(target_sr)
    ).astype(np.float32)


def original_speech_windows(
    segments: Sequence,
    *,
    pad_sec: float = 0.04,
) -> list[tuple[float, float]]:
    """ASR speech spans only — never expanded TTS placement windows."""
    from app.services.timeline_align import speech_window

    pad = max(0.0, float(pad_sec))
    out: list[tuple[float, float]] = []
    for seg in segments:
        sp0, sp1 = speech_window(seg)
        out.append((max(0.0, float(sp0) - pad), float(sp1) + pad))
    return out


def _speech_gate(
    n: int,
    sample_rate: int,
    windows: Sequence[tuple[float, float]],
    *,
    fade_ms: float = 110.0,
) -> np.ndarray:
    """1.0 inside original-speech windows, 0.0 elsewhere, with raised-cosine fades.

    Raised cosine (sin²) avoids the level-slope discontinuity of a linear ramp —
    that slope step is exactly what sounded like a click/ripple at window edges.
    """
    gate = np.zeros(n, dtype=np.float32)
    fade = max(1, int(float(fade_ms) * sample_rate / 1000.0))
    for start, end in windows:
        a = max(0, int(float(start) * sample_rate))
        b = min(n, int(float(end) * sample_rate))
        if b <= a + 2:
            continue
        local = np.ones(b - a, dtype=np.float32)
        f = min(fade, (b - a) // 2)
        if f > 1:
            t = np.linspace(0.0, np.pi, f, dtype=np.float32)
            ramp = 0.5 - 0.5 * np.cos(t)  # smoothstep: 0→1 with zero slope at ends
            local[:f] = ramp
            local[-f:] = ramp[::-1]
        gate[a:b] = np.maximum(gate[a:b], local)
    return gate


def _under_speech_bed(
    original: np.ndarray,
    sample_rate: int,
    *,
    accompaniment: np.ndarray | None,
    vocals: np.ndarray | None,
    vocal_amount: float,
    hard_mute: bool = False,
) -> np.ndarray:
    """Audio used *only* under original speech windows."""
    n = original.size
    if hard_mute:
        # Countdown digits: mute lead voice under the cue only.
        return np.zeros(n, dtype=np.float32)
    if accompaniment is not None and getattr(accompaniment, "size", 0):
        under = _align_length(accompaniment, n)
        # Laughter/cries often sit in the vocal stem — put them back.
        if vocals is not None and getattr(vocals, "size", 0):
            under = _restore_nonspeech_events(
                under,
                _align_length(vocals, n),
                sample_rate=sample_rate,
                event_gain=0.95,
            )
        return under
    if vocals is not None and getattr(vocals, "size", 0):
        vo = _align_length(vocals, n)
        amount = float(np.clip(vocal_amount, 0.0, 1.0))
        cleaned = (original - amount * vo).astype(np.float32)
        # Keep laughs that were subtracted with the vocal stem.
        return _restore_nonspeech_events(
            cleaned, vo, sample_rate=sample_rate, event_gain=0.95
        )
    return original.copy()


def _bridge_windows(
    windows: Sequence[tuple[float, float]],
    bridge_sec: float = 0.6,
) -> list[tuple[float, float]]:
    """Merge windows separated by short gaps.

    Ducking that pops back to 100 % in every 150 ms inter-word gap pumps
    audibly; a real ducker holds through short pauses.
    """
    merged: list[list[float]] = []
    for start, end in sorted((float(s), float(e)) for s, e in windows):
        if merged and start - merged[-1][1] <= float(bridge_sec):
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _match_under_level(
    orig: np.ndarray,
    under: np.ndarray,
    windows: Sequence[tuple[float, float]],
    sample_rate: int,
    *,
    skirt_sec: float = 0.4,
    lo: float = 0.55,
    hi: float = 1.8,
) -> np.ndarray:
    """Scale the under-speech bed to the original level at window edges.

    Demucs accomp / vocal-subtract is a *different* audio than the original mix:
    even with smooth gates its loudness step at window borders was audible as
    ripple. Measure orig in the skirts around each window vs under inside and
    apply one clamped global gain.
    """
    orig_m = _mono(orig)
    under_m = _mono(under)
    n = min(orig_m.size, under_m.size)
    sr = max(1, int(sample_rate))
    if n == 0 or not windows:
        return under
    skirt = max(1, int(float(skirt_sec) * sr))
    orig_sq = 0.0
    orig_n = 0
    under_sq = 0.0
    under_n = 0
    for start, end in windows:
        a = max(0, int(float(start) * sr))
        b = min(n, int(float(end) * sr))
        if b <= a + 2:
            continue
        for sa, sb in ((max(0, a - skirt), a), (b, min(n, b + skirt))):
            if sb > sa:
                seg = orig_m[sa:sb]
                orig_sq += float(np.square(seg, dtype=np.float64).sum())
                orig_n += sb - sa
        seg_u = under_m[a:b]
        under_sq += float(np.square(seg_u, dtype=np.float64).sum())
        under_n += b - a
    if orig_n < sr // 5 or under_n < sr // 5 or under_sq <= 0.0:
        return under
    orig_rms = (orig_sq / orig_n) ** 0.5
    under_rms = (under_sq / under_n) ** 0.5
    if orig_rms <= 1e-6 or under_rms <= 1e-6:
        return under
    gain = min(hi, max(lo, orig_rms / under_rms))
    if abs(gain - 1.0) < 0.03:
        return under
    if gain >= hi * 0.98 or gain <= lo * 1.02:
        # Extreme mismatch = the under-bed is deliberately (near-)muted there
        # (mask mode without Demucs). Boosting it back re-opens the hole.
        return under
    logger.info("bg_rewrite: under-bed level match gain=%.2f", gain)
    return (under_m * gain).astype(np.float32)


def _frame_rms(audio: np.ndarray, sample_rate: int, frame_ms: float) -> np.ndarray:
    mono = _mono(audio)
    if mono.size == 0 or sample_rate <= 0:
        return np.zeros(0, dtype=np.float32)
    frame = max(1, int(float(frame_ms) * int(sample_rate) / 1000.0))
    squared = np.square(mono, dtype=np.float32)
    kernel = np.ones(frame, dtype=np.float32) / float(frame)
    return np.sqrt(
        np.maximum(np.convolve(squared, kernel, mode="same"), 0.0)
    ).astype(np.float32)


def _nonspeech_event_mask(
    vocals: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """1.0 on laughter/breath/clicks in the vocal stem, ~0 on sustained dialogue."""
    vo = _mono(vocals)
    if vo.size == 0:
        return np.zeros(0, dtype=np.float32)
    sr = max(1, int(sample_rate))
    fast = _frame_rms(vo, sr, 10.0)
    slow = _frame_rms(vo, sr, 120.0)
    peak = np.abs(vo)
    crest = peak / np.maximum(slow, 1e-5)
    transient = (fast > np.maximum(slow * 1.55, 6e-4)) | (crest > 3.8)
    burst = (fast > np.maximum(slow * 1.25, 1.2e-3)) & (crest > 2.2)
    event = (transient | burst).astype(np.float32)
    spread = max(1, int(0.05 * sr))
    if spread > 1:
        event = np.clip(
            np.convolve(event, np.ones(spread, dtype=np.float32), mode="same"),
            0.0,
            1.0,
        )
    smooth = max(1, int(0.02 * sr))
    if smooth > 1:
        event = np.convolve(
            event, np.ones(smooth, dtype=np.float32) / float(smooth), mode="same"
        )
    return np.clip(event, 0.0, 1.0).astype(np.float32)


def _restore_nonspeech_events(
    bed: np.ndarray,
    vocals: np.ndarray,
    *,
    sample_rate: int,
    event_gain: float = 0.9,
) -> np.ndarray:
    """Add laughter/breath/clicks from the vocal stem back onto the bed."""
    bg = _mono(bed)
    vo = _mono(vocals)
    n = min(bg.size, vo.size)
    if n == 0:
        return bg
    mask = _nonspeech_event_mask(vo[:n], sample_rate)
    out = bg.copy()
    out[:n] = out[:n] + vo[:n] * mask * float(event_gain)
    peak = float(np.max(np.abs(out)) or 1.0)
    if peak > 0.99:
        out *= 0.98 / peak
    return out.astype(np.float32)


def render_original_background(
    original: np.ndarray,
    dubbed_voice: np.ndarray,
    sample_rate: int,
    speech_windows: Sequence[tuple[float, float]] | None,
    *,
    accompaniment: np.ndarray | None = None,
    accompaniment_sr: int | None = None,
    vocals: np.ndarray | None = None,
    vocals_sr: int | None = None,
    vocal_amount: float = 0.9,
    countdown_mute: bool = False,
    background_speed: float = 1.0,
    bg_gain: float = 1.0,
    voice_gain: float = 1.0,
    duck_windows: Sequence[tuple[float, float]] | None = None,
    speech_duck: float = 1.0,
) -> np.ndarray:
    """Build final audio: original bed outside speech + clean bed under speech + TTS.

    ``countdown_mute`` only clears audio *under digit speech windows*.
    Laughter / breath / room after the countdown stay from the original mix.
    Never wipe the full soundtrack.

    ``speech_duck`` (<1) further attenuates the bed only under ``duck_windows``
    (placed TTS). Outside those windows the bed returns to full level.
    """
    voice = _mono(dubbed_voice)
    orig = _mono(original)
    n = max(orig.size, voice.size, 1)
    orig = _align_length(orig, n)
    voice = _fit(voice, n)

    acc = accompaniment
    if acc is not None and accompaniment_sr is not None:
        acc = _resample_mono(acc, int(accompaniment_sr), int(sample_rate))
    elif acc is not None:
        acc = _mono(acc)
    vo = vocals
    if vo is not None and vocals_sr is not None:
        vo = _resample_mono(vo, int(vocals_sr), int(sample_rate))
    elif vo is not None:
        vo = _mono(vo)

    windows = list(speech_windows or [])
    gate = _speech_gate(n, sample_rate, windows)
    under = _under_speech_bed(
        orig,
        sample_rate,
        accompaniment=None if countdown_mute else acc,
        vocals=None if countdown_mute else vo,
        vocal_amount=vocal_amount,
        hard_mute=bool(countdown_mute),
    )
    if not countdown_mute and acc is not None:
        # Demucs accomp is an independent stem — its absolute level is
        # unreliable vs the original mix, and the step was heard as ripple.
        # (vocal-subtract derives from the original itself: level consistent,
        # no matching — removed lead voice energy must NOT be re-added.)
        under = _match_under_level(orig, under, windows, sample_rate)

    # Outside speech: bit-copy of original (laughs, heels, music…).
    # Inside speech: accomp / vocal-subtract / hard mute for countdown digits.
    bed = orig * (1.0 - gate) + under * gate

    # NOTE: laughs/breaths are re-injected ONLY inside speech windows (in
    # _under_speech_bed). A second global injection here added the vocal stem
    # on top of the original mix — which already contains those events —
    # doubling/comb-filtering every laugh and moan ("background goes crazy").

    if abs(float(background_speed) - 1.0) >= 0.02:
        bed = _slow_bed(bed, sample_rate, float(background_speed))

    duck = float(speech_duck)
    duck_wins = list(duck_windows) if duck_windows is not None else windows
    if 0.0 < duck < 0.999 and duck_wins:
        # Hold duck through short pauses (no per-gap pumping), then ramp smoothly.
        duck_wins = _bridge_windows(duck_wins, bridge_sec=0.6)
        duck_gate = _speech_gate(n, sample_rate, duck_wins, fade_ms=180.0)
        # Inside TTS: bed * duck; outside: unchanged.
        bed = bed * (1.0 - duck_gate * (1.0 - duck))

    out = bed * float(bg_gain) + voice * float(voice_gain)

    outside = float(np.mean(1.0 - gate)) if gate.size else 1.0
    logger.info(
        "bg_rewrite: n=%d windows=%d outside_frac=%.3f speed=%.2f "
        "countdown_under_mute=%s has_accomp=%s has_vocals=%s "
        "bg_gain=%.2f voice_gain=%.2f speech_duck=%.2f duck_wins=%d "
        "orig_rms=%.4f bed_rms=%.4f voice_rms=%.4f",
        n,
        len(windows),
        outside,
        float(background_speed),
        bool(countdown_mute),
        acc is not None and getattr(acc, "size", 0) > 0,
        vo is not None and getattr(vo, "size", 0) > 0,
        float(bg_gain),
        float(voice_gain),
        duck,
        len(duck_wins),
        float(np.sqrt(np.mean(np.square(orig))) or 0.0),
        float(np.sqrt(np.mean(np.square(bed))) or 0.0),
        float(np.sqrt(np.mean(np.square(voice))) or 0.0),
    )

    peak = float(np.max(np.abs(out)) or 1.0)
    if peak > 0.98:
        out *= 0.97 / peak
    return out.astype(np.float32)


# --- Back-compat aliases used by older call sites / tests --------------------


def build_preserved_background(
    original: np.ndarray,
    sample_rate: int,
    speech_windows: Sequence | None = None,
    *,
    vocals: np.ndarray | None = None,
    vocals_sr: int | None = None,
    mode: str = "language_swap",
    dialogue_subtract: float = 0.85,
    pad_sec: float = 0.035,
    target_length: int | None = None,
    accompaniment: np.ndarray | None = None,
    accompaniment_sr: int | None = None,
) -> np.ndarray:
    """Compat: return bed only (without TTS). Prefer ``render_original_background``."""
    windows: list[tuple[float, float]] = []
    for seg in speech_windows or []:
        if hasattr(seg, "start") and hasattr(seg, "end"):
            windows.append(
                (
                    max(0.0, float(seg.start) - float(pad_sec)),
                    float(seg.end) + float(pad_sec),
                )
            )
        else:
            windows.append((float(seg[0]), float(seg[1])))
    silent = np.zeros(
        int(target_length) if target_length is not None else _mono(original).size,
        dtype=np.float32,
    )
    mixed = render_original_background(
        original,
        silent,
        sample_rate,
        windows,
        accompaniment=accompaniment,
        accompaniment_sr=accompaniment_sr,
        vocals=vocals,
        vocals_sr=vocals_sr,
        vocal_amount=dialogue_subtract,
        countdown_mute=(mode == "countdown_mute"),
        background_speed=1.0,
    )
    # Strip the silent voice (already zero) — mixed is the bed.
    return mixed


def mix_language_replacement(
    background: np.ndarray,
    dubbed_voice: np.ndarray,
    *,
    bg_gain: float = 1.0,
    voice_gain: float = 1.0,
) -> np.ndarray:
    bed = _fit(_mono(background) * float(bg_gain), max(_mono(background).size, _mono(dubbed_voice).size))
    voice = _fit(_mono(dubbed_voice) * float(voice_gain), bed.size)
    out = bed + voice
    peak = float(np.max(np.abs(out)) or 1.0)
    if peak > 0.98:
        out *= 0.97 / peak
    return out.astype(np.float32)


def speech_windows_from_placements(
    segments,
    placements,
    clips,
    *,
    pad_before: float = 0.03,
    pad_after: float = 0.04,
) -> list[tuple[float, float]]:
    """Deprecated for bed building — returns original ASR windows."""
    del placements, clips, pad_before, pad_after
    return original_speech_windows(segments, pad_sec=0.04)
