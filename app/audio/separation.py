"""Изоляция фона для умного дубляжа: Demucs (вокал → минус) или маска по STT."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import soundfile as sf

from app.audio import convert_to_wav

if TYPE_CHECKING:
    from app.services.transcription import TimedSegment

logger = logging.getLogger(__name__)

SeparationMode = Literal["auto", "demucs", "mask"]
_demucs_available: bool | None = None
_demucs_models: dict[str, object] = {}


@dataclass
class DubStems:
    background: np.ndarray
    vocals: np.ndarray
    sample_rate: int
    method: str


def demucs_available() -> bool:
    global _demucs_available
    if _demucs_available is None:
        try:
            import demucs.pretrained  # noqa: F401

            _demucs_available = True
        except ImportError:
            _demucs_available = False
    return _demucs_available


def _to_mono(audio: np.ndarray) -> np.ndarray:
    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    return wav.reshape(-1)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if int(orig_sr) == int(target_sr):
        return np.asarray(audio, dtype=np.float32)
    import librosa

    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim == 1:
        return librosa.resample(wav, orig_sr=int(orig_sr), target_sr=int(target_sr)).astype(
            np.float32
        )
    channels = [
        librosa.resample(wav[:, c], orig_sr=int(orig_sr), target_sr=int(target_sr))
        for c in range(wav.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def mask_speech_regions(
    audio: np.ndarray,
    sample_rate: int,
    segments: list[TimedSegment],
    *,
    speech_gain: float = 0.008,
    fade_ms: float = 35.0,
    pad_ms: float = 60.0,
) -> np.ndarray:
    """Приглушает исходную речь по таймкодам STT (fallback без Demucs)."""
    wav = np.asarray(audio, dtype=np.float32)
    if wav.size == 0 or not segments:
        return wav.copy()

    n = wav.shape[0]
    env = np.ones(n, dtype=np.float32)
    fade = max(1, int(fade_ms * sample_rate / 1000))
    pad = pad_ms * sample_rate / 1000

    for seg in segments:
        a = max(0, int((float(seg.start) - pad) * sample_rate))
        b = min(n, int((float(seg.end) + pad) * sample_rate))
        if b <= a:
            continue
        seg_env = np.full(b - a, speech_gain, dtype=np.float32)
        f = min(fade, (b - a) // 2)
        if f > 0:
            # Raised-cosine ramps: linear ramps leave a slope step (click).
            t = np.linspace(0.0, np.pi, f, dtype=np.float32)
            down = 1.0 - (1.0 - speech_gain) * (0.5 - 0.5 * np.cos(t))
            up = down[::-1]
            seg_env[:f] = np.minimum(seg_env[:f], down)
            seg_env[-f:] = np.minimum(seg_env[-f:], up)
        env[a:b] = np.minimum(env[a:b], seg_env)

    if wav.ndim == 1:
        return wav * env
    return wav * env[:, None]


def build_language_swap_bed(
    original: np.ndarray,
    vocals: np.ndarray | None,
    sample_rate: int,
    windows: list,
    *,
    vocal_subtract: float = 0.55,
    pad_sec: float = 0.04,
    band_reduce: float = 0.12,
) -> np.ndarray:
    """Compat wrapper → ``app.audio.background_preserve``."""
    del band_reduce
    from app.audio.background_preserve import build_preserved_background

    return build_preserved_background(
        original,
        sample_rate,
        windows,
        vocals=vocals,
        mode="language_swap",
        dialogue_subtract=float(vocal_subtract),
        pad_sec=float(pad_sec),
    )


def subtract_vocal_leak(
    background: np.ndarray,
    vocals: np.ndarray,
    leak: float = 0.32,
) -> np.ndarray:
    """Conservative residual cancellation for a Demucs accompaniment stem.

    The vocal stem also contains breaths, laughter, chewing and transients.
    Large subtraction factors create phasey/metallic "water" noise and erase
    those events, so cancellation is intentionally capped.
    """
    bg = np.asarray(background, dtype=np.float32)
    vo = np.asarray(vocals, dtype=np.float32)
    amount = float(np.clip(leak, 0.0, 0.35))
    if amount <= 0.001 or vo.size == 0:
        return bg.copy()
    n = min(bg.shape[0], vo.shape[0])
    if bg.ndim == 1:
        vo_m = _to_mono(vo)[:n]
        out = bg[:n] - amount * vo_m
        if bg.shape[0] > n:
            out = np.concatenate([out, bg[n:]])
        return out.astype(np.float32)
    vo_use = vo[:n]
    if vo_use.ndim == 1:
        vo_use = np.repeat(vo_use[:, None], bg.shape[1], axis=1)
    elif vo_use.shape[1] != bg.shape[1]:
        vo_use = np.repeat(np.mean(vo_use, axis=1, keepdims=True), bg.shape[1], axis=1)
    out = bg[:n] - amount * vo_use
    if bg.shape[0] > n:
        out = np.concatenate([out, bg[n:]], axis=0)
    return out.astype(np.float32)


def _frame_rms_envelope(
    audio: np.ndarray, sample_rate: int, frame_ms: float = 12.0
) -> np.ndarray:
    """Sample-rate RMS envelope for speech/event discrimination."""
    mono = _to_mono(audio)
    if mono.size == 0:
        return np.zeros(0, dtype=np.float32)
    frame = max(1, int(float(frame_ms) * int(sample_rate) / 1000.0))
    squared = np.square(mono, dtype=np.float32)
    kernel = np.ones(frame, dtype=np.float32) / float(frame)
    return np.sqrt(
        np.maximum(np.convolve(squared, kernel, mode="same"), 0.0)
    ).astype(np.float32)


def _event_preservation_mask(
    original: np.ndarray,
    vocals: np.ndarray,
    sample_rate: int,
    segments: list[TimedSegment],
    *,
    pad_ms: float = 55.0,
) -> np.ndarray:
    """Soft mask that keeps non-speech content found in the vocal stem.

    Demucs ``vocals`` is not equivalent to replaceable dialogue: laughter,
    crying, breaths, chewing, heel clicks and impacts frequently land there.
    Outside ASR dialogue windows the stem is retained. Inside dialogue windows
    only short/high-crest events are retained, suppressing sustained speech.
    """
    vo = _to_mono(vocals)
    original_mono = _to_mono(original)
    n = min(vo.size, original_mono.size)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    fast = _frame_rms_envelope(vo[:n], sample_rate, 8.0)
    slow = _frame_rms_envelope(vo[:n], sample_rate, 90.0)
    peak = np.abs(vo[:n])
    crest = peak / np.maximum(slow, 1e-5)
    transient = (fast > np.maximum(slow * 1.65, 8e-4)) | (crest > 4.5)
    strong_threshold = max(
        0.025, float(np.percentile(fast, 96)) if fast.size else 0.025
    )
    event = transient | (fast > strong_threshold)

    speech = np.zeros(n, dtype=bool)
    pad = int(max(0.0, float(pad_ms)) * sample_rate / 1000.0)
    for seg in segments:
        a = max(0, int(float(seg.start) * sample_rate) - pad)
        b = min(n, int(float(seg.end) * sample_rate) + pad)
        if b > a:
            speech[a:b] = True

    mask = np.where(speech, event.astype(np.float32), 1.0).astype(np.float32)
    # Expand attacks/tails and then smooth to avoid clicks.
    spread = max(1, int(0.035 * sample_rate))
    if spread > 1:
        mask = np.clip(
            np.convolve(mask, np.ones(spread, dtype=np.float32), mode="same"),
            0.0,
            1.0,
        )
    smooth = max(1, int(0.012 * sample_rate))
    if smooth > 1:
        mask = np.convolve(
            mask,
            np.ones(smooth, dtype=np.float32) / float(smooth),
            mode="same",
        )
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def preserve_background_events(
    background: np.ndarray,
    vocals: np.ndarray,
    original: np.ndarray,
    sample_rate: int,
    segments: list[TimedSegment],
    *,
    event_gain: float = 0.82,
) -> np.ndarray:
    """Restore non-speech events from vocals onto the accompaniment stem.

    Music/SFX already in the Demucs accompaniment remain untouched. Short
    events, laughter, crying, chewing and breaths are restored, while sustained
    ASR-aligned source dialogue remains removed.
    """
    bg = np.asarray(background, dtype=np.float32)
    vo = np.asarray(vocals, dtype=np.float32)
    original_arr = np.asarray(original, dtype=np.float32)
    n = min(bg.shape[0], vo.shape[0], original_arr.shape[0])
    if n <= 0:
        return bg.copy()
    mask = _event_preservation_mask(
        original_arr[:n], vo[:n], sample_rate, segments
    )
    out = bg.copy()
    if bg.ndim == 1:
        out[:n] += _to_mono(vo[:n]) * mask * float(event_gain)
    else:
        vo_use = vo[:n]
        if vo_use.ndim == 1:
            vo_use = np.repeat(vo_use[:, None], bg.shape[1], axis=1)
        elif vo_use.shape[1] != bg.shape[1]:
            vo_use = np.repeat(
                np.mean(vo_use, axis=1, keepdims=True), bg.shape[1], axis=1
            )
        out[:n] += vo_use * mask[:, None] * float(event_gain)
    peak = float(np.max(np.abs(out)) or 1.0)
    if peak > 0.99:
        out *= 0.98 / peak
    return out.astype(np.float32)


def gate_speech_band(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = -40.0,
    band_hz: tuple[float, float] = (200.0, 3200.0),
    reduce_to: float = 0.02,
    frame_ms: float = 25.0,
) -> np.ndarray:
    """Глушит остаточный голос в полосе речи, если RMS выше порога (EN bleed)."""
    wav = np.asarray(audio, dtype=np.float32)
    mono = _to_mono(wav)
    if mono.size < sample_rate // 10:
        return wav.copy() if wav is not audio else wav
    try:
        from scipy.signal import butter, sosfilt

        low, high = band_hz
        nyq = 0.5 * float(sample_rate)
        lo = max(20.0, low) / nyq
        hi = min(high, nyq * 0.98) / nyq
        if not (0.0 < lo < hi < 1.0):
            return wav.copy()
        sos = butter(2, [lo, hi], btype="band", output="sos")
        band = sosfilt(sos, mono).astype(np.float32)
    except Exception:
        # без scipy — грубый highpass через diff
        band = np.diff(mono, prepend=mono[:1]).astype(np.float32)

    frame = max(1, int(frame_ms * sample_rate / 1000.0))
    thr = float(10 ** (threshold_db / 20.0))
    env = np.ones(mono.size, dtype=np.float32)
    i = 0
    while i < mono.size:
        j = min(mono.size, i + frame)
        rms = float(np.sqrt(np.mean(np.square(band[i:j]))) or 0.0)
        if rms >= thr:
            env[i:j] = min(float(reduce_to), float(env[i]))
        i = j
    # сгладить огибающую
    fade = max(1, frame // 2)
    if fade > 1 and mono.size > fade * 2:
        kernel = np.ones(fade, dtype=np.float32) / float(fade)
        env = np.convolve(env, kernel, mode="same")
        env = np.clip(env, float(reduce_to), 1.0)

    if wav.ndim == 1:
        return (wav * env).astype(np.float32)
    return (wav * env[:, None]).astype(np.float32)


def strip_vocals_hard(
    background: np.ndarray,
    vocals: np.ndarray | None,
    sample_rate: int,
    *,
    leak: float = 1.05,
) -> np.ndarray:
    """Максимально вычищает вокал из фона + гейт остатков в speech-band."""
    bg = np.asarray(background, dtype=np.float32)
    if vocals is not None and getattr(vocals, "size", 0):
        bg = subtract_vocal_leak(bg, vocals, leak=max(0.9, float(leak)))
    bg = gate_speech_band(bg, sample_rate, threshold_db=-42.0, reduce_to=0.01)
    return bg


def silence_bed_under_voice(
    bed: np.ndarray,
    voice: np.ndarray,
    sample_rate: int,
    *,
    voice_thresh: float = 0.016,
    pad_ms: float = 55.0,
    fade_ms: float = 12.0,
) -> np.ndarray:
    """Жёстко глушит фон там, где играет озвучка — убирает эхо/гул оригинала под TTS."""
    bg = np.asarray(bed, dtype=np.float32).copy()
    vo = _to_mono(voice)
    n = min(bg.shape[0], vo.size)
    if n < sample_rate // 20:
        return bg

    win = max(1, int(0.02 * sample_rate))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    power = np.convolve(vo[:n] * vo[:n], kernel, mode="same")
    rms = np.sqrt(np.maximum(power, 0.0))
    active = rms >= float(voice_thresh)
    if not np.any(active):
        return bg

    pad = max(0, int(pad_ms * sample_rate / 1000.0))
    if pad:
        kernel_p = np.ones(2 * pad + 1, dtype=np.float32)
        active = np.convolve(active.astype(np.float32), kernel_p, mode="same") > 0.0

    env = np.ones(n, dtype=np.float32)
    env[active] = 0.0
    fade = max(1, int(fade_ms * sample_rate / 1000.0))
    if fade > 1:
        # Pad with edge values so mode=same does not dip the bed at clip ends.
        padded = np.pad(env, fade, mode="edge")
        kernel_f = np.ones(fade, dtype=np.float32) / float(fade)
        smoothed = np.convolve(padded, kernel_f, mode="same")
        env = np.clip(smoothed[fade : fade + n], 0.0, 1.0)

    if bg.ndim == 1:
        bg[:n] *= env
    else:
        bg[:n] *= env[:, None]
    return bg


def debleed_bed_in_windows(
    bed: np.ndarray,
    vocals: np.ndarray | None,
    sample_rate: int,
    segments: list[TimedSegment],
    *,
    leak: float = 0.95,
    pad_ms: float = 160.0,
) -> np.ndarray:
    """В окнах речи сильнее вычитает остаточный вокал (реверб/эхо Demucs other)."""
    bg = np.asarray(bed, dtype=np.float32).copy()
    if vocals is None or not getattr(vocals, "size", 0) or not segments:
        return bg
    vo = _to_mono(np.asarray(vocals, dtype=np.float32))
    n = min(bg.shape[0], vo.size)
    pad = pad_ms / 1000.0
    for seg in segments:
        a = max(0, int((float(seg.start) - pad) * sample_rate))
        b = min(n, int((float(seg.end) + pad) * sample_rate))
        if b <= a:
            continue
        if bg.ndim == 1:
            bg[a:b] = bg[a:b] - float(leak) * vo[a:b]
        else:
            vo_slice = vo[a:b]
            for ch in range(bg.shape[1]):
                bg[a:b, ch] = bg[a:b, ch] - float(leak) * vo_slice
    return bg.astype(np.float32)


def _load_demucs(device: str):
    import torch
    from demucs.pretrained import get_model

    dev = torch.device(
        device if device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    key = f"htdemucs:{dev}"
    model = _demucs_models.get(key)
    if model is None:
        model = get_model("htdemucs")
        model.eval()
        model.to(dev)
        _demucs_models[key] = model
    return model, dev


def _separate_stems_demucs(
    wav_path: Path,
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Возвращает (background, vocals) как (samples, channels) на native SR Demucs."""
    import torch
    from demucs.apply import apply_model

    model, dev = _load_demucs(device)
    audio, sr = sf.read(str(wav_path), always_2d=True)
    wav = np.asarray(audio, dtype=np.float32).T  # channels × samples
    if wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)

    ref = wav.mean(axis=1, keepdims=True)
    std = float(wav.std()) or 1.0
    wav_norm = (wav - ref) / std
    wav_t = torch.from_numpy(wav_norm).float().unsqueeze(0).to(dev)

    model_sr = int(model.samplerate)
    if int(sr) != model_sr:
        import torchaudio

        wav_t = torchaudio.functional.resample(wav_t, int(sr), model_sr)

    with torch.no_grad():
        sources = apply_model(
            model,
            wav_t,
            device=dev,
            progress=False,
            num_workers=0,
            overlap=0.25,
        )[0]

    names = list(getattr(model, "sources", ("drums", "bass", "other", "vocals")))
    idx = {name: i for i, name in enumerate(names)}
    bg = sources[idx["drums"]] + sources[idx["bass"]] + sources[idx["other"]]
    vocals = sources[idx["vocals"]]
    bg_np = (bg.cpu().numpy() * std + ref).T.astype(np.float32)
    vo_np = (vocals.cpu().numpy() * std + ref).T.astype(np.float32)
    del wav_t, sources, wav_norm, wav, bg, vocals
    return bg_np, vo_np, model_sr


def extract_stems_for_dub(
    video_path: Path,
    segments: list[TimedSegment],
    work_dir: Path,
    *,
    mode: SeparationMode = "auto",
    device: str = "cpu",
    sample_rate: int = 24000,
    speech_gain: float = 0.28,
    vocal_leak: float = 0.32,
) -> DubStems:
    """Фон без речи + вокальный стебель для клона."""
    work_dir.mkdir(parents=True, exist_ok=True)
    work_wav = work_dir / "source_stems.wav"
    convert_to_wav(video_path, work_wav, sample_rate=44100, mono=False)

    source, file_sr = sf.read(str(work_wav), always_2d=True)
    source = np.asarray(source, dtype=np.float32)
    if int(file_sr) != 44100:
        source = _resample(source, int(file_sr), 44100)

    use_demucs = mode == "demucs" or (mode == "auto" and demucs_available())
    background: np.ndarray | None = None
    vocals: np.ndarray | None = None
    method = "mask"
    native_sr = 44100

    if use_demucs:
        try:
            background, vocals, native_sr = _separate_stems_demucs(
                work_wav, device=device
            )
            # Light leak cancellation only — heavy subtraction removes instruments
            # that share energy with the vocal stem and hollows out the soundtrack.
            background = subtract_vocal_leak(
                background, vocals, leak=min(0.45, max(0.0, float(vocal_leak)))
            )
            method = "demucs"
            logger.info(
                "Stems via Demucs: bg=%s vocals=%s sr=%d",
                background.shape,
                vocals.shape,
                native_sr,
            )
        except Exception:
            logger.exception("Demucs separation failed, falling back to STT mask")
            if mode == "demucs":
                raise
            background = None
            vocals = None

    if background is None:
        background = mask_speech_regions(
            source,
            44100,
            segments,
            speech_gain=min(0.012, float(speech_gain)),
            fade_ms=90.0,
            pad_ms=220.0,
        )
        vocals = source
        native_sr = 44100
        logger.info("Background via STT speech mask (%d segments)", len(segments))

    del source
    background = _resample(background, native_sr, sample_rate)
    vocals = _resample(_to_mono(vocals), native_sr, sample_rate)
    work_wav.unlink(missing_ok=True)
    return DubStems(
        background=background,
        vocals=vocals,
        sample_rate=sample_rate,
        method=method,
    )


def extract_background_for_dub(
    video_path: Path,
    segments: list[TimedSegment],
    out_path: Path,
    *,
    mode: SeparationMode = "auto",
    device: str = "cpu",
    sample_rate: int = 24000,
    speech_gain: float = 0.008,
) -> Path:
    """Совместимость: только фон на диск."""
    stems = extract_stems_for_dub(
        video_path,
        segments,
        out_path.parent,
        mode=mode,
        device=device,
        sample_rate=sample_rate,
        speech_gain=max(0.18, speech_gain) if speech_gain < 0.05 else speech_gain,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), stems.background, sample_rate, subtype="PCM_16")
    return out_path


def mix_dub_tracks(
    background: np.ndarray,
    voice: np.ndarray,
    *,
    bg_volume: float = 0.55,
    voice_volume: float = 1.0,
    duck_floor: float = 0.28,
    sample_rate: int = 24000,
    speech_windows: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Склеивает фон и озвучку; под речью фон приседает, чтобы не слышать призраки."""
    bg = np.asarray(background, dtype=np.float32)
    vo = _to_mono(voice)
    n = max(bg.shape[0], vo.size)
    if n == 0:
        return np.zeros(1, dtype=np.float32)

    stereo = bg.ndim == 2
    if stereo:
        bed = np.zeros((n, bg.shape[1]), dtype=np.float32)
        bed[: bg.shape[0]] = bg * float(bg_volume)
    else:
        bed = np.zeros(n, dtype=np.float32)
        bed[: bg.shape[0]] = bg.reshape(-1) * float(bg_volume)

    fg = np.zeros(n, dtype=np.float32)
    fg[: vo.size] = vo * float(voice_volume)

    if 0.0 <= duck_floor < 0.999 and fg.size:
        win = max(1, int(0.03 * sample_rate))
        kernel = np.ones(win, dtype=np.float32) / float(win)
        power = np.convolve(fg * fg, kernel, mode="same")
        rms = np.sqrt(np.maximum(power, 0.0))
        peak = float(np.percentile(rms, 93)) if rms.size else 0.0
        if peak > 1e-5:
            amount = np.clip(rms / peak, 0.0, 1.0)
            gain = 1.0 - (1.0 - float(duck_floor)) * amount
            if stereo:
                bed *= gain[:, None]
            else:
                bed *= gain

    if speech_windows:
        mute = np.ones(n, dtype=np.float32)
        floor = min(0.04, max(0.0, float(duck_floor)))
        fade = max(1, int(0.04 * sample_rate))
        for start, end in speech_windows:
            a = max(0, int(float(start) * sample_rate))
            b = min(n, int(float(end) * sample_rate))
            if b <= a:
                continue
            mute[a:b] = np.minimum(mute[a:b], floor)
            f = min(fade, (b - a) // 2)
            if f > 1:
                ramp = np.linspace(1.0, floor, f, dtype=np.float32)
                mute[a : a + f] = np.minimum(mute[a : a + f], ramp)
                mute[b - f : b] = np.minimum(mute[b - f : b], ramp[::-1])
        if stereo:
            bed *= mute[:, None]
        else:
            bed *= mute

    if stereo:
        mixed = bed.copy()
        mixed[:, 0] += fg
        mixed[:, 1] += fg
    else:
        mixed = bed + fg

    peak = float(np.max(np.abs(mixed)) or 1.0)
    if peak > 0.95:
        mixed *= 0.94 / peak
    return mixed.astype(np.float32)
