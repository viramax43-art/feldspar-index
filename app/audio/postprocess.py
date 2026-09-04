"""Постобработка синтезированного аудио."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from app.audio import convert_pcm16_to_ogg_opus, convert_wav_to_ogg_opus
from app.audio.preprocess import normalize_loudness


def apply_edge_fade(
    audio: np.ndarray,
    sample_rate: int,
    fade_ms: float = 4.0,
) -> np.ndarray:
    """Короткий fade in/out на краях фразы (мс), защита от щелчков после обрезки."""
    wav = np.asarray(audio, dtype=np.float32).copy()
    n = min(int(sample_rate * fade_ms / 1000.0), wav.size // 2)
    if n <= 0:
        return wav
    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    wav[:n] *= fade_in
    wav[-n:] *= fade_out
    return wav


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 10,
    threshold_db: float = -38.0,
    leading_padding_ms: int = 12,
    trailing_padding_ms: int = 25,
) -> np.ndarray:
    """
    Обрезает ведущую/конечную тишину по RMS-окнам (не по абсолютному нулю,
    чтобы не срезать тихие согласные с/ф/ш). Оставляет небольшой запас.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio

    frame_size = max(1, int(sample_rate * frame_ms / 1000))
    frame_count = int(np.ceil(len(audio) / frame_size))
    rms_values = np.zeros(frame_count, dtype=np.float32)
    for index in range(frame_count):
        start = index * frame_size
        end = min(start + frame_size, len(audio))
        frame = audio[start:end]
        if frame.size:
            rms_values[index] = np.sqrt(np.mean(np.square(frame, dtype=np.float32)))

    max_rms = float(np.max(rms_values))
    if max_rms <= 1e-7:
        return np.empty(0, dtype=np.float32)

    relative_threshold = max_rms * (10.0 ** (threshold_db / 20.0))
    threshold = max(relative_threshold, 1e-4)
    active_frames = np.flatnonzero(rms_values >= threshold)
    if active_frames.size == 0:
        return np.empty(0, dtype=np.float32)

    first_sample = int(active_frames[0] * frame_size)
    last_sample = min(int((active_frames[-1] + 1) * frame_size), len(audio))
    leading_padding = int(sample_rate * leading_padding_ms / 1000)
    trailing_padding = int(sample_rate * trailing_padding_ms / 1000)
    first_sample = max(0, first_sample - leading_padding)
    last_sample = min(len(audio), last_sample + trailing_padding)
    return audio[first_sample:last_sample].copy()


def float_audio_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def apply_tempo(audio: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    """Изменение темпа без смены высоты тона (ffmpeg atempo / WSOLA).

    Встроенный XTTS speed=… даёт металлическую «рябь» — поэтому темп
    меняем здесь, после синтеза на speed=1.0.
    speed < 1 → медленнее, speed > 1 → быстрее. Диапазон 0.5…2.0.
    """
    import subprocess

    from app.audio import require_ffmpeg

    wav = np.asarray(audio, dtype=np.float32)
    speed = float(np.clip(speed, 0.5, 2.0))
    if wav.size == 0 or abs(speed - 1.0) < 1e-3:
        return wav

    pcm = float_audio_to_pcm16(wav)
    ffmpeg = require_ffmpeg()
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-filter:a",
        f"atempo={speed:.4f}",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "pipe:1",
    ]
    result = subprocess.run(cmd, input=pcm, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        # Fallback: librosa phase vocoder (хуже для речи, но лучше чем рябь XTTS)
        try:
            import librosa

            return librosa.effects.time_stretch(wav, rate=speed).astype(np.float32)
        except Exception:
            return wav
    return (np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32767.0)


def light_denoise_synth(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Лёгкое шумоподавление выхода TTS — убирает тихий фон, который иногда генерирует XTTS.

    prop_decrease=0.3 — мягко, чтобы не исказить голос.
    """
    try:
        import noisereduce as nr
    except ImportError:
        return audio
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < 1e-6:
        return audio
    denoised = nr.reduce_noise(
        y=audio, sr=sample_rate, stationary=True, prop_decrease=0.3,
    ).astype(np.float32)
    rms_after = float(np.sqrt(np.mean(np.square(denoised))))
    if rms_after < rms * 0.4:
        return audio
    return denoised


def process_phrase(
    audio: np.ndarray,
    sample_rate: int,
    fade_ms: float = 4.0,
    trim: bool = True,
    normalize: bool = True,
    target_db: float = -18.0,
    trim_threshold_db: float = -38.0,
    leading_padding_ms: int = 12,
    trailing_padding_ms: int = 25,
    soft_normalize: bool = True,
    headroom_db: float = 6.0,
    denoise_output: bool = True,
    tempo: float = 1.0,
) -> np.ndarray:
    """
    Одна фраза: денойз → trim → темп (atempo) → нормализация → fade.

    tempo — скорость воспроизведения без смены высоты (1.0 = нативный темп XTTS).
    Не путать с XTTS speed=… (он даёт рябь). Сдвиги меньше 3% пропускаем.
    """
    wav = np.asarray(audio, dtype=np.float32)
    if wav.size == 0:
        return wav
    if denoise_output:
        wav = light_denoise_synth(wav, sample_rate)
    if trim:
        wav = trim_silence(
            wav,
            sample_rate,
            threshold_db=trim_threshold_db,
            leading_padding_ms=leading_padding_ms,
            trailing_padding_ms=trailing_padding_ms,
        )
    if wav.size == 0:
        return wav
    if abs(float(tempo) - 1.0) > 0.03:
        wav = apply_tempo(wav, sample_rate, float(tempo))
    if normalize:
        wav = normalize_loudness(wav, target_db=target_db, soft=soft_normalize, headroom_db=headroom_db)
    wav = apply_edge_fade(wav, sample_rate, fade_ms=fade_ms)
    return wav


def merge_phrase_pcm(
    chunks: list[np.ndarray],
    pauses: list[float],
    sample_rate: int = 24000,
    fade_ms: float = 4.0,
    enable_ai_marker: bool = False,
    preprocess: bool = True,
) -> np.ndarray:
    """
    Склейка фраз в единый PCM с контролируемыми паузами между ними.
    preprocess=True — сырые чанки (smoke/тесты): trim+normalize+fade здесь.
    preprocess=False — фразы уже обработаны phrase queue: только склейка.
    """
    if not chunks:
        raise ValueError("Нет аудио для сохранения")

    processed: list[np.ndarray] = []
    for chunk in chunks:
        phrase = np.asarray(chunk, dtype=np.float32)
        if preprocess:
            phrase = process_phrase(phrase, sample_rate, fade_ms=fade_ms)
        processed.append(phrase)

    merged_parts: list[np.ndarray] = []
    for idx, phrase in enumerate(processed):
        merged_parts.append(phrase)
        if idx < len(processed) - 1:
            pause = pauses[idx] if idx < len(pauses) else 0.12
            merged_parts.append(np.zeros(int(sample_rate * pause), dtype=np.float32))

    audio = (
        np.concatenate(merged_parts)
        if merged_parts
        else np.empty(0, dtype=np.float32)
    )
    if enable_ai_marker:
        audio = prepend_ai_marker(audio, sample_rate)
    return audio


def insert_pauses_between_chunks(
    chunks: list[np.ndarray],
    sample_rate: int,
    pause_sec: float,
) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float32)
    pause = np.zeros(int(sample_rate * pause_sec), dtype=np.float32)
    parts: list[np.ndarray] = []
    for idx, chunk in enumerate(chunks):
        parts.append(chunk.astype(np.float32))
        if idx < len(chunks) - 1:
            parts.append(pause)
    return np.concatenate(parts)


def prepend_ai_marker(
    audio: np.ndarray,
    sample_rate: int,
    marker_duration_sec: float = 0.35,
    tone_hz: float = 880.0,
) -> np.ndarray:
    """Короткий звуковой маркер перед синтезированной речью."""
    t = np.linspace(0, marker_duration_sec, int(sample_rate * marker_duration_sec), endpoint=False)
    marker = (0.15 * np.sin(2 * np.pi * tone_hz * t)).astype(np.float32)
    fade = np.linspace(1.0, 0.0, len(marker), dtype=np.float32)
    marker = marker * fade
    gap = np.zeros(int(0.1 * sample_rate), dtype=np.float32)
    return np.concatenate([marker, gap, audio])


def write_ai_metadata_flag(wav_path: Path, marker_text: str) -> None:
    meta_path = wav_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "synthetic": True,
                "generator": "app/hybrid",
                "marker_text": marker_text,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def finalize_synthesis_output(
    chunks: list[np.ndarray],
    pauses: list[float],
    output_wav: Path | None,
    output_ogg: Path | None,
    sample_rate: int = 24000,
    enable_ai_marker: bool = False,
    ai_marker_text: str = "",
    prefer_pcm_ogg: bool = True,
    preprocess: bool = True,
) -> Path | None:
    """
    Собирает фразы в памяти. OGG кодируется из PCM stdin без WAV,
    если prefer_pcm_ogg=True и нужен только OGG (или WAV опционален).
    preprocess=False, если фразы уже прошли trim/fade в phrase queue.
    """
    t0 = time.perf_counter()
    audio = merge_phrase_pcm(
        chunks,
        pauses,
        sample_rate=sample_rate,
        enable_ai_marker=enable_ai_marker,
        preprocess=preprocess,
    )
    if enable_ai_marker and output_wav is not None:
        write_ai_metadata_flag(output_wav, ai_marker_text)

    encode_ms = 0.0
    if output_ogg is not None and prefer_pcm_ogg:
        t_enc = time.perf_counter()
        convert_pcm16_to_ogg_opus(float_audio_to_pcm16(audio), sample_rate, output_ogg)
        encode_ms = (time.perf_counter() - t_enc) * 1000
        if output_wav is not None:
            output_wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_wav, audio, sample_rate, subtype="PCM_16")
            result = output_wav
        else:
            result = output_ogg
    else:
        if output_wav is None:
            raise ValueError("Нужен output_wav или prefer_pcm_ogg с output_ogg")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_wav, audio, sample_rate, subtype="PCM_16")
        if output_ogg is not None:
            t_enc = time.perf_counter()
            convert_wav_to_ogg_opus(output_wav, output_ogg)
            encode_ms = (time.perf_counter() - t_enc) * 1000
        result = output_wav

    duration_sec = len(audio) / float(sample_rate)
    total_ms = (time.perf_counter() - t0) * 1000
    # Метрики логирует вызывающий код; возвращаем путь
    _ = (encode_ms, duration_sec, total_ms)
    return result


def load_wav(path: Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    if sample_rate and sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    return audio, sr
