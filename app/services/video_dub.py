"""Замена языка в видео: STT → перевод → TTS-клон → vocal swap на полной оригинальной дорожке.

Обычное видео: весь оригинальный микс (музыка, шаги, щелчки, чавканье, смех…)
остаётся; в окнах речи вычитается только vocal-stem и кладётся перевод.
Demucs нужен для клона и вычитания речи, не как «фон» без SFX.
"""

from __future__ import annotations

import asyncio
import gc
import html
import json
import logging
import math
import re
import shutil
import subprocess
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from app.audio import convert_to_wav, find_ffprobe, require_ffmpeg, safe_user_path
from app.audio.separation import (
    extract_stems_for_dub,
    mix_dub_tracks,
)
from app.audio.postprocess import apply_tempo, float_audio_to_pcm16, trim_silence
from app.audio.preprocess import apply_bandpass
from app.config import Settings
from app.services.gigachat import GigaChatService
from app.services.synthesis import SynthesisService
from app.services.transcription import TimedSegment, TranscriptionService
from app.text.accent import AccentService
from app.text.language import detect_transcript_language, leftover_source_language, resolve_dub_source_language
from app.text.reply_lang import LANG_NAMES, normalize_reply_lang, xtts_chunk_limit
from app.text.ssml import (
    intonation_from_prosody,
    parse_ssml,
    transfer_ssml_for_slot,
)

logger = logging.getLogger(__name__)

ProgressCb = Callable[[int, int, str], Awaitable[None]]


@dataclass
class DubResult:
    video_path: Path
    segments: list[TimedSegment]
    translated: list[str]
    srt_path: Path | None = None
    clone_refs: list[Path] = field(default_factory=list)
    clone_sec: float = 0.0
    # Per-cue wav dump (set when render was called with cue_audio_dir).
    cue_audio_dir: Path | None = None
    cue_audio_sr: int = 0
    # Where each cue was placed on the timeline (start, end).
    placements: list[tuple[float, float]] = field(default_factory=list)


def select_clone_segments(
    segments: list[TimedSegment],
    *,
    max_sec: float = 18.0,
    max_clips: int = 5,
    min_clip_sec: float = 1.2,
    prefer_whisper: bool = False,
    source_audio: np.ndarray | None = None,
    source_sr: int | None = None,
) -> list[TimedSegment]:
    """Лучшие куски речи из видео для XTTS conditioning (не профиль пользователя)."""
    candidates = [
        seg
        for seg in segments
        if seg.duration >= min_clip_sec * 0.55 and (seg.text or "").strip()
    ]
    from app.text.vocalizations import is_background_vocalization

    candidates = [
        seg for seg in candidates if not is_background_vocalization(seg.text or "")
    ]
    # Whisper bonus only for ASMR/countdown. On normal videos it picks quiet,
    # breathy clips and the clone sounds like a different speaker.
    # SNR bonus: prefer cleaner speech so clone refs aren't full of bed noise.
    from app.audio.clone_denoise import estimate_snr_db

    def _snr_bonus(s: TimedSegment) -> float:
        if source_audio is None or not source_sr:
            return 0.0
        a = max(0, int(float(s.start) * int(source_sr)))
        b = min(len(source_audio), int(float(s.end) * int(source_sr)))
        if b - a < int(0.25 * int(source_sr)):
            return 0.0
        try:
            snr = estimate_snr_db(source_audio[a:b], int(source_sr))
        except Exception:
            return 0.0
        # Map ~6–24 dB into 0–1.2 bonus; dirty clips get almost nothing.
        return max(0.0, min(1.2, (snr - 6.0) / 15.0))

    def _score(s: TimedSegment) -> float:
        rms = float(s.rms or 0.0)
        whisper_bonus = (
            0.35 if prefer_whisper and 0.005 <= rms <= 0.06 else 0.0
        )
        return (
            whisper_bonus
            + min(s.duration, 10.0) * 0.12
            + min(rms, 0.08) * 1.5
            + _snr_bonus(s)
        )

    candidates.sort(key=_score, reverse=True)
    picked: list[TimedSegment] = []
    total = 0.0
    for seg in candidates:
        if len(picked) >= max_clips:
            break
        if total >= max_sec:
            break
        take = min(seg.duration, max_sec - total, 10.0)
        if take < min_clip_sec * 0.5:
            continue
        if take < seg.duration - 0.05:
            picked.append(
                TimedSegment(
                    start=seg.start,
                    end=seg.start + take,
                    text=seg.text,
                    style=seg.style,
                    rms=seg.rms,
                )
            )
        else:
            picked.append(seg)
        total += take
    # Keep score order: Fish/XTTS consume the FIRST reference, so the cleanest,
    # loudest clip must come first (time order would hand them a noisy intro).
    return picked


def _clean_clone_clip(
    clip: np.ndarray,
    sample_rate: int,
    *,
    from_original: bool = False,
    preserve_whisper: bool = False,
    enable_denoise: bool = True,
    denoise_snr_db: float = 14.0,
    denoise_prop: float = 0.72,
) -> np.ndarray:
    """Зачистка референса для клона: HP + шумодав только на грязных клипах."""
    del from_original  # same path for original and Demucs vocals
    wav = np.asarray(clip, dtype=np.float32).reshape(-1)
    if wav.size < sample_rate // 4:
        return wav
    low = 50 if preserve_whisper else 80
    high = 16000 if preserve_whisper else 12000
    try:
        wav = apply_bandpass(wav, sample_rate, low_hz=low, high_hz=high, order=2)
    except Exception:
        logger.debug("clone bandpass skipped", exc_info=True)
    if enable_denoise:
        from app.audio.clone_denoise import denoise_for_voice_clone

        wav, info = denoise_for_voice_clone(
            wav,
            sample_rate,
            snr_threshold_db=float(denoise_snr_db),
            prop_decrease=float(denoise_prop),
            soft=bool(preserve_whisper),
        )
        if info.get("applied"):
            logger.info(
                "Clone ref denoise: snr=%.1f dB prop=%.2f",
                float(info.get("snr_db") or 0.0),
                float(info.get("prop") or 0.0),
            )
        else:
            logger.debug(
                "Clone ref denoise skipped (%s, snr=%s)",
                info.get("reason"),
                info.get("snr_db"),
            )
    peak = float(np.max(np.abs(wav)) or 0.0)
    if peak > 0.98:
        wav = wav * (0.92 / peak)
    return wav.astype(np.float32)



def extract_clone_references(
    video_path: Path,
    segments: list[TimedSegment],
    out_dir: Path,
    *,
    sample_rate: int = 22050,
    max_sec: float = 18.0,
    max_clips: int = 5,
    min_clip_sec: float = 1.2,
    fallback_sec: float = 10.0,
    source_audio: np.ndarray | None = None,
    source_sr: int | None = None,
    from_original: bool = False,
    prefer_whisper: bool = False,
    enable_denoise: bool = True,
    denoise_snr_db: float = 14.0,
    denoise_prop: float = 0.72,
) -> tuple[list[Path], float]:
    """Вырезает речь говорящего → wav-референсы для zero-shot клона.

    Countdown/ASMR: original mono (breath/room). Ordinary: Demucs vocals when
    available — the full mix embeds music into XTTS conditioning and shifts timbre.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if source_audio is not None:
        audio = np.asarray(source_audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        src_sr = int(source_sr or sample_rate)
        if src_sr != sample_rate:
            import librosa

            audio = librosa.resample(audio, orig_sr=src_sr, target_sr=sample_rate)
    else:
        source_wav = out_dir / "source_clone.wav"
        convert_to_wav(video_path, source_wav, sample_rate=sample_rate, mono=True)
        audio, file_sr = sf.read(str(source_wav), always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if int(file_sr) != sample_rate:
            import librosa

            audio = librosa.resample(audio, orig_sr=int(file_sr), target_sr=sample_rate)
    duration = len(audio) / float(sample_rate)
    from app.text.digit_speech import looks_like_countdown

    picked = select_clone_segments(
        segments,
        max_sec=max_sec,
        max_clips=max_clips,
        min_clip_sec=min_clip_sec,
        prefer_whisper=prefer_whisper,
        source_audio=audio,
        source_sr=sample_rate,
    )
    # Countdown: короткие цифры не годны как ref — берём компактный шёпотный участок,
    # НЕ весь span после energy-snap (иначе 10–500с тишины/SFX портят клон).
    preserve_whisper = False
    if looks_like_countdown(segments) and segments:
        starts = sorted(float(s.start) for s in segments)
        s0 = max(0.0, starts[0] - 0.08)
        # First ~3 digit windows or 6s — enough timbre, keeps ref quiet-but-usable.
        early_end = starts[min(2, len(starts) - 1)] + 1.2
        # Cap hard: long refs leak source-language phonetics into other languages.
        s1 = min(duration, max(s0 + 3.0, min(early_end, s0 + min(6.0, float(max_sec)))))
        picked = [
            TimedSegment(
                start=s0, end=s1, text="countdown_whisper", style="calm", rms=0.03
            )
        ]
        min_clip_sec = min(float(min_clip_sec), 1.0)
        preserve_whisper = True
    paths: list[Path] = []
    total = 0.0
    clean_kw = dict(
        from_original=from_original,
        preserve_whisper=preserve_whisper,
        enable_denoise=bool(enable_denoise),
        denoise_snr_db=float(denoise_snr_db),
        denoise_prop=float(denoise_prop),
    )
    if picked:
        for i, seg in enumerate(picked, start=1):
            a = max(0, int(seg.start * sample_rate))
            b = min(len(audio), int(seg.end * sample_rate))
            if b - a < int(min_clip_sec * sample_rate * 0.45):
                continue
            clip = _clean_clone_clip(
                audio[a:b].copy(),
                sample_rate,
                **clean_kw,
            )
            if float(np.sqrt(np.mean(np.square(clip)))) < 1e-4:
                continue
            # Long refs leak source-language words into later videos / cues.
            max_n = int(min(6.0, float(max_sec)) * sample_rate)
            if clip.size > max_n:
                clip = clip[:max_n]
            path = out_dir / f"clone_ref_{i:02d}.wav"
            sf.write(str(path), clip, sample_rate, subtype="PCM_16")
            src_txt = (seg.text or "").strip()
            if src_txt and src_txt != "countdown_whisper":
                path.with_suffix(".txt").write_text(src_txt[:240], encoding="utf-8")
            paths.append(path)
            total += clip.size / float(sample_rate)
    if not paths:
        # Fallback: best speech-like window by RMS×duration — NOT the first
        # N seconds of the track (music intro / moans there make the clone sing
        # or detach the timbre entirely).
        from app.text.vocalizations import is_background_vocalization

        def _best_window(pool: list[TimedSegment]) -> TimedSegment | None:
            best_s: TimedSegment | None = None
            best_sc = -1.0
            for seg in pool:
                dur = float(seg.end) - float(seg.start)
                if dur < 1.0:
                    continue
                score = float(seg.rms or 0.0) * min(dur, 8.0)
                if score > best_sc:
                    best_sc, best_s = score, seg
            return best_s

        non_filler = [
            s
            for s in (segments or [])
            if not is_background_vocalization(s.text or "")
        ]
        best_seg = _best_window(non_filler) or _best_window(list(segments or []))
        fb_sec = max(3.0, float(fallback_sec))
        if best_seg is not None:
            a = max(0, int(float(best_seg.start) * sample_rate))
            b = min(
                len(audio),
                int(
                    min(float(best_seg.end), float(best_seg.start) + fb_sec)
                    * sample_rate
                ),
            )
            logger.info(
                "Clone ref fallback: speech window %.1f-%.1fs (rms=%.4f)",
                float(best_seg.start),
                float(best_seg.end),
                float(best_seg.rms or 0.0),
            )
        else:
            a, b = 0, min(len(audio), int(fb_sec * sample_rate))
        if b - a < sample_rate:
            raise ValueError("В видео слишком мало речи для клонирования голоса")
        clip = _clean_clone_clip(
            audio[a:b].copy(), sample_rate, **clean_kw
        )
        max_n = int(min(6.0, float(max_sec)) * sample_rate)
        if clip.size > max_n:
            clip = clip[:max_n]
        path = out_dir / "clone_ref_01.wav"
        sf.write(str(path), clip, sample_rate, subtype="PCM_16")
        if best_seg is not None and (best_seg.text or "").strip():
            path.with_suffix(".txt").write_text(
                (best_seg.text or "").strip()[:240], encoding="utf-8"
            )
        paths = [path]
        total = clip.size / float(sample_rate)
    logger.info(
        "Clone refs from video: %d clips / %.1fs (video=%.1fs denoise=%s)",
        len(paths),
        total,
        duration,
        enable_denoise,
    )
    return paths, total


def fit_wav_to_duration(
    wav: np.ndarray,
    sample_rate: int,
    target_sec: float,
    *,
    min_speed: float = 0.94,
    max_speed: float = 1.06,
    fill_short: bool = False,
    allow_overflow_sec: float = 0.0,
    stretch_short: bool = False,
    trim_tail: bool = True,
) -> np.ndarray:
    """Подгонка реплики под бюджет слота без «ряби».

    Короткий клип не растягиваем на всю длительность слота, если только
    stretch_short: тогда слегка замедляем речь под длительность оригинала.
    Длинный — слегка ускоряем только если он вылезает за слот + запас
    (allow_overflow_sec). Хвост с речью не режем, если trim_tail=False.
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    target_sec = max(0.08, float(target_sec))
    slack = max(0.0, float(allow_overflow_sec))
    keep_sec = target_sec + slack
    need = max(1, int(round(target_sec * sample_rate)))
    keep = max(need, int(round(keep_sec * sample_rate)))
    if wav.size == 0:
        return np.zeros(need if fill_short else 1, dtype=np.float32)

    current = wav.size / float(sample_rate)
    ratio = current / target_sec  # >1 → речь длиннее слота
    fitted = wav

    if current > keep_sec * 1.02:
        speed = float(min(current / keep_sec, max(1.0, max_speed)))
        if speed >= 1.02:
            fitted = apply_tempo(fitted, sample_rate, speed)
    elif slack <= 1e-6 and current > target_sec * 1.02:
        speed = float(min(ratio, max(1.0, max_speed)))
        if speed >= 1.02:
            fitted = apply_tempo(fitted, sample_rate, speed)
    elif (fill_short or stretch_short) and current < target_sec * 0.94:
        speed = float(max(current / target_sec, min(1.0, min_speed)))
        if speed < 0.99:
            fitted = apply_tempo(fitted, sample_rate, speed)

    if fitted.size > keep:
        if not trim_tail:
            fitted = _drop_trailing_silence(fitted, sample_rate, keep)
            return fitted
        fade = min(int(0.045 * sample_rate), max(1, fitted.size // 10), keep)
        fitted = fitted[:keep].copy()
        if fade > 1:
            fitted[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return fitted
    if fill_short and fitted.size < need:
        return np.pad(fitted, (0, need - fitted.size))
    return fitted


def _drop_trailing_silence(
    wav: np.ndarray, sample_rate: int, keep: int
) -> np.ndarray:
    """Укорачивает только тишину в хвосте; речь не трогаем."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size <= keep:
        return wav
    frame = max(1, int(0.01 * sample_rate))
    last = wav.size
    while last - frame >= keep:
        chunk = wav[last - frame : last]
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        if rms > 1.2e-3:
            break
        last -= frame
    if last <= keep:
        return wav[:keep].copy()
    return wav[:last].copy()


def _pause_hints_from_words(words: list | None) -> list[float]:
    usable: list[tuple[float, float]] = []
    for item in words or []:
        if len(item) < 3:
            continue
        usable.append((float(item[1]), float(item[2])))
    hints: list[float] = []
    for i in range(len(usable) - 1):
        gap = usable[i + 1][0] - usable[i][1]
        if gap >= 0.08:
            hints.append(gap)
    return hints


MAX_WORD_GAP_SEC = 0.09
MAX_INJECTED_PAUSE_SEC = 0.05


def _word_slots(
    words: list | None,
    *,
    ref_start: float,
    ref_end: float,
) -> list[tuple[float, float]]:
    span = max(0.08, float(ref_end) - float(ref_start))
    slots: list[tuple[float, float]] = []
    for item in words or []:
        if len(item) < 3:
            continue
        start = float(item[1]) - float(ref_start)
        end = float(item[2]) - float(ref_start)
        start = min(max(0.0, start), span)
        end = min(max(start + 0.04, end), span)
        if end <= start:
            continue
        if slots and start <= slots[-1][1] + 0.02:
            slots[-1] = (slots[-1][0], max(slots[-1][1], end))
        else:
            slots.append((start, end))
    return slots


def _speech_islands(
    wav: np.ndarray, sample_rate: int, *, min_sec: float = 0.05
) -> list[tuple[int, int]]:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size < sample_rate // 8:
        return [(0, wav.size)] if wav.size else []
    frame = max(1, int(0.01 * sample_rate))
    peak = float(np.max(np.abs(wav)) or 1.0)
    thresh = max(0.022 * peak, 1.6e-3)
    min_len = max(frame, int(min_sec * sample_rate))
    islands: list[tuple[int, int]] = []
    i = 0
    n = wav.size
    while i < n:
        if abs(float(wav[i])) <= thresh:
            i += frame
            continue
        j = i
        while j < n and abs(float(wav[j])) > thresh:
            j += 1
        if j - i >= min_len:
            islands.append((i, j))
        i = max(j, i + frame)
    if not islands:
        return [(0, wav.size)]
    return islands[:24]


def _merge_ranges(ranges: list, count: int) -> list:
    items = list(ranges)
    if count <= 0:
        return items
    while len(items) > count:
        best_i = 0
        best = None
        for i in range(len(items) - 1):
            gap = items[i + 1][0] - items[i][1]
            if best is None or gap < best:
                best = gap
                best_i = i
        left = items[best_i]
        right = items[best_i + 1]
        items[best_i] = (left[0], right[1])
        del items[best_i + 1]
    return items


def align_speech_to_words(
    wav: np.ndarray,
    sample_rate: int,
    *,
    words: list | None,
    ref_start: float,
    ref_end: float,
    min_speed: float = 1.0,
    max_speed: float = 1.0,
    max_gap_sec: float = MAX_WORD_GAP_SEC,
) -> np.ndarray:
    """Ставит куски TTS на старт исходных слов, без растяжки каждого куска."""
    del min_speed, max_speed, max_gap_sec
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size < 8:
        return wav
    slots = _word_slots(words, ref_start=ref_start, ref_end=ref_end)
    islands = _speech_islands(wav, sample_rate)
    if len(slots) < 2 or len(islands) < 2:
        return wav
    n = min(len(slots), len(islands))
    slots = _merge_ranges(slots, n)
    islands = _merge_ranges(islands, n)
    parts: list[np.ndarray] = []
    cursor = 0
    for i, island in enumerate(islands):
        piece = wav[int(island[0]) : int(island[1])]
        if piece.size == 0:
            continue
        target = int(round(float(slots[i][0]) * sample_rate))
        if target > cursor:
            parts.append(np.zeros(target - cursor, dtype=np.float32))
            cursor = target
        parts.append(piece)
        cursor += piece.size
    return np.concatenate(parts) if parts else wav


def _low_energy_spans(
    wav: np.ndarray, sample_rate: int, *, min_sec: float = 0.045
) -> list[tuple[int, int]]:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size < sample_rate // 5:
        return []
    frame = max(1, int(0.01 * sample_rate))
    peak = float(np.max(np.abs(wav)) or 1.0)
    thresh = max(0.018 * peak, 1.4e-3)
    min_len = max(frame, int(min_sec * sample_rate))
    edge = max(frame * 4, int(0.04 * sample_rate))
    spans: list[tuple[int, int]] = []
    i = edge
    n = wav.size - edge
    while i < n:
        if abs(float(wav[i])) > thresh:
            i += frame
            continue
        j = i
        while j < n and abs(float(wav[j])) <= thresh:
            j += 1
        if j - i >= min_len:
            spans.append((i, j))
        i = max(j, i + frame)
    return spans[:8]


def inflate_interior_pauses(
    wav: np.ndarray,
    sample_rate: int,
    extra_sec: float,
    *,
    pause_hints: list[float] | None = None,
    max_gap_sec: float = MAX_INJECTED_PAUSE_SEC,
) -> np.ndarray:
    """Чуть расширяет уже существующие паузы под ритм оригинала."""
    extra = min(max(0.0, float(extra_sec)), 0.55)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if extra < 0.04 or wav.size < 8:
        return wav
    spans = _low_energy_spans(wav, sample_rate)
    if not spans:
        return wav
    cuts = [(a + b) // 2 for a, b in spans]
    cap = max(0.02, float(max_gap_sec))
    per = min(cap, extra / max(1, len(cuts)))
    parts: list[np.ndarray] = []
    prev = 0
    for cut in cuts:
        cut = min(max(cut, prev), wav.size)
        parts.append(wav[prev:cut])
        parts.append(np.zeros(max(1, int(round(per * sample_rate))), dtype=np.float32))
        prev = cut
    parts.append(wav[prev:])
    return np.concatenate(parts) if parts else wav


# антиперекрытие; 0.5с между фразами — если речь короче слота, не резерв под обрезку
MIN_PHRASE_GAP_SEC = 0.12
DEFAULT_SLOT_SLACK_SEC = 0.15


def cue_sync_budget(
    segments: list[TimedSegment],
    index: int,
    media_duration: float,
    *,
    gap_sec: float = MIN_PHRASE_GAP_SEC,
) -> tuple[float, float, float, float]:
    """(speech_start, speech_dur, pause_room, hard_cap).

    hard_cap = до старта следующей реплики (можно занять паузу).
    pause_room = hard_cap − speech_dur ≥ 0.
    """
    from app.services.timeline_align import speech_window

    seg = segments[index]
    sp0, sp1 = speech_window(seg)
    speech_dur = max(0.12, float(sp1) - float(sp0))
    gap = max(0.04, float(gap_sec))
    if index + 1 < len(segments):
        next0, _ = speech_window(segments[index + 1])
        hard_cap = max(0.12, float(next0) - float(sp0) - gap)
    else:
        hard_cap = max(0.12, float(media_duration) - float(sp0) - gap)
    hard_cap = max(hard_cap, speech_dur)
    pause_room = max(0.0, hard_cap - speech_dur)
    return float(sp0), speech_dur, pause_room, hard_cap


def match_clip_to_source_duration(
    wav: np.ndarray,
    sample_rate: int,
    target_sec: float,
    *,
    min_speed: float = 0.85,
    max_speed: float = 1.18,
    pause_hints: list[float] | None = None,
    overflow_sec: float = 0.25,
    words: list | None = None,
    ref_start: float = 0.0,
    ref_end: float | None = None,
    trim_tail: bool = False,
    stretch_short: bool = False,
) -> np.ndarray:
    """Подгонка под слот: ускорение + паузы; без mid-word trim по умолчанию."""
    del ref_start, ref_end
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    target_sec = max(0.12, float(target_sec))
    if wav.size == 0:
        return wav
    hints = list(pause_hints or [])
    if not hints and words:
        hints = _pause_hints_from_words(words)
    current = wav.size / float(sample_rate)
    # Короче целевого окна губ — чуть раздуваем внутренние паузы / мягкий stretch.
    if current < target_sec * 0.88:
        need_extra = min(target_sec - current, 0.45)
        if hints and need_extra >= 0.05:
            wav = inflate_interior_pauses(
                wav,
                sample_rate,
                min(need_extra, sum(hints[:4]) if hints else need_extra),
                pause_hints=hints,
                max_gap_sec=0.16,
            )
            current = wav.size / float(sample_rate)
    lo = max(0.75, float(min_speed))
    hi = min(2.0, float(max_speed))
    overflow = max(0.0, float(overflow_sec))
    fitted = fit_wav_to_duration(
        wav,
        sample_rate,
        target_sec,
        min_speed=lo,
        max_speed=hi,
        stretch_short=bool(stretch_short) and current < target_sec * 0.90,
        allow_overflow_sec=overflow,
        trim_tail=bool(trim_tail),
        fill_short=False,
    )
    # If всё ещё длиннее hard target+overflow — только тишина в хвосте, не речь.
    keep = max(1, int(round((target_sec + overflow) * sample_rate)))
    if fitted.size > keep:
        if trim_tail:
            fade = min(int(0.04 * sample_rate), max(1, fitted.size // 10), keep)
            fitted = fitted[:keep].copy()
            if fade > 1:
                fitted[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        else:
            fitted = _drop_trailing_silence(fitted, sample_rate, keep)
    return fitted


def anchor_placements_to_segments(
    segments: list[TimedSegment],
    durations: list[float],
    *,
    gap_sec: float = MIN_PHRASE_GAP_SEC,
    media_duration: float | None = None,
) -> list[tuple[float, float]]:
    """Жёсткий sync: старт = STT/words start, без сдвига следующих фраз."""
    from app.services.timeline_align import speech_window

    out: list[tuple[float, float]] = []
    gap = max(0.04, float(gap_sec))
    n = len(segments)
    for i, seg in enumerate(segments):
        speech_start, speech_end = speech_window(seg)
        t0 = float(speech_start)
        if i + 1 < n:
            next_start, _ = speech_window(segments[i + 1])
            hard_end = float(next_start) - gap
        elif media_duration is not None:
            hard_end = float(media_duration) - gap
        else:
            hard_end = max(float(speech_end), t0 + 0.12)
        hard_end = max(t0 + 0.08, hard_end)
        dur = max(0.0, float(durations[i]))
        # Никогда не заезжаем на следующую реплику — лучше обрезать хвост.
        dur = min(dur, max(0.08, hard_end - t0))
        out.append((t0, t0 + dur))
    return out


def center_align_digit_placements(
    segments: list[TimedSegment],
    durations: list[float],
    *,
    media_duration: float | None = None,
    preroll_first_sec: float = 0.08,
    min_gap_sec: float = 0.12,
) -> list[tuple[float, float]]:
    """Центры TTS-слов → центры STT-слов; следующую цифру не сдвигаем вперёд.

    Если предыдущая реплика длиннее паузы — обрезаем её хвост, а не уезжаем
    от губ оригинала на «один»/«ноль». Не форсим ранний preroll: длинный TTS
    раньше уезжал на −0.3с от губ.
    """
    from app.services.timeline_align import speech_window

    n = len(segments)
    out: list[tuple[float, float]] = []
    media = float(media_duration) if media_duration and media_duration > 0 else None
    ideals: list[tuple[float, float]] = []
    for i, seg in enumerate(segments):
        sp0, sp1 = speech_window(seg)
        center = 0.5 * (float(sp0) + float(sp1))
        speech_dur = max(0.12, float(sp1) - float(sp0))
        # Prefer the original lip window; don't let a long TTS expand placement.
        dur = max(0.12, min(float(durations[i]), speech_dur * 1.15))
        t0 = center - 0.5 * dur
        if i == 0:
            # Tiny optional lead-in only; never force an early jump off the lips.
            earliest = max(0.05, float(sp0) - max(0.0, float(preroll_first_sec)))
            t0 = max(earliest, min(t0, float(sp0)))
        else:
            t0 = max(0.0, t0)
        t1 = t0 + dur
        if media is not None and t1 > media - 0.02:
            t1 = media - 0.02
            t0 = max(0.05, t1 - dur)
        ideals.append((float(t0), float(t1)))

    for i, (t0, t1) in enumerate(ideals):
        if i > 0:
            prev_t0, prev_t1 = out[i - 1]
            # Never delay this digit past its lip-sync center to clear the previous.
            min_start = prev_t0 + 0.08
            if t0 < prev_t1 + min_gap_sec:
                # Prefer shortening the previous cue.
                limit = max(prev_t0 + 0.12, t0 - min_gap_sec)
                if prev_t1 > limit:
                    out[i - 1] = (prev_t0, limit)
                t0 = max(t0, min_start)
                if t0 < out[i - 1][1] + min_gap_sec:
                    t0 = out[i - 1][1] + min_gap_sec
                t1 = t0 + max(0.12, min(float(durations[i]), t1 - ideals[i][0] + 0.12))
                if media is not None:
                    t1 = min(t1, media - 0.02)
        out.append((float(t0), float(t1)))
    return out


def _fade_clip_edges(
    wav: np.ndarray, sample_rate: int, fade_ms: float = 18.0
) -> np.ndarray:
    piece = np.asarray(wav, dtype=np.float32).reshape(-1)
    if piece.size < 8:
        return piece
    fade = min(int(fade_ms * sample_rate / 1000.0), max(1, piece.size // 8))
    if fade <= 1:
        return piece
    out = piece.copy()
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    out[:fade] *= ramp
    out[-fade:] *= ramp[::-1]
    return out


def _overlay_voice(
    timeline: np.ndarray, piece: np.ndarray, start: int, *, xfade: int
) -> None:
    if piece.size == 0 or start >= timeline.size:
        return
    if start < 0:
        piece = piece[-start:]
        start = 0
        if piece.size == 0:
            return
    end = min(timeline.size, start + piece.size)
    if end <= start:
        return
    # если таймлайн короче клипа, режем только если в хвосте тишина
    if start + piece.size > timeline.size:
        keep = timeline.size - start
        piece = _drop_trailing_silence(piece, 48000, keep)
        end = min(timeline.size, start + piece.size)
        if end <= start:
            return
    piece = piece[: end - start]
    overlap = timeline[start:end]
    mix = piece.copy()
    n_xfade = min(int(xfade), piece.size // 4, overlap.size)
    if n_xfade > 1 and float(np.max(np.abs(overlap[:n_xfade]))) > 1e-4:
        fade_out = np.linspace(1.0, 0.0, n_xfade, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, n_xfade, dtype=np.float32)
        mix[:n_xfade] = overlap[:n_xfade] * fade_out + piece[:n_xfade] * fade_in
    timeline[start:end] = mix


def layout_dub_placements(
    segments: list[TimedSegment],
    durations: list[float],
    media_duration: float,
    *,
    slack_sec: float = DEFAULT_SLOT_SLACK_SEC,
    gap_sec: float = MIN_PHRASE_GAP_SEC,
) -> list[tuple[float, float]]:
    """(start, end) каждой реплики на таймлайне.

    Полный клип сохраняем, сдвигая старт/финиш в пределах slack.
    Соседние фразы не пересекаются (зазор gap). Обрезка — только если
    после сдвигов клип всё равно не влезает в своё окно.
    """
    n = len(segments)
    if n == 0:
        return []
    if len(durations) != n:
        raise ValueError("Число длительностей не совпало с сегментами")
    media = max(0.08, float(media_duration))
    slack = max(0.0, float(slack_sec))
    gap = max(0.0, float(gap_sec))
    out: list[tuple[float, float]] = []
    prev_end = -gap

    for i, seg in enumerate(segments):
        dur = max(0.0, float(durations[i]))
        orig_s = float(seg.start)
        orig_e = float(seg.end)
        earliest = max(0.0, orig_s - slack)
        latest_start = min(media, orig_s + slack)
        latest_end = min(media, orig_e + slack + 0.45)
        earliest = max(earliest, prev_end + gap)

        if earliest > latest_start + 1e-6 and out:
            limit_prev = max(out[-1][0] + 0.08, latest_start - gap)
            pt, pu = out[-1]
            if pu > limit_prev + 1e-6:
                out[-1] = (pt, min(pu, limit_prev))
                prev_end = out[-1][1]
                earliest = max(0.0, orig_s - slack, prev_end + gap)

        earliest = min(max(0.0, earliest), media)
        if dur <= 1e-4:
            t = min(max(orig_s, earliest), media)
            out.append((t, t))
            prev_end = t
            continue

        t = orig_s
        if t < earliest:
            t = earliest
        if t > latest_start:
            t = latest_start
        if t < earliest:
            t = earliest

        u = t + dur
        if u > latest_end + 1e-6:
            t2 = max(earliest, latest_end - dur)
            if t2 + dur <= latest_end + 1e-4:
                t, u = t2, t2 + dur
            else:
                t = earliest
                u = min(t + dur, latest_end)
        u = min(u, media)
        if u < t:
            u = t
        out.append((t, u))
        prev_end = u

    for i in range(n - 1, -1, -1):
        dur = max(0.0, float(durations[i]))
        if dur <= 1e-4:
            continue
        t, u = out[i]
        used = u - t
        if used >= dur - 0.02:
            continue
        orig_s = float(segments[i].start)
        orig_e = float(segments[i].end)
        earliest = max(0.0, orig_s - slack)
        if i > 0:
            earliest = max(earliest, out[i - 1][1] + gap)
        latest_end = min(media, orig_e + slack + 0.45)
        if i + 1 < n:
            latest_end = min(latest_end, out[i + 1][0] - gap)
        t2 = max(earliest, min(t, latest_end - min(dur, latest_end - earliest)))
        u2 = min(latest_end, t2 + dur)
        if u2 < t2:
            continue
        if u2 - t2 > used + 0.01:
            out[i] = (t2, u2)

    # не режем хвост предыдущей — сдвигаем следующую
    for _ in range(max(1, n)):
        moved = False
        for i in range(n - 1):
            t, u = out[i]
            want = t + max(0.0, float(durations[i]))
            orig_e = float(segments[i].end)
            u = min(max(u, min(want, media)), min(media, orig_e + slack + 0.45))
            out[i] = (t, max(t, u))
            nt, nu = out[i + 1]
            ndur = max(0.0, nu - nt, float(durations[i + 1]))
            min_start = out[i][1] + gap
            if nt + 1e-6 >= min_start:
                continue
            orig_s = float(segments[i + 1].start)
            hard_latest = min(media, orig_s + slack + 0.45)
            nt2 = min(max(nt, min_start), hard_latest)
            if nt2 + 1e-6 < min_start:
                # места мало — всё равно не режем предыдущую, чуть заедем за slack
                nt2 = min(media, min_start)
            nu2 = min(media, nt2 + ndur)
            if abs(nt2 - nt) > 1e-4 or abs(nu2 - nu) > 1e-4:
                out[i + 1] = (nt2, max(nt2, nu2))
                moved = True
        if not moved:
            break
    for i in range(n):
        orig_s = float(segments[i].start)
        t, u = out[i]
        dur = max(0.0, u - t, float(durations[i]))
        if dur <= 1e-4:
            continue
        earliest = 0.0 if i == 0 else out[i - 1][1] + gap
        if orig_s + 1e-4 < earliest:
            continue
        t2 = orig_s
        u2 = t2 + dur
        if u2 > media + 1e-4:
            continue
        if i + 1 < n and u2 > out[i + 1][0] - gap + 1e-4:
            continue
        out[i] = (t2, max(t2, u2))
    return out


def layout_complete_speech_placements(
    segments: list[TimedSegment],
    durations: list[float],
    media_duration: float,
    *,
    gap_sec: float = MIN_PHRASE_GAP_SEC,
    silence_gaps: list[tuple[float, float]] | None = None,
    max_early_sec: float = 1.5,
) -> list[tuple[float, float]]:
    """Place full TTS phrases without tempo: borrow following silence, keep min gaps.

    Anchors are word-level speech onsets. Overflow extends into the next silence
    gap (and may push later cues forward). Short TTS leaves unused silence as
    pause. A global pull-back into leading silence is used only when the chain
    would otherwise run past media end.
    """
    from app.services.timeline_align import speech_window

    if len(segments) != len(durations):
        raise ValueError("Число длительностей не совпало с сегментами")
    if not segments:
        return []
    gap = max(0.0, float(gap_sec))
    media = max(0.0, float(media_duration))
    gaps = list(silence_gaps or [])

    def _silence_room_after(t: float, until: float) -> float:
        """How much contiguous silence is available in [t, until)."""
        room = 0.0
        cursor = float(t)
        limit = float(until)
        for g0, g1 in gaps:
            if g1 <= cursor + 1e-6:
                continue
            if g0 >= limit - 1e-6:
                break
            a = max(cursor, g0)
            b = min(limit, g1)
            if b > a:
                # Only count if silence starts near cursor (contiguous borrow).
                if a <= cursor + 0.08:
                    room += b - a
                    cursor = b
                else:
                    break
        return max(0.0, room)

    starts: list[float] = []
    ends: list[float] = []
    cursor = 0.0
    for idx, (seg, raw_duration) in enumerate(zip(segments, durations)):
        duration = max(0.0, float(raw_duration))
        speech_start, speech_end = speech_window(seg)
        anchor = max(0.0, float(speech_start))
        start = max(anchor, cursor)
        # Prefer staying on the lip onset when previous cue finished early.
        if start > anchor + 1e-4 and cursor <= anchor + 1e-4:
            start = anchor
        end = start + duration
        # Log-only: silence after original speech that this cue can occupy.
        next_anchor = media
        if idx + 1 < len(segments):
            ns, _ = speech_window(segments[idx + 1])
            next_anchor = max(0.0, float(ns))
        borrow = _silence_room_after(float(speech_end), next_anchor)
        if duration > max(0.08, float(speech_end) - float(speech_start)) + 0.05:
            # Longer than original speech — silence borrow is the intended slack.
            _ = borrow
        starts.append(start)
        ends.append(end)
        cursor = end + (gap if duration > 1e-4 else 0.0)

    overflow = max(0.0, ends[-1] - media)
    if overflow > 0.05:
        # Pull-back pass: compute the latest start of each cue that still lets
        # the whole tail fit into the media, then re-place forward — cues move
        # left into the free room BEFORE them (pause after the previous cue),
        # bounded per cue (max_early_sec) so lip sync drift stays small.
        # Never creates overlaps; residual overflow = dub genuinely longer.
        early_cap = max(0.0, float(max_early_sec))
        n = len(segments)
        latest_start = [0.0] * n
        latest_end = media
        for idx in range(n - 1, -1, -1):
            dur = max(0.0, float(durations[idx]))
            ls = latest_end - dur
            latest_start[idx] = ls
            latest_end = ls - (gap if dur > 1e-4 else 0.0)
        cursor = 0.0
        for idx in range(n):
            dur = max(0.0, float(durations[idx]))
            if dur <= 1e-4:
                continue
            sp_start, _ = speech_window(segments[idx])
            earliest = max(0.0, float(sp_start) - early_cap, cursor)
            desired = min(starts[idx], latest_start[idx])
            start = max(earliest, desired)
            starts[idx] = start
            ends[idx] = start + dur
            cursor = ends[idx] + gap
        residual = max(0.0, ends[-1] - media)
        if residual > 0.25:
            logger.warning(
                "Dub layout overflows media by %.2fs even after pull-back "
                "(dub content longer than video)",
                residual,
            )

    return list(zip(starts, ends))


def lock_placements_to_speech(
    segments: list[TimedSegment],
    durations: list[float],
    media_duration: float,
) -> list[tuple[float, float]]:
    """Pin every cue to its own ASR speech onset — no cascade shift.

    Voice-pick re-dubs reuse long first-pass clips; a silence-borrow layout then
    shove later cues by tens of seconds while the bed is still muted on the
    *original* ASR windows → audible silence holes. Anchoring to speech_window
    keeps lip-sync and fills those windows.
    """
    from app.services.timeline_align import speech_window

    media = max(0.1, float(media_duration))
    out: list[tuple[float, float]] = []
    for seg, dur in zip(segments, durations):
        sp0, _sp1 = speech_window(seg)
        start = max(0.0, min(float(sp0), media))
        end = min(media, start + max(0.0, float(dur)))
        out.append((start, end))
    return out


def clamp_clip_to_slot(
    wav: np.ndarray,
    sample_rate: int,
    *,
    speech_dur: float,
    hard_cap: float,
    max_factor: float = 2.2,
    abs_cap_sec: float = 14.0,
) -> np.ndarray:
    """Trim absurdly long TTS (Fish hallucination / reused runaway) to the slot."""
    audio = np.asarray(wav, dtype=np.float32).reshape(-1)
    if audio.size < 8:
        return audio
    nat = audio.size / float(sample_rate)
    budget = max(
        0.35,
        min(
            float(abs_cap_sec),
            max(float(speech_dur) * float(max_factor), float(hard_cap) * 1.15),
        ),
    )
    if nat <= budget + 0.05:
        return audio
    keep = max(1, int(budget * sample_rate))
    trimmed = _drop_trailing_silence(audio[:keep], sample_rate, max(1, int(0.06 * sample_rate)))
    if trimmed.size < max(1, int(0.08 * sample_rate)):
        trimmed = audio[:keep]
    logger.info(
        "Clamped dub clip %.2fs → %.2fs (speech=%.2fs hard=%.2fs)",
        nat,
        trimmed.size / float(sample_rate),
        speech_dur,
        hard_cap,
    )
    return trimmed


def layout_silence_borrow_placements(
    segments: list[TimedSegment],
    durations: list[float],
    media_duration: float,
    *,
    gap_sec: float = MIN_PHRASE_GAP_SEC,
    silence_gaps: list[tuple[float, float]] | None = None,
    max_early_sec: float = 1.5,
) -> list[tuple[float, float]]:
    """Alias for silence-aware full-phrase layout (no tempo)."""
    return layout_complete_speech_placements(
        segments,
        durations,
        media_duration,
        gap_sec=gap_sec,
        silence_gaps=silence_gaps,
        max_early_sec=max_early_sec,
    )


def match_voice_level_to_source(
    voice: np.ndarray,
    source_vocals: np.ndarray | None,
    sample_rate: int,
    windows: list[tuple[float, float]],
    *,
    min_gain: float = 0.75,
    max_gain: float = 1.4,
    floor_src_rms: float = 0.018,
) -> np.ndarray:
    """Подгоняет громкость новой речи к RMS оригинального вокала в тех же окнах.

    floor_src_rms: если оригинал слишком тихий (ASMR/шёпот), не давим дубляж
    ниже слышимого пола — иначе «почти не озвучивается».
    """
    out = np.asarray(voice, dtype=np.float32).reshape(-1)
    if source_vocals is None or out.size < 16 or not windows:
        return out
    src = np.asarray(source_vocals, dtype=np.float32).reshape(-1)
    if src.ndim > 1:
        src = np.mean(src, axis=1)
    if src.size < 16:
        return out

    vo_parts: list[np.ndarray] = []
    src_parts: list[np.ndarray] = []
    for start, end in windows:
        a = max(0, int(float(start) * sample_rate))
        b = max(a + 1, int(float(end) * sample_rate))
        vo_slice = out[a : min(out.size, b)]
        src_slice = src[a : min(src.size, b)]
        if vo_slice.size < sample_rate // 20 or src_slice.size < sample_rate // 20:
            continue
        vo_parts.append(vo_slice)
        src_parts.append(src_slice)
    if not vo_parts:
        return out

    vo_rms = float(np.sqrt(np.mean(np.square(np.concatenate(vo_parts)))) or 0.0)
    src_rms = float(np.sqrt(np.mean(np.square(np.concatenate(src_parts)))) or 0.0)
    if vo_rms < 1e-5 or src_rms < 1e-5:
        return out
    target = max(float(src_rms), float(floor_src_rms))
    gain = float(np.clip(target / vo_rms, min_gain, max_gain))
    out = out * gain
    peak = float(np.max(np.abs(out)) or 1.0)
    if peak > 0.97:
        out = out * (0.97 / peak)
    return out.astype(np.float32)


def polish_dub_clip(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    """Лёгкая обрезка краёв — без вырезания естественных пауз внутри фразы."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size < sample_rate // 8:
        return wav
    trimmed = trim_silence(
        wav,
        sample_rate,
        frame_ms=12,
        threshold_db=-40.0,
        leading_padding_ms=12,
        trailing_padding_ms=28,
    )
    return trimmed if trimmed.size >= sample_rate // 25 else wav


def _plain_chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def estimate_tts_sec(text: str, *, language: str = "ru") -> float:
    """Грубая длительность XTTS без пауз (нужна, чтобы не замедлять плотный перевод)."""
    chars = _plain_chars(text)
    cps = 12.5 if language == "ru" else 14.0
    return max(0.25, chars / cps)


_REPEAT_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9']+", re.UNICODE)


def _repeat_tokens(text: str) -> list[str]:
    return _REPEAT_TOKEN_RE.findall((text or "").lower())


def _collapse_immediate_word_repeats(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    out = [tokens[0]]
    for token in tokens[1:]:
        if token == out[-1]:
            continue
        out.append(token)
    return out


def _collapse_repeated_phrases(
    tokens: list[str],
    *,
    min_repeats: int = 10,
    reduce_divisor: int = 3,
    max_unit: int = 4,
) -> list[str]:
    """Collapse long consecutive phrase loops: x x x ... -> fewer repeats."""
    if not tokens:
        return []
    n = len(tokens)
    i = 0
    out: list[str] = []
    while i < n:
        best: tuple[int, int] | None = None  # (unit_len, repeats)
        # Prefer longer units so "oh my" isn't treated as independent words.
        for unit_len in range(min(max_unit, (n - i) // 2), 0, -1):
            unit = tokens[i : i + unit_len]
            repeats = 1
            j = i + unit_len
            while j + unit_len <= n and tokens[j : j + unit_len] == unit:
                repeats += 1
                j += unit_len
            if repeats >= min_repeats:
                best = (unit_len, repeats)
                break
        if best is None:
            out.append(tokens[i])
            i += 1
            continue
        unit_len, repeats = best
        keep = max(1, repeats // max(2, int(reduce_divisor)))
        out.extend(tokens[i : i + unit_len] * keep)
        i += unit_len * repeats
    return out


def compact_repetitions_to_budget(
    text: str,
    *,
    budget_sec: float,
    language: str,
    tolerance: float = 1.03,
) -> str:
    """Shorten only repetitive loops until TTS estimate is close to slot budget."""
    spoken = (text or "").strip()
    if not spoken:
        return spoken
    target = max(0.25, float(budget_sec)) * max(1.0, float(tolerance))
    if estimate_tts_sec(spoken, language=language) <= target:
        return spoken
    tokens = _repeat_tokens(spoken)
    if len(tokens) < 2:
        return spoken

    working = list(tokens)
    changed = False
    used_long_repeat_rule = False
    for _ in range(4):
        # 1) if phrase appears >10 times, keep floor(n/3)
        reduced = _collapse_repeated_phrases(working, min_repeats=11, reduce_divisor=3)
        reduced_changed = reduced != working
        if reduced_changed:
            working = reduced
            changed = True
            used_long_repeat_rule = True
        # 2) "да да да" -> "да" for short loops
        dedup_changed = False
        if not reduced_changed and not used_long_repeat_rule:
            dedup = _collapse_immediate_word_repeats(working)
            dedup_changed = dedup != working
            if dedup_changed:
                working = dedup
                changed = True
        if estimate_tts_sec(" ".join(working), language=language) <= target:
            break
        if not reduced_changed and not dedup_changed:
            break

    return " ".join(working) if changed and working else spoken


def segment_time_budget(
    segments: list[TimedSegment],
    index: int,
    media_duration: float,
    *,
    gap_pad: float = MIN_PHRASE_GAP_SEC,
    max_overflow: float | None = None,
) -> float:
    """Бюджет слота: до следующей фразы минус небольшой зазор.

    max_overflow=None — можно занять паузу до следующего start (перевод часто длиннее).
    """
    seg = segments[index]
    if index + 1 < len(segments):
        hard_end = float(segments[index + 1].start) - gap_pad
    else:
        hard_end = float(media_duration) - gap_pad
    if max_overflow is None:
        soft_end = hard_end
    else:
        soft_end = min(hard_end, float(seg.end) + float(max_overflow))
    budget = soft_end - float(seg.start)
    return max(0.12, budget)


_STYLE_MARK = {
    "question": "❓",
    "expressive": "❗",
    "calm": "…",
    "neutral": "•",
}


def format_cue_sheet(
    segments: list[TimedSegment],
    *,
    translations: list[str] | None = None,
    title: str = "Партитура",
    media_duration: float | None = None,
) -> list[str]:
    """Сообщения Telegram с таймкодами, тоном и (опционально) переводом."""
    total = sum(seg.duration for seg in segments)
    span = ""
    if segments:
        span = f" · окно {segments[0].start:.1f}–{segments[-1].end:.1f}с"
    media = ""
    if media_duration and media_duration > 0:
        cover = 100.0 * total / media_duration
        media = f" · ролик {media_duration:.0f}с · покрытие речи ~{cover:.0f}%"
    head = (
        f"<b>{html.escape(title)}</b>\n"
        f"{len(segments)} фраз · {total:.1f}с речи{span}{media}\n"
    )
    blocks = [head]
    for i, seg in enumerate(segments):
        mark = _STYLE_MARK.get(seg.style, "•")
        extras: list[str] = []
        rate = float(getattr(seg, "rate", 1.0) or 1.0)
        volume = float(getattr(seg, "volume", 1.0) or 1.0)
        if abs(rate - 1.0) > 0.04:
            extras.append(f"×{rate:.2f}")
        if abs(volume - 1.0) > 0.08:
            extras.append(f"{int(round(volume * 100))}%")
        extra = f" {' '.join(extras)}" if extras else ""
        line = (
            f"{mark} <code>{seg.start:04.1f}–{seg.end:04.1f}</code> "
            f"({seg.duration:.1f}с){extra} <i>{html.escape(seg.style)}</i>\n"
            f"{html.escape(seg.text)}"
        )
        if translations and i < len(translations):
            line += f"\n→ <b>{html.escape(translations[i])}</b>"
        blocks.append(line + "\n")

    messages: list[str] = []
    current = ""
    for block in blocks:
        if len(current) + len(block) > 3500:
            if current.strip():
                messages.append(current.strip())
            current = block
        else:
            current += ("\n" if current else "") + block
    if current.strip():
        messages.append(current.strip())
    return messages or [head]


_NUM_LINE = re.compile(
    r"^\s*(?:#|№)?\s*(\d{1,3})\s*[.)\-:]+\s*(.+?)\s*$"
)
_SRT_INDEX = re.compile(r"^\d+$")
# DeepL-safe cue tags: <c i="12">text</c> (also <t id="12">…</t>).
_DEEPL_TAG = re.compile(
    r"""<\s*(?:c|t|seg)\s+(?:i|id)\s*=\s*["']?(\d{1,4})["']?\s*>"""
    r"""(.*?)"""
    r"""</\s*(?:c|t|seg)\s*>""",
    re.IGNORECASE | re.DOTALL,
)
# Cue-sheet meta line (timing / style) — never speech for DeepL.
_CUE_META_LINE = re.compile(
    r"""(?ix)
    (?:
        \d{1,3}[.,]\d+\s*[–—\-~]\s*\d{1,3}[.,]\d+   # 00.9–02.0
      | \(\s*\d+[.,]?\d*\s*с\s*\)                      # (1.1с)
      | ×\s*\d+[.,]\d+                                 # ×0.78
      | \b(?:expressive|neutral|calm|question|soft|whisper)\b
    )
    """,
)
_KEEP_MARK = re.compile(
    r"^\s*(?:\[{1,3}|§+|⟦+)\s*0*(\d{1,4})\s*(?:\]{1,3}|§+|⟧+)\s*(.*)$"
)


def format_translate_pack(segments: list[TimedSegment]) -> str:
    """DeepL-safe pack: only speech inside XML tags; IDs stay put.

    Timing / style markers live on the bot side (cue sheet) and must NOT go
    into the translator — DeepL mangles them and merges cues. Paste this pack
    into DeepL as-is; it translates tag bodies and keeps ``i`` attributes.
    """
    parts: list[str] = []
    for i, seg in enumerate(segments, start=1):
        spoken = (seg.text or "").strip()
        # Escape XML specials inside speech so the pack stays well-formed.
        spoken = (
            spoken.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        parts.append(f'<c i="{i}">{spoken}</c>')
    # Blank line between cues stops DeepL from gluing neighboring phrases.
    return "\n\n".join(parts)


def split_plain_chunks(text: str, limit: int = 3500) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    cur: list[str] = []
    size = 0
    # Prefer splitting on blank lines (between <c> cues) then on newlines.
    blocks = text.split("\n\n") if "\n\n" in text else text.split("\n")
    sep = "\n\n" if "\n\n" in text else "\n"
    for block in blocks:
        add = len(block) + (len(sep) if cur else 0)
        if cur and size + add > limit:
            parts.append(sep.join(cur))
            cur = [block]
            size = len(block)
        else:
            cur.append(block)
            size += add
    if cur:
        parts.append(sep.join(cur))
    return parts or [text[:limit]]


def _unescape_xml_text(value: str) -> str:
    return (
        (value or "")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&amp;", "&")
        .strip()
    )


def _extract_deepl_tags(text: str, count: int) -> list[str] | None:
    hits = list(_DEEPL_TAG.finditer(text))
    if not hits:
        return None
    out = [""] * count
    filled = 0
    for match in hits:
        idx = int(match.group(1))
        body = _unescape_xml_text(match.group(2))
        # Collapse accidental newlines DeepL inserts inside a cue.
        body = re.sub(r"\s*\n\s*", " ", body).strip()
        if not body:
            continue
        if 1 <= idx <= count:
            out[idx - 1] = body
            filled += 1
        elif 0 <= idx < count and not out[idx]:
            out[idx] = body
            filled += 1
    return out if filled else None


def _extract_cue_sheet_speech(lines: list[str]) -> list[str]:
    """If the user pasted the timing cue sheet into DeepL, keep only speech."""
    speech: list[str] = []
    meta_hits = 0
    for ln in lines:
        if _CUE_META_LINE.search(ln):
            meta_hits += 1
            continue
        # Headers like "Партитура" / "10 фраз · …"
        if re.search(r"\bфраз\b|\bпартитур", ln, re.I):
            continue
        if ln.startswith("→"):
            ln = ln.lstrip("→").strip()
        if ln:
            speech.append(ln)
    # Only treat as cue-sheet paste when we actually saw timing/meta lines.
    if meta_hits < 2:
        return []
    return speech


def parse_user_translation(
    raw: str,
    count: int,
    *,
    already_filled: list[str] | None = None,
) -> list[str] | None:
    """Разбор ответа переводчика: DeepL ``<c i>``, `01. …`, голые строки или SRT.

    Partial pastes are OK for large videos:
    - numbered / tagged lines fill those slots;
    - plain lines (no numbers) fill the next empty slots after ``already_filled``.
    Returned list has length ``count``; empty slots stay "" so merge keeps prior text.
    """
    if not raw or count <= 0:
        return None
    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return None
    if count == 1:
        tagged = _extract_deepl_tags(text, 1)
        if tagged and tagged[0]:
            return tagged
        return [re.sub(r"^\s*\d{1,3}\s*[\.\)\:\-]+\s*", "", text).strip() or text]

    # 1) DeepL XML pack — preferred.
    tagged = _extract_deepl_tags(text, count)
    if tagged is not None:
        return tagged

    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]

    # 2) Accidental cue-sheet paste (with/without DeepL mangling of meta).
    sheet_speech = _extract_cue_sheet_speech(lines)
    if sheet_speech:
        if len(sheet_speech) == count:
            return sheet_speech
        if 0 < len(sheet_speech) < count:
            return _place_plain_chunk(sheet_speech, count, already_filled)

    if any("-->" in ln for ln in lines):
        bodies: list[str] = []
        buf: list[str] = []
        for ln in lines:
            if _SRT_INDEX.match(ln) or "-->" in ln:
                if buf:
                    bodies.append(" ".join(buf))
                    buf = []
                continue
            buf.append(ln)
        if buf:
            bodies.append(" ".join(buf))
        if len(bodies) == count:
            return bodies
        if 0 < len(bodies) < count:
            return _place_plain_chunk(bodies, count, already_filled)

    numbered: dict[int, str] = {}
    plain: list[str] = []
    for ln in lines:
        keep = _KEEP_MARK.match(ln)
        if keep:
            numbered[int(keep.group(1))] = keep.group(2).strip()
            continue
        match = _NUM_LINE.match(ln)
        if match:
            numbered[int(match.group(1))] = match.group(2).strip()
        else:
            plain.append(ln)
    if numbered:
        out = [""] * count
        hits = 0
        for idx, val in numbered.items():
            if 1 <= idx <= count and val:
                out[idx - 1] = val
                hits += 1
            elif 0 <= idx < count and val and not out[idx]:
                out[idx] = val
                hits += 1
        if hits:
            if plain:
                merged_base = [
                    (already_filled[i] if already_filled and i < len(already_filled) else "")
                    or out[i]
                    for i in range(count)
                ]
                extra = _place_plain_chunk(plain, count, merged_base)
                for i, val in enumerate(extra):
                    if val.strip() and not out[i].strip():
                        out[i] = val
            return out
    if len(plain) == count:
        return plain
    if 0 < len(plain) < count:
        return _place_plain_chunk(plain, count, already_filled)
    if len(lines) == count:
        cleaned: list[str] = []
        for ln in lines:
            match = _NUM_LINE.match(ln)
            cleaned.append(match.group(2).strip() if match else ln)
        return cleaned
    return None


def _place_plain_chunk(
    chunk: list[str],
    count: int,
    already_filled: list[str] | None,
) -> list[str]:
    """Map a contiguous plain-line paste onto the next empty cue slots."""
    out = [""] * count
    start = 0
    if already_filled:
        for i in range(min(count, len(already_filled))):
            if not (already_filled[i] or "").strip():
                start = i
                break
        else:
            # Everything already filled — overwrite from the top of this chunk.
            start = 0
    for offset, line in enumerate(chunk):
        idx = start + offset
        if idx >= count:
            break
        text = (line or "").strip()
        if text:
            out[idx] = text
    return out


def merge_translations(base: list[str], incoming: list[str]) -> list[str]:
    size = max(len(base), len(incoming))
    out = [""] * size
    for i, val in enumerate(base):
        if val.strip():
            out[i] = val.strip()
    for i, val in enumerate(incoming):
        if val.strip():
            out[i] = val.strip()
    return out


def missing_translation_indices(pasted: list[str], total: int) -> list[int]:
    """1-based cue numbers still empty."""
    missing: list[int] = []
    for i in range(total):
        val = pasted[i] if i < len(pasted) else ""
        if not (val or "").strip():
            missing.append(i + 1)
    return missing


def format_srt_timestamp(seconds: float) -> str:
    ms_total = max(0, int(round(float(seconds) * 1000)))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(
    path: Path,
    segments: list[TimedSegment],
    translations: list[str],
) -> Path:
    lines: list[str] = []
    for i, (seg, text) in enumerate(zip(segments, translations), start=1):
        spoken = (text or seg.text).strip()
        if not spoken:
            continue
        lines.append(str(i))
        lines.append(
            f"{format_srt_timestamp(seg.start)} --> {format_srt_timestamp(seg.end)}"
        )
        lines.append(spoken)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _repair_translation_json(blob: str) -> str:
    """Чинит типичный JSON от GigaChat: лишние кавычки и пропущенная \" перед ,{."""
    repaired = re.sub(r'("text"\s*:\s*)"+', r'\1"', blob)
    repaired = re.sub(
        r'("text"\s*:\s*"[^{}"\n]*?)\s*,\s*\{',
        r'\1"},{',
        repaired,
    )
    return repaired


def _extract_translation_texts_loose(raw: str) -> list[str]:
    """Достаёт поля text по очереди, даже если JSON битый."""
    out: list[str] = []
    for match in re.finditer(r'"text"\s*:\s*', raw):
        i = match.end()
        while i < len(raw) and raw[i] in " \t\r\n":
            i += 1
        if i < len(raw) and raw[i] == '"':
            i += 1
            while i < len(raw) and raw[i] == '"':
                i += 1
        buf: list[str] = []
        while i < len(raw):
            ch = raw[i]
            if ch == "\\" and i + 1 < len(raw):
                buf.append(raw[i + 1])
                i += 2
                continue
            if ch == '"':
                nxt = raw[i + 1 : i + 24].lstrip()
                if not nxt or nxt[0] in ",}]":
                    break
                buf.append(ch)
                i += 1
                continue
            if ch == "{" and buf:
                break
            buf.append(ch)
            i += 1
        value = "".join(buf).strip().strip('"')
        if value:
            out.append(value)
    return out


def parse_translation_json(raw: str, count: int) -> list[str] | None:
    """Достаёт перевод из ответа LLM; допускает markdown, строки, неполный массив."""
    if not raw or count <= 0:
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        blob = match.group(0)
    else:
        start = text.find("[")
        if start < 0:
            blob = text
        else:
            blob = text[start:].rstrip().rstrip(",")
    candidates = [blob, _repair_translation_json(blob)]
    trimmed = blob.rstrip().rstrip(",")
    if trimmed.endswith("]"):
        # возможно битый закрывающий ] без кавычек — тоже пробуем починить
        base = trimmed[:-1].rstrip().rstrip(",")
    else:
        base = trimmed
    candidates.extend(
        [
            base + "]",
            base + '"]',
            base + '"}]',
            base + "}]",
            base + '"}]}',
            _repair_translation_json(base) + "]",
            _repair_translation_json(base) + "}]",
        ]
    )
    payload = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if payload is None or not isinstance(payload, list) or not payload:
        loose = _extract_translation_texts_loose(text)
        if len(loose) < max(1, count // 2):
            return None
        payload = [{"i": i, "text": t} for i, t in enumerate(loose)]

    by_index: dict[int, str] = {}
    sequential: list[str] = []
    for i, item in enumerate(payload):
        if isinstance(item, str):
            value = item.strip()
            if value:
                by_index[i] = value
                sequential.append(value)
            continue
        if isinstance(item, dict):
            value = str(item.get("text") or item.get("translation") or "").strip()
            if not value:
                continue
            idx = item.get("i", item.get("index", i))
            try:
                idx_i = int(idx)
            except (TypeError, ValueError):
                idx_i = i
            by_index[idx_i] = value
            sequential.append(value)

    if not sequential and not by_index:
        return None

    out: list[str] = []
    for i in range(count):
        if i in by_index and by_index[i]:
            out.append(by_index[i])
        elif i < len(sequential) and sequential[i]:
            out.append(sequential[i])
        else:
            out.append("")
    # если часть пустая — не валидно целиком (fallback переведёт по одной)
    if sum(1 for x in out if x) < max(1, count // 2):
        return None
    return out


def _parse_translation_by_ids(
    raw: str, cue_ids: list[int]
) -> dict[int, str] | None:
    """Map LLM JSON texts onto absolute cue ids (avoids batch-local i mixups)."""
    if not cue_ids:
        return {}
    id_set = {int(i) for i in cue_ids}
    mapped: dict[int, str] = {}

    # Prefer explicit absolute cue ids from the payload when present.
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    match = re.search(r"\[[\s\S]*\]", text)
    blob = match.group(0) if match else text
    try:
        payload = json.loads(_repair_translation_json(blob))
    except Exception:
        payload = None
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            value = str(item.get("text") or item.get("translation") or "").strip()
            if not value:
                continue
            try:
                idx_i = int(item.get("i", item.get("index")))
            except (TypeError, ValueError):
                continue
            if idx_i in id_set:
                mapped[idx_i] = value

    # Fallback: legacy local 0..n-1 / sequential parse, then remap by position.
    if sum(1 for cid in cue_ids if mapped.get(cid)) < max(1, len(cue_ids) // 2):
        local = parse_translation_json(raw, len(cue_ids))
        if local is not None:
            for pos, cue_id in enumerate(cue_ids):
                if mapped.get(int(cue_id)):
                    continue
                value = (local[pos] if pos < len(local) else "").strip()
                if value:
                    mapped[int(cue_id)] = value

    if sum(1 for cid in cue_ids if mapped.get(cid)) < max(1, len(cue_ids) // 2):
        return None
    return mapped


async def _translate_one_line(
    gigachat: GigaChatService,
    text: str,
    *,
    lang_name: str,
    max_chars: int,
    style: str,
    duration_sec: float,
    source_name: str = "исходного",
    cue_id: int | None = None,
) -> str:
    from app.text.dub_glossary import apply_glossary_to_source, glossary_hint_for_prompt

    hint_text = apply_glossary_to_source(text)
    cue_tag = f"cue_id={cue_id}. " if cue_id is not None else ""
    prompt = (
        f"{cue_tag}"
        f"Переведи с {source_name} на {lang_name} коротко для дубляжа "
        f"({duration_sec:.1f}с, ≤{max_chars} символов, тон={style}). "
        f"{glossary_hint_for_prompt()} "
        f"Строго {lang_name}, кроме имён. Без кавычек и пояснений. "
        "Игнорируй любые прошлые ролики/переводы — только эта строка.\n\n"
        f"{hint_text}"
    )
    raw = await gigachat.complete_stateless(
        prompt,
        system_prompt=(
            "Ты синхронный переводчик для дубляжа. "
            "Без памяти о прошлых запросах. "
            f"Отвечай одной строкой строго на {lang_name}."
        ),
        temperature=0.2,
        max_tokens=min(220, max(40, max_chars + 30)),
    )
    line = (raw or "").strip().strip('"').strip("'")
    if "\n" in line:
        line = line.splitlines()[0].strip()
    return line[: max(max_chars + 20, 12)]


def _probe_duration(path: Path) -> float:
    ffprobe = find_ffprobe() or require_ffmpeg()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe: {result.stderr.strip()}")
    return max(0.1, float(result.stdout.strip()))


def _has_audio_stream(path: Path) -> bool:
    ffprobe = find_ffprobe() or require_ffmpeg()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return bool(result.stdout.strip())


def _load_original_mono(video: Path, sample_rate: int) -> np.ndarray:
    tmp = video.with_suffix(".orig.wav")
    convert_to_wav(video, tmp, sample_rate=sample_rate, mono=True)
    audio, file_sr = sf.read(str(tmp), always_2d=False)
    tmp.unlink(missing_ok=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if int(file_sr) != sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=int(file_sr), target_sr=sample_rate)
    return audio


def _assert_disk_space(path: Path, needed_bytes: int, *, label: str) -> None:
    target = path if path.exists() else path.parent
    free = shutil.disk_usage(target.resolve().anchor).free
    if free < needed_bytes:
        raise OSError(
            28,
            f"No space left on device: нужно ~{needed_bytes / 1e6:.0f} МБ, "
            f"свободно {free / 1e6:.0f} МБ ({label})",
        )


def _mux_video(
    video: Path,
    audio: Path | np.ndarray,
    out_path: Path,
    *,
    bg_volume: float = 0.15,
    voice_volume: float = 1.35,
    duck_original: bool = False,
    sample_rate: int | None = None,
) -> None:
    """Картинка без перекодирования + финальная аудиодорожка.

    numpy-массив уходит в ffmpeg через stdin (PCM), без промежуточного WAV.
    """
    ffmpeg = require_ffmpeg()
    stdin_bytes: bytes | None = None
    if isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate обязателен для PCM mux")
        wav = np.ascontiguousarray(audio, dtype=np.float32)
        channels = 1 if wav.ndim == 1 else int(wav.shape[-1])
        stdin_bytes = float_audio_to_pcm16(wav)
        del wav
        audio_args = [
            "-f",
            "s16le",
            "-ar",
            str(int(sample_rate)),
            "-ac",
            str(channels),
            "-i",
            "pipe:0",
        ]
        duck_original = False
    else:
        audio_args = ["-i", str(audio)]

    if duck_original and _has_audio_stream(video) and bg_volume > 0.001:
        filter_complex = (
            f"[0:a]volume={bg_volume:.3f}[bg];"
            f"[1:a]volume={voice_volume:.3f}[fg];"
            f"[bg][fg]amix=inputs=2:duration=first:dropout_transition=2[a]"
        )
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(video),
            *audio_args,
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            str(out_path),
        ]
    else:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(video),
            *audio_args,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            str(out_path),
        ]
    if stdin_bytes is None:
        result = subprocess.run(cmd, capture_output=True, text=True)
        err = (result.stderr or "").strip()
    else:
        result = subprocess.run(cmd, input=stdin_bytes, capture_output=True)
        err = result.stderr.decode("utf-8", errors="replace").strip()
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux: {err}")


class VideoDubService:
    def __init__(
        self,
        settings: Settings,
        transcription: TranscriptionService,
        gigachat: GigaChatService,
        synthesis: SynthesisService,
        accents: AccentService,
    ) -> None:
        self.settings = settings
        self.transcription = transcription
        self.gigachat = gigachat
        self.synthesis = synthesis
        self.accents = accents
        self._stt_source_lang: str | None = None

    async def analyze(
        self,
        media_path: Path,
        *,
        quiet_audio: bool = False,
    ) -> tuple[list[TimedSegment], float]:
        segments, _wav, duration = await self.transcription.transcribe_timed(
            media_path,
            language=self.settings.video_dub_stt_language,
            max_seconds=self.settings.video_dub_max_seconds,
            for_video=True,
            quiet_audio=bool(quiet_audio),
        )
        self._stt_source_lang = getattr(
            self.transcription, "last_video_stt_language", None
        )
        if not segments:
            raise ValueError("В видео не удалось распознать речь")
        return segments, duration

    async def translate_segments(
        self,
        segments: list[TimedSegment],
        language: str,
        *,
        media_duration: float | None = None,
        on_progress: ProgressCb | None = None,
        user_id: int | None = None,
    ) -> list[str]:
        lang = normalize_reply_lang(language)
        from app.text.digit_speech import looks_like_countdown

        # Drop chat / prior-dub GigaChat memory so cues don't inherit old phrases.
        if user_id is not None:
            try:
                self.gigachat.reset(int(user_id))
            except Exception:
                logger.exception("Failed to reset GigaChat history for user %s", user_id)

        blob = " ".join(seg.text for seg in segments[:80])
        source = resolve_dub_source_language(
            blob,
            target_lang=lang,
            stt_language=self._stt_source_lang,
            countdown=looks_like_countdown(segments),
            default="en",
        )
        if source == lang:
            logger.info("Dub STT language=%s matches target — skip translation", source)
            from app.text.digit_speech import translate_digit_cue

            return [
                translate_digit_cue(seg.text or "", language=lang) or seg.text
                for seg in segments
            ]
        src_name = LANG_NAMES.get(source, source)
        name = LANG_NAMES.get(lang, lang)
        logger.info("Translating %d cues %s → %s", len(segments), source, lang)

        from app.text.digit_speech import is_digit_like_text, translate_digit_cue
        from app.text.vocalizations import is_background_vocalization

        media = float(media_duration) if media_duration and media_duration > 0 else (
            max(float(segments[-1].end) + 1.0, float(segments[-1].start) + 1.0)
            if segments
            else 30.0
        )
        budgets: list[int] = []
        slot_secs: list[float] = []
        out: list[str] = [""] * len(segments)
        digit_done = 0
        filler_skip: set[int] = set()
        for i, seg in enumerate(segments):
            if is_background_vocalization(seg.text or ""):
                filler_skip.add(i)
                budgets.append(0)
                slot_secs.append(0.0)
                continue
            dig = translate_digit_cue(seg.text or "", language=lang)
            if dig:
                out[i] = dig
                digit_done += 1
            max_chars = max(12, xtts_chunk_limit(lang) - 8)
            cps = 7.0 if lang in {"ja", "ko"} else 12.0
            _, speech_dur, _pause_room, hard_cap = cue_sync_budget(
                segments, i, media, gap_sec=MIN_PHRASE_GAP_SEC
            )
            slot = max(speech_dur, hard_cap * 0.92)
            slot_secs.append(round(slot, 2))
            budgets.append(max(8, min(max_chars, int(slot * cps))))
        if filler_skip:
            logger.info(
                "Skip background vocalizations (uh-huh/oh…): %d/%d",
                len(filler_skip),
                len(segments),
            )
        if digit_done:
            logger.info("Digit cues resolved without LLM: %d/%d", digit_done, len(segments))
        if digit_done + len(filler_skip) == len(segments):
            from app.text.digit_speech import assert_translation_word_parity

            return assert_translation_word_parity(segments, out)

        batch_size = 16
        need_idx = [
            i
            for i, t in enumerate(out)
            if not (t or "").strip() and i not in filler_skip
        ]
        # батчим только не-цифровые
        batches: list[list[int]] = []
        for start in range(0, len(need_idx), batch_size):
            batches.append(need_idx[start : start + batch_size])
        total_batches = max(1, len(batches))
        job_nonce = uuid.uuid4().hex[:8]
        for batch_i, idxs in enumerate(batches):
            # Keep each batch isolated even if chat history is re-enabled later.
            if user_id is not None:
                try:
                    self.gigachat.reset(int(user_id))
                except Exception:
                    pass
            batch = [segments[i] for i in idxs]
            if on_progress is not None:
                await on_progress(
                    batch_i + 1,
                    total_batches,
                    f"{src_name} → {name} {batch_i + 1}/{total_batches}",
                )
            lines = []
            for j, seg in enumerate(batch):
                gi = idxs[j]
                from app.text.dub_glossary import apply_glossary_to_source

                lines.append(
                    {
                        "i": int(gi),
                        "duration_sec": slot_secs[gi],
                        "max_chars": budgets[gi],
                        "style": seg.style,
                        "text": apply_glossary_to_source(seg.text or ""),
                    }
                )
            from app.text.dub_glossary import glossary_hint_for_prompt

            prompt = (
                f"job={job_nonce} batch={batch_i + 1}/{total_batches}. "
                f"Переведи реплики с {src_name} на {name}.\n"
                "Это НОВЫЙ изолированный запрос: не используй тексты из прошлых "
                "видео, чатов или предыдущих батчей.\n"
                "Каждая строка должна сохранять смысл исходной и уложиться в "
                "duration_sec секунд устной речи (не длиннее max_chars символов).\n"
                "Формулируй короче оригинала, если не влезает по времени — "
                "сжимай, не выкидывай смысл.\n"
                "Если в тексте отдельные числа/countdown (5 4 3…) — переводи КАЖДОЕ "
                "число отдельным словом (пять, четыре…), НИКОГДА не делай дроби "
                "вроде 5.4 / пять целых.\n"
                f"{glossary_hint_for_prompt()}\n"
                f"Весь text — строго {name}, кроме имён собственных. "
                "Не копируй исходную фразу и не смешивай языки.\n"
                "Сохрани тон: question / calm / expressive.\n"
                "Не добавляй пояснений. Верни ТОЛЬКО JSON-массив вида "
                '[{"i":<cue_id>,"text":"..."}] той же длины и с теми же i, '
                "что в запросе. Внутри text не используй кавычки — перефразируй.\n\n"
                + json.dumps(lines, ensure_ascii=False)
            )
            max_tokens = min(3500, max(600, 140 * len(batch) + 200))
            try:
                raw = await self.gigachat.complete_stateless(
                    prompt,
                    system_prompt=(
                        "Ты синхронный переводчик для дубляжа видео. "
                        "Без памяти о прошлых роликах и диалогах. "
                        "Сжимаешь формулировки, не теряя смысл. "
                        f"Отвечай только JSON, все фразы на {name}."
                    ),
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
            except Exception:
                logger.exception("Batch %d translation request failed", batch_i)
                raw = ""
            parsed_map = _parse_translation_by_ids(raw, idxs)
            if parsed_map is None:
                logger.warning(
                    "Batch translation parse failed (%d segs @%s). Raw head: %s",
                    len(batch),
                    idxs[:3],
                    (raw or "")[:400],
                )
                parsed_map = {}
            for j, seg in enumerate(batch):
                gi = idxs[j]
                text = (parsed_map.get(gi) or "").strip()
                if leftover_source_language(text, source=source, target=lang):
                    logger.warning(
                        "Cue %d still in source language, retry: %s",
                        gi,
                        text[:80],
                    )
                    text = ""
                if text:
                    out[gi] = text
                    continue
                try:
                    retry = await _translate_one_line(
                        self.gigachat,
                        seg.text,
                        lang_name=name,
                        source_name=src_name,
                        max_chars=budgets[gi],
                        style=seg.style,
                        duration_sec=seg.duration,
                        cue_id=gi,
                    )
                    retry = (retry or "").strip()
                    if leftover_source_language(retry, source=source, target=lang):
                        logger.warning(
                            "Cue %d translation stayed in %s: %s",
                            gi,
                            source,
                            retry[:80],
                        )
                        retry = ""
                    out[gi] = retry
                except Exception:
                    logger.exception("Per-line translation failed for seg %d", gi)
                    out[gi] = ""
        from app.text.dub_glossary import sanitize_dub_translation

        return [
            sanitize_dub_translation(segments[i].text or "", out[i] or "")
            for i in range(len(segments))
        ]

    async def render(
        self,
        user_id: int,
        video_path: Path,
        segments: list[TimedSegment],
        translated: list[str],
        language: str,
        duration_sec: float,
        *,
        on_progress: ProgressCb | None = None,
        clone_refs_override: list[Path] | None = None,
        reuse_clip_paths: dict[int, Path] | None = None,
        cue_audio_dir: Path | None = None,
        lock_to_speech: bool = False,
    ) -> DubResult:
        """clone_refs_override: voice from THESE refs (e.g. a chosen dub cue)
        instead of re-extracting from the video — used by voice-pick re-dub.
        reuse_clip_paths: cue index → wav reused as-is (expressive keepers).
        cue_audio_dir: dump every synthesized cue wav here (voice picking).
        lock_to_speech: pin each cue to its ASR onset (no cascade layout).
        Required for voice-pick — otherwise long reused clips shove later cues
        by minutes and the muted ASR windows become silence holes."""
        if len(translated) != len(segments):
            raise ValueError("Число переводов не совпало с сегментами")
        from app.text.digit_speech import (
            assert_translation_word_parity,
            ensure_full_countdown,
            looks_like_countdown,
            translate_digit_cue,
        )

        if looks_like_countdown(segments):
            segments = ensure_full_countdown(
                segments, media_duration=duration_sec
            )
            translated = assert_translation_word_parity(segments, translated)
            # добить пустые переводы
            translated = [
                translate_digit_cue(s.text or "") or (t or s.text or "")
                for s, t in zip(segments, translated)
            ]
            logger.info(
                "Countdown cues locked: %s",
                [s.text for s in segments],
            )
        from app.text.dub_glossary import sanitize_dub_translation

        translated = [
            sanitize_dub_translation(s.text or "", t or "")
            for s, t in zip(segments, translated)
        ]
        # Mixed cues («ах, ах, да!»): moan tokens are not spoken — strip them so
        # TTS time goes only to real words and the timeline is not stretched.
        from app.text.vocalizations import (
            is_background_vocalization,
            remove_vocalization_tokens,
        )

        translated = [
            remove_vocalization_tokens(t) if (t or "").strip() else t
            for t in translated
        ]
        if len(translated) != len(segments):
            raise ValueError("Число переводов не совпало с сегментами")

        kept = [
            (seg, text)
            for seg, text in zip(segments, translated)
            if not is_background_vocalization(seg.text or "")
        ]
        if len(kept) != len(segments):
            logger.info(
                "Render: drop %d background vocalization cues (uh-huh/oh…)",
                len(segments) - len(kept),
            )
            segments = [p[0] for p in kept]
            translated = [p[1] for p in kept]
        lang = normalize_reply_lang(language)
        source_lang = resolve_dub_source_language(
            " ".join((s.text or "") for s in segments),
            target_lang=lang,
            stt_language=self._stt_source_lang,
            countdown=looks_like_countdown(segments),
            default="en",
        )
        cross_lingual = source_lang != lang
        if cross_lingual:
            logger.info(
                "Cross-lingual dub %s -> %s (keep video voice clone)",
                source_lang,
                lang,
            )
        # Isolate this video from leftover Fish/cache state of the previous one.
        reset = getattr(self.synthesis, "reset_after_dub", None)
        if callable(reset):
            try:
                reset(user_id)
            except Exception:
                logger.debug("pre-render TTS reset failed", exc_info=True)
        sr = int(self.synthesis.primary.sample_rate or self.settings.output_sample_rate)
        timeline = np.zeros(max(1, int(duration_sec * sr) + sr // 4), dtype=np.float32)
        out_dir = safe_user_path(self.settings.users_dir, user_id, "outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = uuid.uuid4().hex[:8]
        gap_sec = max(
            0.08,
            float(self.settings.video_dub_phrase_gap_sec),
            float(getattr(self.settings, "video_dub_min_phrase_gap_sec", 0.12)),
        )
        slack_sec = max(0.0, float(self.settings.video_dub_slot_slack_sec))
        total = len(segments)

        clone_dir = out_dir / f"clone_{stamp}"
        replace_mode = self.settings.video_dub_mix_mode == "replace"
        background: np.ndarray | None = None
        original: np.ndarray | None = None
        original_mono: np.ndarray | None = None
        vocals: np.ndarray | None = None
        vocals_sr = sr
        stems_method = "none"

        if replace_mode:
            if on_progress is not None:
                await on_progress(0, total, "Выделяю фон и голос…")
            stem_dir = out_dir / f"stems_{stamp}"
            stems = await asyncio.to_thread(
                extract_stems_for_dub,
                video_path,
                segments,
                stem_dir,
                mode=self.settings.video_dub_separation,
                device=self.settings.video_dub_separation_device,
                sample_rate=sr,
                speech_gain=float(self.settings.video_dub_speech_mask_gain),
                vocal_leak=float(self.settings.video_dub_vocal_leak),
            )
            background = stems.background
            vocals = stems.vocals
            vocals_sr = stems.sample_rate
            stems_method = str(stems.method or "mask")
            logger.info("Language-replace stems method=%s", stems_method)
            original_mono = await asyncio.to_thread(_load_original_mono, video_path, sr)
        elif _has_audio_stream(video_path):
            original_mono = await asyncio.to_thread(_load_original_mono, video_path, sr)
            original = original_mono

        if original_mono is None and _has_audio_stream(video_path):
            original_mono = await asyncio.to_thread(_load_original_mono, video_path, sr)

        if on_progress is not None:
            await on_progress(0, total, "Клонирую голос из видео…")
        use_seamless = self.synthesis.primary.name == "seamless_m4t"
        from app.text.digit_speech import looks_like_countdown

        countdown_mode = looks_like_countdown(segments)
        clone_refs: list[Path] = []
        clone_sec = 0.0
        source_for_s2st = vocals
        source_for_s2st_sr = vocals_sr
        if not use_seamless:
            vox_clone = self.synthesis.primary.name == "voxcpm2"
            if clone_refs_override:
                clone_refs = [p for p in clone_refs_override if Path(p).exists()]
                if not clone_refs:
                    raise ValueError("clone_refs_override: файлы референса не найдены")
                clone_sec = 0.0
                for p in clone_refs:
                    try:
                        info = sf.info(str(p))
                        clone_sec += float(info.frames) / float(info.samplerate or 1)
                    except Exception:
                        pass
                logger.info(
                    "Clone refs override (voice pick): %s",
                    [str(p) for p in clone_refs],
                )
            else:
                # Countdown/ASMR: original mono (breath + room are part of the style).
                # Ordinary videos: Demucs vocals when available. Clone denoise runs only
                # on low-SNR clips (see app.audio.clone_denoise).
                if countdown_mode or vocals is None:
                    clone_src = original_mono if original_mono is not None else vocals
                    clone_sr = sr if original_mono is not None else int(vocals_sr)
                else:
                    clone_src = vocals
                    clone_sr = int(vocals_sr)
                clone_refs, clone_sec = await asyncio.to_thread(
                    extract_clone_references,
                    video_path,
                    segments,
                    clone_dir,
                    sample_rate=int(self.settings.reference_sample_rate),
                    max_sec=(
                        min(8.0, float(self.settings.video_dub_clone_max_sec))
                        if vox_clone
                        else float(self.settings.video_dub_clone_max_sec)
                    ),
                    max_clips=1 if vox_clone else int(self.settings.video_dub_clone_max_clips),
                    min_clip_sec=float(self.settings.video_dub_clone_min_clip_sec),
                    fallback_sec=float(self.settings.video_dub_clone_fallback_sec),
                    source_audio=clone_src,
                    source_sr=clone_sr,
                    from_original=True,
                    prefer_whisper=bool(countdown_mode),
                    enable_denoise=bool(self.settings.video_dub_clone_denoise),
                    denoise_snr_db=float(self.settings.video_dub_clone_denoise_snr_db),
                    denoise_prop=float(self.settings.video_dub_clone_denoise_prop),
                )
            # vocals оставляем до конца TTS — нужен перенос интонации (F0)
        else:
            if on_progress is not None:
                await on_progress(0, total, "SeamlessM4T: речь → речь…")
            if source_for_s2st is None:
                if original is not None:
                    source_for_s2st = original
                    source_for_s2st_sr = sr
                elif _has_audio_stream(video_path):
                    source_for_s2st = await asyncio.to_thread(
                        _load_original_mono, video_path, sr
                    )
                    source_for_s2st_sr = sr
                    original = source_for_s2st

        clips: list[np.ndarray | None] = []
        clip_durations: list[float] = []
        # Measured real TTS pace (chars/sec) on this video — Fish free tier often
        # speaks slower than the 12.5 cps estimate, which overflows the timeline.
        measured_cps: list[float] = []
        retranslate_left = max(
            0, int(getattr(self.settings, "video_dub_overflow_retranslate", 8))
        )
        src_name = LANG_NAMES.get(source_lang, source_lang)
        tgt_name = LANG_NAMES.get(lang, lang)
        for i, (seg, text) in enumerate(zip(segments, translated)):
            spoken = text.strip()
            if on_progress is not None:
                tip = spoken[:40] if spoken else (seg.text or "")[:40]
                await on_progress(i + 1, total, tip)
            # Stoны / fillers in translation (or missed earlier) — leave original bed.
            from app.text.vocalizations import is_background_vocalization

            if (
                is_background_vocalization(seg.text or "")
                or is_background_vocalization(spoken)
            ) and not use_seamless:
                clips.append(None)
                clip_durations.append(0.0)
                continue
            if not spoken and not use_seamless:
                clips.append(None)
                clip_durations.append(0.0)
                continue
            # Voice-pick re-dub: expressive keepers reuse their original dub.
            reuse_path = (reuse_clip_paths or {}).get(i)
            if reuse_path is not None and Path(reuse_path).exists():
                try:
                    rwav, rsr = sf.read(str(reuse_path), always_2d=False)
                    rwav = np.asarray(rwav, dtype=np.float32)
                    if rwav.ndim > 1:
                        rwav = np.mean(rwav, axis=1)
                    if rwav.size >= 8:
                        if int(rsr) != sr:
                            import librosa

                            rwav = librosa.resample(
                                rwav, orig_sr=int(rsr), target_sr=sr
                            )
                        sp0, speech_dur, pause_room, hard_cap = cue_sync_budget(
                            segments, i, duration_sec, gap_sec=gap_sec
                        )
                        rwav = clamp_clip_to_slot(
                            rwav,
                            sr,
                            speech_dur=speech_dur,
                            hard_cap=hard_cap,
                        )
                        clips.append(rwav)
                        clip_durations.append(rwav.size / float(sr))
                        continue
                except Exception:
                    logger.exception("Reuse clip failed cue %d — re-synthesize", i)
            try:
                if use_seamless:
                    if source_for_s2st is None:
                        raise RuntimeError("Нет исходной дорожки для Seamless S2ST")
                    a = max(0, int(float(seg.start) * source_for_s2st_sr))
                    b = min(
                        source_for_s2st.shape[0],
                        int(float(seg.end) * source_for_s2st_sr),
                    )
                    if b <= a + 8:
                        raise RuntimeError("Пустой слот исходника")
                    src_slice = np.asarray(source_for_s2st[a:b], dtype=np.float32)
                    if src_slice.ndim > 1:
                        src_slice = np.mean(src_slice, axis=1)
                    engine = self.synthesis.primary
                    wav = await asyncio.to_thread(
                        engine.synthesize_s2st,
                        src_slice,
                        int(source_for_s2st_sr),
                        lang,
                    )
                    file_sr = int(engine.sample_rate)
                else:
                    from app.text.digit_speech import (
                        build_asmr_digit_ssml,
                        is_digit_like_text,
                        translate_digit_cue,
                    )

                    digit_ru = translate_digit_cue(spoken, language=lang) or (
                        translate_digit_cue(seg.text or "", language=lang)
                    )
                    is_digit = bool(digit_ru) or is_digit_like_text(spoken)
                    if digit_ru:
                        spoken = digit_ru
                    elif lang == "ru":
                        spoken = await self.accents.add_accents(spoken)
                    # Защита от «пять целых четыре» если LLM/preprocess склеили
                    if re.search(r"цел(ая|ых|ые)", spoken, re.IGNORECASE):
                        from app.text.digit_speech import extract_digit_sequence

                        fixed = extract_digit_sequence(seg.text or "")
                        if fixed:
                            # один cue = одно число; не склеивать обратно
                            spoken = fixed[0] if is_digit else " ".join(fixed)
                            is_digit = True
                    sp0, speech_dur, pause_room, hard_cap = cue_sync_budget(
                        segments, i, duration_sec, gap_sec=gap_sec
                    )
                    budget = hard_cap
                    if not is_digit:
                        compacted = compact_repetitions_to_budget(
                            spoken,
                            budget_sec=budget,
                            language=lang,
                        )
                        if compacted != spoken:
                            logger.info(
                                "Cue %d compacted repeats: est %.2fs -> %.2fs (budget %.2fs)",
                                i,
                                estimate_tts_sec(spoken, language=lang),
                                estimate_tts_sec(compacted, language=lang),
                                budget,
                            )
                            spoken = compacted
                    est = estimate_tts_sec(spoken, language=lang)
                    if measured_cps:
                        # Use the measured (slower) pace if it predicts overflow.
                        cps_med = sorted(measured_cps)[len(measured_cps) // 2]
                        cps_base = 12.5 if lang == "ru" else 14.0
                        cps_eff = min(cps_base, max(4.0, cps_med))
                        est = max(est, _plain_chars(spoken) / cps_eff)
                    if (
                        not is_digit
                        and est > budget * 0.98
                        and retranslate_left > 0
                        and budget > 0.8
                        and (seg.text or "").strip()
                        and source_lang != lang
                    ):
                        # Slot won't fit at the real TTS pace — ask the LLM for a
                        # tighter phrasing BEFORE spending a TTS call.
                        retranslate_left -= 1
                        from app.text.dub_glossary import sanitize_dub_translation

                        try:
                            tighter = await _translate_one_line(
                                self.gigachat,
                                seg.text or "",
                                lang_name=tgt_name,
                                source_name=src_name,
                                max_chars=max(8, int(budget * 11.0 * 0.92)),
                                style=seg.style,
                                duration_sec=budget,
                                cue_id=i,
                            )
                        except Exception:
                            logger.exception("Overflow re-translate failed cue %d", i)
                            tighter = ""
                        tighter = remove_vocalization_tokens(
                            sanitize_dub_translation(seg.text or "", tighter or "")
                        )
                        if (
                            tighter
                            and tighter != spoken
                            and not leftover_source_language(
                                tighter, source=source_lang, target=lang
                            )
                        ):
                            new_est = estimate_tts_sec(tighter, language=lang)
                            if new_est < est:
                                logger.info(
                                    "Cue %d re-translated tighter: est %.2fs -> %.2fs "
                                    "(budget %.2fs)",
                                    i,
                                    est,
                                    new_est,
                                    budget,
                                )
                                spoken = tighter
                                est = new_est
                    dense = (not is_digit) and est > budget * 0.90
                    # Never infer ASMR from an absolute RMS threshold alone.  A
                    # normally spoken, quiet/normalised cue used to be classified as
                    # whisper and synthesized at 0.86x; on long videos that produced
                    # the characteristic low, robotic "monster" voice.  The special
                    # whisper treatment is reserved for the explicit countdown path.
                    asmr = countdown_mode and is_digit
                    cloud_fish = self.synthesis.primary.name == "openrouter_fish"
                    # Never speed/slow audio or video — fit via silence borrow only.
                    speed_ovr = 1.0
                    if is_digit:
                        is_last_digit = i + 1 >= len(segments) or not is_digit_like_text(
                            segments[i + 1].text or ""
                        )
                        if cloud_fish and asmr:
                            # Keep Fish whisper character; duration fit happens after.
                            target_ssml = spoken
                            tone = "calm"
                            pause_cap = 0.05
                        elif cloud_fish:
                            target_ssml = spoken
                            tone = "neutral"
                            pause_cap = 0.05
                        else:
                            target_ssml = build_asmr_digit_ssml(
                                spoken,
                                pause_after_ms=0,
                                rate=0.76,
                                volume=0.64,
                                soft_tail=is_last_digit and spoken in {"ноль", "нуль"},
                            )
                            tone = "calm"
                            # Не раздувать паузы внутри клипа — sync по STT start
                            pause_cap = 0.08
                    else:
                        from app.text.dub_glossary import inject_step_pause_ssml

                        spoken_ssml = inject_step_pause_ssml(spoken)
                        target_ssml = transfer_ssml_for_slot(
                            seg.ssml or "", spoken_ssml, dense=dense
                        )
                        if spoken_ssml != spoken and "<break" in spoken_ssml:
                            target_ssml = spoken_ssml if spoken_ssml.startswith("<speak") else (
                                f"<speak>{spoken_ssml}</speak>"
                            )
                        plain, prosody = parse_ssml(target_ssml)
                        if not plain.strip():
                            plain = spoken
                        spoken = plain
                        tone = intonation_from_prosody(seg.style, prosody)
                        if asmr:
                            tone = "calm"
                        elif cloud_fish and not countdown_mode:
                            # [expressive]/[softly] tags destabilize the cloned
                            # timbre on conversational videos (voice "sings",
                            # clone detaches). Fish modulates from text+ref anyway.
                            tone = "neutral"
                        pause_cap = 0.10 if dense or speed_ovr > 1.08 else 0.32
                    plain, prosody = parse_ssml(target_ssml)
                    if not plain.strip():
                        plain = spoken
                    clone_ref_txt = ""
                    for ref_p in clone_refs:
                        side = Path(ref_p).with_suffix(".txt")
                        if side.exists():
                            try:
                                clone_ref_txt = side.read_text(encoding="utf-8").strip()
                            except OSError:
                                clone_ref_txt = ""
                            if clone_ref_txt:
                                break
                    wav, file_sr = await self.synthesis.synthesize_pcm(
                        user_id,
                        plain,
                        language=lang,
                        intonation=tone,
                        speaker_wavs=clone_refs,
                        ssml=target_ssml,
                        speed_override=speed_ovr,
                        max_pause_sec=pause_cap,
                        allow_fallback=False,
                        cross_lingual=cross_lingual,
                        ref_transcript=clone_ref_txt or None,
                    )
                    wav = np.asarray(wav, dtype=np.float32)
                if wav.ndim > 1:
                    wav = np.mean(wav, axis=1)
                if wav.size == 0:
                    raise RuntimeError("Не удалось озвучить сегмент")
                if int(file_sr) != sr:
                    import librosa

                    wav = librosa.resample(
                        wav, orig_sr=int(file_sr), target_sr=sr
                    )
                wav = polish_dub_clip(wav, sr)
                # ASMR: шёпот тише разговорного (~−6…−8 dB)
                from app.text.digit_speech import is_digit_like_text

                digit_cue = is_digit_like_text(seg.text or "")
                if countdown_mode and (digit_cue or float(seg.rms or 0.0) < 0.055):
                    # Soft ASMR duck — but not into inaudible when source is tiny.
                    duck = 0.72 if float(seg.rms or 0.0) < 0.025 else 0.55
                    wav = wav * duck

                # Prosody energy match only on countdown/ASMR. On ordinary videos
                # the source slice often includes music/SFX and flattens TTS dynamics
                # into a pumped, robotic delivery.
                sp0, speech_dur, pause_room, hard_cap = cue_sync_budget(
                    segments, i, duration_sec, gap_sec=gap_sec
                )
                if countdown_mode and not use_seamless:
                    prosody_src = original_mono if original_mono is not None else vocals
                    prosody_sr = sr if original_mono is not None else int(vocals_sr)
                    if prosody_src is not None:
                        from app.audio.prosody_transfer import transfer_prosody_from_source
                        import librosa

                        a = max(0, int(float(sp0) * prosody_sr))
                        b = min(
                            int(prosody_src.shape[0]),
                            int((float(sp0) + float(speech_dur)) * prosody_sr),
                        )
                        if b > a + 8:
                            src_slice = np.asarray(prosody_src[a:b], dtype=np.float32)
                            if src_slice.ndim > 1:
                                src_slice = np.mean(src_slice, axis=1)
                            if int(prosody_sr) != sr:
                                src_slice = librosa.resample(
                                    src_slice, orig_sr=int(prosody_sr), target_sr=sr
                                )
                            wav = transfer_prosody_from_source(wav, src_slice, sr)
                natural_dur = wav.size / float(sr)
                # No atempo: only drop trailing TTS silence so natural speech stays.
                if wav.size > int(0.08 * sr):
                    keep_floor = max(1, int(0.06 * sr))
                    wav = _drop_trailing_silence(wav, sr, keep_floor)
                    if wav.size < keep_floor:
                        pass
                    natural_dur = wav.size / float(sr)
                # Track the real TTS pace to calibrate estimates of later cues.
                if (
                    not countdown_mode
                    and not digit_cue
                    and 0.5 <= natural_dur <= 15.0
                ):
                    chars_n = _plain_chars(spoken)
                    if chars_n >= 6:
                        measured_cps.append(chars_n / natural_dur)
                if digit_cue:
                    logger.info(
                        "Digit natural i=%d dur=%.2fs speech=%.2fs (no tempo)",
                        i,
                        natural_dur,
                        speech_dur,
                    )
                elif natural_dur > hard_cap * 1.02:
                    logger.info(
                        "Cue %d longer than slot (%.2fs > %.2fs) — silence borrow",
                        i,
                        natural_dur,
                        hard_cap,
                    )
                # Safety: Fish sometimes hallucinates 40s+ from a short cue —
                # that cascades the whole timeline into silence holes.
                wav = clamp_clip_to_slot(
                    wav,
                    sr,
                    speech_dur=speech_dur,
                    hard_cap=hard_cap,
                    abs_cap_sec=12.0 if lock_to_speech else 18.0,
                )
                natural_dur = wav.size / float(sr)
            except Exception:
                logger.exception("TTS failed for dub cue %d, skip instead of other voice", i)
                clips.append(None)
                clip_durations.append(0.0)
                continue
            clips.append(wav)
            clip_durations.append(wav.size / float(sr))
            if cue_audio_dir is not None:
                try:
                    cue_audio_dir.mkdir(parents=True, exist_ok=True)
                    sf.write(
                        str(cue_audio_dir / f"cue_{i:03d}.wav"),
                        wav,
                        sr,
                        subtype="PCM_16",
                    )
                except OSError:
                    logger.warning("Cue audio dump failed %d", i, exc_info=True)

        if use_seamless:
            source_for_s2st = None
            gc.collect()
        # Detect inter-phrase silence on the original bed for borrow layout.
        silence_gaps: list[tuple[float, float]] = []
        if original_mono is not None and not countdown_mode:
            from app.audio.background_preserve import original_speech_windows
            from app.audio.silence_gaps import silence_gaps_for_dub

            asr_win = original_speech_windows(segments, pad_sec=0.02)
            silence_gaps = silence_gaps_for_dub(
                original_mono,
                sr,
                asr_win,
                min_silence_sec=max(
                    0.10,
                    float(
                        getattr(
                            self.settings,
                            "video_dub_min_phrase_gap_sec",
                            gap_sec,
                        )
                    ),
                ),
            )
            logger.info(
                "Silence gaps for layout: %d intervals (media=%.1fs)",
                len(silence_gaps),
                duration_sec,
            )
        layout_gap = max(
            float(gap_sec),
            float(
                getattr(self.settings, "video_dub_min_phrase_gap_sec", gap_sec)
            ),
        )
        # Countdown: centre on lip peaks. Voice-pick: pin to ASR onset.
        # Ordinary: full TTS + silence borrow.
        if countdown_mode:
            placements = center_align_digit_placements(
                segments,
                clip_durations,
                media_duration=duration_sec,
                preroll_first_sec=0.06,
            )
        elif lock_to_speech or clone_refs_override or reuse_clip_paths:
            placements = lock_placements_to_speech(
                segments, clip_durations, duration_sec
            )
            logger.info(
                "Voice-pick layout: locked %d cues to ASR speech onsets",
                sum(1 for d in clip_durations if d > 1e-4),
            )
        else:
            placements = layout_silence_borrow_placements(
                segments,
                clip_durations,
                duration_sec,
                gap_sec=layout_gap,
                silence_gaps=silence_gaps,
                max_early_sec=float(
                    getattr(self.settings, "video_dub_layout_max_early_sec", 1.5)
                ),
            )
        last_audio = float(duration_sec)
        for i, wav in enumerate(clips):
            if wav is None or wav.size == 0:
                continue
            last_audio = max(last_audio, placements[i][0] + wav.size / float(sr))
        need_n = max(timeline.size, int(math.ceil(last_audio * sr)) + int(0.08 * sr))
        if need_n > timeline.size:
            timeline = np.pad(timeline, (0, need_n - timeline.size))
        # Short crossfade: 50ms smeared overlapping speech into metallic ripple.
        xfade = max(1, int(0.01 * sr))
        for i, wav in enumerate(clips):
            if wav is None or wav.size == 0:
                continue
            t0, _t1 = placements[i]
            piece = wav
            # Full natural TTS — layout reserved silence; do not hard-clamp.
            placements[i] = (t0, t0 + piece.size / float(sr))
            t1 = placements[i][1]
            used = piece.size / float(sr)
            slot = max(0.08, float(t1) - float(t0))
            shifted = t0 - float(segments[i].start)
            overflow = max(
                0.0,
                used
                - max(0.08, float(segments[i].end) - float(segments[i].start)),
            )
            if overflow > 0.15 or abs(shifted) > 0.20:
                logger.info(
                    "Dub layout i=%s orig=%.2f-%.2f placed=%.2f-%.2f "
                    "wav=%.2fs window=%.2fs shift=%+.2fs overflow=%.2fs",
                    i,
                    segments[i].start,
                    segments[i].end,
                    t0,
                    t1,
                    used,
                    slot,
                    shifted,
                    overflow,
                )
            piece = _fade_clip_edges(piece, sr, 18.0 if countdown_mode else 4.0)
            _overlay_voice(
                timeline, piece, int(round(t0 * sr)), xfade=xfade
            )

        placed_segments = [
            TimedSegment(
                start=placements[i][0],
                end=max(placements[i][0], placements[i][1]),
                text=segments[i].text,
                style=segments[i].style,
                rms=segments[i].rms,
                ssml=segments[i].ssml,
                rate=segments[i].rate,
                volume=segments[i].volume,
                pause_after=segments[i].pause_after,
                words=list(segments[i].words or []),
            )
            for i in range(len(segments))
        ]

        if on_progress is not None:
            await on_progress(total, total, "Склеиваю видео…")

        # Match dubbed speech loudness to original vocals before peak limiting.
        if (
            not countdown_mode
            and vocals is not None
            and replace_mode
        ):
            level_windows = [
                (float(placements[i][0]), float(placements[i][1]))
                for i in range(len(segments))
                if clips[i] is not None and clips[i].size > 0
            ]
            if int(vocals_sr) != sr:
                import librosa

                vocals_for_level = librosa.resample(
                    np.asarray(vocals, dtype=np.float32).reshape(-1),
                    orig_sr=int(vocals_sr),
                    target_sr=sr,
                )
            else:
                vocals_for_level = np.asarray(vocals, dtype=np.float32).reshape(-1)
            timeline = match_voice_level_to_source(
                timeline, vocals_for_level, sr, level_windows
            )

        peak = float(np.max(np.abs(timeline)) or 1.0)
        if peak > 0.98:
            timeline *= 0.97 / peak

        bg_volume = float(self.settings.video_dub_bg_volume)
        voice_volume = float(self.settings.video_dub_voice_volume)

        if replace_mode and original_mono is not None:
            # Full rewrite: original outside speech, Demucs accomp under speech, + TTS.
            from app.audio.background_preserve import (
                original_speech_windows,
                render_original_background,
            )
            from app.text.digit_speech import looks_like_countdown

            countdown = looks_like_countdown(segments)
            # Mute original vocals ONLY where we actually place TTS. Skipped cues
            # (moans / оаоа / fillers) keep the original sound in their window.
            voiced_segments = [
                segments[i]
                for i in range(len(segments))
                if clips[i] is not None and clips[i].size > 0
            ]
            windows = original_speech_windows(voiced_segments, pad_sec=0.04)
            # Duck under *placed* TTS; ASR windows still drive vocal rewrite.
            duck_windows = [
                (float(placements[i][0]), float(placements[i][1]))
                for i in range(len(segments))
                if clips[i] is not None and clips[i].size > 0
            ]
            speech_duck = float(
                getattr(self.settings, "video_dub_speech_duck", 0.70)
            )
            # Demucs accompaniment only fills *under* original speech — never
            # replaces the full soundtrack (that dropped heels/laughs/SFX).
            accomp = None
            accomp_sr = None
            if background is not None and not countdown:
                if stems_method == "demucs":
                    accomp = background
                    accomp_sr = sr
                else:
                    # Mask mode (no Demucs): `background` already has speech
                    # regions attenuated — use it as the under-speech bed.
                    # `vocals` in mask mode is the FULL MIX, not a vocal stem:
                    # passing it down makes orig−0.9×orig (bed gutted to 10%)
                    # plus transient re-injection of speech consonants — the
                    # "crazy background" artifact. Never subtract it.
                    accomp = background
                    accomp_sr = sr
                    vocals = None
            final_audio = render_original_background(
                original_mono,
                timeline,
                sr,
                windows,
                accompaniment=accomp,
                accompaniment_sr=accomp_sr,
                vocals=vocals,
                vocals_sr=int(vocals_sr) if vocals is not None else None,
                vocal_amount=0.9,
                countdown_mute=bool(countdown),
                background_speed=1.0,
                bg_gain=bg_volume,
                voice_gain=voice_volume,
                duck_windows=duck_windows,
                speech_duck=speech_duck,
            )
        elif original is not None:
            final_audio = mix_dub_tracks(
                original,
                timeline,
                bg_volume=bg_volume,
                voice_volume=voice_volume,
            )
        else:
            final_audio = timeline * voice_volume
            peak = float(np.max(np.abs(final_audio)) or 1.0)
            if peak > 0.98:
                final_audio *= 0.97 / peak
        background = None
        original = None
        vocals = None
        timeline = None
        gc.collect()

        video_out = out_dir / f"dub_{stamp}.mp4"
        srt_path = out_dir / f"dub_{stamp}.srt"
        write_srt(srt_path, placed_segments, translated)
        _assert_disk_space(
            out_dir,
            video_path.stat().st_size + int(duration_sec * 32_000) + 80 * 1024 * 1024,
            label="склейка MP4",
        )
        try:
            await asyncio.to_thread(
                _mux_video,
                video_path,
                final_audio,
                video_out,
                duck_original=False,
                sample_rate=sr,
            )
        except Exception:
            video_out.unlink(missing_ok=True)
            raise
        del final_audio
        gc.collect()
        # stems/clone занимают гигабайты — чистим сразу после mux
        for junk in (clone_dir, out_dir / f"stems_{stamp}"):
            if junk.exists():
                shutil.rmtree(junk, ignore_errors=True)
        return DubResult(
            video_path=video_out,
            segments=segments,
            translated=translated,
            srt_path=srt_path,
            clone_refs=clone_refs,
            clone_sec=clone_sec,
            cue_audio_dir=cue_audio_dir,
            cue_audio_sr=sr if cue_audio_dir is not None else 0,
            placements=[(float(a), float(b)) for a, b in placements],
        )
