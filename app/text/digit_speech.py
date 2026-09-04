"""Счёт / отдельные цифры в речи: не дроби, а слова с паузами (ASMR countdown)."""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass

from app.services.transcription import TimedSegment

logger = logging.getLogger(__name__)

# EN / RU / digits → каноническое русское слово для TTS
_DIGIT_WORD: dict[str, str] = {
    "0": "ноль",
    "1": "один",
    "2": "два",
    "3": "три",
    "4": "четыре",
    "5": "пять",
    "6": "шесть",
    "7": "семь",
    "8": "восемь",
    "9": "девять",
    "10": "десять",
    "zero": "ноль",
    "one": "один",
    "two": "два",
    "three": "три",
    "four": "четыре",
    "five": "пять",
    "six": "шесть",
    "seven": "семь",
    "eight": "восемь",
    "nine": "девять",
    "ten": "десять",
    "oh": "ноль",
    "o": "ноль",
    "ноль": "ноль",
    "нуль": "ноль",
    "один": "один",
    "одна": "один",
    "два": "два",
    "две": "два",
    "три": "три",
    "четыре": "четыре",
    "пять": "пять",
    "шесть": "шесть",
    "семь": "семь",
    "восемь": "восемь",
    "девять": "девять",
    "десять": "десять",
}

_RU_TO_VAL: dict[str, int] = {
    "ноль": 0,
    "один": 1,
    "два": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
}

_VAL_TO_RU: dict[int, str] = {v: k for k, v in _RU_TO_VAL.items()}

# Классический ASMR countdown
DEFAULT_COUNTDOWN: tuple[int, ...] = (5, 4, 3, 2, 1, 0)

_MIN_DIGIT_SEC: dict[str, float] = {
    "пять": 0.75,
    "четыре": 0.52,
    "три": 0.55,
    "два": 0.42,
    "один": 0.50,
    "ноль": 0.42,
    "шесть": 0.45,
    "семь": 0.40,
    "восемь": 0.48,
    "девять": 0.45,
    "десять": 0.48,
}

_TOKEN_CLEAN = re.compile(r"^[\s\"'`«»]+|[\s\"'`«»,.;:!?…]+$", re.UNICODE)


@dataclass(frozen=True)
class DigitToken:
    word_ru: str
    start: float
    end: float
    raw: str


def normalize_digit_token(raw: str) -> str | None:
    """Возвращает русское слово-цифру или None, если это не одиночная цифра/число 0–10."""
    t = _TOKEN_CLEAN.sub("", (raw or "").strip().lower().replace("ё", "е"))
    if not t:
        return None
    if t in _DIGIT_WORD:
        return _DIGIT_WORD[t]
    # "5." / "5," / "(5)"
    t2 = re.sub(r"[^\w]", "", t)
    if t2 in _DIGIT_WORD:
        return _DIGIT_WORD[t2]
    return None


def digit_value(raw: str) -> int | None:
    ru = normalize_digit_token(raw)
    if ru is None:
        return None
    return _RU_TO_VAL.get(ru)


def is_digit_like_text(text: str) -> bool:
    return normalize_digit_token(text) is not None


def is_digit_word_token(text: str) -> bool:
    """Арабская цифра или EN/RU слово-цифра."""
    return normalize_digit_token(text) is not None


def extract_digit_sequence(text: str) -> list[str] | None:
    """Если вся фраза — ряд цифр (5 4 3 / five four), вернуть список RU-слов."""
    parts = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text or "")
    if len(parts) < 2:
        one = normalize_digit_token(text or "")
        return [one] if one else None
    out: list[str] = []
    for p in parts:
        w = normalize_digit_token(p)
        if w is None:
            return None
        out.append(w)
    return out if len(out) >= 2 else None


def looks_like_countdown(segments: list[TimedSegment]) -> bool:
    """Обратный отсчёт (↓ к нулю), не обычное видео с парой чисел в тексте.

    Важно: при True ``ensure_full_countdown`` *заменяет* всю партитуру рядом
    цифр. Поэтому требуем, чтобы реплики почти целиком были digit-cues.
    """
    if not segments:
        return False

    digs = [s for s in segments if is_digit_like_text(s.text or "")]
    non = [s for s in segments if not is_digit_like_text(s.text or "")]

    # Одна фраза с рядом цифр (классический packed countdown)
    if len(segments) == 1:
        seq = extract_digit_sequence(segments[0].text or "")
        if seq and len(seq) >= 3:
            vals = [_RU_TO_VAL[w] for w in seq]
            desc = sum(1 for a, b in zip(vals, vals[1:]) if a > b)
            return desc >= len(vals) // 2
        return False

    # Смешанное видео (есть обычные фразы) — никогда не countdown-mode
    if non:
        # короткие междометия/мусор не спасают, но ≥2 непустых слова — речь
        real_non = [
            s
            for s in non
            if len(re.findall(r"\w+", s.text or "", flags=re.UNICODE)) >= 2
        ]
        if real_non:
            return False
        if len(non) > max(1, len(digs) // 4):
            return False

    if len(segments) >= 2:
        digit_ratio = len(digs) / float(len(segments))
        dig_dur = sum(float(s.duration) for s in digs)
        tot_dur = sum(float(s.duration) for s in segments) or 1e-6
        if digit_ratio < 0.85 or (dig_dur / tot_dur) < 0.75:
            return False

    vals = [digit_value(s.text or "") for s in digs]
    vals = [v for v in vals if v is not None]
    if len(vals) < 3:
        return False
    desc = sum(1 for a, b in zip(vals, vals[1:]) if a > b)
    asc = sum(1 for a, b in zip(vals, vals[1:]) if a < b)
    # нужен нисходящий характер и конец у 0/1
    if desc > asc and (vals[-1] <= 1) and max(vals) >= 3:
        return True
    if desc >= 2 and max(vals) - min(vals) >= 3 and vals[-1] <= 1:
        return True
    joined = " ".join(s.text or "" for s in digs)
    if re.search(r"\d", joined) and desc >= asc and len(vals) >= 3 and vals[-1] <= 1:
        return True
    return False


def detect_countdown_energy_slots(
    audio,
    sample_rate: int,
    *,
    count: int,
    min_sep_sec: float = 0.75,
) -> list[tuple[float, float]]:
    """Find ``count`` speech islands with a multi-feature score (not RMS alone).

    Whisper often packs sparse ASMR countdown words into the first few seconds;
    peaks across the whole clip restore the real lip timeline.

    Primary cues (weighted): relative energy, speech-band ratio, spectral flux
    (onset), zero-crossing rate, spectral entropy. Extra: centroid + flux
    continuity used inside band/entropy terms.
    """
    import numpy as np

    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono.ndim > 1:
        mono = np.mean(mono, axis=1)
    sr = int(sample_rate)
    need = max(2, int(count))
    if mono.size < sr // 2 or need < 2:
        return []

    hop = max(1, int(0.01 * sr))
    win = max(hop * 2, int(0.05 * sr))
    n_frames = max(1, (mono.size - win) // hop)
    if n_frames < 8:
        return []

    rms = np.zeros(n_frames, dtype=np.float32)
    zcr = np.zeros(n_frames, dtype=np.float32)
    band = np.zeros(n_frames, dtype=np.float32)
    centroid = np.zeros(n_frames, dtype=np.float32)
    entropy = np.zeros(n_frames, dtype=np.float32)
    flux = np.zeros(n_frames, dtype=np.float32)
    prev_mag: np.ndarray | None = None
    freqs = np.fft.rfftfreq(win, d=1.0 / float(sr))
    speech_mask = (freqs >= 250.0) & (freqs <= 3800.0)

    for i in range(n_frames):
        frame = mono[i * hop : i * hop + win]
        rms[i] = float(np.sqrt(np.mean(np.square(frame))) or 0.0)
        if frame.size > 1:
            zcr[i] = float(np.mean(np.abs(np.diff(np.signbit(frame)))))
        spec = np.abs(np.fft.rfft(frame * np.hanning(frame.size)))
        power = np.square(spec) + 1e-12
        total = float(np.sum(power))
        band[i] = float(np.sum(power[speech_mask]) / total)
        centroid[i] = float(np.sum(freqs * power) / total)
        p = power / total
        entropy[i] = float(-np.sum(p * np.log(p + 1e-12)) / max(np.log(len(p)), 1e-6))
        if prev_mag is not None:
            flux[i] = float(np.mean(np.maximum(0.0, spec - prev_mag)))
        prev_mag = spec

    def _norm(x: np.ndarray) -> np.ndarray:
        lo = float(np.percentile(x, 10))
        hi = float(np.percentile(x, 95))
        if hi <= lo + 1e-9:
            return np.zeros_like(x)
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    # Whisper: elevated ZCR + mid speech-band energy; voiced speech: energy+flux.
    score = (
        0.28 * _norm(rms)
        + 0.22 * _norm(band)
        + 0.20 * _norm(flux)
        + 0.15 * _norm(zcr)
        + 0.15 * (1.0 - _norm(entropy))  # speech less flat-random than pure noise
    )
    # Mild preference for human-ish centroids (not sub-bass thumps / hiss).
    cen_n = _norm(centroid)
    score = score * (0.85 + 0.30 * cen_n)

    thr = float(np.percentile(score, 78))
    thr = max(thr, float(np.max(score)) * 0.22, 0.08)
    times = np.arange(n_frames, dtype=np.float64) * (hop / float(sr))

    candidates: list[tuple[float, float, int]] = []
    for i in range(1, n_frames - 1):
        val = float(score[i])
        if val < thr:
            continue
        if val >= float(score[i - 1]) and val >= float(score[i + 1]):
            candidates.append((float(times[i]), val, i))
    candidates.sort(key=lambda item: -item[1])

    picked: list[tuple[float, float, int]] = []
    for t, val, idx in candidates:
        if all(abs(t - pt) >= float(min_sep_sec) for pt, _, _ in picked):
            picked.append((t, val, idx))
        if len(picked) >= need + 2:
            break
    if len(picked) < need:
        # Quiet ASMR: relax threshold once and retry.
        thr2 = thr * 0.72
        candidates = []
        for i in range(1, n_frames - 1):
            val = float(score[i])
            if val < thr2:
                continue
            if val >= float(score[i - 1]) and val >= float(score[i + 1]):
                candidates.append((float(times[i]), val, i))
        candidates.sort(key=lambda item: -item[1])
        picked = []
        for t, val, idx in candidates:
            if all(abs(t - pt) >= float(min_sep_sec) for pt, _, _ in picked):
                picked.append((t, val, idx))
            if len(picked) >= need + 2:
                break
    if len(picked) < need:
        return []
    picked = sorted(picked, key=lambda item: item[0])[:need]

    slots: list[tuple[float, float]] = []
    media = mono.size / float(sr)
    for t, val, idx in picked:
        floor = max(thr * 0.35, val * 0.28)
        lo = hi = idx
        while lo > 0 and float(score[lo]) >= floor:
            lo -= 1
        while hi < n_frames - 1 and float(score[hi]) >= floor:
            hi += 1
        s = float(times[max(0, lo)])
        e = float(times[min(len(times) - 1, hi)]) + hop / float(sr)
        if e - s < 0.35:
            s = max(0.05, t - 0.28)
            e = min(media - 0.02, t + 0.35)
        if e - s > 1.6:
            s = max(0.05, t - 0.55)
            e = min(media - 0.02, t + 0.70)
        slots.append((s, max(s + 0.20, e)))
    return slots


def snap_countdown_cues_to_energy(
    segments: list[TimedSegment],
    audio,
    sample_rate: int,
) -> list[TimedSegment]:
    """Re-time countdown digits onto real speech-energy peaks."""
    if not looks_like_countdown(segments):
        return segments
    digs = [s for s in segments if is_digit_like_text(s.text or "")]
    if len(digs) < 3:
        return segments
    slots = detect_countdown_energy_slots(
        audio, sample_rate, count=len(digs), min_sep_sec=0.70
    )
    if len(slots) < len(digs):
        logger.warning(
            "Countdown energy snap skipped: found %d slots for %d digits",
            len(slots),
            len(digs),
        )
        return segments

    out: list[TimedSegment] = []
    for seg, (s, e) in zip(digs, slots):
        need = min_digit_duration(seg.text or "")
        if e - s < need:
            c = 0.5 * (s + e)
            s = max(0.05, c - need * 0.5)
            e = s + need
        ru = normalize_digit_token(seg.text or "") or (seg.text or "").strip()
        out.append(
            TimedSegment(
                start=float(s),
                end=float(e),
                text=ru,
                style="calm",
                words=[(ru, float(s), float(e))],
                rms=float(seg.rms or 0.0),
                ssml=seg.ssml,
                rate=seg.rate,
                volume=seg.volume,
                pause_after=seg.pause_after,
            )
        )
    span0 = float(digs[0].start), float(digs[-1].end)
    span1 = float(out[0].start), float(out[-1].end)
    logger.info(
        "Countdown energy snap: %.1f–%.1fs → %.1f–%.1fs (%d cues)",
        span0[0],
        span0[1],
        span1[0],
        span1[1],
        len(out),
    )
    return out


def _interpolate_slot(
    val: int,
    known: dict[int, tuple[float, float]],
    *,
    media_duration: float | None,
) -> tuple[float, float]:
    """Оценка (start, end) для пропущенной цифры по соседним таймкодам."""
    if val in known:
        return known[val]
    lower = [k for k in known if k < val]
    higher = [k for k in known if k > val]
    speech = 0.28
    if known:
        durs = [e - s for s, e in known.values()]
        speech = float(np_median(durs)) if durs else 0.28
        speech = max(0.18, min(0.55, speech))

    if lower and higher:
        lo = max(lower)
        hi = min(higher)
        t0_lo, t1_lo = known[lo]
        t0_hi, _ = known[hi]
        # линейная интерполяция между lo и hi по значению
        span_v = float(hi - lo) or 1.0
        frac = (val - lo) / span_v
        center_lo = 0.5 * (t0_lo + t1_lo)
        center_hi = 0.5 * (t0_hi + known[hi][1])
        center = center_lo + frac * (center_hi - center_lo)
        s = center - speech * 0.5
        return s, s + speech

    if higher:
        hi = min(higher)
        t0_hi, t1_hi = known[hi]
        # шаг по медиане интервалов между известными
        step = _median_step(known) or 1.0
        # сколько шагов вверх от val до hi
        n = hi - val
        end = t0_hi - 0.15 - (n - 1) * step
        start = end - speech
        if start < 0.05:
            start = 0.05
            end = start + speech
        return start, end

    if lower:
        lo = max(lower)
        t0_lo, t1_lo = known[lo]
        step = _median_step(known) or 1.0
        n = val - lo
        start = t1_lo + 0.15 + (n - 1) * step
        end = start + speech
        if media_duration is not None:
            end = min(end, float(media_duration) - 0.05)
            start = min(start, end - 0.12)
        return start, end

    # ничего нет — равномерно в [0.4, 6]
    span = 5.5 if media_duration is None else max(3.0, float(media_duration) * 0.55)
    # val 5→0 → index 0..5 for default
    try:
        idx = list(DEFAULT_COUNTDOWN).index(val)
    except ValueError:
        idx = max(0, 5 - val)
    start = 0.45 + idx * (span / 6.0)
    return start, start + speech


def np_median(xs: list[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return 0.28
    mid = n // 2
    if n % 2:
        return float(ys[mid])
    return 0.5 * (ys[mid - 1] + ys[mid])


def _median_step(known: dict[int, tuple[float, float]]) -> float | None:
    keys = sorted(known.keys(), reverse=True)
    gaps: list[float] = []
    for a, b in zip(keys, keys[1:]):
        # a > b (descending countdown)
        t1_a = known[a][1]
        t0_b = known[b][0]
        gap = t0_b - t1_a
        if 0.2 <= gap <= 2.5:
            gaps.append(gap + (known[a][1] - known[a][0]))  # center-ish step
        gaps.append(abs(known[b][0] - known[a][0]))
    if not gaps:
        return None
    return max(0.7, min(1.35, np_median(gaps)))


def min_digit_duration(word_ru: str) -> float:
    return float(_MIN_DIGIT_SEC.get((word_ru or "").strip().lower(), 0.30))


def expand_digit_windows(
    segments: list[TimedSegment],
    *,
    media_duration: float | None = None,
) -> list[TimedSegment]:
    """Расширяет слишком короткие STT-окна цифр до минимума по слогам (вокруг центра)."""
    out: list[TimedSegment] = []
    for seg in segments:
        if not is_digit_like_text(seg.text or ""):
            out.append(seg)
            continue
        ru = normalize_digit_token(seg.text or "") or (seg.text or "").strip()
        need = min_digit_duration(ru)
        s = float(seg.start)
        e = float(seg.end)
        if seg.words:
            s = float(seg.words[0][1])
            e = float(seg.words[-1][2])
        dur = max(0.05, e - s)
        if dur < need:
            c = 0.5 * (s + e)
            s = c - need * 0.5
            e = c + need * 0.5
        if s < 0.05:
            e += 0.05 - s
            s = 0.05
        if media_duration is not None and e > float(media_duration) - 0.02:
            shift = e - (float(media_duration) - 0.02)
            s = max(0.05, s - shift)
            e = float(media_duration) - 0.02
        out.append(
            TimedSegment(
                start=s,
                end=max(s + need * 0.9, e),
                text=ru,
                style="calm",
                words=[(ru, s, max(s + need * 0.9, e))],
                rms=float(seg.rms or 0.0),
                ssml=seg.ssml,
                rate=seg.rate,
                volume=seg.volume,
                pause_after=seg.pause_after,
            )
        )
    for i in range(1, len(out)):
        if not is_digit_like_text(out[i].text or ""):
            continue
        prev = out[i - 1]
        cur = out[i]
        if float(cur.start) < float(prev.end) + 0.2:
            need = max(
                min_digit_duration(cur.text or ""),
                float(cur.end) - float(cur.start),
            )
            ns = float(prev.end) + 0.25
            out[i] = TimedSegment(
                start=ns,
                end=ns + need,
                text=cur.text,
                style="calm",
                words=[(cur.text, ns, ns + need)],
                rms=float(cur.rms or 0.0),
            )
    return out


def ensure_full_countdown(
    segments: list[TimedSegment],
    *,
    media_duration: float | None = None,
    target: tuple[int, ...] = DEFAULT_COUNTDOWN,
) -> list[TimedSegment]:
    """Гарантирует полный ряд цифр (по умолчанию 5…0) с таймкодами.

    Если ASR потерял «пять»/«три» — восстанавливаем по соседним словам.
    """
    if not looks_like_countdown(segments):
        return segments

    known: dict[int, tuple[float, float]] = {}
    rms_vals: list[float] = []
    for seg in segments:
        val = digit_value(seg.text or "")
        if val is None:
            # попробуем words
            for token, s, e in seg.words or []:
                v = digit_value(token)
                if v is not None and v not in known:
                    known[v] = (float(s), max(float(s) + 0.08, float(e)))
            continue
        rms_vals.append(float(seg.rms or 0.0))
        s, e = float(seg.start), float(seg.end)
        if seg.words:
            s = float(seg.words[0][1])
            e = float(seg.words[-1][2])
        if val not in known:
            known[val] = (s, max(s + 0.08, e))

    if len(known) < 2 and not any(is_digit_like_text(s.text or "") for s in segments):
        return segments

    # Если нашли только часть ряда — всё равно достраиваем target,
    # когда похоже на countdown (есть ≥2 из target или max≥3).
    overlap = [v for v in target if v in known]
    if len(overlap) < 2 and (not known or max(known) < 3):
        return segments

    # Расширяем target от max известного вниз до 0, но не уже DEFAULT если max≥4
    hi = max(max(known), max(target))
    if hi >= 4:
        want = tuple(range(hi, -1, -1))
    else:
        want = target

    out: list[TimedSegment] = []
    base_rms = np_median(rms_vals) if rms_vals else 0.03
    missing: list[int] = []
    for val in want:
        ru = _VAL_TO_RU[val]
        start, end = _interpolate_slot(val, known, media_duration=media_duration)
        if val not in known:
            missing.append(val)
        out.append(
            TimedSegment(
                start=float(start),
                end=max(float(start) + 0.12, float(end)),
                text=ru,
                style="calm",
                words=[(ru, float(start), max(float(start) + 0.12, float(end)))],
                rms=base_rms,
            )
        )

    # монотонность таймкодов: каждый следующий позже предыдущего
    for i in range(1, len(out)):
        prev = out[i - 1]
        cur = out[i]
        min_start = float(prev.end) + 0.35
        if float(cur.start) < min_start:
            dur = max(0.18, float(cur.end) - float(cur.start))
            out[i] = TimedSegment(
                start=min_start,
                end=min_start + dur,
                text=cur.text,
                style="calm",
                words=[(cur.text, min_start, min_start + dur)],
                rms=float(cur.rms or base_rms),
            )

    if missing:
        logger.info(
            "Countdown restored missing digits %s → %s cues",
            missing,
            [c.text for c in out],
        )
    elif len(out) != len(segments):
        logger.info(
            "Countdown normalized %d → %d cues: %s",
            len(segments),
            len(out),
            [c.text for c in out],
        )
    return expand_digit_windows(out, media_duration=media_duration)


def _sequence_is_countdown(ru_words: list[str]) -> bool:
    vals = [_RU_TO_VAL[w] for w in ru_words if w in _RU_TO_VAL]
    if len(vals) < 2:
        return False
    desc = sum(1 for a, b in zip(vals, vals[1:]) if a > b)
    asc = sum(1 for a, b in zip(vals, vals[1:]) if a < b)
    if desc > asc and vals[-1] <= 1:
        return True
    if desc >= asc and len(vals) >= 4 and vals[-1] <= 1:
        return True
    return False


def split_countdown_cues(
    segments: list[TimedSegment],
    *,
    max_gap_sec: float = 2.2,
) -> list[TimedSegment]:
    """Каждая цифра countdown → отдельная реплика (не «5,4» и не одна фраза)."""
    out: list[TimedSegment] = []
    for seg in segments:
        words = list(seg.words or [])
        if len(words) >= 2:
            digit_words: list[tuple[str, float, float, str]] = []
            ok = True
            arabic = 0
            for token, start, end in words:
                ru = normalize_digit_token(token)
                if ru is None:
                    ok = False
                    break
                if re.fullmatch(r"\d+", _TOKEN_CLEAN.sub("", token)):
                    arabic += 1
                digit_words.append((ru, float(start), float(end), token))
            ru_only = [d[0] for d in digit_words]
            # All tokens are digit-words → one cue each (incl. "five four" fragments).
            is_cd = ok and len(digit_words) >= 2
            if is_cd:
                for i in range(1, len(digit_words)):
                    gap = digit_words[i][1] - digit_words[i - 1][2]
                    if gap > max_gap_sec:
                        is_cd = False
                        break
            if is_cd:
                for ru, start, end, raw in digit_words:
                    out.append(
                        TimedSegment(
                            start=start,
                            end=max(start + 0.08, end),
                            text=ru,
                            style="calm",
                            words=[(raw, start, end)],
                            rms=float(seg.rms or 0.0),
                        )
                    )
                continue
        seq = extract_digit_sequence(seg.text or "")
        if seq and len(seq) >= 2 and (
            re.search(r"\d", seg.text or "")
            or _sequence_is_countdown(seq)
            or all(normalize_digit_token(p) for p in re.findall(r"\S+", seg.text or ""))
        ):
            if not words or len(words) != len(seq):
                span = max(0.2, float(seg.end) - float(seg.start))
                step = span / len(seq)
                for i, ru in enumerate(seq):
                    s = float(seg.start) + i * step
                    e = s + min(0.4, step * 0.45)
                    out.append(
                        TimedSegment(
                            start=s,
                            end=max(s + 0.08, e),
                            text=ru,
                            style="calm",
                            words=[(ru, s, e)],
                            rms=float(seg.rms or 0.0),
                        )
                    )
                continue
        if is_digit_like_text(seg.text or "") and (
            re.search(r"\d", seg.text or "")
            or looks_like_countdown([seg])
        ):
            ru = normalize_digit_token(seg.text or "") or (seg.text or "").strip()
            out.append(
                TimedSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=ru,
                    style="calm",
                    words=list(seg.words or []),
                    rms=float(seg.rms or 0.0),
                    ssml=seg.ssml,
                    rate=seg.rate,
                    volume=seg.volume,
                    pause_after=seg.pause_after,
                )
            )
            continue
        out.append(seg)
    return out


def build_asmr_digit_ssml(
    word_ru: str,
    *,
    pause_after_ms: int = 0,
    rate: float = 0.78,
    volume: float = 0.68,
    soft_tail: bool = False,
) -> str:
    """Падающая интонация + мягкая громкость для шёпотного countdown.

    Паузы между цифрами НЕ вшиваем в wav (иначе темп ломается) —
    размещение по таймкодам STT. После «ноль» — короткий break.
    """
    # pitch % — наш parse_ssml / XTTS; rate медленный для шёпота
    pitch = "-8%" if soft_tail else "-5%"
    body = (
        f'<prosody rate="{rate:.2f}" volume="{volume:.2f}" pitch="{pitch}">'
        f"{word_ru}"
        f"</prosody>"
    )
    if soft_tail:
        body += '<break time="220ms"/>'
    elif pause_after_ms >= 80:
        jitter = int(random.uniform(-40, 60))
        body += f'<break time="{max(80, int(pause_after_ms) + jitter)}ms"/>'
    return f"<speak>{body}</speak>"


def translate_digit_cue(text: str, language: str = "ru") -> str | None:
    """Без LLM: цифра → слово целевого языка (сейчас RU)."""
    del language
    return normalize_digit_token(text)


def assert_translation_word_parity(
    source_segments: list[TimedSegment],
    translated: list[str],
) -> list[str]:
    """Для countdown: число слов перевода = число исходных digit-cues."""
    if not looks_like_countdown(source_segments):
        return translated
    if len(translated) != len(source_segments):
        logger.warning(
            "Countdown translate parity: %d cues vs %d texts — realigning",
            len(source_segments),
            len(translated),
        )
    out: list[str] = []
    for i, seg in enumerate(source_segments):
        ru = translate_digit_cue(seg.text or "") or (
            translate_digit_cue(translated[i]) if i < len(translated) else None
        )
        if ru is None and i < len(translated):
            ru = (translated[i] or "").strip() or (seg.text or "")
        out.append(ru or (seg.text or ""))
    return out
