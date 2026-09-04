"""Уточнение концов слов по затуханию огибающей (шёпот / ASMR).

ASR/forced-align обрезает слово, когда энергия падает ниже абсолютного
порога тишины, хотя хвост шёпота ещё слышен 150–300 мс. Берём пик слова
и ищем момент, когда амплитуда падает до –40 dB относительно пика.
"""

from __future__ import annotations

import logging

import numpy as np

from app.services.transcription import TimedSegment

logger = logging.getLogger(__name__)


def _decay_end(
    mono: np.ndarray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
    *,
    rel_db: float = -40.0,
    max_extend_sec: float = 0.45,
    min_extend_sec: float = 0.12,
    hard_limit_sec: float | None = None,
) -> float:
    sr = int(sample_rate)
    a = max(0, int(float(start_sec) * sr))
    b0 = max(a + 1, int(float(end_sec) * sr))
    b_max = min(
        mono.size,
        int((float(end_sec) + float(max_extend_sec)) * sr),
    )
    if hard_limit_sec is not None:
        b_max = min(b_max, int(float(hard_limit_sec) * sr))
    if b_max <= a + 2:
        return max(float(end_sec), float(start_sec) + 0.08)

    # пик в исходном окне (+ чуть хвоста ASR)
    peak_b = min(b_max, max(b0 + int(0.05 * sr), a + 1))
    window = mono[a:peak_b]
    peak = float(np.max(np.abs(window)) or 0.0)
    if peak < 1e-5:
        return float(end_sec) + float(min_extend_sec)

    thr = peak * float(10 ** (rel_db / 20.0))
    # ищем последний сэмпл ≥ thr в расширенном окне
    search = mono[a:b_max]
    abs_s = np.abs(search)
    above = np.where(abs_s >= thr)[0]
    if above.size == 0:
        new_end = float(end_sec) + float(min_extend_sec)
    else:
        new_end = (a + int(above[-1]) + 1) / float(sr)

    # минимум: ASR end + min_extend (хвост шёпота)
    new_end = max(new_end, float(end_sec) + float(min_extend_sec))
    new_end = min(new_end, float(end_sec) + float(max_extend_sec))
    if hard_limit_sec is not None:
        new_end = min(new_end, float(hard_limit_sec) - 0.02)
    return max(float(start_sec) + 0.08, new_end)


def refine_segments_by_envelope(
    segments: list[TimedSegment],
    audio: np.ndarray,
    sample_rate: int,
    *,
    rel_db: float = -40.0,
    max_extend_sec: float = 0.45,
    min_extend_sec: float = 0.12,
) -> list[TimedSegment]:
    """Сдвигает end слов/фраз по relative decay; не заезжает на следующее слово."""
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono.ndim > 1:
        mono = np.mean(mono, axis=1)
    if mono.size < sample_rate // 4 or not segments:
        return segments

    sr = int(sample_rate)
    media_end = mono.size / float(sr)
    out: list[TimedSegment] = []
    extended = 0.0

    for i, seg in enumerate(segments):
        next_start = (
            float(segments[i + 1].start)
            if i + 1 < len(segments)
            else media_end
        )
        # не пересекаем соседа (оставляем 80 мс зазор)
        hard = min(media_end, next_start - 0.08)

        words = list(seg.words or [])
        new_words: list[tuple[str, float, float]] = []
        if words:
            for j, (tok, ws, we) in enumerate(words):
                if j + 1 < len(words):
                    w_hard = float(words[j + 1][1]) - 0.04
                else:
                    w_hard = hard
                we2 = _decay_end(
                    mono,
                    sr,
                    float(ws),
                    float(we),
                    rel_db=rel_db,
                    max_extend_sec=max_extend_sec,
                    min_extend_sec=min_extend_sec,
                    hard_limit_sec=w_hard,
                )
                extended += max(0.0, we2 - float(we))
                new_words.append((tok, float(ws), we2))
            new_start = float(new_words[0][1])
            new_end = float(new_words[-1][2])
        else:
            new_start = float(seg.start)
            new_end = _decay_end(
                mono,
                sr,
                float(seg.start),
                float(seg.end),
                rel_db=rel_db,
                max_extend_sec=max_extend_sec,
                min_extend_sec=min_extend_sec,
                hard_limit_sec=hard,
            )
            extended += max(0.0, new_end - float(seg.end))

        out.append(
            TimedSegment(
                start=new_start,
                end=max(new_start + 0.08, new_end),
                text=seg.text,
                style=seg.style,
                words=new_words or list(seg.words or []),
                rms=seg.rms,
                ssml=seg.ssml,
                rate=seg.rate,
                volume=seg.volume,
                pause_after=seg.pause_after,
            )
        )

    if extended >= 0.15:
        logger.info(
            "Envelope refine: +%.2fs total speech tail across %d cues",
            extended,
            len(out),
        )
    return out
