"""Нарезка таймлайна дубляжа по нейросетевым word-timestamps (Whisper/WhisperX)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.services.transcription import TimedSegment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WordToken:
    text: str
    start: float
    end: float


def flatten_word_timeline(segments: list[TimedSegment]) -> list[WordToken]:
    words: list[WordToken] = []
    for seg in segments:
        if seg.words:
            for token, start, end in seg.words:
                t = (token or "").strip()
                if not t:
                    continue
                s = float(start)
                e = max(s + 0.04, float(end))
                if words and s < words[-1].end - 0.02:
                    s = words[-1].end
                words.append(WordToken(text=t, start=s, end=e))
            continue
        text = (seg.text or "").strip()
        if not text:
            continue
        parts = re.findall(r"\S+", text)
        if not parts:
            continue
        span = max(0.08, float(seg.end) - float(seg.start))
        step = span / len(parts)
        for i, part in enumerate(parts):
            s = float(seg.start) + i * step
            words.append(WordToken(text=part, start=s, end=s + step))
    return words


def build_dub_cues_from_words(
    words: list[WordToken],
    *,
    min_pause_sec: float = 0.30,
    max_cue_sec: float = 6.5,
    min_cue_sec: float = 1.0,
    hard_max_cue_sec: float = 9.0,
) -> list[TimedSegment]:
    """Группирует слова в естественные реплики по паузам (forced-align / WhisperX)."""
    if not words:
        return []
    min_pause_sec = max(0.18, float(min_pause_sec))
    max_cue_sec = max(2.0, float(max_cue_sec))
    min_cue_sec = max(0.45, float(min_cue_sec))
    hard_max = max(max_cue_sec, float(hard_max_cue_sec))

    cues: list[TimedSegment] = []
    buf: list[WordToken] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        text = " ".join(w.text for w in buf).strip()
        if not text:
            buf = []
            return
        start = float(buf[0].start)
        end = float(buf[-1].end)
        style = "question" if text.endswith("?") else "expressive" if text.endswith("!") else "neutral"
        cues.append(
            TimedSegment(
                start=start,
                end=max(start + 0.08, end),
                text=text,
                style=style,
                words=[(w.text, w.start, w.end) for w in buf],
            )
        )
        buf = []

    for i, w in enumerate(words):
        if not buf:
            buf.append(w)
            continue
        gap = float(w.start) - float(buf[-1].end)
        cur_dur = float(buf[-1].end) - float(buf[0].start)
        punct_break = buf[-1].text.endswith((".", "?", "!", "…", ";"))
        # Prefer sentence boundaries. Short mid-phrase hesitations (0.3–0.5s)
        # must not restart TTS prosody — that sounds robotic on long speech.
        long_pause = gap >= min_pause_sec * 1.5
        need_break = (
            cur_dur >= max_cue_sec
            or (punct_break and gap >= min_pause_sec * 0.45 and cur_dur >= min_cue_sec * 0.75)
            or (long_pause and cur_dur >= min_cue_sec)
            or (gap >= min_pause_sec and punct_break and cur_dur >= min_cue_sec * 0.65)
        )
        if need_break and cur_dur >= min_cue_sec * 0.65:
            flush()
        elif cur_dur >= hard_max:
            flush()
        buf.append(w)
    flush()
    return cues


_SENT_END_RE = re.compile(r"[.!?…»)”'\"]+\s*$")


def merge_sentence_fragments(
    cues: list[TimedSegment],
    *,
    max_gap_sec: float = 2.2,
    max_merged_sec: float = 14.0,
) -> list[TimedSegment]:
    """Join fragments of ONE sentence back into a single cue.

    WhisperX/Whisper words often carry no punctuation, so mid-sentence
    hesitations were split into separate timelines ("So," / "I see you" /
    "are really, really nutty boy. And"). A cue whose text does NOT end with
    sentence-final punctuation is continued by the next cue (small gap only).
    Word timestamps are kept, so inter-word pauses survive as interior break
    markers downstream (``enrich_segments_ssml`` → SSML breaks / "…").
    """
    if len(cues) < 2:
        return cues
    from app.text.digit_speech import is_digit_like_text

    out: list[TimedSegment] = []
    for cue in cues:
        if out:
            prev = out[-1]
            prev_text = (prev.text or "").rstrip()
            gap = float(cue.start) - float(prev.end)
            merged_dur = float(cue.end) - float(prev.start)
            if (
                prev_text
                and not _SENT_END_RE.search(prev_text)
                and not is_digit_like_text(prev_text)
                and not is_digit_like_text(cue.text or "")
                and -0.05 <= gap <= max_gap_sec
                and merged_dur <= max_merged_sec
            ):
                words = list(prev.words or []) + list(cue.words or [])
                tail = (cue.text or "").strip()
                joiner = "" if prev_text.endswith(("-", "—", "–")) else " "
                text = f"{prev_text}{joiner}{tail}".strip()
                out[-1] = TimedSegment(
                    start=float(prev.start),
                    end=max(float(prev.end), float(cue.end)),
                    text=text,
                    style=(
                        "question"
                        if text.endswith("?")
                        else "expressive"
                        if text.endswith("!")
                        else prev.style
                    ),
                    words=words,
                    rms=max(float(prev.rms or 0.0), float(cue.rms or 0.0)),
                    ssml="",
                    rate=prev.rate,
                    volume=prev.volume,
                    pause_after=cue.pause_after,
                    no_speech_prob=max(
                        float(prev.no_speech_prob or 0.0),
                        float(cue.no_speech_prob or 0.0),
                    ),
                    avg_logprob=min(
                        float(prev.avg_logprob or 0.0),
                        float(cue.avg_logprob or 0.0),
                    ),
                )
                continue
        out.append(cue)
    return out


def merge_micro_cues(
    cues: list[TimedSegment],
    *,
    min_cue_sec: float = 0.55,
) -> list[TimedSegment]:
    """Склеивает крошечные реплики с соседними — иначе слот 0.08с и сдвиг таймлайна."""
    if len(cues) < 2:
        return cues
    min_cue_sec = max(0.25, float(min_cue_sec))
    out: list[TimedSegment] = []
    for cue in cues:
        if (
            out
            and float(cue.duration) < min_cue_sec
            and float(out[-1].duration) < min_cue_sec * 2.5
        ):
            prev = out[-1]
            words = list(prev.words or []) + list(cue.words or [])
            text = f"{prev.text} {cue.text}".strip()
            style = (
                "question"
                if text.endswith("?")
                else "expressive"
                if text.endswith("!")
                else prev.style
            )
            out[-1] = TimedSegment(
                start=float(prev.start),
                end=max(float(prev.end), float(cue.end)),
                text=text,
                style=style,
                words=words,
                rms=max(float(prev.rms or 0.0), float(cue.rms or 0.0)),
                ssml=prev.ssml or cue.ssml,
                rate=prev.rate,
                volume=prev.volume,
                pause_after=cue.pause_after,
            )
            continue
        if out and float(cue.duration) < min_cue_sec:
            prev = out[-1]
            gap = float(cue.start) - float(prev.end)
            if gap <= 0.85:
                words = list(prev.words or []) + list(cue.words or [])
                text = f"{prev.text} {cue.text}".strip()
                out[-1] = TimedSegment(
                    start=float(prev.start),
                    end=max(float(prev.end), float(cue.end)),
                    text=text,
                    style=prev.style if not text.endswith("?") else "question",
                    words=words,
                    rms=max(float(prev.rms or 0.0), float(cue.rms or 0.0)),
                    ssml=prev.ssml or cue.ssml,
                    rate=prev.rate,
                    volume=prev.volume,
                    pause_after=cue.pause_after,
                )
                continue
        out.append(cue)
    return out


def speech_window(
    seg: TimedSegment,
) -> tuple[float, float]:
    """Окно артикуляции по word-timestamps (не расширенный mute-слот)."""
    words = list(seg.words or [])
    if words:
        start = float(words[0][1])
        end = float(words[-1][2])
        return start, max(start + 0.08, end)
    return float(seg.start), max(float(seg.start) + 0.08, float(seg.end))


def expand_cue_windows(
    cues: list[TimedSegment],
    media_duration: float | None = None,
    *,
    gap_sec: float = 0.12,
) -> list[TimedSegment]:
    """Не трогает end речи: word-окна остаются для lip-sync.

    Раньше end растягивался до следующей реплики — TTS заполнял паузы и
    сдвигал следующие фразы. Бюджет до соседа считается отдельно в video_dub.
    """
    del media_duration, gap_sec
    return list(cues)


def rebuild_video_dub_segments(
    segments: list[TimedSegment],
    *,
    min_pause_sec: float = 0.30,
    max_cue_sec: float = 6.5,
    min_cue_sec: float = 1.0,
    media_duration: float | None = None,
) -> list[TimedSegment]:
    """Пересобирает STT-сегменты в реплики, синхронные с word-таймлайном."""
    from app.text.digit_speech import (
        ensure_full_countdown,
        is_digit_like_text,
        looks_like_countdown,
        split_countdown_cues,
    )

    words = flatten_word_timeline(segments)
    if len(words) < 2:
        cues = list(segments)
    else:
        # Countdown: арабские И словесные (five/four/…) — режем по слову.
        from app.text.digit_speech import _sequence_is_countdown, normalize_digit_token

        digit_rus = []
        for w in words:
            ru = normalize_digit_token(w.text)
            if ru:
                digit_rus.append(ru)
        digitish = len(digit_rus)
        arabicish = sum(
            1 for w in words if re.fullmatch(r"\d+", w.text.strip(".,!?"))
        )
        tight = (
            digitish >= 4
            and digitish >= int(0.8 * len(words))
            and (
                arabicish >= 2
                or digitish == len(words)
                or _sequence_is_countdown(digit_rus)
            )
        )
        if tight:
            cues = build_dub_cues_from_words(
                words,
                min_pause_sec=min(0.18, float(min_pause_sec)),
                max_cue_sec=1.2,
                min_cue_sec=0.12,
                hard_max_cue_sec=1.8,
            )
        else:
            cues = build_dub_cues_from_words(
                words,
                min_pause_sec=min_pause_sec,
                max_cue_sec=max_cue_sec,
                min_cue_sec=min_cue_sec,
            )
        if not cues:
            cues = list(segments)
    # Split multi-digit lines when present; restore missing digits ONLY for
    # pure countdown clips — never wipe a normal transcript.
    cues = split_countdown_cues(cues)
    from app.text.vocalizations import drop_background_vocalizations

    # Uh-huh / oh oh / uh-uh… — leave original as bed, do not TTS.
    cues = drop_background_vocalizations(cues)
    # Mid-sentence fragments → one cue; inter-word pauses stay in seg.words
    # and become interior break markers instead of separate timelines.
    before_merge = len(cues)
    cues = merge_sentence_fragments(cues)
    if len(cues) != before_merge:
        logger.info(
            "Merged sentence fragments: %d → %d cues", before_merge, len(cues)
        )
    if looks_like_countdown(cues):
        cues = ensure_full_countdown(cues, media_duration=media_duration)
        if any(is_digit_like_text(c.text) for c in cues):
            from app.text.digit_speech import expand_digit_windows

            return expand_digit_windows(cues, media_duration=media_duration)
    return merge_micro_cues(cues, min_cue_sec=max(0.35, float(min_cue_sec) * 0.45))
