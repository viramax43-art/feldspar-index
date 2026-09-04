"""SSML-прослойка: снимаем разметку с оригинальной речи и переносим в синтез.

XTTS не читает теги, поэтому SSML — канонический формат обмена:
parse → rate / volume / emphasis / break → параметры TTS и усиление PCM.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_SPEAK_TAG = re.compile(r"</?speak\b[^>]*>", re.IGNORECASE)
_PROSODY = re.compile(
    r"<prosody\b([^>]*)>(.*?)</prosody>", re.IGNORECASE | re.DOTALL
)
_EMPHASIS = re.compile(
    r"<emphasis\b([^>]*)>(.*?)</emphasis>", re.IGNORECASE | re.DOTALL
)
_BREAK = re.compile(
    r'<break\b[^>]*?time\s*=\s*["\'](\d+(?:\.\d+)?)(ms|s)["\'][^>]*?/?>',
    re.IGNORECASE,
)
_ATTR = re.compile(r'(\w+)\s*=\s*["\']([^"\']+)["\']')
_TAGS = re.compile(r"<[^>]+>")

_RATE_WORDS = {
    "x-slow": 0.80,
    "slow": 0.88,
    "medium": 1.0,
    "fast": 1.12,
    "x-fast": 1.22,
}
_VOLUME_WORDS = {
    "silent": 0.0,
    "x-soft": 0.70,
    "soft": 0.82,
    "medium": 1.0,
    "loud": 1.18,
    "x-loud": 1.32,
}


@dataclass
class Prosody:
    """Уровень высказывания: переносится с оригинала на перевод."""

    rate: float = 1.0
    volume: float = 1.0
    pitch: str = "medium"
    emphasis: str = "none"
    pause_after_ms: int = 0
    interior_breaks_ms: list[int] = field(default_factory=list)

    def clamped_rate(self, lo: float = 0.75, hi: float = 1.12) -> float:
        return max(lo, min(hi, float(self.rate)))

    def clamped_volume(self, lo: float = 0.55, hi: float = 1.28) -> float:
        return max(lo, min(hi, float(self.volume)))


@dataclass
class PacedRun:
    """Кусок перевода, выровненный по речи/паузе оригинала."""

    text: str
    target_sec: float
    pause_after_sec: float = 0.0


def strip_ssml(text: str) -> str:
    if not text:
        return ""
    cleaned = _BREAK.sub(" ", text)
    cleaned = _TAGS.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _parse_rate(raw: str) -> float | None:
    value = (raw or "").strip().lower()
    if value in _RATE_WORDS:
        return _RATE_WORDS[value]
    if value.endswith("%"):
        try:
            pct = float(value[:-1])
        except ValueError:
            return None
        if value.startswith(("+", "-")):
            return max(0.7, min(1.35, 1.0 + pct / 100.0))
        return max(0.7, min(1.35, pct / 100.0))
    try:
        num = float(value)
    except ValueError:
        return None
    if 0.5 < num < 2.5:
        return num
    return None


def _parse_volume(raw: str) -> float | None:
    value = (raw or "").strip().lower()
    if value in _VOLUME_WORDS:
        return _VOLUME_WORDS[value]
    if value.endswith("db"):
        try:
            db = float(value[:-2].replace(" ", ""))
        except ValueError:
            return None
        return max(0.55, min(1.5, 10 ** (db / 20.0)))
    if value.endswith("%"):
        try:
            pct = float(value[:-1])
        except ValueError:
            return None
        if value.startswith(("+", "-")):
            return max(0.55, min(1.5, 1.0 + pct / 100.0))
        return max(0.55, min(1.5, pct / 100.0))
    return None


def _attrs(blob: str) -> dict[str, str]:
    return {m.group(1).lower(): m.group(2) for m in _ATTR.finditer(blob or "")}


def parse_ssml(raw: str) -> tuple[str, Prosody]:
    """Достаёт plain-текст и просодию. Не-SSML строка возвращается как есть."""
    text = (raw or "").strip()
    if not text:
        return "", Prosody()
    looks_ssml = "<" in text and ("speak" in text.lower() or "prosody" in text.lower())
    if not looks_ssml:
        return text, Prosody()

    prosody = Prosody()
    trailing = re.search(
        r"</prosody>\s*<break\b[^>]*time\s*=\s*[\"'](\d+(?:\.\d+)?)(ms|s)[\"']",
        text,
        re.IGNORECASE,
    )
    if trailing:
        amount, unit = trailing.group(1), trailing.group(2)
        ms = int(round(float(amount) * (1000.0 if unit.lower() == "s" else 1.0)))
        prosody.pause_after_ms = max(0, min(2000, ms))

    body = text
    match = _PROSODY.search(body)
    if match:
        attrs = _attrs(match.group(1))
        if "rate" in attrs:
            parsed = _parse_rate(attrs["rate"])
            if parsed is not None:
                prosody.rate = parsed
        if "volume" in attrs:
            parsed = _parse_volume(attrs["volume"])
            if parsed is not None:
                prosody.volume = parsed
        if "pitch" in attrs:
            prosody.pitch = attrs["pitch"].strip().lower() or "medium"
        body = match.group(2)

    emp = _EMPHASIS.search(body)
    if emp:
        attrs = _attrs(emp.group(1))
        prosody.emphasis = (attrs.get("level") or "moderate").strip().lower()
        body = _EMPHASIS.sub(r"\2", body, count=1)

    breaks = _BREAK.findall(body)
    times: list[int] = []
    for amount, unit in breaks:
        ms = int(round(float(amount) * (1000.0 if unit.lower() == "s" else 1.0)))
        times.append(max(0, min(2000, ms)))
    prosody.interior_breaks_ms = [t for t in times if t >= 180]
    return strip_ssml(body), prosody


def _fmt_rate(rate: float) -> str:
    return f"{int(round(rate * 100))}%"


def _fmt_volume(volume: float) -> str:
    if abs(volume - 1.0) < 0.03:
        return "medium"
    db = 20.0 * math.log10(max(1e-6, volume))
    sign = "+" if db >= 0 else ""
    return f"{sign}{db:.1f}dB"


def to_ssml(text: str, prosody: Prosody) -> str:
    inner = (text or "").strip() or " "
    if prosody.emphasis in {"moderate", "strong", "reduced"}:
        inner = f'<emphasis level="{prosody.emphasis}">{inner}</emphasis>'
    gaps = [int(g) for g in prosody.interior_breaks_ms if int(g) >= 180][:3]
    if gaps and "<break" not in inner.lower():
        inner = _insert_breaks(inner, gaps)
    inner = (
        f'<prosody rate="{_fmt_rate(prosody.rate)}" '
        f'volume="{_fmt_volume(prosody.volume)}" '
        f'pitch="{prosody.pitch}">{inner}</prosody>'
    )
    if prosody.pause_after_ms >= 80:
        inner += f'<break time="{int(prosody.pause_after_ms)}ms"/>'
    return f"<speak>{inner}</speak>"


def _insert_breaks(text: str, gaps_ms: list[int]) -> str:
    """Insert ALL interior breaks, spread evenly across the sentence.

    One break per real pause in the original (merged sentences keep their
    inter-word pauses as break markers instead of separate timelines).
    """
    parts = text.split()
    if not parts:
        return text
    k = min(len(gaps_ms), max(1, len(parts) // 3))
    if k <= 0:
        return text
    # Chunk boundaries at ~1/(k+1), 2/(k+1), … of the word list.
    bounds = sorted(
        {
            max(1, min(len(parts) - 1, round(len(parts) * (i + 1) / (k + 1))))
            for i in range(k)
        }
    )
    out: list[str] = []
    prev = 0
    for gi, bound in enumerate(bounds):
        out.append(" ".join(parts[prev:bound]))
        out.append(f'<break time="{int(gaps_ms[min(gi, len(gaps_ms) - 1)])}ms"/>')
        prev = bound
    out.append(" ".join(parts[prev:]))
    return " ".join(chunk for chunk in out if chunk)


def transfer_ssml(source_ssml: str, translated: str) -> str:
    """Оборачивает перевод в просодию оригинала."""
    _, prosody = parse_ssml(source_ssml)
    if not source_ssml.strip():
        return to_ssml(translated, Prosody())
    return to_ssml(translated, prosody)


def transfer_ssml_for_slot(
    source_ssml: str,
    translated: str,
    *,
    dense: bool,
) -> str:
    """Перенос просодии в слот дубляжа: плотный перевод не замедляем и не рвём break'ами."""
    if (source_ssml or "").strip():
        _, prosody = parse_ssml(source_ssml)
    else:
        prosody = Prosody()
    if dense:
        prosody.rate = min(1.06, max(1.0, float(prosody.rate)))
        prosody.interior_breaks_ms = []
        prosody.pause_after_ms = 0
    elif prosody.rate < 1.0:
        prosody.rate = max(0.92, float(prosody.rate))
    elif prosody.rate > 1.0:
        prosody.rate = min(1.08, float(prosody.rate))
    return to_ssml(translated, prosody)


def _allocate_tokens(tokens: list[str], weights: list[int]) -> list[list[str]]:
    n = len(tokens)
    k = len(weights)
    if k <= 1:
        return [tokens]
    total = sum(max(1, w) for w in weights) or k
    out: list[list[str]] = []
    idx = 0
    for i, raw_w in enumerate(weights):
        left = k - i - 1
        if i == k - 1:
            out.append(tokens[idx:])
            break
        share = max(0, int(round(n * max(1, raw_w) / total)))
        max_take = max(0, n - idx - left)
        take = min(max(1 if idx < n and max_take else 0, share), max_take)
        out.append(tokens[idx : idx + take])
        idx += take
    return out


def pace_translation(
    translated: str,
    words: list[tuple[str, float, float]] | None,
    *,
    fallback_sec: float,
    pause_sec: float = 0.18,
    max_runs: int = 6,
) -> list[PacedRun]:
    """Режет перевод в тех же местах, где в оригинале пауза между словами."""
    text = (translated or "").strip()
    if not text:
        return []
    tokens = text.split()
    usable: list[tuple[str, float, float]] = []
    for item in words or []:
        if len(item) < 3:
            continue
        usable.append((str(item[0]), float(item[1]), float(item[2])))
    if len(usable) < 2 or len(tokens) < 2:
        span = usable[-1][2] - usable[0][1] if usable else float(fallback_sec)
        return [PacedRun(text, max(0.12, span), 0.0)]

    gaps: list[tuple[int, float]] = []
    for i in range(len(usable) - 1):
        gap = usable[i + 1][1] - usable[i][2]
        if gap >= pause_sec:
            gaps.append((i, gap))
    if not gaps:
        span = max(0.12, usable[-1][2] - usable[0][1])
        return [PacedRun(text, span, 0.0)]

    keep = sorted(gaps, key=lambda item: -item[1])[: max(0, max_runs - 1)]
    split_after = {i for i, _gap in keep}

    groups: list[list[tuple[str, float, float]]] = [[usable[0]]]
    group_pauses: list[float] = []
    for i in range(len(usable) - 1):
        nxt = usable[i + 1]
        if i in split_after:
            group_pauses.append(max(0.0, nxt[1] - usable[i][2]))
            groups.append([nxt])
        else:
            groups[-1].append(nxt)

    pieces = _allocate_tokens(tokens, [max(1, len(group)) for group in groups])
    runs: list[PacedRun] = []
    for k, piece in enumerate(pieces):
        blurt = " ".join(piece).strip()
        if not blurt:
            if runs and k < len(group_pauses):
                runs[-1].pause_after_sec += group_pauses[k]
            continue
        group = groups[k] if k < len(groups) else groups[-1]
        target = max(0.12, group[-1][2] - group[0][1])
        pause = group_pauses[k] if k < len(group_pauses) else 0.0
        runs.append(PacedRun(blurt, target, pause))
    if not runs:
        return [PacedRun(text, max(0.12, float(fallback_sec)), 0.0)]
    runs[-1].pause_after_sec = 0.0
    return runs


def voxcpm_style_bits(
    prosody: Prosody | None,
    intonation: str,
    text: str = "",
    *,
    stable: bool = False,
) -> str:
    """Теги стиля VoxCPM2. stable — один тон на все реплики дубляжа."""
    bits: list[str] = []
    blob = (text or "").rstrip()
    pitch = (prosody.pitch if prosody is not None else "") or ""
    if stable:
        if blob.endswith("?") or pitch.startswith("+8"):
            bits.append("questioning")
        bits.append("natural")
        return ", ".join(bits)
    tone = (intonation or "neutral").lower()
    if blob.endswith("?") or pitch.startswith("+8"):
        bits.append("questioning")
    elif tone == "calm":
        bits.append("calm")
    elif tone == "expressive":
        bits.append("expressive")
    rate = float(prosody.rate) if prosody is not None else 1.0
    if rate <= 0.94:
        bits.append("slowly")
    elif rate >= 1.10:
        bits.append("quickly")
    seen: set[str] = set()
    ordered: list[str] = []
    for bit in bits:
        if bit not in seen:
            seen.add(bit)
            ordered.append(bit)
    return ", ".join(ordered)


def apply_interior_breaks_plain(text: str, breaks_ms: list[int]) -> str:
    """Для движков без SSML: многоточие ≈ короткая пауза в оригинале.

    Все реальные паузы (до двух) расставляются равномерно — склеенные
    предложения сохраняют ритм оригинала вместо разбиения на таймлайны.
    """
    if not text or not breaks_ms:
        return text
    if any(ch in text for ch in ".…"):
        return text
    n_breaks = min(2, len([b for b in breaks_ms if int(b) >= 180]))
    if n_breaks <= 0:
        return text
    parts = text.split()
    if len(parts) < 4:
        return text
    k = min(n_breaks, max(1, len(parts) // 3))
    bounds = sorted(
        {
            max(1, min(len(parts) - 1, round(len(parts) * (i + 1) / (k + 1))))
            for i in range(k)
        }
    )
    out: list[str] = []
    prev = 0
    for bound in bounds:
        out.append(" ".join(parts[prev:bound]))
        prev = bound
    out.append(" ".join(parts[prev:]))
    return " … ".join(chunk for chunk in out if chunk)


def intonation_from_prosody(style: str, prosody: Prosody) -> str:
    if style == "question" or style == "expressive":
        return "expressive"
    if prosody.emphasis in {"moderate", "strong"} or prosody.volume >= 1.12:
        return "expressive"
    if prosody.volume <= 0.85 or style == "calm":
        return "calm"
    return "neutral"


def enrich_segments_ssml(segments: list, audio, sample_rate: int) -> None:
    """Пишет ssml / rate / volume / pause_after на TimedSegment по исходному аудио."""
    import numpy as np

    from app.audio.prosody_transfer import pitch_hint_from_audio

    if not segments:
        return
    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    sr = int(sample_rate) or 16000
    rates: list[float] = []
    volumes: list[float] = []
    for seg in segments:
        chars = len(re.sub(r"\s+", "", seg.text or ""))
        if chars >= 8 and seg.duration > 0.2:
            rates.append(chars / seg.duration)
        if seg.rms > 1e-6:
            volumes.append(float(seg.rms))
    med_rate = float(np.median(rates)) if rates else 12.0
    med_vol = float(np.median(volumes)) if volumes else 0.04

    for i, seg in enumerate(segments):
        chars = len(re.sub(r"\s+", "", seg.text or ""))
        raw_rate = (chars / seg.duration) / med_rate if chars >= 4 and med_rate > 1e-6 else 1.0
        raw_vol = float(seg.rms) / med_vol if med_vol > 1e-8 else 1.0
        if i + 1 < len(segments):
            gap_ms = int(round(max(0.0, segments[i + 1].start - seg.end) * 1000))
        else:
            gap_ms = 0
        interior = _interior_breaks(seg, wav, sr)
        a = max(0, int(float(seg.start) * sr))
        b = min(len(wav), int(float(seg.end) * sr))
        clip = wav[a:b] if b > a else wav[:0]
        f0_pitch = pitch_hint_from_audio(clip, sr) if clip.size > sr // 4 else "medium"
        if getattr(seg, "style", "") == "question":
            pitch = "+10%" if f0_pitch == "medium" else f0_pitch
        elif getattr(seg, "style", "") == "expressive":
            pitch = f0_pitch if f0_pitch != "medium" else "+4%"
        elif getattr(seg, "style", "") == "calm":
            pitch = f0_pitch if f0_pitch != "medium" else "-4%"
        else:
            pitch = f0_pitch
        emphasis = "none"
        if raw_vol >= 1.35 or getattr(seg, "style", "") == "expressive":
            emphasis = "moderate"
        if raw_vol >= 1.7:
            emphasis = "strong"
        prosody = Prosody(
            rate=max(0.78, min(1.28, raw_rate)),
            volume=max(0.72, min(1.35, raw_vol)),
            pitch=pitch,
            emphasis=emphasis,
            pause_after_ms=min(1200, gap_ms) if gap_ms >= 90 else 0,
            interior_breaks_ms=interior[:3],
        )
        seg.rate = prosody.rate
        seg.volume = prosody.volume
        seg.pause_after = prosody.pause_after_ms / 1000.0
        seg.ssml = to_ssml(seg.text or "", prosody)


def _interior_breaks(seg, wav, sr: int) -> list[int]:
    words = getattr(seg, "words", None) or []
    gaps: list[int] = []
    if len(words) >= 3:
        for (_w0, _s0, e0), (_w1, s1, _e1) in zip(words, words[1:]):
            gap = float(s1) - float(e0)
            if 0.28 <= gap <= 1.6:
                gaps.append(int(round(gap * 1000)))
        return gaps[:3]
    a = max(0, int(seg.start * sr))
    b = min(len(wav), int(seg.end * sr))
    clip = wav[a:b]
    if clip.size < sr:
        return []
    win = max(1, int(0.02 * sr))
    hop = win
    thr = max(1e-4, float((clip * clip).mean()) * 0.12)
    silence_run = 0
    found: list[int] = []
    edge = int(0.12 * sr)
    for i in range(0, clip.size - win, hop):
        if i < edge or i > clip.size - edge:
            silence_run = 0
            continue
        rms = float((clip[i : i + win] ** 2).mean())
        if rms < thr:
            silence_run += hop
        else:
            if silence_run >= int(0.28 * sr):
                found.append(int(round(silence_run / sr * 1000)))
            silence_run = 0
        if len(found) >= 2:
            break
    return found
