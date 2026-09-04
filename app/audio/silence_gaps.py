"""Silence / pause detection for dub layout (borrow gaps instead of tempo)."""

from __future__ import annotations

import numpy as np


def detect_silence_intervals(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_silence_sec: float = 0.12,
    threshold_db: float = -42.0,
    frame_ms: float = 20.0,
    pad_sec: float = 0.02,
) -> list[tuple[float, float]]:
    """Return contiguous low-energy intervals ``(start_sec, end_sec)``."""
    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    sr = max(1, int(sample_rate))
    if wav.size < sr // 10:
        return []
    frame = max(1, int(float(frame_ms) * sr / 1000.0))
    hop = max(1, frame // 2)
    thr = 10.0 ** (float(threshold_db) / 20.0)
    silent_flags: list[bool] = []
    times: list[float] = []
    for i in range(0, max(1, wav.size - frame + 1), hop):
        chunk = wav[i : i + frame]
        rms = float(np.sqrt(np.mean(np.square(chunk))) or 0.0)
        silent_flags.append(rms < thr)
        times.append(i / float(sr))
    if not silent_flags:
        return []
    # Extend last flag to media end.
    times.append(wav.size / float(sr))
    silent_flags.append(silent_flags[-1])

    min_len = max(0.05, float(min_silence_sec))
    pad = max(0.0, float(pad_sec))
    out: list[tuple[float, float]] = []
    run_start: float | None = None
    for idx, flag in enumerate(silent_flags):
        t0 = times[idx]
        t1 = times[min(idx + 1, len(times) - 1)]
        if flag:
            if run_start is None:
                run_start = t0
        elif run_start is not None:
            end = t0
            if end - run_start >= min_len:
                out.append((max(0.0, run_start + pad), max(0.0, end - pad)))
            run_start = None
    if run_start is not None:
        end = times[-1]
        if end - run_start >= min_len:
            out.append((max(0.0, run_start + pad), max(0.0, end - pad)))
    return [(a, b) for a, b in out if b > a + 1e-3]


def subtract_intervals(
    base: list[tuple[float, float]],
    cut: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Subtract ``cut`` windows from ``base`` silence intervals."""
    if not base:
        return []
    if not cut:
        return list(base)
    result: list[tuple[float, float]] = []
    for s0, s1 in base:
        pieces = [(s0, s1)]
        for c0, c1 in cut:
            nxt: list[tuple[float, float]] = []
            for a, b in pieces:
                if c1 <= a or c0 >= b:
                    nxt.append((a, b))
                    continue
                if c0 > a:
                    nxt.append((a, min(b, c0)))
                if c1 < b:
                    nxt.append((max(a, c1), b))
            pieces = [(a, b) for a, b in nxt if b > a + 1e-4]
        result.extend(pieces)
    return result


def silence_gaps_for_dub(
    audio: np.ndarray,
    sample_rate: int,
    speech_windows: list[tuple[float, float]] | None = None,
    *,
    min_silence_sec: float = 0.12,
) -> list[tuple[float, float]]:
    """Silence intervals with STT/speech windows removed (true inter-phrase gaps)."""
    raw = detect_silence_intervals(
        audio,
        sample_rate,
        min_silence_sec=min_silence_sec,
    )
    return subtract_intervals(raw, list(speech_windows or []))
