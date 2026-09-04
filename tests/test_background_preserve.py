"""Tests for rewritten original-background language replacement."""

import numpy as np
import pytest

from app.audio.background_preserve import (
    original_speech_windows,
    render_original_background,
)
from app.services.transcription import TimedSegment


def test_outside_speech_is_bit_identical_original():
    sr = 16000
    n = sr * 4
    original = np.linspace(0.05, 0.8, n, dtype=np.float32)
    # Laugh / heel sequence after dialogue.
    original[int(2.5 * sr) : int(3.2 * sr)] = 0.62
    voice = np.zeros(n, dtype=np.float32)
    voice[sr : 2 * sr] = 0.3
    # Accompaniment under speech is quieter music only.
    accomp = np.ones(n, dtype=np.float32) * 0.2
    windows = [(1.0, 2.0)]
    out = render_original_background(
        original,
        voice,
        sr,
        windows,
        accompaniment=accomp,
        background_speed=1.0,
    )
    # Before / after speech: exact original (no voice there).
    assert np.allclose(out[:sr], original[:sr], atol=1e-5)
    assert np.allclose(
        out[int(2.5 * sr) : int(3.2 * sr)],
        original[int(2.5 * sr) : int(3.2 * sr)],
        atol=1e-5,
    )


def test_under_speech_uses_accompaniment_not_original_dialogue():
    sr = 8000
    n = sr * 2
    original = np.ones(n, dtype=np.float32) * 0.5  # dialogue+music
    accomp = np.ones(n, dtype=np.float32) * 0.15  # music only
    voice = np.zeros(n, dtype=np.float32)
    voice[sr // 2 : sr] = 0.4
    windows = [(0.5, 1.0)]
    out = render_original_background(
        original,
        voice,
        sr,
        windows,
        accompaniment=accomp,
        background_speed=1.0,
    )
    mid = int(0.75 * sr)
    # Under speech: accomp level-matched toward the original bed (≤1.8× gain
    # clamp) + voice — still clearly NOT the original dialogue 0.5+0.4=0.9.
    assert 0.55 <= float(out[mid]) <= 0.75
    # Outside: pure original.
    assert float(out[100]) == pytest.approx(0.5, abs=0.02)


def test_sfx_after_speech_survives_even_if_tts_is_long():
    """Bed holes follow ASR windows, not long TTS overflows."""
    sr = 16000
    n = sr * 5
    original = np.zeros(n, dtype=np.float32)
    original[:] = 0.1
    # Click/laugh at 2.2s — after a 1.0–2.0s ASR cue.
    original[int(2.2 * sr) : int(2.25 * sr)] = 0.9
    # TTS spills past 2.0s into the laugh region.
    voice = np.zeros(n, dtype=np.float32)
    voice[sr : int(2.4 * sr)] = 0.25
    windows = original_speech_windows(
        [TimedSegment(1.0, 2.0, "talk")], pad_sec=0.0
    )
    out = render_original_background(
        original,
        voice,
        sr,
        windows,
        accompaniment=np.zeros(n, dtype=np.float32),
        background_speed=1.0,
    )
    click = out[int(2.2 * sr) : int(2.25 * sr)]
    # Original click remains (plus any TTS that overlaps) — not wiped.
    assert float(np.max(np.abs(click))) >= 0.9


def test_background_speed_half_slows_onset():
    """speed=0.5 must delay a transient that used to sit early in the buffer."""
    sr = 8000
    n = sr * 2
    original = np.zeros(n, dtype=np.float32)
    # Click at 0.5s.
    original[int(0.5 * sr) : int(0.5 * sr) + 4] = 1.0
    voice = np.zeros(n, dtype=np.float32)
    out_fast = render_original_background(
        original, voice, sr, [], background_speed=1.0
    )
    out_slow = render_original_background(
        original, voice, sr, [], background_speed=0.5
    )
    peak_fast = int(np.argmax(np.abs(out_fast)))
    peak_slow = int(np.argmax(np.abs(out_slow)))
    assert peak_slow > peak_fast
    assert peak_slow == pytest.approx(peak_fast * 2, rel=0.15)


def test_countdown_keeps_laughter_after_digits():
    """Countdown must NOT wipe the full bed — only mute under digit windows."""
    sr = 8000
    n = sr * 3
    original = np.zeros(n, dtype=np.float32)
    # Digit speech 0.3–1.0s, laughter at 1.5–2.0s.
    original[int(0.3 * sr) : int(1.0 * sr)] = 0.2
    original[int(1.5 * sr) : int(2.0 * sr)] = 0.55
    voice = np.zeros(n, dtype=np.float32)
    voice[int(0.3 * sr) : int(1.0 * sr)] = 0.35
    out = render_original_background(
        original,
        voice,
        sr,
        [(0.3, 1.0)],
        countdown_mute=True,
        background_speed=1.0,
    )
    # Under digits: mostly new voice (original muted).
    assert float(np.mean(np.abs(out[int(0.5 * sr) : int(0.8 * sr)]))) == pytest.approx(
        0.35, abs=0.08
    )
    # After digits: original laughter must survive.
    assert float(np.mean(np.abs(out[int(1.6 * sr) : int(1.9 * sr)]))) == pytest.approx(
        0.55, abs=0.05
    )


def test_restores_laugh_from_vocal_stem_under_speech():
    sr = 1000
    n = 2000
    original = np.zeros(n, dtype=np.float32)
    accomp = np.zeros(n, dtype=np.float32)
    vocals = np.zeros(n, dtype=np.float32)
    # Sustained dialogue in window.
    original[200:800] = 0.05
    vocals[200:800] = 0.05
    # Laugh burst also in vocal stem inside/near window.
    vocals[900:1100] = 0.4
    original[900:1100] = 0.4
    voice = np.zeros(n, dtype=np.float32)
    out = render_original_background(
        original,
        voice,
        sr,
        [(0.2, 0.85)],
        accompaniment=accomp,
        vocals=vocals,
        background_speed=1.0,
    )
    # Outside the tight speech window the original laugh region is kept.
    assert float(np.mean(np.abs(out[950:1050]))) > 0.2


def test_countdown_is_voice_only():
    # Legacy name: full-track wipe is gone; digits are muted under windows only.
    sr = 8000
    original = np.ones(sr, dtype=np.float32) * 0.4
    voice = np.zeros(sr, dtype=np.float32)
    voice[100:500] = 0.3
    out = render_original_background(
        original, voice, sr, [(0.0, 0.05)], countdown_mute=True, background_speed=1.0
    )
    # Outside the tiny window original remains.
    assert float(out[int(0.5 * sr)]) == pytest.approx(0.4, abs=0.05)


def test_fallback_without_stems_keeps_original_outside():
    sr = 8000
    original = np.ones(sr * 2, dtype=np.float32) * 0.45
    voice = np.zeros(sr * 2, dtype=np.float32)
    out = render_original_background(
        original,
        voice,
        sr,
        [(0.2, 0.8)],
        accompaniment=None,
        vocals=None,
        background_speed=1.0,
    )
    assert float(out[int(1.2 * sr)]) == pytest.approx(0.45, abs=0.02)


def test_speech_duck_lowers_bed_only_under_tts():
    sr = 8000
    n = sr * 3
    original = np.ones(n, dtype=np.float32) * 0.5
    voice = np.zeros(n, dtype=np.float32)
    # ASR rewrite window (unused for duck assert — no accomp).
    asr = [(1.0, 1.5)]
    # TTS placement longer than ASR.
    duck = [(1.0, 2.0)]
    out = render_original_background(
        original,
        voice,
        sr,
        asr,
        accompaniment=None,
        duck_windows=duck,
        speech_duck=0.70,
        bg_gain=1.0,
    )
    # Outside TTS: full bed.
    assert float(out[int(0.5 * sr)]) == pytest.approx(0.5, abs=0.03)
    assert float(out[int(2.5 * sr)]) == pytest.approx(0.5, abs=0.03)
    # Inside TTS duck window: ~0.5 * 0.70.
    assert float(out[int(1.7 * sr)]) == pytest.approx(0.35, abs=0.05)


def test_gate_edges_are_smooth_no_ripple_step():
    """Bed↔under transitions must not contain sample-level steps (ripple)."""
    sr = 16000
    n = sr * 4
    rng = np.random.default_rng(7)
    original = (0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / sr)).astype(np.float32)
    original += 0.02 * rng.standard_normal(n).astype(np.float32)
    accomp = (0.12 * np.sin(2 * np.pi * 110 * np.arange(n) / sr)).astype(np.float32)
    voice = np.zeros(n, dtype=np.float32)
    out = render_original_background(
        original,
        voice,
        sr,
        [(1.0, 2.0), (2.6, 3.4)],
        accompaniment=accomp,
        background_speed=1.0,
    )
    # No click: sample-to-sample delta at edges ≈ the signal's own slew rate
    # (a hard gate step would produce a delta ≫ anything in the source).
    ref_delta = float(np.max(np.abs(np.diff(original))))
    for edge in (1.0, 2.0, 2.6, 3.4):
        i = int(edge * sr)
        delta = np.abs(np.diff(out[i - 400 : i + 400]))
        assert float(np.max(delta)) < ref_delta * 1.5


def test_under_bed_level_matched_to_original():
    """A moderately quieter accompaniment is lifted toward the original bed."""
    from app.audio.background_preserve import _match_under_level

    sr = 8000
    n = sr * 2
    original = np.ones(n, dtype=np.float32) * 0.4
    under = np.ones(n, dtype=np.float32) * 0.3
    matched = _match_under_level(original, under, [(0.5, 1.5)], sr)
    # 0.4/0.3 ≈ 1.33 — moderate demucs-level mismatch is compensated
    assert float(matched[sr]) == pytest.approx(0.4, abs=0.03)
    # Extreme mismatch = deliberately muted bed (mask mode) → NOT re-boosted
    muted = _match_under_level(original, np.ones(n, dtype=np.float32) * 0.05, [(0.5, 1.5)], sr)
    assert float(muted[sr]) == pytest.approx(0.05, abs=0.01)
    # Already-matched level → untouched
    same = _match_under_level(original, np.ones(n, dtype=np.float32) * 0.4, [(0.5, 1.5)], sr)
    assert float(same[sr]) == pytest.approx(0.4, abs=0.02)
    # No windows / silence → untouched
    quiet = np.ones(n, dtype=np.float32) * 0.2
    assert float(_match_under_level(original, quiet, [], sr)[sr]) == pytest.approx(0.2)
