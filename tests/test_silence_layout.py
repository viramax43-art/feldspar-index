"""Silence-gap detection and no-tempo dub layout."""

import numpy as np
import pytest

from app.audio.silence_gaps import (
    detect_silence_intervals,
    silence_gaps_for_dub,
    subtract_intervals,
)
from app.services.transcription import TimedSegment
from app.services.video_dub import layout_silence_borrow_placements


def test_detect_silence_intervals_finds_gap():
    sr = 8000
    audio = np.ones(sr * 3, dtype=np.float32) * 0.2
    audio[sr : 2 * sr] = 0.0
    gaps = detect_silence_intervals(audio, sr, min_silence_sec=0.2, threshold_db=-40)
    assert gaps
    assert any(g0 < 1.2 and g1 > 1.8 for g0, g1 in gaps)


def test_subtract_speech_from_silence():
    base = [(0.0, 1.0), (2.0, 4.0)]
    cut = [(0.4, 0.6), (2.5, 3.0)]
    out = subtract_intervals(base, cut)
    assert any(abs(a - 0.0) < 1e-6 and abs(b - 0.4) < 1e-6 for a, b in out)
    assert any(a >= 3.0 - 1e-3 for a, _b in out)


def test_silence_gaps_for_dub_removes_speech():
    sr = 8000
    audio = np.zeros(sr * 4, dtype=np.float32)
    audio[0:sr] = 0.3
    audio[2 * sr : 3 * sr] = 0.3
    gaps = silence_gaps_for_dub(
        audio, sr, speech_windows=[(0.0, 1.0), (2.0, 3.0)], min_silence_sec=0.15
    )
    assert any(g0 <= 1.05 and g1 >= 1.9 for g0, g1 in gaps)


def test_layout_long_tts_borrows_forward_keeps_gap():
    segments = [
        TimedSegment(1.0, 1.5, "one", words=[("one", 1.0, 1.5)]),
        TimedSegment(3.0, 3.4, "two", words=[("two", 3.0, 3.4)]),
    ]
    durations = [1.6, 0.4]
    silence = [(1.5, 3.0)]
    places = layout_silence_borrow_placements(
        segments,
        durations,
        media_duration=5.0,
        gap_sec=0.12,
        silence_gaps=silence,
    )
    assert places[0][0] == pytest.approx(1.0, abs=0.02)
    assert places[0][1] == pytest.approx(2.6, abs=0.02)
    assert places[1][0] >= places[0][1] + 0.12 - 1e-3
    assert places[1][0] == pytest.approx(3.0, abs=0.05) or places[1][0] >= 2.72


def test_layout_short_tts_leaves_pause():
    segments = [
        TimedSegment(1.0, 2.0, "hi", words=[("hi", 1.0, 2.0)]),
        TimedSegment(3.0, 3.5, "bye", words=[("bye", 3.0, 3.5)]),
    ]
    durations = [0.4, 0.3]
    places = layout_silence_borrow_placements(
        segments, durations, 5.0, gap_sec=0.12, silence_gaps=[(2.0, 3.0)]
    )
    assert places[0][1] - places[0][0] == pytest.approx(0.4, abs=0.01)
    assert places[1][0] == pytest.approx(3.0, abs=0.02)
