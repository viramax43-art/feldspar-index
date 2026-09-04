"""Smoke tests for F0/energy prosody transfer."""

from __future__ import annotations

import numpy as np

from app.audio.prosody_transfer import pitch_hint_from_audio, transfer_prosody_from_source


def test_transfer_prosody_preserves_length():
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    tts = 0.2 * np.sin(2 * np.pi * 180 * t).astype(np.float32)
    src = 0.25 * np.sin(2 * np.pi * 120 * t).astype(np.float32)
    out = transfer_prosody_from_source(tts, src, sr)
    assert out.shape == tts.shape
    assert float(np.max(np.abs(out))) <= 1.0


def test_pitch_hint_returns_string():
    sr = 16000
    t = np.linspace(0, 0.8, int(0.8 * sr), dtype=np.float32)
    # rising chirp-ish
    freq = 120 + 80 * t
    phase = np.cumsum(2 * np.pi * freq / sr)
    wav = (0.2 * np.sin(phase)).astype(np.float32)
    hint = pitch_hint_from_audio(wav, sr)
    assert isinstance(hint, str)
    assert hint
