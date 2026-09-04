"""Tests for envelope word ends and dub glossary."""

import numpy as np

from app.audio.envelope_align import refine_segments_by_envelope
from app.services.transcription import TimedSegment
from app.text.dub_glossary import (
    inject_step_pause_ssml,
    sanitize_dub_translation,
)


def test_sanitize_pickles_not_tickling():
    src = "pickles pickles pickles"
    bad = "щекочущие щекочи щекочущие"
    out = sanitize_dub_translation(src, bad)
    assert "щекоч" not in out.lower()
    assert "огурц" in out.lower()


def test_sanitize_handful_not_shoes():
    out = sanitize_dub_translation("a handful of onions", "из каждой туфли лук")
    assert "туфл" not in out.lower()
    assert "горсть" in out.lower()


def test_sanitize_kovyazh():
    out = sanitize_dub_translation("beef patty", "ковяж котлета")
    assert "говяжий" in out.lower()
    assert "ковяж" not in out.lower()


def test_inject_step_break():
    ssml = inject_step_pause_ssml("Шаг один: добавь масло")
    assert "600ms" in ssml
    assert "Шаг один" in ssml


def test_envelope_extends_whisper_tail():
    sr = 16000
    # 0.5с тишины + слово 0.3с + длинный тихий хвост 0.35с
    t = np.arange(int(1.4 * sr), dtype=np.float32) / sr
    wav = np.zeros_like(t)
    # основная речь 0.50–0.80
    a0, a1 = int(0.50 * sr), int(0.80 * sr)
    wav[a0:a1] = 0.4 * np.sin(2 * np.pi * 220 * t[a0:a1])
    # затухающий хвост шёпота до ~1.15с (выше –40 dB от пика)
    peak = 0.4
    thr = peak * (10 ** (-40 / 20))
    for i in range(a1, int(1.15 * sr)):
        env = peak * np.exp(-(i - a1) / (0.12 * sr))
        if env < thr:
            break
        wav[i] = env * np.sin(2 * np.pi * 180 * (i / sr))

    segs = [
        TimedSegment(
            0.50,
            0.80,
            "five",
            words=[("five", 0.50, 0.80)],
            rms=0.03,
        )
    ]
    out = refine_segments_by_envelope(segs, wav, sr, rel_db=-40.0, min_extend_sec=0.12)
    assert out[0].end >= 0.92  # хвост длиннее ASR 0.80
    assert out[0].duration > 0.40
