"""Тесты обработки аудио."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.audio import is_supported_audio, safe_user_path
from app.audio.quality import evaluate_reference


def test_supported_audio_formats():
    assert is_supported_audio(Path("test.wav"))
    assert is_supported_audio(Path("test.ogg"))
    assert not is_supported_audio(Path("test.txt"))


def test_safe_user_path(tmp_path):
    base = tmp_path / "users"
    base.mkdir()
    path = safe_user_path(base, 12345, "references", "ref_001.wav")
    assert path.parent.name == "references"
    assert "12345" in str(path)


def test_safe_user_path_rejects_traversal(tmp_path):
    base = tmp_path / "users"
    base.mkdir()
    with pytest.raises(ValueError):
        safe_user_path(base, 12345, "..", "hack.wav")


def test_duration_check(tmp_path):
    wav = tmp_path / "short.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    report = evaluate_reference(
        {"duration_sec": 0.5, "speech_ratio": 0.8, "clipping_ratio": 0, "rms": 0.05},
        min_duration=3.0,
    )
    assert not report.accepted
    assert report.score < 70


def test_quality_accepted():
    report = evaluate_reference(
        {
            "duration_sec": 8.0,
            "speech_ratio": 0.85,
            "clipping_ratio": 0.0,
            "rms": 0.04,
        }
    )
    assert report.accepted
    assert report.score >= 55


def test_vad_accepts_22050_sample_rate():
    """Silero VAD не принимает 22050 напрямую — должен быть внутренний ресэмпл."""
    from app.audio.preprocess import _vad_sample_rate, speech_ratio

    assert _vad_sample_rate(22050) == 16000
    # Тишина — ratio ~0, главное что без ValueError
    audio = np.zeros(22050 * 2, dtype=np.float32)
    ratio = speech_ratio(audio, 22050)
    assert 0.0 <= ratio <= 1.0

