import numpy as np

from app.audio.preprocess import boost_quiet_stt_audio


def test_boost_quiet_stt_raises_sparse_whisper():
    sr = 16000
    # 5s near-silence with two soft speech bursts
    audio = np.random.randn(sr * 5).astype(np.float32) * 0.0003
    t = np.linspace(0, 0.4, int(0.4 * sr), endpoint=False)
    burst = (0.004 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    audio[sr : sr + burst.size] += burst
    audio[3 * sr : 3 * sr + burst.size] += burst

    out = boost_quiet_stt_audio(audio, sr, target_db=-12.0, max_gain_db=42.0)
    speech = out[sr : sr + burst.size]
    speech_rms = float(np.sqrt(np.mean(np.square(speech))))
    assert speech_rms > 0.05
    assert float(np.max(np.abs(out))) <= 1.0


def test_is_probable_non_speech():
    from app.services.transcription import TimedSegment, is_probable_non_speech

    # Whisper hallucination on a moan: high no_speech + low logprob → drop
    moan = TimedSegment(1.0, 2.0, "Oh! Oh!", no_speech_prob=0.83, avg_logprob=-1.4)
    assert is_probable_non_speech(moan)
    # Confident speech stays
    speech = TimedSegment(
        1.0, 2.0, "Привет всем", no_speech_prob=0.05, avg_logprob=-0.2
    )
    assert not is_probable_non_speech(speech)
    # Unknown confidence (aligner didn't report) → never dropped
    unknown = TimedSegment(1.0, 2.0, "что угодно")
    assert not is_probable_non_speech(unknown)
    # High no_speech but decent logprob (quiet speech) → keep
    quiet = TimedSegment(1.0, 2.0, "тихая фраза", no_speech_prob=0.7, avg_logprob=-0.4)
    assert not is_probable_non_speech(quiet)


def test_stt_worker_keeps_separate_models_per_size(monkeypatch):
    import sys
    from unittest.mock import MagicMock

    from app.services import stt_worker

    created: list[str] = []

    class FakeWhisper:
        def __init__(self, size, **_kwargs):
            created.append(str(size))
            self.size = size

    fake_mod = MagicMock()
    fake_mod.WhisperModel = FakeWhisper
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_mod)
    monkeypatch.setattr(stt_worker, "_MODELS", {})

    small_a = stt_worker._load_model(
        {"model_size": "small", "device": "cpu", "compute_type": "int8"}
    )
    small_b = stt_worker._load_model(
        {"model_size": "small", "device": "cpu", "compute_type": "int8"}
    )
    medium = stt_worker._load_model(
        {"model_size": "medium", "device": "cpu", "compute_type": "int8"}
    )
    assert small_a is small_b
    assert medium is not small_a
    assert created == ["small", "medium"]
