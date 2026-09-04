import numpy as np

from app.audio.clone_denoise import (
    denoise_for_voice_clone,
    estimate_snr_db,
    extract_noise_sample,
)


def _speech_with_pauses(
    sr: int = 22050,
    *,
    noise_std: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    quiet = np.zeros(int(0.2 * sr), dtype=np.float32)
    for freq in (180.0, 240.0, 300.0):
        t = np.linspace(0, 0.35, int(sr * 0.35), endpoint=False, dtype=np.float32)
        # Amplitude envelope so frames aren't flat.
        env = np.linspace(0.2, 1.0, t.size, dtype=np.float32)
        parts.append((0.4 * env * np.sin(2 * np.pi * freq * t)).astype(np.float32))
        parts.append(quiet.copy())
    wav = np.concatenate(parts)
    if noise_std > 0:
        wav = (wav + rng.normal(0, noise_std, wav.size).astype(np.float32)).astype(
            np.float32
        )
    return wav


def test_estimate_snr_clean_vs_noisy():
    sr = 22050
    clean = _speech_with_pauses(sr, noise_std=0.0)
    dirty = _speech_with_pauses(sr, noise_std=0.1, seed=1)
    assert estimate_snr_db(clean, sr) > estimate_snr_db(dirty, sr) + 3.0


def test_denoise_skips_clean_clip():
    sr = 22050
    clean = _speech_with_pauses(sr, noise_std=0.0)
    out, info = denoise_for_voice_clone(clean, sr, snr_threshold_db=12.0)
    assert info["applied"] is False
    assert info["reason"] == "clean_enough"
    assert out.shape == clean.shape


def test_denoise_applies_on_noisy_clip():
    sr = 22050
    dirty = _speech_with_pauses(sr, noise_std=0.09, seed=2)
    out, info = denoise_for_voice_clone(
        dirty, sr, snr_threshold_db=40.0, prop_decrease=0.7
    )
    assert info["applied"] is True
    assert out.shape == dirty.shape
    assert float(info.get("prop") or 0.0) > 0.0
    # Output stays finite and not crushed to silence.
    assert float(np.max(np.abs(out))) > 0.05
    assert float(np.sqrt(np.mean(np.square(out)))) > 1e-3


def test_extract_noise_sample_from_padded_clip():
    sr = 16000
    quiet = np.zeros(int(0.3 * sr), dtype=np.float32)
    quiet += (np.random.default_rng(3).normal(0, 0.01, quiet.size)).astype(np.float32)
    speech = _speech_with_pauses(sr, noise_std=0.0)
    clip = np.concatenate([quiet, speech, quiet])
    noise = extract_noise_sample(clip, sr)
    assert noise is not None
    assert noise.size >= int(0.12 * sr)
