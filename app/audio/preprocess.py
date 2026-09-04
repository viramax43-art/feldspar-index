"""Предобработка голосовых референсов."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from app.audio import convert_to_wav, is_supported_audio, require_ffmpeg

logger = logging.getLogger(__name__)

# Silero VAD хранит внутренний RNN-state — один экземпляр нельзя шарить между потоками
_vad_init_lock = threading.Lock()
_vad_local = threading.local()
# Silero VAD поддерживает только 8000 и 16000 Hz (или кратные 16000)
VAD_SAMPLE_RATE = 16000


def _load_vad():
    """Отдельная копия модели на каждый поток (thread-local)."""
    model = getattr(_vad_local, "model", None)
    utils = getattr(_vad_local, "utils", None)
    if model is not None and utils is not None:
        return model, utils

    with _vad_init_lock:
        model = getattr(_vad_local, "model", None)
        utils = getattr(_vad_local, "utils", None)
        if model is None or utils is None:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            model.eval()
            _vad_local.model = model
            _vad_local.utils = utils
            logger.debug("Silero VAD загружен для потока %s", threading.get_ident())
    return _vad_local.model, _vad_local.utils


def warm_up_vad() -> None:
    """Предзагрузка VAD в текущем потоке (до старта пула воркеров)."""
    _load_vad()
    _ = _get_speech_timestamps(
        np.zeros(VAD_SAMPLE_RATE, dtype=np.float32),
        VAD_SAMPLE_RATE,
        min_speech_duration_ms=100,
    )


def load_audio_mono(path: Path, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def normalize_loudness(
    audio: np.ndarray,
    target_db: float = -20.0,
    *,
    soft: bool = False,
    headroom_db: float = 6.0,
) -> np.ndarray:
    """Нормализация громкости.

    soft=False (по умолч.)  — жёсткая нормализация к target_db (для референсов).
    soft=True               — мягкая нормализация: фраза подтягивается к target_db
                              только если её RMS выходит за коридор ±headroom_db,
                              иначе естественная громкость сохраняется.
                              Это сохраняет интонационную динамику между фразами.
    """
    if audio.size == 0:
        return audio
    rms = np.sqrt(np.mean(np.square(audio)))
    if rms < 1e-8:
        return audio
    target_rms = 10 ** (target_db / 20.0)

    if soft:
        current_db = 20.0 * np.log10(max(rms, 1e-8))
        deviation = current_db - target_db
        if abs(deviation) <= headroom_db:
            # Громкость в пределах коридора — не трогаем, только пик-лимитер
            scaled = audio
        else:
            # Подтягиваем к ближайшему краю коридора, а не к центру
            edge_db = target_db + (headroom_db if deviation > 0 else -headroom_db)
            edge_rms = 10 ** (edge_db / 20.0)
            scaled = audio * (edge_rms / rms)
    else:
        scaled = audio * (target_rms / rms)

    peak = np.max(np.abs(scaled))
    if peak > 0.99:
        scaled = scaled * (0.99 / peak)
    return scaled.astype(np.float32)


def boost_quiet_stt_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_db: float = -12.0,
    max_gain_db: float = 42.0,
) -> np.ndarray:
    """Поднять редкий шёпот для Whisper по RMS активных кадров, не всего ролика.

    Глобальный RMS длинного ASMR почти равен тишине — жёсткая normalize_loudness
    либо почти не трогает речь, либо задирает шум. Берём 75-й перцентиль кадров
    выше шумового пола и тянем их к target_db.
    """
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return mono
    sr = max(1, int(sample_rate))
    hop = max(1, int(0.02 * sr))
    win = max(hop, int(0.05 * sr))
    if mono.size < win:
        return normalize_loudness(mono, target_db=target_db, soft=False)

    n_frames = max(1, (mono.size - win) // hop)
    rms = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame = mono[i * hop : i * hop + win]
        rms[i] = float(np.sqrt(np.mean(np.square(frame))) or 0.0)

    noise = float(np.percentile(rms, 25))
    thresh = max(noise * 2.2, 1e-5)
    speech = rms[rms >= thresh]
    if speech.size < 4:
        speech = rms[rms >= max(noise * 1.4, 1e-6)]
    if speech.size < 1:
        ref = float(np.sqrt(np.mean(np.square(mono))) or 1e-8)
    else:
        ref = float(np.percentile(speech, 75))

    target_rms = 10 ** (float(target_db) / 20.0)
    gain = target_rms / max(ref, 1e-8)
    gain = min(gain, 10 ** (float(max_gain_db) / 20.0))
    out = mono * float(gain)
    peak = float(np.max(np.abs(out)) or 0.0)
    if peak > 0.99:
        out = out * (0.99 / peak)
    logger.info(
        "Quiet STT speech-boost: ref_rms=%.5f gain=%.1fx (%.1f dB) → peak=%.3f",
        ref,
        gain,
        20.0 * float(np.log10(max(gain, 1e-8))),
        float(np.max(np.abs(out)) or 0.0),
    )
    return out.astype(np.float32)


def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= threshold))


def apply_bandpass(
    audio: np.ndarray,
    sample_rate: int,
    low_hz: int = 80,
    high_hz: int = 10000,
    order: int = 5,
) -> np.ndarray:
    """Bandpass-фильтр: убирает низкочастотный гул и высокочастотный шум.

    Диапазон 80–10000 Hz покрывает всю речевую полосу с запасом,
    отсекая сетевой гул 50/60 Hz, гудение и шипение выше 10 kHz.
    """
    from scipy.signal import butter, sosfilt

    nyq = sample_rate / 2.0
    low = max(low_hz / nyq, 0.001)
    high = min(high_hz / nyq, 0.999)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def apply_denoise(
    audio: np.ndarray,
    sample_rate: int,
    prop_decrease: float = 0.6,
    stationary: bool = True,
) -> np.ndarray:
    """Шумоподавление через noisereduce.

    prop_decrease=0.6 — агрессивнее, чем было (0.35), но не искажает голос.
    Двухпроходная стратегия: сначала стационарный шум, потом фильтр.
    """
    try:
        import noisereduce as nr
    except ImportError as exc:
        raise RuntimeError("noisereduce не установлен") from exc
    reduced = nr.reduce_noise(
        y=audio,
        sr=sample_rate,
        stationary=stationary,
        prop_decrease=prop_decrease,
    )
    return reduced.astype(np.float32)


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


def _vad_sample_rate(sample_rate: int) -> int:
    if sample_rate in (8000, 16000) or sample_rate % 16000 == 0:
        return sample_rate
    return VAD_SAMPLE_RATE


def _get_speech_timestamps(
    audio: np.ndarray,
    sample_rate: int,
    **kwargs: int,
) -> list[dict[str, int]]:
    """VAD с автоматическим ресэмплингом под Silero (8/16 kHz)."""
    model, utils = _load_vad()
    get_speech_timestamps = utils[0]
    vad_sr = _vad_sample_rate(sample_rate)
    vad_audio = (
        _resample_audio(audio, sample_rate, vad_sr)
        if vad_sr != sample_rate
        else audio
    )
    speech_chunks = get_speech_timestamps(
        torch.from_numpy(vad_audio),
        model,
        sampling_rate=vad_sr,
        **kwargs,
    )
    if vad_sr != sample_rate and speech_chunks:
        scale = sample_rate / vad_sr
        speech_chunks = [
            {
                "start": min(int(c["start"] * scale), len(audio)),
                "end": min(int(c["end"] * scale), len(audio)),
            }
            for c in speech_chunks
        ]
    return speech_chunks


def remove_long_pauses(
    audio: np.ndarray,
    sample_rate: int,
    max_silence_sec: float = 0.6,
) -> np.ndarray:
    speech_chunks = _get_speech_timestamps(
        audio,
        sample_rate,
        min_speech_duration_ms=250,
        min_silence_duration_ms=int(max_silence_sec * 1000),
    )
    if not speech_chunks:
        return audio
    pieces: list[np.ndarray] = []
    gap = int(0.08 * sample_rate)
    for idx, chunk in enumerate(speech_chunks):
        start = chunk["start"]
        end = chunk["end"]
        if end <= start:
            continue
        pieces.append(audio[start:end])
        if idx < len(speech_chunks) - 1:
            pieces.append(np.zeros(gap, dtype=np.float32))
    return np.concatenate(pieces) if pieces else audio


def speech_ratio(audio: np.ndarray, sample_rate: int) -> float:
    if audio.size == 0:
        return 0.0
    speech_chunks = _get_speech_timestamps(
        audio,
        sample_rate,
        min_speech_duration_ms=200,
    )
    speech_samples = sum(c["end"] - c["start"] for c in speech_chunks)
    return float(speech_samples / max(len(audio), 1))


def preprocess_telegram_voice(
    source_path: Path,
    output_path: Path,
    sample_rate: int = 22050,
    enable_denoise: bool = True,
    enable_bandpass: bool = True,
) -> dict:
    """Конвертирует Telegram OGG/OPUS в WAV с полной очисткой для TTS-клонирования.

    Пайплайн:
    1. FFmpeg → mono WAV
    2. Bandpass 80–10000 Hz (убирает гул и шипение)
    3. Шумоподавление noisereduce (стационарный шум)
    4. VAD — удаление длинных пауз
    5. Нормализация громкости
    """
    require_ffmpeg()
    if not is_supported_audio(source_path):
        raise ValueError(f"Неподдерживаемый формат: {source_path.suffix}")

    temp_wav = output_path.with_suffix(".raw.wav")
    convert_to_wav(source_path, temp_wav, sample_rate=sample_rate, mono=True)
    audio = load_audio_mono(temp_wav, sample_rate)

    original_duration = len(audio) / sample_rate

    # 1. Bandpass — убираем частоты вне речевого диапазона
    if enable_bandpass:
        audio = apply_bandpass(audio, sample_rate)

    # 2. Шумоподавление — до VAD, чтобы шум не мешал определению пауз
    denoise_applied = False
    rms_pre = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    if enable_denoise and rms_pre > 1e-4:
        denoised = apply_denoise(audio, sample_rate, prop_decrease=0.6)
        rms_post = float(np.sqrt(np.mean(np.square(denoised)))) if denoised.size else 0.0
        if rms_post >= rms_pre * 0.4:
            audio = denoised
            denoise_applied = True

    # 3. VAD — убираем длинные паузы
    audio = remove_long_pauses(audio, sample_rate)

    # 4. Нормализация громкости
    audio = normalize_loudness(audio)

    clipping_ratio = detect_clipping(audio)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    ratio = speech_ratio(audio, sample_rate)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate, subtype="PCM_16")
    if temp_wav.exists():
        temp_wav.unlink(missing_ok=True)

    duration = len(audio) / sample_rate
    return {
        "duration_sec": round(duration, 2),
        "original_duration_sec": round(original_duration, 2),
        "speech_ratio": round(ratio, 3),
        "clipping_ratio": round(clipping_ratio, 4),
        "rms": round(rms, 5),
        "denoise_applied": denoise_applied,
    }


def merge_references(
    reference_paths: list[Path],
    output_path: Path,
    sample_rate: int = 22050,
    crossfade_ms: int = 80,
) -> Path:
    """Объединяет несколько референсов с коротким кроссфейдом."""
    if not reference_paths:
        raise ValueError("Нет референсов для объединения")
    if len(reference_paths) == 1:
        audio = load_audio_mono(reference_paths[0], sample_rate)
        sf.write(output_path, audio, sample_rate, subtype="PCM_16")
        return output_path

    crossfade = int(sample_rate * crossfade_ms / 1000)
    merged: np.ndarray | None = None
    for path in reference_paths:
        chunk = load_audio_mono(path, sample_rate)
        if merged is None:
            merged = chunk
            continue
        if crossfade > 0 and len(merged) >= crossfade and len(chunk) >= crossfade:
            fade_out = np.linspace(1.0, 0.0, crossfade, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, crossfade, dtype=np.float32)
            overlap = merged[-crossfade:] * fade_out + chunk[:crossfade] * fade_in
            merged = np.concatenate([merged[:-crossfade], overlap, chunk[crossfade:]])
        else:
            gap = np.zeros(int(0.1 * sample_rate), dtype=np.float32)
            merged = np.concatenate([merged, gap, chunk])

    assert merged is not None
    merged = normalize_loudness(merged)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, merged, sample_rate, subtype="PCM_16")
    return output_path
