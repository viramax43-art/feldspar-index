"""Утилиты для работы с аудио."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FFMPEG_DIR_CANDIDATES = (
    PROJECT_ROOT / "bin",
    PROJECT_ROOT / "bin" / "ffmpeg",
    PROJECT_ROOT / "bin" / "ffmpeg" / "bin",
)


def _exe(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def _bundled(name: str) -> str | None:
    exe = _exe(name)
    env_key = "FFMPEG_PATH" if name == "ffmpeg" else "FFPROBE_PATH"
    env_val = (os.environ.get(env_key) or "").strip()
    if env_val:
        path = Path(env_val)
        if path.is_file():
            return str(path)
        nested = path / exe
        if nested.is_file():
            return str(nested)
    for folder in _FFMPEG_DIR_CANDIDATES:
        candidate = folder / exe
        if candidate.is_file():
            return str(candidate)
    return None


def find_ffmpeg() -> str | None:
    bundled = _bundled("ffmpeg")
    if bundled:
        return bundled
    return shutil.which("ffmpeg")


def find_ffprobe() -> str | None:
    bundled = _bundled("ffprobe")
    if bundled:
        return bundled
    which = shutil.which("ffprobe")
    if which:
        return which
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        sibling = Path(ffmpeg).with_name(_exe("ffprobe"))
        if sibling.is_file():
            return str(sibling)
    return None


def require_ffmpeg() -> str:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg не найден. Установите FFmpeg и добавьте его в PATH."
        )
    return ffmpeg


def safe_user_path(base_dir: Path, user_id: int, *parts: str) -> Path:
    """Безопасный путь в директории пользователя без пользовательского ввода."""
    if user_id <= 0:
        raise ValueError("Некорректный user_id")
    user_dir = (base_dir / str(user_id)).resolve()
    base_resolved = base_dir.resolve()
    if not str(user_dir).startswith(str(base_resolved)):
        raise ValueError("Попытка выхода за пределы базовой директории")
    target = user_dir
    for part in parts:
        safe_part = re.sub(r"[^\w\-.]", "_", part)
        target = target / safe_part
    target_resolved = target.resolve()
    if not str(target_resolved).startswith(str(user_dir)):
        raise ValueError("Некорректный путь к файлу")
    return target_resolved


def convert_to_wav(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 22050,
    mono: bool = True,
) -> None:
    ffmpeg = require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channels = "1" if mono else "2"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-ar",
        str(sample_rate),
        "-ac",
        channels,
        "-sample_fmt",
        "s16",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg ошибка конвертации: {result.stderr.strip()}")


def apply_quiet_stt_ffmpeg_boost(wav_path: Path) -> None:
    """dynaudnorm + mild lift for whisper-quiet / ASMR sources (in-place)."""
    ffmpeg = require_ffmpeg()
    tmp = wav_path.with_name(wav_path.stem + ".quietboost.wav")
    # highpass drops rumble; dynaudnorm lifts sparse whispers; volume adds headroom
    af = "highpass=f=80,dynaudnorm=f=75:g=31:p=0.9:s=30,volume=3dB"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(wav_path),
        "-af",
        af,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg quiet boost: {result.stderr.strip()}")
    tmp.replace(wav_path)


def convert_wav_to_ogg_opus(input_path: Path, output_path: Path) -> None:
    ffmpeg = require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-c:a",
        "libopus",
        "-b:a",
        "128k",
        "-vbr",
        "on",
        "-application",
        "audio",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg ошибка OGG: {result.stderr.strip()}")


def convert_pcm16_to_ogg_opus(
    pcm_s16le: bytes,
    sample_rate: int,
    output_path: Path,
) -> None:
    """Кодирование PCM16 LE из памяти в OGG/OPUS без промежуточного WAV.

    128k + application=audio — высокое качество для голосового клона.
    voip @ 64k давал заметные артефакты и потерю обертонов.
    """
    ffmpeg = require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-c:a",
        "libopus",
        "-b:a",
        "128k",
        "-vbr",
        "on",
        "-application",
        "audio",
        str(output_path),
    ]
    result = subprocess.run(cmd, input=pcm_s16le, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg ошибка OGG (stdin PCM): {err}")


def is_supported_audio(path: Path) -> bool:
    return path.suffix.lower() in {".wav", ".ogg", ".opus", ".oga", ".mp3", ".m4a"}
