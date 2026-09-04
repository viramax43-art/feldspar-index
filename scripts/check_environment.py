#!/usr/bin/env python3
"""Проверка окружения для Assistant."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys


def _nvidia_smi() -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return out.strip()
    except Exception:
        return None


def main() -> int:
    print("=== Assistant — проверка окружения ===\n")
    print(f"Python: {sys.version.split()[0]} ({platform.system()} {platform.release()})")

    # На ноутбуках с iGPU + NVIDIA предпочитаем дискретную GPU
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    smi = _nvidia_smi()
    if smi:
        print(f"nvidia-smi: {smi}")
    else:
        print("nvidia-smi: не найден (драйвер NVIDIA не установлен или не в PATH)")

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"PyTorch CUDA build: {torch.version.cuda or 'нет (CPU-only сборка)'}")

        if "+cpu" in torch.__version__ or torch.version.cuda is None:
            print("❌ Установлен CPU-only PyTorch — он не видит NVIDIA GPU.")
            print("   Переустановите сборку с CUDA:")
            print("   pip uninstall -y torch torchaudio")
            print(
                "   pip install torch torchaudio "
                "--index-url https://download.pytorch.org/whl/cu124"
            )
            cuda = False
        else:
            cuda = torch.cuda.is_available()

        print(f"CUDA доступна в PyTorch: {cuda}")
        if cuda:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            vram_gb = props.total_memory / (1024**3)
            print(f"Активный GPU: {props.name} (cuda:{idx})")
            print(f"VRAM: {vram_gb:.2f} GB")
            if vram_gb < 4.5:
                print(
                    "⚠️  ~4 GB VRAM (GTX 1650 и аналоги) — используйте короткие тексты, "
                    "MAX_TEXT_LENGTH≤800, USE_FP16=true"
                )
            elif vram_gb < 8:
                print("ℹ️  6–8 GB VRAM — достаточно для XTTS-v2 при умеренных текстах")
            else:
                print("✅ VRAM достаточно для комфортной работы")
        else:
            if smi:
                print(
                    "⚠️  GPU NVIDIA видна драйверу, но не PyTorch. "
                    "Нужна CUDA-сборка torch (см. команду выше)."
                )
            else:
                print("⚠️  CUDA недоступна — синтез на CPU будет в 10–50 раз медленнее")
    except ImportError:
        print("❌ PyTorch не установлен")
        return 1

    ffmpeg = shutil.which("ffmpeg")
    print(f"FFmpeg: {'найден — ' + ffmpeg if ffmpeg else 'НЕ НАЙДЕН'}")
    if not ffmpeg:
        print("❌ Установите FFmpeg и добавьте в PATH")

    from pathlib import Path

    silero_path = Path("assets/tts/silero/v5_5_ru.pt")
    if silero_path.exists() and silero_path.stat().st_size > 1_000_000:
        print(f"Silero TTS: найден — {silero_path} ({silero_path.stat().st_size // 1024} KB)")
    else:
        print(f"Silero TTS: НЕ НАЙДЕН ({silero_path})")
        print("  Скачайте fallback: python scripts/download_silero.py")

    try:
        import TTS  # noqa: F401

        print("coqui-tts: установлен")
        # Полная загрузка XTTS тяжёлая — только если --load-xtts
        if "--load-xtts" in sys.argv:
            from TTS.api import TTS as CoquiTTS

            print("Проверка загрузки XTTS-v2 (может занять несколько минут)...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
            tts.to(device)
            print(f"✅ Модель XTTS-v2 загружена на {device}")
        else:
            print("XTTS: пакет OK (для полной загрузки: python scripts/check_environment.py --load-xtts)")
    except ImportError:
        print("❌ coqui-tts не установлен (pip install coqui-tts)")
        return 1
    except Exception as exc:
        print(f"❌ Ошибка загрузки модели: {exc}")
        return 1

    print("\n=== Рекомендации по скорости ===")
    print("• Hybrid: XTTS (профиль/CUDA) + Silero CPU fallback")
    if torch.cuda.is_available():
        print("• XTTS короткая фраза (~20 слов): ~3–8 с на GPU")
        print("• Silero короткая фраза: обычно <1 с на CPU")
        print("• GTX 1650 (4 GB): короткие тексты, USE_FP16=true, Silero при OOM")
    else:
        print("• Без CUDA: предпочтите TTS_ENGINE=silero или ожидайте медленный XTTS на CPU")

    print("\nГотово.")
    ok = bool(ffmpeg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
