#!/usr/bin/env python3
"""
Опциональное дообучение XTTS GPT-энкодера.

ВАЖНО:
- Основной режим проекта — zero-shot cloning без fine-tuning.
- Fine-tuning GPT-энкодера XTTS имеет смысл только при 10+ минутах чистой речи
  с точными транскриптами, что редко достижимо из Telegram voice messages.
- На маленьком датасете (3–10 сообщений) fine-tuning часто ухудшает обобщение
  и может занять 40–90 минут на GPU.
- Для большинства пользователей рекомендуется оставаться в zero-shot режиме.

Этот скрипт подготавливает конфигурацию и выводит инструкции по запуску
официального Gradio-демо fine-tuning из coqui-tts, поскольку полный training
pipeline требует значительных ресурсов и ручной настройки гиперпараметров.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def estimate_training_time(num_samples: int, avg_duration_sec: float, gpu_vram_gb: float = 8) -> dict:
    total_audio_min = num_samples * avg_duration_sec / 60
    # Эвристика: ~1.5–3 мин на минуту аудио на 8GB GPU
    factor = 2.0 if gpu_vram_gb >= 8 else 3.5
    estimated_min = max(15, total_audio_min * factor)
    return {
        "total_audio_minutes": round(total_audio_min, 1),
        "estimated_minutes": round(estimated_min, 0),
        "realistic_under_90min": estimated_min <= 90 and num_samples >= 5 and total_audio_min >= 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Опциональный fine-tuning XTTS")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Папка после prepare_dataset.py")
    parser.add_argument("--output-dir", default="./finetune_output", type=Path)
    parser.add_argument("--gpu-vram-gb", type=float, default=8.0)
    args = parser.parse_args()

    meta_path = args.dataset_dir / "metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"Не найден {meta_path}. Сначала запустите prepare_dataset.py")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    estimate = estimate_training_time(len(metadata), avg_duration_sec=8.0, gpu_vram_gb=args.gpu_vram_gb)

    print("=== Опциональный fine-tuning XTTS ===\n")
    print("Нужен ли fine-tuning?")
    print("- НЕТ, если у вас 3–10 Telegram voice messages (типичный сценарий бота).")
    print("- ВОЗМОЖНО, если есть 10+ минут студийных записей с точными транскриптами.")
    print()
    print(f"Записей в датасете: {len(metadata)}")
    print(f"Оценка времени: ~{estimate['estimated_minutes']:.0f} мин")
    print(f"Реалистично за 90 мин: {'да' if estimate['realistic_under_90min'] else 'нет'}")
    print()

    if not estimate["realistic_under_90min"]:
        print("Рекомендация: оставайтесь в zero-shot режиме (/addvoice + /finishvoice).")
        print("Fine-tuning на маленьком зашумлённом датасете может ухудшить качество.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "steps": [
            "Установите coqui-tts: pip install coqui-tts",
            "Клонируйте репозиторий idiap/coqui-ai-TTS",
            "Установите зависимости демо: pip install -r TTS/demos/xtts_ft_demo/requirements.txt",
            "Запустите: python TTS/demos/xtts_ft_demo/xtts_demo.py",
            "Шаг 1: загрузите WAV из dataset_dir/wavs",
            "Шаг 2: fine-tuning GPT encoder (~40 мин на Colab T4)",
            "Шаг 3: inference с fine-tuned checkpoint",
            "Сохраняйте checkpoint в output_dir для продолжения после прерывания",
        ],
        "warnings": [
            "Используйте train/validation split (демо делает это автоматически)",
            "Следите за переобучением на <5 мин аудио",
            "Telegram OGG/OPUS — плохой источник для обучения",
        ],
    }
    plan_path = args.output_dir / "finetune_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"План сохранён: {plan_path}")
    print("\nСледующие шаги:")
    for i, step in enumerate(plan["steps"], 1):
        print(f"  {i}. {step}")


if __name__ == "__main__":
    main()
