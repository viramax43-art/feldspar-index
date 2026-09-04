#!/usr/bin/env python3
"""Fine-tuning XTTS-v2 GPT encoder на голосе конкретного пользователя.

Использование:
    # 1. Подготовка датасета (если ещё не готов)
    python scripts/prepare_dataset.py \
        --input-dir data/users/<USER_ID>/references \
        --output-dir data/finetune/dataset \
        --sample-rate 22050

    # 2. Файн-тюнинг (GPU обязателен, минимум 6 GB VRAM)
    # На маленьком диске: один last_full.pth + lean best_model.pth,
    # без TensorBoard и без копий checkpoint_N / best_model_N.
    python scripts/finetune_xtts.py \
        --dataset-dir data/finetune/dataset \
        --output-dir data/finetune/output \
        --speaker-wav data/users/<USER_ID>/references/ref_001.wav \
        --epochs 30 \
        --batch-size 2 \
        --lr 5e-6

    # 3. Тестирование
    python scripts/finetune_xtts.py --test \
        --checkpoint data/finetune/output/<run>/best_model.pth \
        --config data/finetune/output/<run>/config.json \
        --speaker-wav data/users/<USER_ID>/references/ref_001.wav \
        --text "Привет! Как дела?"

Требования:
    - GPU с ≥ 6 GB VRAM (CUDA)
    - coqui-tts >= 0.22.0
    - Датасет: WAV 22050 Hz mono + metadata.csv (id|text)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Обязательно до импорта TTS
os.environ.setdefault("COQUI_TOS_AGREED", "1")

# Полный trainer-checkpoint ~5.6 GB (веса + AdamW). Lean (только веса) ~2.1 GB.
_FULL_CKPT_GB = 6.0
_LEAN_CKPT_GB = 2.5


def disk_free_gb(path: Path | str = ".") -> float:
    target = Path(path)
    anchor = target.anchor or str(target.resolve().drive or target)
    try:
        return shutil.disk_usage(anchor if anchor else str(target)).free / 1e9
    except OSError:
        return shutil.disk_usage(str(target.resolve().parent)).free / 1e9


def prune_training_junk(root: Path, *, preserve: set[Path] | None = None) -> int:
    """Удаляет дубли checkpoint/tensorboard. Качество модели не трогает."""
    preserve = {p.resolve() for p in (preserve or set()) if p}
    removed = 0
    if not root.exists():
        return 0
    patterns = (
        "checkpoint_*.pth",
        "best_model_[0-9]*.pth",
        "events.out.tfevents*",
        "*.pth.partial",
        "last_full.pth.partial",
    )
    for pattern in patterns:
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            if path.resolve() in preserve:
                continue
            try:
                removed += path.stat().st_size
                path.unlink()
                logger.info("Диск: удалил %s", path)
            except OSError as exc:
                logger.warning("Не удалось удалить %s: %s", path, exc)
    return removed


def _lean_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": state["model"],
        "config": state.get("config"),
        "step": state.get("step"),
        "epoch": state.get("epoch"),
        "date": state.get("date"),
        "model_loss": state.get("model_loss"),
    }


def _save_overwrite(obj: Any, dest: Path) -> None:
    """Пишет файл, сначала удаляя старый — без второго 5+ GB временного копии."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    import torch

    torch.save(obj, dest)


def install_disk_safe_checkpointing(trainer: Any) -> None:
    """Один last_full.pth (resume) + lean best_model.pth. Без копии best_model_N.pth."""
    from trainer.io import save_model

    output_path = Path(trainer.output_path)
    full_path = output_path / "last_full.pth"
    lean_path = output_path / "best_model.pth"

    def _persist(state: dict[str, Any]) -> None:
        free = disk_free_gb(output_path)
        wrote_full = False
        if free >= _FULL_CKPT_GB + 0.5:
            logger.info("Диск свободно %.1f GB — пишу last_full.pth (resume)", free)
            _save_overwrite(state, full_path)
            wrote_full = True
        else:
            logger.warning(
                "Мало места (%.1f GB): last_full.pth пропускаю, только веса",
                free,
            )
        free = disk_free_gb(output_path)
        if free >= _LEAN_CKPT_GB:
            _save_overwrite(_lean_payload(state), lean_path)
            logger.info(
                "Сохранён lean best_model.pth (%.2f GB)",
                lean_path.stat().st_size / 1e9,
            )
        elif wrote_full:
            logger.warning("Нет места на lean checkpoint (%.1f GB свободно)", free)
        else:
            logger.error("Нет места сохранить checkpoint (%.1f GB)", free)
        prune_training_junk(output_path, preserve={full_path, lean_path})

    def save_best_model() -> None:
        eval_loss = trainer._pick_target_avg_loss(trainer.keep_avg_eval)
        train_loss = trainer._pick_target_avg_loss(trainer.keep_avg_train) or float("inf")
        current: dict[str, Any] = {"train_loss": train_loss, "eval_loss": eval_loss}
        best = trainer.best_loss
        if isinstance(current, dict) and isinstance(best, dict):
            if current.get("eval_loss") is not None and best.get("eval_loss") is not None:
                is_better = current["eval_loss"] < best["eval_loss"]
            else:
                is_better = current["train_loss"] < best["train_loss"]
        else:
            is_better = float(current) < float(best)  # type: ignore[arg-type]
        if not is_better:
            return
        logger.info("Новый лучший loss — сохраняю компактный checkpoint")
        save_model(
            trainer.config,
            trainer._get_model(),
            str(lean_path),
            current_step=trainer.total_steps_done,
            epoch=trainer.epochs_done,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler if trainer.use_amp_scaler else None,
            model_loss=current,
            save_func=lambda state, _path: _persist(state),
        )
        trainer.best_loss = current

    def save_checkpoint() -> None:
        eval_loss = trainer._pick_target_avg_loss(trainer.keep_avg_eval)
        train_loss = trainer._pick_target_avg_loss(trainer.keep_avg_train)
        save_model(
            trainer.config,
            trainer._get_model(),
            str(full_path),
            current_step=trainer.total_steps_done,
            epoch=trainer.epochs_done,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler if trainer.use_amp_scaler else None,
            model_loss={"train_loss": train_loss, "eval_loss": eval_loss},
            save_func=lambda state, _path: _persist(state),
        )

    trainer.save_best_model = save_best_model
    trainer.save_checkpoint = save_checkpoint


def export_lean_checkpoint(src: Path, dest: Path) -> None:
    import torch

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Непонятный checkpoint: {src}")
    if "optimizer" not in ckpt:
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        return
    _save_overwrite(_lean_payload(ckpt), dest)
    del ckpt


def finalize_run(
    run_dir: Path,
    *,
    keep_full: bool,
    cleanup_dataset: bool,
    dataset_dir: Path,
) -> None:
    full_path = run_dir / "last_full.pth"
    lean_path = run_dir / "best_model.pth"
    if full_path.exists() and (
        not lean_path.exists() or lean_path.stat().st_size > 3_000_000_000
    ):
        logger.info("Экспортирую lean best_model.pth без optimizer...")
        export_lean_checkpoint(full_path, lean_path)
    if not keep_full and full_path.exists():
        size = full_path.stat().st_size
        full_path.unlink()
        logger.info("Удалил last_full.pth (%.2f GB) — обучение завершено", size / 1e9)
    prune_training_junk(run_dir, preserve={lean_path})
    if cleanup_dataset and dataset_dir.exists():
        size = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file())
        shutil.rmtree(dataset_dir, ignore_errors=True)
        logger.info("Удалил датасет %s (%.2f GB)", dataset_dir, size / 1e9)


def ensure_finetune_assets(model_dir: Path) -> None:
    """dvae.pth и mel_stats.pth нужны для GPT fine-tune, но не всегда в кеше TTS."""
    import urllib.request

    assets = {
        "mel_stats.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth",
        "dvae.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth",
    }
    for name, url in assets.items():
        dest = model_dir / name
        if dest.exists() and dest.stat().st_size > 500:
            continue
        logger.info("Скачиваю %s для fine-tune...", name)
        urllib.request.urlretrieve(url, dest)
        logger.info("Сохранено: %s (%s bytes)", dest, dest.stat().st_size)


def find_base_model_files() -> tuple[Path, Path, Path]:
    """Находит файлы базовой модели XTTS-v2 (скачанной через TTS)."""
    try:
        from TTS.utils.manage import ModelManager
        manager = ModelManager()
        model_path, config_path, _ = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
        model_dir = Path(model_path).parent if Path(model_path).is_file() else Path(model_path)
    except Exception:
        from TTS.api import TTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        model_dir = Path(tts.model_path).parent if hasattr(tts, 'model_path') else None
        del tts
        if model_dir is None:
            raise RuntimeError("Не удалось найти базовую модель XTTS-v2")

    candidates = [
        model_dir,
        Path.home() / ".local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2",
        Path.home() / "AppData/Local/tts/tts_models--multilingual--multi-dataset--xtts_v2",
        Path(os.environ.get("MODEL_CACHE_DIR", "")) / "tts_models--multilingual--multi-dataset--xtts_v2",
    ]

    for d in candidates:
        if not d or not d.exists():
            continue
        ckpt = d / "model.pth"
        vocab = d / "vocab.json"
        config = d / "config.json"
        if ckpt.exists() and vocab.exists() and config.exists():
            ensure_finetune_assets(d)
            logger.info("Базовая модель найдена: %s", d)
            return ckpt, vocab, config

    raise RuntimeError(
        "Не найдена базовая модель XTTS-v2. Запустите бот хотя бы раз, "
        "чтобы модель скачалась автоматически."
    )


def prepare_metadata_csv(dataset_dir: Path) -> Path:
    """Конвертирует metadata.json → metadata.csv (формат LJSpeech: id|text)."""
    meta_json = dataset_dir / "metadata.json"
    meta_csv = dataset_dir / "metadata.csv"

    if meta_csv.exists():
        logger.info("metadata.csv уже существует: %s", meta_csv)
        return meta_csv

    if not meta_json.exists():
        raise FileNotFoundError(
            f"Ни metadata.csv, ни metadata.json не найдены в {dataset_dir}"
        )

    with meta_json.open(encoding="utf-8") as f:
        items = json.load(f)

    with meta_csv.open("w", encoding="utf-8") as f:
        for item in items:
            stem = Path(item["audio_file"]).stem
            text = item.get("text", "").strip()
            if text:
                f.write(f"{stem}|{text}\n")

    logger.info("Создан metadata.csv с %d записями", len(items))
    return meta_csv


def create_whisper_transcripts(dataset_dir: Path, language: str = "ru") -> Path:
    """Автоматическая транскрипция WAV-файлов через faster-whisper."""
    wavs_dir = dataset_dir / "wavs"
    meta_csv = dataset_dir / "metadata.csv"

    if meta_csv.exists():
        lines = [l for l in meta_csv.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
        has_text = all(len(l.split("|")) >= 3 and l.split("|", 2)[2].strip() for l in lines)
        if has_text:
            logger.info("Транскрипции уже есть в metadata.csv (%d строк)", len(lines))
            return meta_csv
        # Старый формат id|text → id|text|text
        if lines and all(len(l.split("|")) == 2 for l in lines):
            converted = []
            for line in lines:
                stem, text = line.split("|", 1)
                text = text.strip().replace("|", ",")
                if text:
                    converted.append(f"{stem}|{text}|{text}")
            if converted:
                meta_csv.write_text("\n".join(converted) + "\n", encoding="utf-8")
                logger.info("Сконвертирован metadata.csv в LJSpeech 3-колонки (%d)", len(converted))
                return meta_csv

    logger.info("Транскрибирование WAV-файлов через faster-whisper...")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper не установлен. Установите: pip install faster-whisper"
        )

    model = WhisperModel("small", device="cpu", compute_type="int8")
    wavs = sorted(wavs_dir.glob("*.wav"))
    logger.info("Найдено %d WAV-файлов для транскрипции", len(wavs))

    with meta_csv.open("w", encoding="utf-8") as f:
        for wav in wavs:
            segments, _ = model.transcribe(str(wav), language=language, beam_size=3)
            text = " ".join(s.text.strip() for s in segments).strip()
            if text:
                text = text.replace("|", ",")
                # LJSpeech formatter в coqui-tts требует 3 колонки: id|raw|normalized
                f.write(f"{wav.stem}|{text}|{text}\n")
                logger.info("  %s → %s", wav.name, text[:80])

    del model
    logger.info("Транскрипция завершена → %s", meta_csv)
    return meta_csv


def train(args: argparse.Namespace) -> None:
    """Запуск fine-tuning XTTS-v2 GPT encoder."""
    import torch

    if not torch.cuda.is_available():
        logger.error("CUDA не доступна! Fine-tuning требует GPU.")
        sys.exit(1)

    logger.info("GPU: %s (%.1f GB VRAM)", torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_memory / 1e9)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    batch_size = args.batch_size
    grad_accum = args.grad_accum
    max_wav_length = 255995  # ~11.6 s
    if vram_gb < 6.0:
        # GTX 1650 4GB и аналоги: ультра-экономный режим
        batch_size = 1
        grad_accum = max(grad_accum, 32)
        max_wav_length = 132300  # ~6 s — иначе OOM
        logger.warning(
            "Мало VRAM (%.1f GB). Ставлю batch=%d, grad_accum=%d, max_wav=%.1fs",
            vram_gb, batch_size, grad_accum, max_wav_length / 22050,
        )
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preserve = {Path(p) for p in (args.restore_path,) if p}
    junk = prune_training_junk(output_dir, preserve=preserve)
    if junk:
        logger.info("Очистил %.2f GB старых checkpoint/tensorboard", junk / 1e9)
    logger.info("Свободно на диске: %.1f GB", disk_free_gb(output_dir))
    if disk_free_gb(output_dir) < _LEAN_CKPT_GB + 1.0:
        raise RuntimeError(
            f"Мало места на диске ({disk_free_gb(output_dir):.1f} GB). "
            "Освободите хотя бы ~4 GB перед обучением."
        )

    wavs_dir = dataset_dir / "wavs"
    if not wavs_dir.exists():
        raise FileNotFoundError(f"Папка с WAV-файлами не найдена: {wavs_dir}")

    meta_csv = create_whisper_transcripts(dataset_dir, language=args.language)

    # Минимум 500 семплов для нормального файн-тюна
    n_lines = sum(1 for line in meta_csv.read_text(encoding="utf-8").splitlines() if "|" in line)
    min_samples = args.min_samples
    if n_lines < min_samples:
        raise RuntimeError(
            f"В датасете только {n_lines} записей с текстом "
            f"(минимум {min_samples}). "
            f"Соберите ещё голосовые:\n"
            f"  python scripts/collect_account_voices.py --limit 1000 --max-seconds 7200 --build-profile\n"
            f"затем подготовьте датасет:\n"
            f"  python scripts/prepare_dataset.py --input-dir data/users/<ID>/references "
            f"--output-dir data/finetune/dataset"
        )

    # Находим базовую модель
    base_ckpt, vocab_path, base_config = find_base_model_files()

    logger.info("Базовый чекпоинт: %s", base_ckpt)
    logger.info("Vocab: %s", vocab_path)
    logger.info("Датасет: %s", dataset_dir)
    logger.info("Параметры: epochs=%d, batch=%d, lr=%s, grad_accum=%d",
                args.epochs, batch_size, args.lr, grad_accum)

    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.xtts_config import XttsAudioConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
    from trainer import Trainer, TrainerArgs
    from trainer.logging.dummy_logger import DummyLogger

    # Конфиг датасета
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="voice_clone",
        path=str(dataset_dir),
        meta_file_train=str(meta_csv.name),
        language=args.language,
    )

    # Аргументы модели
    model_args = GPTArgs(
        max_conditioning_length=132300,  # ~6 sec @ 22050 Hz
        min_conditioning_length=66150,   # ~3 sec
        debug_loading_failures=False,
        max_wav_length=max_wav_length,
        max_text_length=180,
        mel_norm_file=str(base_ckpt.parent / "mel_stats.pth"),
        dvae_checkpoint=str(base_ckpt.parent / "dvae.pth"),
        xtts_checkpoint=str(base_ckpt),
        tokenizer_file=str(vocab_path),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )

    config = GPTTrainerConfig(
        output_path=str(output_dir),
        model_args=model_args,
        run_name="xtts_finetune",
        project_name="voice_caller",
        run_description="XTTS-v2 fine-tuning on user voice",
        dashboard_logger="tensorboard",
        audio=audio_config,
        batch_size=batch_size,
        batch_group_size=48,
        eval_batch_size=batch_size,
        num_loader_workers=4 if sys.platform != "win32" else 0,
        eval_split_max_size=256,
        print_step=50,
        plot_step=10**9,
        log_model_step=10**9,
        save_step=args.save_step,
        save_n_checkpoints=1,
        save_checkpoints=True,
        save_all_best=False,
        save_on_interrupt=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=args.lr,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
            "gamma": 0.5,
        },
        # Пустой список: не пишем тестовые wav на диск (на веса не влияет)
        test_sentences=[],
        epochs=args.epochs,
    )

    # Загрузка датасета
    model = GPTTrainer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        [dataset_config],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=0.1,
    )

    logger.info("Train samples: %d, Eval samples: %d", len(train_samples), len(eval_samples))

    restore_path = args.restore_path or None
    if restore_path:
        logger.info("Продолжаю с checkpoint: %s", restore_path)

    trainer = Trainer(
        TrainerArgs(
            restore_path=restore_path,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=grad_accum,
        ),
        config,
        output_path=str(output_dir),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        dashboard_logger=DummyLogger(),
    )
    install_disk_safe_checkpointing(trainer)

    logger.info("Начинаю fine-tuning...")
    logger.info("Чекпоинты run: %s", trainer.output_path)
    finished_ok = False
    try:
        trainer.fit()
        finished_ok = True
    finally:
        finalize_run(
            Path(trainer.output_path),
            keep_full=args.keep_resume_checkpoint or not finished_ok,
            cleanup_dataset=args.cleanup_dataset and finished_ok,
            dataset_dir=dataset_dir,
        )
    logger.info("Fine-tuning завершён! Lean checkpoint: %s", Path(trainer.output_path) / "best_model.pth")


def test_model(args: argparse.Namespace) -> None:
    """Тестирование fine-tuned модели."""
    import torch
    import torchaudio

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Загрузка fine-tuned модели...")
    config = XttsConfig()
    config.load_json(args.config)

    _, vocab_path, _ = find_base_model_files()

    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=args.checkpoint,
        vocab_path=str(vocab_path),
        use_deepspeed=False,
    )
    model.to(device)

    logger.info("Вычисление speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[args.speaker_wav],
        gpt_cond_len=30,
        gpt_cond_chunk_len=6,
        max_ref_length=30,
        sound_norm_refs=True,
    )

    logger.info("Синтез: %s", args.text)
    out = model.inference(
        args.text,
        args.language,
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.85,
        top_k=80,
        top_p=0.92,
        repetition_penalty=1.5,
        length_penalty=1.0,
        speed=1.0,
    )

    wav = torch.tensor(out["wav"]).unsqueeze(0)
    out_path = args.out or "finetune_test.wav"
    torchaudio.save(out_path, wav, 24000)
    logger.info("Сохранено: %s (%.1f сек)", out_path, wav.shape[1] / 24000)


def main() -> None:
    parser = argparse.ArgumentParser(description="XTTS-v2 Fine-Tuning")
    parser.add_argument("--test", action="store_true", help="Режим тестирования")

    # Training args
    parser.add_argument("--dataset-dir", type=str, default="data/finetune/dataset")
    parser.add_argument("--output-dir", type=str, default="data/finetune/output")
    parser.add_argument("--speaker-wav", type=str, default="")
    parser.add_argument("--language", type=str, default="ru")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=500,
        help="Минимум голосовых с транскриптом для обучения (по умолчанию 500)",
    )
    parser.add_argument(
        "--restore-path",
        type=str,
        default="",
        help="Путь к checkpoint для продолжения обучения (например best_model.pth)",
    )
    parser.add_argument(
        "--save-step",
        type=int,
        default=10000,
        help="Как часто сохранять checkpoint (шаги). Один файл перезаписывается.",
    )
    parser.add_argument(
        "--keep-resume-checkpoint",
        action="store_true",
        help="После обучения оставить last_full.pth (~5.6 GB) для resume. По умолчанию удаляется.",
    )
    parser.add_argument(
        "--cleanup-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Удалить data/finetune/dataset после обучения (ссылки/копии wav). Качество не страдает.",
    )

    # Test args
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--text", type=str, default="Привет! Как дела?")
    parser.add_argument("--out", type=str, default="")

    args = parser.parse_args()

    if args.test:
        if not args.checkpoint or not args.config:
            parser.error("--test требует --checkpoint и --config")
        if not args.speaker_wav:
            parser.error("--test требует --speaker-wav")
        test_model(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
