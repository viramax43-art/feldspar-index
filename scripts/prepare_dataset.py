#!/usr/bin/env python3
"""Подготовка датасета из WAV-референсов для опционального fine-tuning."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from app.audio.preprocess import load_audio_mono, normalize_loudness, remove_long_pauses
from app.audio.quality import evaluate_reference
import soundfile as sf


def _try_hardlink(src: Path, dst: Path) -> bool:
    """Жёсткая ссылка не занимает место и не портит качество (тот же inode)."""
    try:
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
        return True
    except OSError:
        return False


def _is_ready_wav(path: Path, sample_rate: int) -> bool:
    if path.suffix.lower() != ".wav":
        return False
    try:
        info = sf.info(str(path))
    except Exception:
        return False
    return (
        info.samplerate == sample_rate
        and info.channels == 1
        and (info.subtype or "").upper() in {"PCM_16", "PCM_24", "PCM_32", "FLOAT"}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовка датасета XTTS fine-tuning")
    parser.add_argument("--input-dir", required=True, type=Path, help="Папка с WAV/OGG")
    parser.add_argument("--output-dir", required=True, type=Path, help="Выходная папка")
    parser.add_argument("--transcript-csv", type=Path, help="CSV: filename,text")
    parser.add_argument("--sample-rate", type=int, default=22050)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = args.output_dir / "wavs"
    wavs_dir.mkdir(exist_ok=True)

    transcripts: dict[str, str] = {}
    if args.transcript_csv and args.transcript_csv.exists():
        with args.transcript_csv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                transcripts[row["filename"]] = row["text"]

    metadata = []
    for src in sorted(args.input_dir.glob("*")):
        if src.suffix.lower() not in {".wav", ".ogg", ".opus", ".oga"}:
            continue
        audio = load_audio_mono(src, args.sample_rate)
        audio = remove_long_pauses(audio, args.sample_rate)
        audio = normalize_loudness(audio)
        duration = len(audio) / args.sample_rate
        quality = evaluate_reference({"duration_sec": duration, "speech_ratio": 0.8, "clipping_ratio": 0, "rms": 0.05})
        if not quality.accepted:
            print(f"Пропуск {src.name}: низкое качество")
            continue
        dst = wavs_dir / f"{src.stem}.wav"
        if _is_ready_wav(src, args.sample_rate) and _try_hardlink(src, dst):
            print(f"hardlink {src.name}")
        else:
            sf.write(dst, audio, args.sample_rate, subtype="PCM_16")
        text = transcripts.get(src.name, transcripts.get(src.stem + src.suffix, ""))
        metadata.append({"audio_file": dst.name, "text": text, "speaker_name": "speaker"})

    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Подготовлено {len(metadata)} файлов в {args.output_dir}")


if __name__ == "__main__":
    main()
