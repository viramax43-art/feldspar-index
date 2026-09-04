#!/usr/bin/env python3
"""Rebuild metadata.csv for existing ft_*.wav clips (short, under XTTS ru limit)."""

from __future__ import annotations

import logging
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("meta")

MAX_CHARS = 180
MIN_CHARS = 10
MIN_SEC = 1.0
MAX_SEC = 11.0


def main() -> None:
    wavs_dir = Path("data/finetune/dataset/wavs")
    meta_path = Path("data/finetune/dataset/metadata.csv")
    files = sorted(wavs_dir.glob("ft_*.wav"))
    log.info("Transcribing %d ft clips...", len(files))
    model = WhisperModel("small", device="cpu", compute_type="int8")

    rows: list[str] = []
    for i, wav in enumerate(files):
        info = sf.info(str(wav))
        dur = info.frames / info.samplerate
        if dur < MIN_SEC or dur > MAX_SEC:
            continue
        segments, _ = model.transcribe(str(wav), language="ru", beam_size=3)
        text = " ".join(s.text.strip() for s in segments).strip().replace("|", ",")
        if not (MIN_CHARS <= len(text) <= MAX_CHARS):
            continue
        rows.append(f"{wav.stem}|{text}|{text}")
        if (i + 1) % 50 == 0:
            log.info("  %d/%d kept=%d", i + 1, len(files), len(rows))

    meta_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    log.info("DONE kept=%d -> %s", len(rows), meta_path)


if __name__ == "__main__":
    main()
