#!/usr/bin/env python3
"""Нарезает референсы на короткие клипы под XTTS fine-tune (без потери качества)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("ft_clips")

TARGET_SR = 22050
MAX_CHARS = 180
MIN_CHARS = 10
MIN_SEC = 1.2
MAX_SEC = 6.0


def _to_mono_22k(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/finetune/dataset"))
    parser.add_argument("--language", default="ru")
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    out_wavs = args.output_dir / "wavs"
    out_wavs.mkdir(parents=True, exist_ok=True)
    for old in out_wavs.glob("*"):
        if old.is_file():
            old.unlink()

    files = sorted(
        p
        for p in args.input_dir.iterdir()
        if p.suffix.lower() in {".wav", ".ogg", ".opus", ".oga"}
    )
    if not files:
        raise SystemExit(f"Нет аудио в {args.input_dir}")

    log.info("Источник: %d файлов. Гружу Whisper...", len(files))
    model = WhisperModel("small", device="cpu", compute_type="int8")

    rows: list[str] = []
    idx = 0
    for fi, src in enumerate(files, 1):
        audio, sr = sf.read(str(src), always_2d=False)
        audio = _to_mono_22k(np.asarray(audio), int(sr))
        segments, _ = model.transcribe(str(src), language=args.language, beam_size=3, word_timestamps=False)

        packed: list[tuple[float, float, str]] = []
        cur_start = None
        cur_end = None
        cur_text: list[str] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            start, end = float(seg.start), float(seg.end)
            if cur_start is None:
                cur_start, cur_end, cur_text = start, end, [text]
                continue
            new_text = " ".join(cur_text + [text])
            new_dur = end - cur_start
            if new_dur <= MAX_SEC and len(new_text) <= MAX_CHARS:
                cur_end = end
                cur_text.append(text)
            else:
                packed.append((cur_start, cur_end, " ".join(cur_text)))
                cur_start, cur_end, cur_text = start, end, [text]
        if cur_start is not None and cur_text:
            packed.append((cur_start, cur_end, " ".join(cur_text)))

        for start, end, text in packed:
            text = text.replace("|", ",").strip()
            dur = end - start
            if dur < MIN_SEC or dur > MAX_SEC + 0.25:
                continue
            if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                continue
            s0 = max(0, int(start * TARGET_SR))
            s1 = min(len(audio), int(end * TARGET_SR))
            clip = audio[s0:s1]
            if len(clip) < int(MIN_SEC * TARGET_SR):
                continue
            name = f"ft_{idx:04d}.wav"
            sf.write(out_wavs / name, clip, TARGET_SR, subtype="PCM_16")
            rows.append(f"{Path(name).stem}|{text}|{text}")
            idx += 1

        if fi % 10 == 0 or fi == len(files):
            log.info("  %d/%d → clips=%d", fi, len(files), idx)

    meta = args.output_dir / "metadata.csv"
    meta.write_text("\n".join(rows) + "\n", encoding="utf-8")
    log.info("DONE clips=%d → %s", len(rows), meta)


if __name__ == "__main__":
    main()
