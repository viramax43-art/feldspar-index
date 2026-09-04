"""Отдельный процесс faster-whisper: в боте ctranslate2 падает access violation вместе с torch/CUDA."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import soundfile as sf

from app.services.transcription import TimedSegment, merge_timed_segments

logger = logging.getLogger("stt_worker")
_MODELS: dict[str, Any] = {}


def _style_for_text(text: str) -> str:
    if text.endswith(("?", "？")):
        return "question"
    if text.endswith(("!", "！")):
        return "expressive"
    return "neutral"


def _normalize_language(language: str | None) -> str | None:
    raw = (language or "").strip().lower()
    if not raw or raw in {"auto", "detect"}:
        return None
    return raw


def _load_model(job: dict[str, Any]) -> Any:
    key = f"{job.get('model_size')}|{job.get('device')}|{job.get('compute_type')}"
    cached = _MODELS.get(key)
    if cached is not None:
        return cached
    from faster_whisper import WhisperModel

    logger.info(
        "worker load faster-whisper %s (%s/%s)",
        job["model_size"],
        job["device"],
        job["compute_type"],
    )
    model = WhisperModel(
        job["model_size"],
        device=job["device"],
        compute_type=job["compute_type"],
        cpu_threads=max(1, int(job.get("cpu_threads") or 1)),
        num_workers=1,
    )
    _MODELS[key] = model
    logger.info("worker model ready")
    return model


def _transcribe_timed(
    model: Any,
    path: Path,
    language: str | None,
    *,
    beam_size: int,
    vad_filter: bool,
    no_speech_threshold: float | None,
    time_offset: float = 0.0,
) -> list[TimedSegment]:
    resolved = _normalize_language(language)
    kwargs: dict[str, Any] = {
        "language": resolved,
        "beam_size": beam_size,
        "vad_filter": vad_filter,
        "word_timestamps": True,
        "condition_on_previous_text": False,
        "task": "transcribe",
        "temperature": 0.0,
        "compression_ratio_threshold": 2.2,
    }
    if no_speech_threshold is not None:
        kwargs["no_speech_threshold"] = no_speech_threshold
    segments, _info = model.transcribe(str(path), **kwargs)
    out: list[TimedSegment] = []
    for segment in segments:
        text = (segment.text or "").strip()
        if not text:
            continue
        low = text.lower()
        if low in {
            "thanks for watching",
            "thank you for watching",
            "please subscribe",
            "субтитры создавал",
            "продолжение следует",
            "подписывайтесь на канал",
        }:
            continue
        start = float(getattr(segment, "start", 0.0) or 0.0) + time_offset
        end = float(getattr(segment, "end", start - time_offset) or (start - time_offset))
        end = end + time_offset
        if end <= start:
            end = start + 0.4
        words: list[tuple[str, float, float]] = []
        for item in getattr(segment, "words", None) or []:
            token = str(getattr(item, "word", "") or "").strip()
            if not token:
                continue
            ws = float(getattr(item, "start", start - time_offset) or 0.0)
            we = float(getattr(item, "end", ws) or ws)
            words.append((token, ws + time_offset, we + time_offset))
        out.append(
            TimedSegment(
                start=start,
                end=end,
                text=text,
                style=_style_for_text(text),
                words=words,
                no_speech_prob=float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
                avg_logprob=float(getattr(segment, "avg_logprob", 0.0) or 0.0),
            )
        )
    return out


def _probe_language(
    model: Any,
    audio: np.ndarray,
    sample_rate: int,
    workdir: Path,
) -> str:
    probe_sec = min(45.0, max(8.0, len(audio) / float(sample_rate)))
    n = int(probe_sec * sample_rate)
    path = workdir / "lang_probe.wav"
    sf.write(str(path), audio[:n], sample_rate, subtype="PCM_16")
    try:
        _, info = model.transcribe(
            str(path),
            language=None,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
            task="transcribe",
        )
        lang = (getattr(info, "language", None) or "en").lower()
        prob = float(getattr(info, "language_probability", 0.0) or 0.0)
        logger.info("detected language=%s (p=%.2f)", lang, prob)
        return lang
    finally:
        path.unlink(missing_ok=True)


def _timed_chunked(
    model: Any,
    audio: np.ndarray,
    sample_rate: int,
    workdir: Path,
    language: str | None,
    *,
    beam_size: int,
    vad_filter: bool,
    no_speech_threshold: float,
    chunk_sec: float,
    overlap_sec: float,
    quiet_audio: bool = False,
    quiet_target_db: float = -12.0,
) -> list[TimedSegment]:
    locked = _normalize_language(language)
    if locked is None:
        locked = _probe_language(model, audio, sample_rate, workdir)
    duration = len(audio) / float(sample_rate)
    chunk_sec = max(10.0, float(chunk_sec))
    overlap_sec = max(0.0, min(float(overlap_sec), chunk_sec / 3))
    step = max(5.0, chunk_sec - overlap_sec)
    all_segs: list[TimedSegment] = []
    starts = [0.0]
    t = step
    while t < duration - 0.5:
        starts.append(t)
        t += step
    boost_fn = None
    if quiet_audio:
        from app.audio.preprocess import boost_quiet_stt_audio

        boost_fn = boost_quiet_stt_audio
    for idx, start_sec in enumerate(starts):
        end_sec = min(duration, start_sec + chunk_sec)
        a = int(start_sec * sample_rate)
        b = int(end_sec * sample_rate)
        if b - a < sample_rate // 2:
            continue
        chunk = np.asarray(audio[a:b], dtype=np.float32)
        if boost_fn is not None:
            chunk = boost_fn(chunk, sample_rate, target_db=float(quiet_target_db))
        chunk_path = workdir / f"vid_chunk_{idx:03d}.wav"
        sf.write(str(chunk_path), chunk, sample_rate, subtype="PCM_16")
        segs: list[TimedSegment] = []
        try:
            segs = _transcribe_timed(
                model,
                chunk_path,
                locked,
                beam_size=beam_size,
                vad_filter=vad_filter,
                no_speech_threshold=no_speech_threshold,
                time_offset=start_sec,
            )
            all_segs.extend(segs)
        finally:
            chunk_path.unlink(missing_ok=True)
        logger.info(
            "chunk %d/%d (%.1f–%.1f, lang=%s quiet=%s): %d segs",
            idx + 1,
            len(starts),
            start_sec,
            end_sec,
            locked,
            quiet_audio,
            len(segs),
        )
    return merge_timed_segments(all_segs)


def _segment_to_dict(seg: TimedSegment) -> dict[str, Any]:
    return {
        "start": seg.start,
        "end": seg.end,
        "text": seg.text,
        "style": seg.style,
        "words": [[w, s, e] for w, s, e in seg.words],
        "no_speech_prob": float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
        "avg_logprob": float(getattr(seg, "avg_logprob", 0.0) or 0.0),
    }


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    model = _load_model(job)
    wav_path = Path(job["wav_path"])
    workdir = wav_path.parent
    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    mode = job.get("mode") or "timed"
    if mode == "chunks":
        chunk_samples = max(sample_rate, int(sample_rate * float(job["chunk_seconds"])))
        parts: list[str] = []
        chunk_count = max(1, int(np.ceil(len(audio) / chunk_samples)))
        language = job.get("stt_language") or "ru"
        for index in range(chunk_count):
            start = index * chunk_samples
            end = min(start + chunk_samples, len(audio))
            chunk_path = workdir / f"chunk_{index:04d}.wav"
            sf.write(str(chunk_path), audio[start:end], sample_rate, subtype="PCM_16")
            try:
                segments, _info = model.transcribe(
                    str(chunk_path),
                    language=_normalize_language(language) or "ru",
                    beam_size=int(job.get("beam_size") or 3),
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                part = " ".join(s.text.strip() for s in segments).strip()
            finally:
                chunk_path.unlink(missing_ok=True)
            if part:
                parts.append(part)
        return {"ok": True, "parts": parts}

    detected_lang: str | None = None
    if job.get("for_video"):
        aligner = str(job.get("aligner") or "whisper").lower().strip()
        if aligner == "whisperx":
            try:
                from app.services.whisperx_align import (
                    transcribe_align_whisperx,
                    transcribe_align_whisperx_chunked,
                    whisperx_available,
                )

                if not whisperx_available():
                    raise ImportError("whisperx not installed")
                duration = len(audio) / float(sample_rate)
                chunk_lim = float(job.get("whisperx_chunk_sec") or 90.0)
                if duration > chunk_lim * 1.15:
                    segments, detected_lang = transcribe_align_whisperx_chunked(
                        audio,
                        sample_rate,
                        workdir,
                        language=job.get("language"),
                        model_size=str(job.get("model_size") or "small"),
                        device=str(job.get("device") or "cpu"),
                        compute_type=str(job.get("compute_type") or "int8"),
                        chunk_sec=chunk_lim,
                        overlap_sec=float(job.get("overlap_sec") or 2.0),
                    )
                    if not detected_lang:
                        detected_lang = _normalize_language(job.get("language"))
                else:
                    segments, detected_lang = transcribe_align_whisperx(
                        wav_path,
                        language=job.get("language"),
                        model_size=str(job.get("model_size") or "small"),
                        device=str(job.get("device") or "cpu"),
                        compute_type=str(job.get("compute_type") or "int8"),
                    )
            except Exception:
                logger.exception("WhisperX failed, fallback to faster-whisper")
                segments = _timed_chunked(
                    model,
                    audio,
                    sample_rate,
                    workdir,
                    job.get("language"),
                    beam_size=int(job["beam_size"]),
                    vad_filter=bool(job["vad_filter"]),
                    no_speech_threshold=float(job["no_speech_threshold"]),
                    chunk_sec=float(job["chunk_sec"]),
                    overlap_sec=float(job["overlap_sec"]),
                    quiet_audio=bool(job.get("quiet_audio")),
                    quiet_target_db=float(job.get("quiet_target_db") or -12.0),
                )
        else:
            segments = _timed_chunked(
                model,
                audio,
                sample_rate,
                workdir,
                job.get("language"),
                beam_size=int(job["beam_size"]),
                vad_filter=bool(job["vad_filter"]),
                no_speech_threshold=float(job["no_speech_threshold"]),
                chunk_sec=float(job["chunk_sec"]),
                overlap_sec=float(job["overlap_sec"]),
                quiet_audio=bool(job.get("quiet_audio")),
                quiet_target_db=float(job.get("quiet_target_db") or -12.0),
            )
    else:
        segments = _transcribe_timed(
            model,
            wav_path,
            job.get("language"),
            beam_size=int(job["beam_size"]),
            vad_filter=True,
            no_speech_threshold=None,
        )
    out: dict[str, Any] = {
        "ok": True,
        "segments": [_segment_to_dict(s) for s in segments],
    }
    if detected_lang:
        out["language"] = str(detected_lang).lower().strip()
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    if len(sys.argv) != 3:
        print("usage: python -m app.services.stt_worker JOB.json OUT.json", file=sys.stderr)
        return 2
    job_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    job = json.loads(job_path.read_text(encoding="utf-8"))
    try:
        result = run_job(job)
    except Exception as exc:
        logger.exception("stt worker failed")
        out_path.write_text(
            json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
            encoding="utf-8",
        )
        return 1
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
