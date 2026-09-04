"""WhisperX: транскрипция + нейросетевой forced-align (wav2vec2) для word timestamps."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.services.transcription import TimedSegment

logger = logging.getLogger(__name__)

# Reuse models inside one STT worker process — reloading every chunk caused
# ACCESS_VIOLATION crashes on long videos (Windows + torch/ctranslate2).
_WX_ASR: dict[str, Any] = {}
_WX_ALIGN: dict[str, tuple[Any, Any]] = {}


def whisperx_available() -> bool:
    try:
        import whisperx  # noqa: F401

        return True
    except ImportError:
        return False


def _style_for_text(text: str) -> str:
    if text.endswith(("?", "？")):
        return "question"
    if text.endswith(("!", "！")):
        return "expressive"
    return "neutral"


def _resolve_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_asr_model(model_size: str, device: str, compute_type: str) -> Any:
    import whisperx

    dev = _resolve_device(device)
    compute = compute_type if dev == "cuda" else "int8"
    key = f"{model_size}|{dev}|{compute}"
    model = _WX_ASR.get(key)
    if model is None:
        logger.info("WhisperX load %s on %s (%s)", model_size, dev, compute)
        model = whisperx.load_model(model_size, device=dev, compute_type=compute)
        _WX_ASR[key] = model
    return model, dev


def _get_align_model(language_code: str, device: str) -> tuple[Any, Any]:
    import whisperx

    key = f"{language_code}|{device}"
    cached = _WX_ALIGN.get(key)
    if cached is None:
        cached = whisperx.load_align_model(language_code=language_code, device=device)
        _WX_ALIGN[key] = cached
    return cached


def _segments_from_aligned(
    aligned: dict[str, Any],
    *,
    time_offset: float = 0.0,
) -> list[TimedSegment]:
    segments: list[TimedSegment] = []
    off = float(time_offset)
    for seg in aligned.get("segments") or []:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        low = text.lower()
        if low in {"thanks for watching", "субтитры создавал", "продолжение следует"}:
            continue
        start = float(seg.get("start") or 0.0) + off
        end = float(seg.get("end") or float(seg.get("start") or 0.0) + 0.4) + off
        words: list[tuple[str, float, float]] = []
        for w in seg.get("words") or []:
            token = str(w.get("word") or w.get("text") or "").strip()
            if not token:
                continue
            ws = float(w.get("start") or 0.0) + off
            we = float(w.get("end") or float(w.get("start") or 0.0) + 0.04) + off
            words.append((token, ws, max(ws + 0.04, we)))
        segments.append(
            TimedSegment(
                start=start,
                end=max(start + 0.08, end),
                text=text,
                style=_style_for_text(text),
                words=words,
                no_speech_prob=float(seg.get("no_speech_prob") or 0.0),
                avg_logprob=float(seg.get("avg_logprob") or 0.0),
            )
        )
    return segments


def transcribe_align_whisperx(
    wav_path: Path,
    *,
    language: str | None = None,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    batch_size: int = 8,
    time_offset: float = 0.0,
) -> tuple[list[TimedSegment], str]:
    """Возвращает сегменты с точными word timestamps и detected language."""
    import whisperx

    audio = whisperx.load_audio(str(wav_path))
    model, dev = _get_asr_model(model_size, device, compute_type)
    lang = (language or "").strip().lower()
    if lang in {"", "auto", "detect"}:
        lang = None
    result: dict[str, Any] = model.transcribe(
        audio, batch_size=int(batch_size), language=lang
    )
    detected = str(result.get("language") or lang or "en").lower()
    align_model, metadata = _get_align_model(detected, dev)
    aligned = whisperx.align(
        result.get("segments") or [],
        align_model,
        metadata,
        audio,
        dev,
        return_char_alignments=False,
    )
    segments = _segments_from_aligned(aligned, time_offset=time_offset)
    logger.info(
        "WhisperX aligned: %d segments, %d words, lang=%s",
        len(segments),
        sum(len(s.words or []) for s in segments),
        detected,
    )
    return segments, detected


def transcribe_align_whisperx_chunked(
    audio: np.ndarray,
    sample_rate: int,
    workdir: Path,
    *,
    language: str | None = None,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    chunk_sec: float = 90.0,
    overlap_sec: float = 2.0,
) -> tuple[list[TimedSegment], str | None]:
    """WhisperX по чанкам для длинных роликов. Returns (segments, detected_lang)."""
    from app.services.transcription import merge_timed_segments

    duration = len(audio) / float(sample_rate)
    chunk_sec = max(30.0, float(chunk_sec))
    overlap_sec = max(0.0, min(float(overlap_sec), chunk_sec / 3))
    step = max(10.0, chunk_sec - overlap_sec)
    starts = [0.0]
    t = step
    while t < duration - 0.5:
        starts.append(t)
        t += step
    all_segs: list[TimedSegment] = []
    locked_lang = (language or "").strip().lower()
    if locked_lang in {"", "auto", "detect"}:
        locked_lang = ""
    detected_lang: str | None = locked_lang or None
    failures = 0
    for idx, start_sec in enumerate(starts):
        end_sec = min(duration, start_sec + chunk_sec)
        a = int(start_sec * sample_rate)
        b = int(end_sec * sample_rate)
        if b - a < sample_rate // 2:
            continue
        chunk_path = workdir / f"wx_chunk_{idx:03d}.wav"
        sf.write(str(chunk_path), audio[a:b], sample_rate, subtype="PCM_16")
        try:
            segs, chunk_lang = transcribe_align_whisperx(
                chunk_path,
                language=detected_lang or language,
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                time_offset=start_sec,
            )
            if not detected_lang and chunk_lang:
                detected_lang = chunk_lang
            all_segs.extend(segs)
            logger.info(
                "WhisperX chunk %d/%d (%.1f-%.1f): %d segs",
                idx + 1,
                len(starts),
                start_sec,
                end_sec,
                len(segs),
            )
        except Exception:
            failures += 1
            logger.exception(
                "WhisperX chunk %d/%d failed (%.1f-%.1f)",
                idx + 1,
                len(starts),
                start_sec,
                end_sec,
            )
            if failures >= 2 and not all_segs:
                raise
            if failures >= 3:
                logger.warning(
                    "WhisperX stopping early after %d chunk failures; keeping %d segs",
                    failures,
                    len(all_segs),
                )
                break
        finally:
            chunk_path.unlink(missing_ok=True)
    if not all_segs:
        raise RuntimeError("WhisperX produced no segments on any chunk")
    return merge_timed_segments(all_segs), detected_lang
