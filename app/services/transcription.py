"""Чанковое распознавание Telegram voice через faster-whisper."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
import soundfile as sf

from app.audio import apply_quiet_stt_ffmpeg_boost, convert_to_wav
from app.config import Settings

logger = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscriptionUpdate:
    part: str
    full_text: str
    chunk_index: int
    chunk_count: int


@dataclass
class TimedSegment:
    start: float
    end: float
    text: str
    style: str = "neutral"
    rms: float = 0.0
    ssml: str = ""
    rate: float = 1.0
    volume: float = 1.0
    pause_after: float = 0.0
    words: list[tuple[str, float, float]] = field(default_factory=list)
    # Whisper confidence (0.0/0.0 = unknown → never treated as hallucination).
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.08, float(self.end) - float(self.start))


def is_probable_non_speech(
    seg: "TimedSegment",
    *,
    no_speech_prob: float = 0.6,
    min_logprob: float = -1.0,
) -> bool:
    """Whisper hallucination on moans/noise: high no_speech + low logprob.

    Confidence 0.0/0.0 means "unknown" (aligner didn't report) → never dropped.
    """
    if not (seg.text or "").strip():
        return False
    if seg.no_speech_prob <= 0.0 and seg.avg_logprob >= 0.0:
        return False
    return (
        float(seg.no_speech_prob) >= float(no_speech_prob)
        and float(seg.avg_logprob) <= float(min_logprob)
    )


def merge_timed_segments(
    segments: list[TimedSegment],
    *,
    min_gap: float = 0.12,
) -> list[TimedSegment]:
    """Склеивает пересекающиеся/дублирующиеся куски после чанкового STT."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    merged: list[TimedSegment] = [ordered[0]]
    for seg in ordered[1:]:
        prev = merged[-1]
        overlap = prev.end - seg.start
        same_text = prev.text.strip().lower() == seg.text.strip().lower()
        if overlap > 0.4 and same_text:
            # почти полный дубль из overlap-окна
            if seg.end > prev.end:
                prev.end = seg.end
                prev.rms = max(prev.rms, seg.rms)
                if getattr(seg, "words", None):
                    prev.words = list(prev.words or []) + list(seg.words)
            continue
        if seg.start <= prev.end + min_gap and not same_text:
            # соседние фразы — оставляем обе, чуть подрезаем старт
            if seg.start < prev.end:
                seg.start = prev.end
            if seg.end <= seg.start + 0.05:
                continue
            merged.append(seg)
            continue
        if seg.start <= prev.end + min_gap and same_text:
            prev.end = max(prev.end, seg.end)
            prev.rms = max(prev.rms, seg.rms)
            if getattr(seg, "words", None):
                prev.words = list(prev.words or []) + list(seg.words)
            continue
        merged.append(seg)
    return merged


def _dedupe_words(
    words: list[tuple[str, float, float]],
) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    for token, start, end in words:
        token = (token or "").strip()
        if not token:
            continue
        start_f = float(start)
        end_f = max(start_f + 0.04, float(end))
        if out and abs(start_f - out[-1][1]) < 0.04 and token.lower() == out[-1][0].lower():
            out[-1] = (out[-1][0], out[-1][1], max(out[-1][2], end_f))
            continue
        out.append((token, start_f, end_f))
    return out


def merge_micro_segments(
    segments: list[TimedSegment],
    *,
    min_duration: float = 0.65,
    max_gap: float = 0.28,
) -> list[TimedSegment]:
    """Склеивает слишком короткие соседние фразы — иначе XTTS звучит рублено."""
    if len(segments) < 2:
        return segments
    min_duration = max(0.35, float(min_duration))
    max_gap = max(0.08, float(max_gap))
    out: list[TimedSegment] = []
    buf: TimedSegment | None = None
    for seg in segments:
        if buf is None:
            buf = TimedSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                style=seg.style,
                rms=seg.rms,
                words=list(seg.words or []),
            )
            continue
        gap = float(seg.start) - float(buf.end)
        if buf.duration < min_duration and gap <= max_gap:
            buf.end = max(buf.end, seg.end)
            buf.text = f"{buf.text} {seg.text}".strip()
            buf.rms = max(buf.rms, seg.rms)
            buf.words = list(buf.words or []) + list(seg.words or [])
            if seg.style in {"question", "expressive"}:
                buf.style = seg.style
            continue
        out.append(buf)
        buf = TimedSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            style=seg.style,
            rms=seg.rms,
            words=list(seg.words or []),
        )
    if buf is not None:
        out.append(buf)
    return out


def split_long_timed_segments(
    segments: list[TimedSegment],
    *,
    max_sec: float = 4.5,
    min_gap: float = 0.20,
    min_part_sec: float = 0.55,
    soft_gap: float = 0.12,
) -> list[TimedSegment]:
    """Дробит длинные STT-фразы по паузам между словами — иначе дубляж не попадает в губы."""
    max_sec = max(1.2, float(max_sec))
    min_gap = max(0.08, float(min_gap))
    min_part = max(0.35, float(min_part_sec))
    soft_gap = max(0.06, min(float(soft_gap), min_gap))
    result: list[TimedSegment] = []
    for seg in segments:
        words = _dedupe_words(list(seg.words or []))
        if seg.duration <= max_sec or len(words) < 3:
            result.append(seg)
            continue
        cuts: list[int] = []
        for i in range(len(words) - 1):
            gap = float(words[i + 1][1]) - float(words[i][2])
            if gap >= min_gap:
                cuts.append(i + 1)
        if not cuts:
            for i in range(len(words) - 1):
                gap = float(words[i + 1][1]) - float(words[i][2])
                if gap >= soft_gap:
                    cuts.append(i + 1)
        if not cuts:
            # равномерная нарезка по ~max_sec
            target = max(2, int(math.ceil(seg.duration / max_sec)))
            step = max(1, len(words) // target)
            cuts = list(range(step, len(words), step))
        bounds = [0] + cuts + [len(words)]
        parts: list[TimedSegment] = []
        for a, b in zip(bounds, bounds[1:]):
            chunk = words[a:b]
            if not chunk:
                continue
            start = float(chunk[0][1])
            end = float(chunk[-1][2])
            if end <= start + 0.05:
                continue
            if parts and (end - start) < min_part:
                prev = parts[-1]
                prev.end = max(prev.end, end)
                prev.words = list(prev.words or []) + chunk
                prev.text = " ".join(w[0] for w in prev.words).strip()
                continue
            text = " ".join(w[0] for w in chunk).strip()
            if not text:
                continue
            parts.append(
                TimedSegment(
                    start=start,
                    end=end,
                    text=text,
                    style=seg.style,
                    rms=seg.rms,
                    words=chunk,
                )
            )
        if len(parts) <= 1:
            result.append(seg)
        else:
            result.extend(parts)
    return result


_ACCESS_VIOLATION = 0xC0000005
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TranscriptionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._isolated_ok = False
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self.work_root = settings.data_dir / "tmp" / "stt"
        # Whisper language before digit/RU normalize (needed for EN→RU countdown).
        self.last_video_stt_language: str | None = None

    def _segments_from_dicts(self, rows: list[dict[str, Any]]) -> list[TimedSegment]:
        out: list[TimedSegment] = []
        for row in rows:
            words_raw = row.get("words") or []
            words: list[tuple[str, float, float]] = []
            for item in words_raw:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    words.append((str(item[0]), float(item[1]), float(item[2])))
            out.append(
                TimedSegment(
                    start=float(row["start"]),
                    end=float(row["end"]),
                    text=str(row.get("text") or ""),
                    style=str(row.get("style") or "neutral"),
                    words=words,
                    no_speech_prob=float(row.get("no_speech_prob") or 0.0),
                    avg_logprob=float(row.get("avg_logprob") or 0.0),
                )
            )
        return out

    def _run_stt_worker(self, job: dict[str, Any], workdir: Path) -> dict[str, Any]:
        """Whisper живёт в отдельном процессе: ctranslate2 + torch/CUDA в одном Python падает AV."""
        workdir.mkdir(parents=True, exist_ok=True)
        job_path = workdir / "stt_job.json"
        out_path = workdir / "stt_out.json"
        attempts: list[dict[str, Any]] = [dict(job)]
        # WhisperX on long Windows jobs can ACCESS_VIOLATION; retry faster-whisper.
        if (
            bool(job.get("for_video"))
            and str(job.get("aligner") or "").lower().strip() == "whisperx"
        ):
            fw = dict(job)
            fw["aligner"] = "whisper"
            attempts.append(fw)

        last_code = 0
        last_err = ""
        for attempt_i, attempt_job in enumerate(attempts):
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            job_path.write_text(
                json.dumps(attempt_job, ensure_ascii=False), encoding="utf-8"
            )
            env = os.environ.copy()
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env["PYTHONUNBUFFERED"] = "1"
            env["CUDA_VISIBLE_DEVICES"] = ""
            flags = 0
            if os.name == "nt":
                flags = subprocess.CREATE_NO_WINDOW
            logger.info(
                "STT worker start mode=%s aligner=%s attempt=%d/%d",
                attempt_job.get("mode") or "timed",
                attempt_job.get("aligner") or "whisper",
                attempt_i + 1,
                len(attempts),
            )
            # Long videos: WhisperX can take >15 min; allow up to 30 min.
            timeout_sec = 1800 if bool(attempt_job.get("for_video")) else 900
            proc = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "app.services.stt_worker",
                    str(job_path),
                    str(out_path),
                ],
                cwd=str(_PROJECT_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                creationflags=flags,
            )
            if proc.stdout:
                logger.info("STT worker stdout:\n%s", proc.stdout[-6000:])
            if proc.stderr:
                logger.warning("STT worker stderr:\n%s", proc.stderr[-4000:])
            code = proc.returncode or 0
            last_code = code
            last_err = (proc.stderr or proc.stdout or "").strip()[-800:]
            if code == 0 and out_path.is_file():
                payload = json.loads(out_path.read_text(encoding="utf-8"))
                if payload.get("ok"):
                    self._isolated_ok = True
                    return payload
                last_err = str(payload.get("error") or "ошибка Whisper")
                logger.warning("STT worker returned ok=false: %s", last_err[:300])
                continue
            if (code & 0xFFFFFFFF) == _ACCESS_VIOLATION:
                logger.warning(
                    "STT worker ACCESS_VIOLATION (aligner=%s); %s",
                    attempt_job.get("aligner"),
                    "retrying with faster-whisper"
                    if attempt_i + 1 < len(attempts)
                    else "no more retries",
                )
                continue
            if attempt_i + 1 < len(attempts):
                logger.warning(
                    "STT worker failed code=%s; retrying next aligner", code
                )
                continue
            break

        if (last_code & 0xFFFFFFFF) == _ACCESS_VIOLATION:
            raise TranscriptionError(
                "Распознавание аварийно завершилось. Отправьте видео ещё раз."
            )
        if last_err:
            raise TranscriptionError(f"Whisper упал (код {last_code}): {last_err}")
        raise TranscriptionError("Whisper не вернул результат")

    @property
    def loaded(self) -> bool:
        return self._model is not None or self._isolated_ok

    async def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is not None:
                return self._model

            def load() -> Any:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise TranscriptionError(
                        "faster-whisper не установлен: pip install -r requirements.txt"
                    ) from exc
                return WhisperModel(
                    self.settings.stt_model_size,
                    device=self.settings.stt_device,
                    compute_type=self.settings.stt_compute_type,
                )

            logger.info(
                "Загрузка faster-whisper %s (%s/%s)...",
                self.settings.stt_model_size,
                self.settings.stt_device,
                self.settings.stt_compute_type,
            )
            try:
                self._model = await asyncio.to_thread(load)
            except TranscriptionError:
                raise
            except Exception as exc:
                raise TranscriptionError(
                    f"Не удалось загрузить faster-whisper: {exc}"
                ) from exc
            logger.info("faster-whisper загружен")
            return self._model

    def _transcribe_file(self, path: Path) -> str:
        assert self._model is not None
        segments, _info = self._model.transcribe(
            str(path),
            language=self.settings.stt_language,
            beam_size=self.settings.stt_beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def _normalize_stt_language(self, language: str | None) -> str | None:
        raw = (language or "").strip().lower()
        if not raw or raw in {"auto", "detect"}:
            return None
        return raw

    def _probe_language(
        self,
        audio: np.ndarray,
        sample_rate: int,
        workdir: Path,
    ) -> str:
        assert self._model is not None
        probe_sec = min(45.0, max(8.0, len(audio) / float(sample_rate)))
        n = int(probe_sec * sample_rate)
        path = workdir / "lang_probe.wav"
        sf.write(str(path), audio[:n], sample_rate, subtype="PCM_16")
        try:
            _, info = self._model.transcribe(
                str(path),
                language=None,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
                task="transcribe",
            )
            lang = (getattr(info, "language", None) or "en").lower()
            prob = float(getattr(info, "language_probability", 0.0) or 0.0)
            logger.info("Video STT detected language=%s (p=%.2f)", lang, prob)
            return lang
        finally:
            path.unlink(missing_ok=True)

    def _style_for_text(self, text: str) -> str:
        if text.endswith(("?", "？")):
            return "question"
        if text.endswith(("!", "！")):
            return "expressive"
        return "neutral"

    def _transcribe_timed(
        self,
        path: Path,
        language: str | None = None,
        *,
        beam_size: int | None = None,
        vad_filter: bool = True,
        no_speech_threshold: float | None = None,
        time_offset: float = 0.0,
        condition_on_previous_text: bool = False,
    ) -> list[TimedSegment]:
        assert self._model is not None
        if language is None:
            resolved = self._normalize_stt_language(self.settings.stt_language)
        else:
            resolved = self._normalize_stt_language(language)
        kwargs: dict[str, Any] = {
            "language": resolved,
            "beam_size": beam_size or self.settings.stt_beam_size,
            "vad_filter": vad_filter,
            "word_timestamps": True,
            "condition_on_previous_text": condition_on_previous_text,
            "task": "transcribe",
        }
        if no_speech_threshold is not None:
            kwargs["no_speech_threshold"] = no_speech_threshold
        if not vad_filter:
            # для песен/музыки VAD выбрасывает почти всё — не режем
            kwargs["vad_filter"] = False
        segments, _info = self._model.transcribe(str(path), **kwargs)
        out: list[TimedSegment] = []
        for segment in segments:
            text = (segment.text or "").strip()
            if not text:
                continue
            # типичные whisper-галлюцинации на тишине/музыке
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
                    style=self._style_for_text(text),
                    words=words,
                    no_speech_prob=float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
                    avg_logprob=float(getattr(segment, "avg_logprob", 0.0) or 0.0),
                )
            )
        return out

    def _transcribe_timed_chunked(
        self,
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
        locked = self._normalize_stt_language(language)
        if locked is None:
            locked = self._probe_language(audio, sample_rate, workdir)
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
        from app.audio.preprocess import boost_quiet_stt_audio

        for idx, start_sec in enumerate(starts):
            end_sec = min(duration, start_sec + chunk_sec)
            a = int(start_sec * sample_rate)
            b = int(end_sec * sample_rate)
            if b - a < sample_rate // 2:
                continue
            chunk = np.asarray(audio[a:b], dtype=np.float32)
            if quiet_audio:
                chunk = boost_quiet_stt_audio(
                    chunk, sample_rate, target_db=float(quiet_target_db)
                )
            chunk_path = workdir / f"vid_chunk_{idx:03d}.wav"
            sf.write(str(chunk_path), chunk, sample_rate, subtype="PCM_16")
            segs: list[TimedSegment] = []
            try:
                segs = self._transcribe_timed(
                    chunk_path,
                    locked,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    no_speech_threshold=no_speech_threshold,
                    time_offset=start_sec,
                    condition_on_previous_text=False,
                )
                all_segs.extend(segs)
            finally:
                chunk_path.unlink(missing_ok=True)
            logger.info(
                "Video STT chunk %d/%d (%.1f–%.1f, lang=%s quiet=%s): %d segs",
                idx + 1,
                len(starts),
                start_sec,
                end_sec,
                locked,
                quiet_audio,
                len(segs),
            )
        return merge_timed_segments(all_segs)

    async def transcribe_timed(
        self,
        input_path: Path,
        *,
        language: str | None = None,
        max_seconds: float | None = None,
        for_video: bool = False,
        quiet_audio: bool = False,
    ) -> tuple[list[TimedSegment], Path, float]:
        """Сегменты с таймкодами + wav 16 kHz и длительность исходника."""
        workdir = self.work_root / f"timed_{uuid.uuid4().hex}"
        workdir.mkdir(parents=True, exist_ok=True)
        wav_path = workdir / "source.wav"
        await asyncio.to_thread(convert_to_wav, input_path, wav_path, 16000, True)
        audio, sample_rate = await asyncio.to_thread(
            sf.read, wav_path, dtype="float32", always_2d=False
        )
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        duration = len(audio) / float(sample_rate)
        if duration <= 0:
            raise TranscriptionError("В медиа нет звука")
        if max_seconds is None:
            limit = self.settings.stt_max_voice_seconds
        elif max_seconds <= 0:
            limit = None
        else:
            limit = max_seconds
        if limit is not None and duration > limit:
            raise TranscriptionError(
                f"Слишком длинное видео ({duration:.0f}с). Лимит: {limit:.0f}с"
            )

        no_speech = float(self.settings.video_dub_stt_no_speech_threshold)
        quiet_target_db = float(self.settings.video_dub_quiet_stt_target_db)
        min_quiet_coverage = float(self.settings.video_dub_quiet_stt_min_coverage)
        if quiet_audio and for_video:
            from app.audio.preprocess import boost_quiet_stt_audio

            before_rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
            try:
                await asyncio.to_thread(apply_quiet_stt_ffmpeg_boost, wav_path)
                audio, sample_rate = await asyncio.to_thread(
                    sf.read, wav_path, dtype="float32", always_2d=False
                )
                audio = np.asarray(audio, dtype=np.float32)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
            except Exception:
                logger.exception("FFmpeg quiet boost failed — numpy speech boost only")
            audio = boost_quiet_stt_audio(
                audio, int(sample_rate), target_db=quiet_target_db
            )
            await asyncio.to_thread(
                sf.write, str(wav_path), audio, int(sample_rate), "PCM_16"
            )
            after_rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
            no_speech = min(
                no_speech,
                float(self.settings.video_dub_quiet_stt_no_speech_threshold),
            )
            logger.info(
                "Quiet STT prep: rms %.5f → %.5f (target_db=%.1f, no_speech=%.2f)",
                before_rms,
                after_rms,
                quiet_target_db,
                no_speech,
            )

        self.last_video_stt_language = None
        aligner = str(self.settings.video_dub_stt_aligner or "whisper").lower().strip()
        use_inprocess = self._model is not None and hasattr(self._model, "transcribe")
        job: dict[str, Any] | None = None

        def _speech_sec(segs: list[TimedSegment]) -> float:
            return sum(float(s.duration) for s in segs)

        async def _run_stt(
            *,
            audio_arr: np.ndarray,
            wav: Path,
            no_speech_thr: float,
            quiet: bool,
            target_db: float,
            force_aligner: str | None = None,
        ) -> tuple[list[TimedSegment], dict[str, Any] | None]:
            nonlocal use_inprocess
            use_aligner = (force_aligner or aligner).lower().strip()
            local_job: dict[str, Any] | None = None
            if use_inprocess:
                await self._ensure_loaded()
                async with self._inference_lock:
                    if for_video:
                        segs = await asyncio.to_thread(
                            self._transcribe_timed_chunked,
                            audio_arr,
                            sample_rate,
                            workdir,
                            language,
                            beam_size=int(self.settings.video_dub_stt_beam_size),
                            vad_filter=bool(self.settings.video_dub_stt_vad),
                            no_speech_threshold=no_speech_thr,
                            chunk_sec=float(self.settings.video_dub_stt_chunk_sec),
                            overlap_sec=float(self.settings.video_dub_stt_overlap_sec),
                            quiet_audio=quiet,
                            quiet_target_db=target_db,
                        )
                    else:
                        segs = await asyncio.to_thread(
                            self._transcribe_timed, wav, language
                        )
                return segs, None
            local_job = {
                "mode": "timed",
                "wav_path": str(wav),
                "model_size": (
                    str(
                        getattr(
                            self.settings,
                            "video_dub_stt_model_size",
                            self.settings.stt_model_size,
                        )
                    )
                    if for_video
                    else self.settings.stt_model_size
                ),
                "device": self.settings.stt_device,
                "compute_type": self.settings.stt_compute_type,
                "cpu_threads": 1,
                "language": language or self.settings.video_dub_stt_language,
                "for_video": bool(for_video),
                "aligner": use_aligner,
                "beam_size": int(
                    self.settings.video_dub_stt_beam_size
                    if for_video
                    else self.settings.stt_beam_size
                ),
                "vad_filter": bool(self.settings.video_dub_stt_vad)
                if for_video
                else True,
                "no_speech_threshold": no_speech_thr,
                "chunk_sec": float(self.settings.video_dub_stt_chunk_sec),
                "overlap_sec": float(self.settings.video_dub_stt_overlap_sec),
                "whisperx_chunk_sec": float(
                    self.settings.video_dub_whisperx_chunk_sec
                ),
                "quiet_audio": bool(quiet),
                "quiet_target_db": float(target_db),
            }
            payload = await asyncio.to_thread(self._run_stt_worker, local_job, workdir)
            segs = self._segments_from_dicts(list(payload.get("segments") or []))
            raw_lang = str(payload.get("language") or "").lower().strip()
            if raw_lang:
                self.last_video_stt_language = raw_lang
            return segs, local_job

        segments, job = await _run_stt(
            audio_arr=audio,
            wav=wav_path,
            no_speech_thr=no_speech,
            quiet=bool(quiet_audio and for_video),
            target_db=quiet_target_db,
            # WhisperX forced-align + VAD often drops whisper-quiet speech.
            force_aligner="whisper" if (quiet_audio and for_video) else None,
        )
        if for_video:
            # Capture source language from raw STT text *before* digit → RU rewrite.
            if not self.last_video_stt_language:
                from app.text.language import detect_transcript_language

                pre_blob = " ".join((s.text or "") for s in segments[:80])
                self.last_video_stt_language = detect_transcript_language(
                    pre_blob, default="en"
                )
            logger.info(
                "Video STT source language=%s (pre-normalize)",
                self.last_video_stt_language,
            )
            speech_raw = _speech_sec(segments)
            # WhisperX/VAD иногда «съедает» речь — тогда таймлайн ломается.
            if (
                job is not None
                and str(aligner).lower() == "whisperx"
                and duration > 3.0
                and speech_raw < max(1.2, duration * 0.22)
            ):
                logger.warning(
                    "WhisperX coverage too low (%.1fs / %.1fs) — fallback faster-whisper",
                    speech_raw,
                    duration,
                )
                try:
                    alt, _ = await _run_stt(
                        audio_arr=audio,
                        wav=wav_path,
                        no_speech_thr=no_speech,
                        quiet=bool(quiet_audio),
                        target_db=quiet_target_db,
                        force_aligner="whisper",
                    )
                    if _speech_sec(alt) > speech_raw * 1.15:
                        segments = alt
                        speech_raw = _speech_sec(segments)
                except Exception:
                    logger.exception("faster-whisper fallback failed")

            # Quiet/ASMR: if still almost empty (only countdown digits etc.), retry harder.
            if (
                quiet_audio
                and duration > 8.0
                and speech_raw < max(2.0, duration * min_quiet_coverage)
            ):
                from app.audio.preprocess import boost_quiet_stt_audio

                hard_db = min(-9.0, quiet_target_db + 3.0)
                hard_no_speech = min(0.08, no_speech)
                logger.warning(
                    "Quiet STT coverage low (%.1fs / %.1fs=%.0f%%) — hard retry "
                    "(target_db=%.1f no_speech=%.2f)",
                    speech_raw,
                    duration,
                    100.0 * speech_raw / max(duration, 1e-6),
                    hard_db,
                    hard_no_speech,
                )
                retry_audio = boost_quiet_stt_audio(
                    audio, int(sample_rate), target_db=hard_db, max_gain_db=48.0
                )
                retry_wav = workdir / "source_quiet_retry.wav"
                await asyncio.to_thread(
                    sf.write, str(retry_wav), retry_audio, int(sample_rate), "PCM_16"
                )
                try:
                    alt, _ = await _run_stt(
                        audio_arr=retry_audio,
                        wav=retry_wav,
                        no_speech_thr=hard_no_speech,
                        quiet=True,
                        target_db=hard_db,
                        force_aligner="whisper",
                    )
                    if _speech_sec(alt) > speech_raw * 1.2:
                        segments = alt
                        audio = retry_audio
                        wav_path = retry_wav
                        speech_raw = _speech_sec(segments)
                        logger.info(
                            "Quiet hard retry kept: %.1fs speech / %d cues",
                            speech_raw,
                            len(segments),
                        )
                except Exception:
                    logger.exception("Quiet STT hard retry failed")

            from app.services.timeline_align import rebuild_video_dub_segments

            before = len(segments)
            segments = rebuild_video_dub_segments(
                segments,
                min_pause_sec=float(self.settings.video_dub_cue_min_pause_sec),
                max_cue_sec=float(self.settings.video_dub_cue_max_sec),
                min_cue_sec=float(self.settings.video_dub_cue_min_sec),
                media_duration=duration,
            )
            if len(segments) != before:
                logger.info(
                    "Video dub cues (neural align): %d → %d phrases",
                    before,
                    len(segments),
                )
            # Drop Whisper hallucinations on moans/noise (sounds that aren't real
            # speech): high no_speech_prob + low avg_logprob. Skipped in quiet
            # mode — whispered speech legitimately scores high no_speech.
            if not quiet_audio:
                from app.text.digit_speech import is_digit_like_text as _is_digit

                drop_p = float(
                    getattr(self.settings, "video_dub_drop_no_speech_prob", 0.6)
                )
                drop_lp = float(
                    getattr(self.settings, "video_dub_drop_min_logprob", -1.0)
                )
                kept_segs: list[TimedSegment] = []
                dropped_halluc = 0
                for s in segments:
                    if is_probable_non_speech(
                        s, no_speech_prob=drop_p, min_logprob=drop_lp
                    ) and not _is_digit(s.text or ""):
                        dropped_halluc += 1
                        logger.info(
                            "Drop hallucinated cue %.2f-%.2f "
                            "(no_speech=%.2f logprob=%.2f): %s",
                            s.start,
                            s.end,
                            s.no_speech_prob,
                            s.avg_logprob,
                            (s.text or "")[:60],
                        )
                        continue
                    kept_segs.append(s)
                if dropped_halluc:
                    segments = kept_segs
                    logger.info(
                        "Dropped %d hallucinated non-speech cues", dropped_halluc
                    )
            # Sparse ASMR countdown: Whisper packs words into the first seconds;
            # snap digits onto real speech-energy peaks across the whole clip.
            # Skip when quiet mode still has tiny coverage — likely missed speech,
            # not a pure countdown video.
            from app.text.digit_speech import (
                is_digit_like_text,
                looks_like_countdown,
                snap_countdown_cues_to_energy,
            )

            coverage = _speech_sec(segments) / max(duration, 1e-6)
            digit_n = sum(1 for s in segments if is_digit_like_text(s.text or ""))
            pure_digits = (
                digit_n >= max(3, int(0.85 * len(segments))) if segments else False
            )
            allow_countdown = looks_like_countdown(segments) and (
                not quiet_audio
                or coverage >= min_quiet_coverage
                or pure_digits
            )
            if looks_like_countdown(segments) and not allow_countdown:
                logger.warning(
                    "Skip countdown snap on quiet/low-coverage result "
                    "(coverage=%.1f%% digits=%d/%d) — keeping raw STT cues",
                    100.0 * coverage,
                    digit_n,
                    len(segments),
                )
            elif allow_countdown:
                segments = snap_countdown_cues_to_energy(
                    segments, audio, sample_rate
                )
            # Шёпот/ASMR: ASR режет хвосты слов — удлиняем end по –40 dB от пика
            from app.audio.envelope_align import refine_segments_by_envelope

            segments = refine_segments_by_envelope(
                segments, audio, sample_rate, rel_db=-40.0
            )
        for seg in segments:
            a = int(seg.start * sample_rate)
            b = int(seg.end * sample_rate)
            clip = audio[max(0, a) : min(len(audio), max(a + 1, b))]
            if clip.size:
                seg.rms = float(np.sqrt(np.mean(np.square(clip))))
                if seg.style == "neutral" and seg.rms > 0.08:
                    seg.style = "expressive"
                elif seg.style == "neutral" and seg.rms < 0.02:
                    seg.style = "calm"
        from app.text.ssml import enrich_segments_ssml

        enrich_segments_ssml(segments, audio, sample_rate)
        speech_sec = sum(s.duration for s in segments)
        logger.info(
            "Timed STT done: %.1fs media → %d phrases / %.1fs speech (video=%s)",
            duration,
            len(segments),
            speech_sec,
            for_video,
        )
        return segments, wav_path, duration

    async def transcribe_chunks(
        self,
        input_path: Path,
    ) -> AsyncIterator[TranscriptionUpdate]:
        workdir = self.work_root / uuid.uuid4().hex
        workdir.mkdir(parents=True, exist_ok=True)
        wav_path = workdir / "source.wav"
        try:
            await asyncio.to_thread(
                convert_to_wav,
                input_path,
                wav_path,
                16000,
                True,
            )
            audio, sample_rate = await asyncio.to_thread(
                sf.read,
                wav_path,
                dtype="float32",
                always_2d=False,
            )
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            duration = len(audio) / float(sample_rate)
            if duration <= 0:
                raise TranscriptionError("Голосовое сообщение пустое")
            if duration > self.settings.stt_max_voice_seconds:
                raise TranscriptionError(
                    f"Голосовое слишком длинное ({duration:.0f}с). "
                    f"Лимит: {self.settings.stt_max_voice_seconds:.0f}с"
                )

            use_inprocess = self._model is not None and hasattr(self._model, "transcribe")
            if not use_inprocess:
                job = {
                    "mode": "chunks",
                    "wav_path": str(wav_path),
                    "model_size": self.settings.stt_model_size,
                    "device": self.settings.stt_device,
                    "compute_type": self.settings.stt_compute_type,
                    "cpu_threads": 1,
                    "chunk_seconds": float(self.settings.stt_chunk_seconds),
                    "stt_language": self.settings.stt_language,
                    "beam_size": int(self.settings.stt_beam_size),
                }
                payload = await asyncio.to_thread(self._run_stt_worker, job, workdir)
                parts = [str(p) for p in payload.get("parts") or [] if str(p).strip()]
                transcript_parts: list[str] = []
                total = max(1, len(parts))
                for index, part in enumerate(parts, start=1):
                    transcript_parts.append(part)
                    yield TranscriptionUpdate(
                        part=part,
                        full_text=" ".join(transcript_parts),
                        chunk_index=index,
                        chunk_count=total,
                    )
                return

            await self._ensure_loaded()
            chunk_samples = max(
                sample_rate,
                int(sample_rate * self.settings.stt_chunk_seconds),
            )
            chunk_count = max(1, math.ceil(len(audio) / chunk_samples))
            transcript_parts: list[str] = []

            for index in range(chunk_count):
                start = index * chunk_samples
                end = min(start + chunk_samples, len(audio))
                chunk = audio[start:end]
                chunk_path = workdir / f"chunk_{index:04d}.wav"
                await asyncio.to_thread(
                    sf.write,
                    chunk_path,
                    chunk,
                    sample_rate,
                    subtype="PCM_16",
                )
                async with self._inference_lock:
                    part = await asyncio.to_thread(
                        self._transcribe_file,
                        chunk_path,
                    )
                chunk_path.unlink(missing_ok=True)
                if not part:
                    continue
                transcript_parts.append(part)
                yield TranscriptionUpdate(
                    part=part,
                    full_text=" ".join(transcript_parts),
                    chunk_index=index + 1,
                    chunk_count=chunk_count,
                )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
